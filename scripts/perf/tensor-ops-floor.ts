/**
 * The per-op overhead floor of the tensor library at the sizes of a quadrature grid (a pair grid has about 4e3 nodes, a triple grid
 * 3e4, a fine experiment 1e5): single tensor ops against the equivalent raw typed-array loop (which allocates its output, as the
 * tensor op does, and is the best a hand-written kernel does here). The rows are the elementwise ops, the broadcast layouts of a
 * grid (outer, row against matrix, column against matrix, trailing and leading axis, rank-3 grids), comparisons and `where`,
 * reductions (flat, along an axis, `logsumexp`, `softmax`, `argmax`) and the gather and scatter family. Part of the tensorisation
 * study (`docs/probprog/ep-perf-tensor-study.md`); an experiment, not wired into the library.
 *
 *   nice -n 15 node --conditions=source --experimental-transform-types --no-warnings scripts/perf/tensor-ops-floor.ts [--only <substring>] [--json <file>]
 */
import { writeFileSync } from 'node:fs'
import {
  add,
  arange,
  argmax,
  cast,
  div,
  exp,
  greater,
  less,
  linspace,
  log,
  logsumexp,
  max,
  maximum,
  min,
  minimum,
  mod,
  mul,
  oneHot,
  pow,
  reshape,
  scatterAdd,
  segmentSum,
  sin,
  slice,
  softmax,
  sub,
  sum,
  type Tensor,
  take,
  takeAlongAxis,
  where,
} from '../../packages/engine/src/lib/tensor/index.ts'
import { summarise } from '../ep-perf-stats.ts'
import { bench, calibrate } from './timing.ts'

const SIZES = [1e3, 1e4, 1e5]

interface Row {
  readonly op: string
  readonly n: number
  readonly tensorUs: number
  readonly rawUs: number
  readonly ratio: number
  readonly tensorSpread: number
  readonly rawSpread: number
}

interface Case {
  readonly op: string
  readonly tensor: () => unknown
  readonly raw: () => unknown
}

const data = (t: unknown): Float64Array => (t as Tensor).data as Float64Array
const at = (a: Float64Array, i: number): number => a[i] as number

/** The ops of the outer-layout rows, each with a loop of its own below (a closure per element would make the reference slow). */
type OuterOp = 'add' | 'sub' | 'mul' | 'div' | 'max' | 'min' | 'pow'

/** A raw loop for `[s,1] op [1,s]` (outer): one hand-written loop per op, the best a kernel does. */
function outerRaw(C: Float64Array, R: Float64Array, side: number, op: OuterOp): () => unknown {
  return () => {
    const o = new Float64Array(side * side)
    for (let i = 0; i < side; i++) {
      const x = at(C, i)
      const base = i * side
      if (op === 'add') for (let j = 0; j < side; j++) o[base + j] = x + at(R, j)
      else if (op === 'sub') for (let j = 0; j < side; j++) o[base + j] = x - at(R, j)
      else if (op === 'mul') for (let j = 0; j < side; j++) o[base + j] = x * at(R, j)
      else if (op === 'div') for (let j = 0; j < side; j++) o[base + j] = x / at(R, j)
      else if (op === 'max') for (let j = 0; j < side; j++) o[base + j] = Math.max(x, at(R, j))
      else if (op === 'min') for (let j = 0; j < side; j++) o[base + j] = Math.min(x, at(R, j))
      else for (let j = 0; j < side; j++) o[base + j] = x === 1 || at(R, j) === 0 ? 1 : x ** at(R, j)
    }
    return o
  }
}

function cases(n: number): Case[] {
  // Inputs are made with the tensor library: a smooth positive vector, a second one, and a permutation-like int32 index.
  const a = add(mul(sin(linspace(0, 20, n)), 0.4), 1.5)
  const b = add(mul(sin(linspace(1, 30, n)), 0.3), 1.2)
  const idx = cast(mod(mul(arange(n), 7919), n), 'int32')
  const A = data(a)
  const B = data(b)
  const I = (idx as Tensor).data as Int32Array
  // The grid shape: an n0 x n1 grid of about n nodes, as a column against a row.
  const side = Math.round(Math.sqrt(n))
  const col = reshape(slice(a, [0], [side]), [side, 1])
  const row = reshape(slice(b, [0], [side]), [1, side])
  const grid = add(col, row)
  const C = data(col)
  const R = data(row)
  const G = data(grid)
  // A rank-3 grid of about n nodes.
  const c3 = Math.floor(n ** (1 / 3) + 1e-9)
  const total3 = c3 * c3 * c3
  const cube = reshape(slice(add(a, 0), [0], [total3]), [c3, c3, c3])
  const Q = data(cube)
  const vec3 = slice(b, [0], [c3])
  const V3 = data(vec3)
  const a3 = reshape(vec3, [c3, 1, 1])
  const b3 = reshape(vec3, [1, c3, 1])
  const c3t = reshape(vec3, [1, 1, c3])
  const a3c3 = add(a3, c3t)
  const AC = data(a3c3)
  const rowMask = greater(grid, 2.7)
  const M = (rowMask as Tensor).data as Uint8Array
  const rowIdx = cast(mod(mul(arange(side), 37), side), 'int32')
  const RI = (rowIdx as Tensor).data as Int32Array
  const alongIdx = cast(mod(mul(arange(side * side), 7), side), 'int32')
  const AI = (alongIdx as Tensor).data as Int32Array
  const labels = cast(mod(mul(arange(side), 7), side), 'int32')
  const LB = (labels as Tensor).data as Int32Array
  const out: Case[] = [
    // ---- broadcast binary ops, the outer layout [s,1] op [1,s]
    { op: 'add broadcast ([s,1] + [1,s])', tensor: () => add(col, row), raw: outerRaw(C, R, side, 'add') },
    { op: 'sub broadcast ([s,1] - [1,s])', tensor: () => sub(col, row), raw: outerRaw(C, R, side, 'sub') },
    { op: 'mul broadcast ([s,1] * [1,s])', tensor: () => mul(col, row), raw: outerRaw(C, R, side, 'mul') },
    { op: 'div broadcast ([s,1] / [1,s])', tensor: () => div(col, row), raw: outerRaw(C, R, side, 'div') },
    {
      op: 'maximum broadcast ([s,1], [1,s])',
      tensor: () => maximum(col, row),
      raw: outerRaw(C, R, side, 'max'),
    },
    {
      op: 'minimum broadcast ([s,1], [1,s])',
      tensor: () => minimum(col, row),
      raw: outerRaw(C, R, side, 'min'),
    },
    {
      op: 'pow broadcast ([s,1] ** [1,s])',
      tensor: () => pow(col, row),
      raw: outerRaw(C, R, side, 'pow'),
    },
    // ---- the other layouts
    {
      op: 'add row against matrix ([s,s] + [1,s])',
      tensor: () => add(grid, row),
      raw: () => {
        const o = new Float64Array(side * side)
        for (let i = 0; i < side; i++) for (let j = 0; j < side; j++) o[i * side + j] = at(G, i * side + j) + at(R, j)
        return o
      },
    },
    {
      op: 'add column against matrix ([s,s] + [s,1])',
      tensor: () => add(grid, col),
      raw: () => {
        const o = new Float64Array(side * side)
        for (let i = 0; i < side; i++) {
          const x = at(C, i)
          for (let j = 0; j < side; j++) o[i * side + j] = at(G, i * side + j) + x
        }
        return o
      },
    },
    {
      op: 'add trailing axis ([c,c,c] + [c])',
      tensor: () => add(cube, vec3),
      raw: () => {
        const o = new Float64Array(total3)
        for (let r = 0; r < c3 * c3; r++) for (let k = 0; k < c3; k++) o[r * c3 + k] = at(Q, r * c3 + k) + at(V3, k)
        return o
      },
    },
    {
      op: 'add leading axis ([c,c,c] + [c,1,1])',
      tensor: () => add(cube, a3),
      raw: () => {
        const o = new Float64Array(total3)
        for (let i = 0; i < c3; i++) {
          const x = at(V3, i)
          for (let r = 0; r < c3 * c3; r++) o[i * c3 * c3 + r] = at(Q, i * c3 * c3 + r) + x
        }
        return o
      },
    },
    {
      op: 'add rank-3 grid ([c,1,1] + [1,1,c])',
      tensor: () => add(a3, c3t),
      raw: () => {
        const o = new Float64Array(c3 * c3)
        for (let i = 0; i < c3; i++) {
          const x = at(V3, i)
          for (let k = 0; k < c3; k++) o[i * c3 + k] = x + at(V3, k)
        }
        return o
      },
    },
    {
      op: 'add rank-3 grid ([c,1,c] + [1,c,1])',
      tensor: () => add(a3c3, b3),
      raw: () => {
        const o = new Float64Array(total3)
        for (let i = 0; i < c3; i++)
          for (let j = 0; j < c3; j++) {
            const y = at(V3, j)
            for (let k = 0; k < c3; k++) o[(i * c3 + j) * c3 + k] = at(AC, i * c3 + k) + y
          }
        return o
      },
    },
    {
      op: 'add rank-3 grid ([c,c,1] + [1,1,c])',
      tensor: () => add(reshape(slice(cube, [0, 0, 0], [c3, c3, 1]), [c3, c3, 1]), c3t),
      raw: () => {
        const o = new Float64Array(c3 * c3 * c3)
        for (let r = 0; r < c3 * c3; r++) {
          const x = at(Q, r * c3)
          for (let k = 0; k < c3; k++) o[r * c3 + k] = x + at(V3, k)
        }
        return o
      },
    },
    {
      op: 'less broadcast ([s,1] < [1,s])',
      tensor: () => less(col, row),
      raw: () => {
        const o = new Uint8Array(side * side)
        for (let i = 0; i < side; i++) {
          const x = at(C, i)
          for (let j = 0; j < side; j++) o[i * side + j] = x < at(R, j) ? 1 : 0
        }
        return o
      },
    },
    {
      op: 'where (cond [s,s], [s,1], [1,s])',
      tensor: () => where(rowMask, col, row),
      raw: () => {
        const o = new Float64Array(side * side)
        for (let i = 0; i < side; i++) {
          const x = at(C, i)
          for (let j = 0; j < side; j++) o[i * side + j] = M[i * side + j] ? x : at(R, j)
        }
        return o
      },
    },
    // ---- reductions
    {
      op: 'sum over axis 1 of [s,s]',
      tensor: () => sum(grid, 1),
      raw: () => {
        const o = new Float64Array(side)
        for (let i = 0; i < side; i++) {
          let t = 0
          for (let j = 0; j < side; j++) t += at(G, i * side + j)
          o[i] = t
        }
        return o
      },
    },
    {
      op: 'sum over axis 0 of [s,s]',
      tensor: () => sum(grid, 0),
      raw: () => {
        const o = new Float64Array(side)
        for (let i = 0; i < side; i++) for (let j = 0; j < side; j++) o[j] = at(o, j) + at(G, i * side + j)
        return o
      },
    },
    {
      op: 'sum over axis 1 of [c,c,c]',
      tensor: () => sum(cube, 1),
      raw: () => {
        const o = new Float64Array(c3 * c3)
        for (let i = 0; i < c3; i++)
          for (let j = 0; j < c3; j++)
            for (let k = 0; k < c3; k++) o[i * c3 + k] = at(o, i * c3 + k) + at(Q, (i * c3 + j) * c3 + k)
        return o
      },
    },
    {
      op: 'max',
      tensor: () => max(a),
      raw: () => {
        let top = Number.NEGATIVE_INFINITY
        for (let i = 0; i < n; i++) if (at(A, i) > top) top = at(A, i)
        return top
      },
    },
    {
      op: 'max over axis 1 of [s,s]',
      tensor: () => max(grid, 1),
      raw: () => {
        const o = new Float64Array(side)
        for (let i = 0; i < side; i++) {
          let t = Number.NEGATIVE_INFINITY
          for (let j = 0; j < side; j++) t = Math.max(at(G, i * side + j), t)
          o[i] = t
        }
        return o
      },
    },
    {
      op: 'max over axis 0 of [s,s]',
      tensor: () => max(grid, 0),
      raw: () => {
        const o = new Float64Array(side).fill(Number.NEGATIVE_INFINITY)
        for (let i = 0; i < side; i++) for (let j = 0; j < side; j++) o[j] = Math.max(at(G, i * side + j), at(o, j))
        return o
      },
    },
    {
      op: 'min over axis 1 of [s,s]',
      tensor: () => min(grid, 1),
      raw: () => {
        const o = new Float64Array(side)
        for (let i = 0; i < side; i++) {
          let t = Number.POSITIVE_INFINITY
          for (let j = 0; j < side; j++) if (at(G, i * side + j) < t) t = at(G, i * side + j)
          o[i] = t
        }
        return o
      },
    },
    {
      op: 'max over axis 1 of [c,c,c]',
      tensor: () => max(cube, 1),
      raw: () => {
        const o = new Float64Array(c3 * c3).fill(Number.NEGATIVE_INFINITY)
        for (let i = 0; i < c3; i++)
          for (let j = 0; j < c3; j++)
            for (let k = 0; k < c3; k++) o[i * c3 + k] = Math.max(at(Q, (i * c3 + j) * c3 + k), at(o, i * c3 + k))
        return o
      },
    },
    {
      op: 'argmax over axis 1 of [s,s]',
      tensor: () => argmax(grid, 1),
      raw: () => {
        const o = new Int32Array(side)
        for (let i = 0; i < side; i++) {
          let best = 0
          let bv = at(G, i * side)
          for (let j = 1; j < side; j++)
            if (at(G, i * side + j) > bv) {
              best = j
              bv = at(G, i * side + j)
            }
          o[i] = best
        }
        return o
      },
    },
    {
      op: 'exp',
      tensor: () => exp(a),
      raw: () => {
        const o = new Float64Array(n)
        for (let i = 0; i < n; i++) o[i] = Math.exp(at(A, i))
        return o
      },
    },
    {
      op: 'log',
      tensor: () => log(a),
      raw: () => {
        const o = new Float64Array(n)
        for (let i = 0; i < n; i++) o[i] = Math.log(at(A, i))
        return o
      },
    },
    {
      op: 'add',
      tensor: () => add(a, b),
      raw: () => {
        const o = new Float64Array(n)
        for (let i = 0; i < n; i++) o[i] = at(A, i) + at(B, i)
        return o
      },
    },
    {
      op: 'mul',
      tensor: () => mul(a, b),
      raw: () => {
        const o = new Float64Array(n)
        for (let i = 0; i < n; i++) o[i] = at(A, i) * at(B, i)
        return o
      },
    },
    {
      op: 'sum',
      tensor: () => sum(a),
      raw: () => {
        let s = 0
        for (let i = 0; i < n; i++) s += at(A, i)
        return s
      },
    },
    {
      op: 'logsumexp',
      tensor: () => logsumexp(a),
      raw: () => {
        let top = Number.NEGATIVE_INFINITY
        for (let i = 0; i < n; i++) if (at(A, i) > top) top = at(A, i)
        let s = 0
        for (let i = 0; i < n; i++) s += Math.exp(at(A, i) - top)
        return top + Math.log(s)
      },
    },
    {
      op: 'logsumexp over axis 1 of [s,s]',
      tensor: () => logsumexp(grid, 1),
      raw: () => {
        const o = new Float64Array(side)
        for (let i = 0; i < side; i++) {
          let top = Number.NEGATIVE_INFINITY
          for (let j = 0; j < side; j++) if (at(G, i * side + j) > top) top = at(G, i * side + j)
          let s = 0
          for (let j = 0; j < side; j++) s += Math.exp(at(G, i * side + j) - top)
          o[i] = Math.log(s) + top
        }
        return o
      },
    },
    {
      op: 'softmax over axis 1 of [s,s]',
      tensor: () => softmax(grid, 1),
      raw: () => {
        const o = new Float64Array(side * side)
        for (let i = 0; i < side; i++) {
          let top = Number.NEGATIVE_INFINITY
          for (let j = 0; j < side; j++) if (at(G, i * side + j) > top) top = at(G, i * side + j)
          let s = 0
          for (let j = 0; j < side; j++) s += Math.exp(at(G, i * side + j) - top)
          const lse = Math.log(s) + top
          for (let j = 0; j < side; j++) o[i * side + j] = Math.exp(at(G, i * side + j) - lse)
        }
        return o
      },
    },
    // ---- gather and scatter
    {
      op: 'take (gather)',
      tensor: () => take(a, idx),
      raw: () => {
        const o = new Float64Array(n)
        for (let i = 0; i < n; i++) o[i] = at(A, I[i] as number)
        return o
      },
    },
    {
      op: 'take rows of [s,s] (axis 0, s indices)',
      tensor: () => take(grid, rowIdx, 0),
      raw: () => {
        const o = new Float64Array(side * side)
        for (let t = 0; t < side; t++) {
          const src = (RI[t] as number) * side
          for (let k = 0; k < side; k++) o[t * side + k] = at(G, src + k)
        }
        return o
      },
    },
    {
      op: 'take columns of [s,s] (axis 1, s indices)',
      tensor: () => take(grid, rowIdx, 1),
      raw: () => {
        const o = new Float64Array(side * side)
        for (let i = 0; i < side; i++)
          for (let t = 0; t < side; t++) o[i * side + t] = at(G, i * side + (RI[t] as number))
        return o
      },
    },
    {
      op: 'takeAlongAxis of [s,s] (axis 1)',
      tensor: () => takeAlongAxis(grid, reshape(alongIdx, [side, side]), 1),
      raw: () => {
        const o = new Float64Array(side * side)
        for (let i = 0; i < side; i++)
          for (let t = 0; t < side; t++) o[i * side + t] = at(G, i * side + (AI[i * side + t] as number))
        return o
      },
    },
    {
      op: 'scatterAdd (n updates into [s])',
      tensor: () => scatterAdd([side], mod(idx, side), a, 0),
      raw: () => {
        const o = new Float64Array(side)
        for (let i = 0; i < n; i++) {
          const j = (I[i] as number) % side
          o[j] = at(o, j) + at(A, i)
        }
        return o
      },
    },
    {
      op: 'segmentSum (n values, s segments)',
      tensor: () => segmentSum(a, mod(idx, side), side),
      raw: () => {
        const o = new Float64Array(side)
        for (let i = 0; i < n; i++) {
          const j = (I[i] as number) % side
          o[j] = at(o, j) + at(A, i)
        }
        return o
      },
    },
    {
      op: 'oneHot (s labels, s classes)',
      tensor: () => oneHot(labels, side),
      raw: () => {
        const o = new Float64Array(side * side)
        for (let i = 0; i < side; i++) o[i * side + (LB[i] as number)] = 1
        return o
      },
    },
    {
      // The shape of an integrand: exp(-0.5 * (a - b)^2) as separate tensor ops against one fused loop.
      op: 'exp(-0.5 (a-b)^2) (4 ops vs 1 loop)',
      tensor: () => exp(mul(-0.5, mul(add(a, mul(b, -1)), add(a, mul(b, -1))))),
      raw: () => {
        const o = new Float64Array(n)
        for (let i = 0; i < n; i++) {
          const d = at(A, i) - at(B, i)
          o[i] = Math.exp(-0.5 * (d * d))
        }
        return o
      },
    },
  ]
  return out
}

function rows(only: string | undefined): Row[] {
  const out: Row[] = []
  for (const n of SIZES) {
    for (const c of cases(n)) {
      if (only !== undefined && !c.op.includes(only)) continue
      const t = summarise(bench(c.tensor, 7, 15))
      const r = summarise(bench(c.raw, 7, 15))
      out.push({
        op: c.op,
        n,
        tensorUs: t.median,
        rawUs: r.median,
        ratio: t.median / r.median,
        tensorSpread: t.spread,
        rawSpread: r.spread,
      })
    }
  }
  return out
}

const arg = (flag: string): string | undefined => {
  const i = process.argv.indexOf(flag)
  return i >= 0 ? process.argv[i + 1] : undefined
}
const calStart = calibrate(3)
const result = rows(arg('--only'))
const calEnd = calibrate(3)
for (const r of result)
  process.stdout.write(
    `${r.op.padEnd(46)} n=${String(r.n).padStart(6)}  tensor ${r.tensorUs.toFixed(2).padStart(10)} us  raw ${r.rawUs.toFixed(2).padStart(10)} us  ratio ${r.ratio.toFixed(2).padStart(6)}  (spread ${(100 * r.tensorSpread).toFixed(0)}% / ${(100 * r.rawSpread).toFixed(0)}%)\n`,
  )
process.stdout.write(
  `calibration ms: start ${calStart.map((x) => x.toFixed(1)).join(' ')}  end ${calEnd.map((x) => x.toFixed(1)).join(' ')}\n`,
)
const json = arg('--json')
if (json !== undefined)
  writeFileSync(json, JSON.stringify({ rows: result, calibration: { start: calStart, end: calEnd } }, null, 1))
