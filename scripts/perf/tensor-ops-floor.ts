/**
 * The per-op overhead floor of the tensor library at the sizes of a quadrature grid (a pair grid has about 4e3 nodes, a triple grid
 * 3e4, a fine experiment 1e5): `exp`, `log`, `add`, `mul`, `sum`, `logsumexp` and an indexing gather, each against the equivalent raw
 * typed-array loop (which allocates its output, as the tensor op does, and is the best a hand-written kernel does here). Part of the
 * tensorisation study (`docs/probprog/ep-perf-tensor-study.md`); an experiment, not wired into the library.
 *
 *   nice -n 15 node --conditions=source --experimental-transform-types --no-warnings scripts/perf/tensor-ops-floor.ts [--json <file>]
 */
import { writeFileSync } from 'node:fs'
import {
  add,
  arange,
  cast,
  exp,
  linspace,
  log,
  logsumexp,
  max,
  mod,
  mul,
  reshape,
  sin,
  slice,
  sum,
  type Tensor,
  take,
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

const data = (t: unknown): Float64Array => (t as Tensor).data as Float64Array

function rows(): Row[] {
  const out: Row[] = []
  for (const n of SIZES) {
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
    const cases: { op: string; tensor: () => unknown; raw: () => unknown }[] = [
      {
        op: 'add broadcast ([s,1] + [1,s])',
        tensor: () => add(col, row),
        raw: () => {
          const o = new Float64Array(side * side)
          for (let i = 0; i < side; i++)
            for (let j = 0; j < side; j++) o[i * side + j] = (C[i] as number) + (R[j] as number)
          return o
        },
      },
      {
        op: 'sum over axis 1 of [s,s]',
        tensor: () => sum(grid, 1),
        raw: () => {
          const o = new Float64Array(side)
          for (let i = 0; i < side; i++) {
            let t = 0
            for (let j = 0; j < side; j++) t += G[i * side + j] as number
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
          for (let i = 0; i < side; i++)
            for (let j = 0; j < side; j++) o[j] = (o[j] as number) + (G[i * side + j] as number)
          return o
        },
      },
      {
        op: 'max',
        tensor: () => max(a),
        raw: () => {
          let top = Number.NEGATIVE_INFINITY
          for (let i = 0; i < n; i++) if ((A[i] as number) > top) top = A[i] as number
          return top
        },
      },
      {
        op: 'exp',
        tensor: () => exp(a),
        raw: () => {
          const o = new Float64Array(n)
          for (let i = 0; i < n; i++) o[i] = Math.exp(A[i] as number)
          return o
        },
      },
      {
        op: 'log',
        tensor: () => log(a),
        raw: () => {
          const o = new Float64Array(n)
          for (let i = 0; i < n; i++) o[i] = Math.log(A[i] as number)
          return o
        },
      },
      {
        op: 'add',
        tensor: () => add(a, b),
        raw: () => {
          const o = new Float64Array(n)
          for (let i = 0; i < n; i++) o[i] = (A[i] as number) + (B[i] as number)
          return o
        },
      },
      {
        op: 'mul',
        tensor: () => mul(a, b),
        raw: () => {
          const o = new Float64Array(n)
          for (let i = 0; i < n; i++) o[i] = (A[i] as number) * (B[i] as number)
          return o
        },
      },
      {
        op: 'sum',
        tensor: () => sum(a),
        raw: () => {
          let s = 0
          for (let i = 0; i < n; i++) s += A[i] as number
          return s
        },
      },
      {
        op: 'logsumexp',
        tensor: () => logsumexp(a),
        raw: () => {
          let top = Number.NEGATIVE_INFINITY
          for (let i = 0; i < n; i++) if ((A[i] as number) > top) top = A[i] as number
          let s = 0
          for (let i = 0; i < n; i++) s += Math.exp((A[i] as number) - top)
          return top + Math.log(s)
        },
      },
      {
        op: 'take (gather)',
        tensor: () => take(a, idx),
        raw: () => {
          const o = new Float64Array(n)
          for (let i = 0; i < n; i++) o[i] = A[I[i] as number] as number
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
            const d = (A[i] as number) - (B[i] as number)
            o[i] = Math.exp(-0.5 * (d * d))
          }
          return o
        },
      },
    ]
    for (const c of cases) {
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

const calStart = calibrate(3)
const result = rows()
const calEnd = calibrate(3)
for (const r of result)
  process.stdout.write(
    `${r.op.padEnd(40)} n=${String(r.n).padStart(6)}  tensor ${r.tensorUs.toFixed(2).padStart(10)} us  raw ${r.rawUs.toFixed(2).padStart(10)} us  ratio ${r.ratio.toFixed(2).padStart(6)}  (spread ${(100 * r.tensorSpread).toFixed(0)}% / ${(100 * r.rawSpread).toFixed(0)}%)\n`,
  )
const json = process.argv.indexOf('--json')
if (json >= 0)
  writeFileSync(
    process.argv[json + 1] as string,
    JSON.stringify({ rows: result, calibration: { start: calStart, end: calEnd } }, null, 1),
  )
