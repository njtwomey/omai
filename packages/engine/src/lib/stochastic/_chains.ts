// Internal helpers of the Markov chain modules: validation, classification and linear-algebra blocks.
import { MdmlError } from '../base/index.ts'
import { digraph, stronglyConnectedComponents } from '../graph/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import {
  asFloat,
  asTensor,
  concat,
  fromStorage,
  isTraced,
  ones,
  sliceAxis,
  type Tensor,
  type TensorLike,
  take,
  transpose,
  type Value,
} from '../tensor/index.ts'
import { primalNumbers } from './_shared.ts'

function readable(x: Value): Tensor | undefined {
  let v: Value = x
  while (isTraced(v)) {
    const p = (v as { primal?: Value }).primal
    if (p === undefined) return undefined
    v = p
  }
  return v
}

function squareSize(p: TensorLike, where: string): { t: Value; n: number } {
  const t = asFloat(asTensor(p))
  if (t.shape.length !== 2 || t.shape[0] !== t.shape[1] || (t.shape[0] ?? 0) === 0) {
    throw new MdmlError('shape', `${where}: expected a non-empty square matrix`)
  }
  return { t, n: t.shape[0] ?? 0 }
}

/** Validated stochastic rows, or undefined rows when the matrix is traced without a concrete value (under vmap). */
export function maybeStochasticRows(p: TensorLike, where: string): { n: number; rows: number[][] | undefined } {
  const { t, n } = squareSize(p, where)
  const c = readable(t)
  return c === undefined ? { n, rows: undefined } : stochasticRows(c, where)
}

/** Validated generator rows, or undefined rows when the matrix is traced without a concrete value (under vmap). */
export function maybeGeneratorRows(q: TensorLike, where: string): { n: number; rows: number[][] | undefined } {
  const { t, n } = squareSize(q, where)
  const c = readable(t)
  return c === undefined ? { n, rows: undefined } : generatorRows(c, where)
}

/** Concrete rows of a matrix, validated as stochastic (rows non-negative, summing to one). */
export function stochasticRows(p: TensorLike, where: string): { n: number; rows: number[][] } {
  const t = asFloat(asTensor(p))
  if (t.shape.length !== 2 || t.shape[0] !== t.shape[1]) {
    throw new MdmlError('shape', `${where}: the transition matrix must be square`)
  }
  const n = t.shape[0] ?? 0
  if (n === 0) throw new MdmlError('shape', `${where}: the transition matrix is empty`)
  const flat = primalNumbers(t)
  const rows: number[][] = []
  for (let i = 0; i < n; i++) {
    const row = flat.slice(i * n, (i + 1) * n)
    let s = 0
    for (const v of row) {
      if (!(v >= -1e-12) || !Number.isFinite(v)) throw new MdmlError('value', `${where}: entries must be >= 0`)
      s += v
    }
    if (Math.abs(s - 1) > 1e-8) throw new MdmlError('value', `${where}: row ${i} sums to ${s}, not 1`)
    rows.push(row)
  }
  return { n, rows }
}

/** Concrete rows of a generator, validated (off-diagonal rates >= 0, rows summing to zero). */
export function generatorRows(q: TensorLike, where: string): { n: number; rows: number[][] } {
  const t = asFloat(asTensor(q))
  if (t.shape.length !== 2 || t.shape[0] !== t.shape[1]) throw new MdmlError('shape', `${where}: Q must be square`)
  const n = t.shape[0] ?? 0
  if (n === 0) throw new MdmlError('shape', `${where}: the generator is empty`)
  const flat = primalNumbers(t)
  const rows: number[][] = []
  for (let i = 0; i < n; i++) {
    const row = flat.slice(i * n, (i + 1) * n)
    let s = 0
    for (let j = 0; j < n; j++) {
      const v = row[j] ?? 0
      if (!Number.isFinite(v) || (j !== i && v < 0)) {
        throw new MdmlError('value', `${where}: off-diagonal rates must be finite and >= 0`)
      }
      s += v
    }
    if (Math.abs(s) > 1e-8 * Math.max(1, Math.abs(row[i] ?? 0))) {
      throw new MdmlError('value', `${where}: row ${i} sums to ${s}, not 0`)
    }
    rows.push(row)
  }
  return { n, rows }
}

/** The raw classification: classes in component order. */
export interface RawClassification {
  readonly classes: number[][]
  readonly classOf: Int32Array
  readonly closed: boolean[]
  readonly periods: number[]
  readonly recurrent: boolean[]
  readonly absorbing: number[]
}

function gcd(a: number, b: number): number {
  let x = Math.abs(a)
  let y = Math.abs(b)
  while (y !== 0) [x, y] = [y, x % y]
  return x
}

/** Communicating classes, closedness, periods and absorbing states of a chain given by its rows. */
export function classify(n: number, rows: readonly (readonly number[])[]): RawClassification {
  const edges: [number, number][] = []
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if ((rows[i]?.[j] ?? 0) > 0) edges.push([i, j])
  const scc = stronglyConnectedComponents(digraph(edges, { nodes: n }))
  const classes = scc.components.map((c) => [...c].sort((a, b) => a - b))
  const classOf = new Int32Array(n)
  for (const [k, c] of classes.entries()) for (const s of c) classOf[s] = k
  const closed = classes.map((c, k) => c.every((i) => (rows[i] ?? []).every((v, j) => v <= 0 || classOf[j] === k)))
  const periods = classes.map((c, k) => {
    // Breadth-first levels inside the class; the period is the gcd of level[u] + 1 - level[v] over its edges.
    const level = new Map<number, number>([[c[0] ?? 0, 0]])
    const queue = [c[0] ?? 0]
    for (let h = 0; h < queue.length; h++) {
      const u = queue[h] ?? 0
      for (let v = 0; v < n; v++) {
        if ((rows[u]?.[v] ?? 0) > 0 && classOf[v] === k && !level.has(v)) {
          level.set(v, (level.get(u) ?? 0) + 1)
          queue.push(v)
        }
      }
    }
    let d = 0
    for (const u of c) {
      for (const v of c) if ((rows[u]?.[v] ?? 0) > 0) d = gcd(d, (level.get(u) ?? 0) + 1 - (level.get(v) ?? 0))
    }
    return d === 0 ? 1 : d
  })
  const recurrent = Array.from({ length: n }, (_, i) => closed[classOf[i] ?? 0] ?? false)
  const absorbing = Array.from({ length: n }, (_, i) => i).filter((i) => (rows[i]?.[i] ?? 0) >= 1 - 1e-12)
  return { classes, classOf, closed, periods, recurrent, absorbing }
}

/** Solve pi M = 0 with sum(pi) = 1 (the last balance equation replaced by the normalisation). */
export function stationarySolve(m: Value, n: number): Value {
  const a = concat([sliceAxis(transpose(m), 0, 0, n - 1), ones([1, n])], 0)
  const b = fromStorage(
    Float64Array.from({ length: n }, (_, i) => (i === n - 1 ? 1 : 0)),
    [n],
    'float64',
  )
  return solve(a, b)
}

/** A matrix from concrete rows. */
export function tensorOf(rows: readonly (readonly number[])[]): Tensor<'float64'> {
  return fromStorage(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0], 'float64')
}

/** The sub-matrix on rows r and columns c (differentiable). */
export function block(p: Value, r: readonly number[], c: readonly number[]): Value {
  const idx = (xs: readonly number[]): Tensor<'int32'> => fromStorage(Int32Array.from(xs), [xs.length], 'int32')
  return take(take(p, idx(r), 0), idx(c), 1)
}

/** The states from which the target set is reachable. */
export function reach(n: number, rows: readonly (readonly number[])[], target: readonly boolean[]): boolean[] {
  const can = [...target]
  let changed = true
  for (let iter = 0; changed && iter <= n; iter++) {
    changed = false
    for (let i = 0; i < n; i++) {
      if (can[i] === true) continue
      if ((rows[i] ?? []).some((v, j) => v > 0 && can[j] === true)) {
        can[i] = true
        changed = true
      }
    }
  }
  return can
}

/** A validated membership mask of target states. */
export function targetMask(n: number, target: readonly number[], where: string): boolean[] {
  if (target.length === 0) throw new MdmlError('value', `${where}: the target set is empty`)
  const mask = new Array<boolean>(n).fill(false)
  for (const s of target) {
    if (!(Number.isInteger(s) && s >= 0 && s < n)) throw new MdmlError('value', `${where}: state ${s} out of range`)
    mask[s] = true
  }
  return mask
}

/** Assemble a vector from per-state scalars (numbers or scalar values). */
export function assemble(parts: readonly (Value | number)[]): Value {
  return concat(
    parts.map((x) => (typeof x === 'number' ? fromStorage(Float64Array.of(x), [1], 'float64') : sliceAxis(x, 0, 0, 1))),
    0,
  )
}
