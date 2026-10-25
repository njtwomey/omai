// Helpers shared by the learn modules (not exported from the domain).
import { MdmlError } from '../base/index.ts'
import { resolveKey } from '../context/index.ts'
import type { Key } from '../random/index.ts'
import type { RunOptions } from '../trace/index.ts'
import type { FitOptions } from './estimator.ts'
import { fitRunOptions } from './explained.ts'
import type { MatrixView } from './validation.ts'

const EPS = 2.220446049250313e-16

/** Default input feature names `x0, x1, ...` (sklearn's), or the given ones checked for length. */
export function inputNames(d: number, input: readonly string[] | undefined, who: string): string[] {
  if (input === undefined) return Array.from({ length: d }, (_, j) => `x${j}`)
  if (input.length !== d) throw new MdmlError('shape', `${who}: ${input.length} feature names for ${d} features`)
  return [...input]
}

/** sklearn's `_handle_zeros_in_scale`: scales below 10 eps become 1 (so constant features pass through). */
export function handleZeros(scale: number): number {
  return scale < 10 * EPS ? 1 : scale
}

/** The non-NaN values of a column. */
export function present(col: ArrayLike<number>): Float64Array {
  let m = 0
  for (let i = 0; i < col.length; i++) if (!Number.isNaN(col[i] ?? Number.NaN)) m++
  const out = new Float64Array(m)
  let k = 0
  for (let i = 0; i < col.length; i++) {
    const v = col[i] ?? Number.NaN
    if (!Number.isNaN(v)) out[k++] = v
  }
  return out
}

/** Apply a function to every column of a view, producing a view of the same shape. */
export function mapColumns(m: MatrixView, f: (col: Float64Array, j: number) => ArrayLike<number>): MatrixView {
  const out = new Float64Array(m.n * m.d)
  const col = new Float64Array(m.n)
  for (let j = 0; j < m.d; j++) {
    for (let i = 0; i < m.n; i++) col[i] = m.data[i * m.d + j] ?? 0
    const r = f(col, j)
    for (let i = 0; i < m.n; i++) out[i * m.d + j] = r[i] ?? 0
  }
  return { n: m.n, d: m.d, data: out }
}

/** Apply a function to every entry of a view, keeping NaN as NaN. */
export function mapEntries(m: MatrixView, f: (v: number, j: number) => number): MatrixView {
  const out = new Float64Array(m.n * m.d)
  for (let i = 0; i < m.n; i++) {
    for (let j = 0; j < m.d; j++) {
      const v = m.data[i * m.d + j] ?? 0
      out[i * m.d + j] = Number.isNaN(v) ? Number.NaN : f(v, j)
    }
  }
  return { n: m.n, d: m.d, data: out }
}

/** The key of a randomised learn function, by the context rule of section 3.2. */
export function keyFor(options: FitOptions, name: string): Key {
  return resolveKey(options, name)
}

/** Run options for a traced learn procedure: the caller's parent run, or the resolved tracer and policy. */
export const runOptionsFor: typeof fitRunOptions = fitRunOptions

/** Fit options for an inner fit, nested under a run and with a derived key; it records no training steps unless `explain` (a refit of the chosen model keeps the caller's choice). */
export function innerFitOptions(
  options: FitOptions,
  key: Key,
  parent?: RunOptions['parent'],
  explain = false,
): FitOptions {
  return {
    key,
    ...(explain ? (options.explain === undefined ? {} : { explain: options.explain }) : { explain: false }),
    ...(options.ctx === undefined ? {} : { ctx: options.ctx }),
    ...(parent === undefined
      ? {
          ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
          ...(options.policy === undefined ? {} : { policy: options.policy }),
        }
      : { parent }),
  }
}

/** Numbers of a typed array as a plain array (for JSON parameters). */
export function plain(xs: ArrayLike<number>): number[] {
  return Array.from(xs)
}
