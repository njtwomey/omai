/**
 * The exposition record of a fitted equation against the truth (`equationFit`): a model that is a sum of named terms
 * with estimated coefficients (sparse regression, symbolic regression, a discovered law) beside the true coefficients
 * when they are known, the fitted function and the true one on a grid and, optionally, the observations. A view
 * typesets the fitted expression with each coefficient marked by how far it is from the truth, plots the fit against
 * the truth and draws the residual beneath. `coefficientErrors` classifies the terms.
 */

import { OmaiError } from '../base/index.ts'
import { seal } from './_check.ts'

/** One term of the model: `estimate * tex`. */
export interface EquationTerm {
  /** The term without its coefficient, as TeX (`x^2`, `\sin x`, `1`). */
  readonly tex: string
  /** The estimated coefficient (zero: the term was dropped). */
  readonly estimate: number
  /** The true coefficient, when known (zero: the term is not in the law). */
  readonly truth?: number
}

/** How a coefficient compares with its truth. */
export type TermStatus =
  /** Within the tolerance of the truth (both zero counts). */
  | 'right'
  /** Present in the law, but the estimate is further than the tolerance from it. */
  | 'off'
  /** In the law but dropped (estimate zero). */
  | 'missing'
  /** Not in the law but kept (truth zero, estimate not). */
  | 'extra'
  /** No truth is known. */
  | 'unknown'

/** A fitted equation, with the truth beside it when known. */
export interface EquationFit {
  readonly kind: 'equationFit'
  /** The left-hand side as TeX (default `y`). */
  readonly target?: string
  /** The terms of a model linear in its coefficients. */
  readonly terms?: readonly EquationTerm[]
  /** The whole fitted expression as TeX, for a model that is not a sum of terms (the output of symbolic regression). */
  readonly expression?: string
  /** The whole true expression as TeX, when known. */
  readonly truthExpression?: string
  /** The grid the curves are on, increasing. */
  readonly x: readonly number[]
  /** The fitted function on the grid. */
  readonly fit: readonly number[]
  /** The true function on the grid. */
  readonly truth?: readonly number[]
  /** The observations the model was fitted to. */
  readonly observed?: { readonly x: readonly number[]; readonly y: readonly number[] }
  /** The relative error within which a coefficient counts as right (default 0.1). */
  readonly tolerance?: number
  /** The names of the axes (default `x` and `y`). */
  readonly axes?: readonly [string, string]
  readonly caption?: string
}

/**
 * An equation fit, with the curves and the observations checked against the grid and every number checked finite.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `shape` when `fit`, `truth` or the observations do not match the grid, or when the record has
 *   neither terms nor an expression; `value` when a coefficient is not finite.
 * @example
 * ```ts
 * import { equationFit } from 'omai-engine/exposition'
 * const r = equationFit({ terms: [{ tex: 'x', estimate: 2, truth: 2 }], x: [0, 1], fit: [0, 2], truth: [0, 2] })
 * r.terms?.length // => 1
 * ```
 */
export function equationFit(record: Omit<EquationFit, 'kind'>): EquationFit {
  const n = record.x.length
  if (n < 2) throw new OmaiError('shape', 'equationFit: x needs at least two points')
  if (record.fit.length !== n)
    throw new OmaiError('shape', `equationFit: fit has ${record.fit.length} entries, expected ${n}`)
  if (record.truth !== undefined && record.truth.length !== n)
    throw new OmaiError('shape', `equationFit: truth has ${record.truth.length} entries, expected ${n}`)
  if (record.observed !== undefined && record.observed.x.length !== record.observed.y.length)
    throw new OmaiError('shape', 'equationFit: observed x and y differ in length')
  if (record.terms === undefined && record.expression === undefined)
    throw new OmaiError('shape', 'equationFit: give the terms or the expression')
  for (const [k, t] of (record.terms ?? []).entries())
    if (!(Number.isFinite(t.estimate) && (t.truth === undefined || Number.isFinite(t.truth))))
      throw new OmaiError('value', `equationFit: terms[${k}] is not finite`)
  return seal({ kind: 'equationFit', ...record })
}

/**
 * Compare each coefficient of the fit with its truth: the signed error, the relative error and the status.
 *
 * @param record - The record.
 * @returns One entry per term (none for a record with only an expression).
 * @example
 * ```ts
 * import { coefficientErrors, equationFit } from 'omai-engine/exposition'
 * const r = equationFit({
 *   terms: [
 *     { tex: '1', estimate: 0.31, truth: 0.3 },
 *     { tex: 'x', estimate: 0.4, truth: 0 },
 *     { tex: 'x^2', estimate: 0, truth: -0.4 },
 *   ],
 *   x: [0, 1],
 *   fit: [0, 1],
 * })
 * coefficientErrors(r).map((e) => e.status) // => ['right', 'extra', 'missing']
 * ```
 */
export function coefficientErrors(record: EquationFit): {
  readonly error: number | undefined
  readonly relative: number | undefined
  readonly status: TermStatus
}[] {
  const tolerance = record.tolerance ?? 0.1
  return (record.terms ?? []).map((t) => {
    if (t.truth === undefined) return { error: undefined, relative: undefined, status: 'unknown' as const }
    const error = t.estimate - t.truth
    const relative =
      t.truth === 0 ? (t.estimate === 0 ? 0 : Number.POSITIVE_INFINITY) : Math.abs(error) / Math.abs(t.truth)
    const status: TermStatus =
      t.truth === 0
        ? t.estimate === 0
          ? 'right'
          : 'extra'
        : t.estimate === 0
          ? 'missing'
          : relative <= tolerance
            ? 'right'
            : 'off'
    return { error, relative, status }
  })
}

/**
 * The residual of the fit on the grid: the fitted function minus the true one (or minus the observations at their own
 * `x` when the truth is unknown, linearly interpolated from the fit).
 *
 * @param record - The record.
 * @returns The `x` of each residual and its value.
 * @example
 * ```ts
 * import { equationFit, fitResidual } from 'omai-engine/exposition'
 * fitResidual(equationFit({ terms: [], x: [0, 1], fit: [1, 3], truth: [1, 2] })).values // => [0, 1]
 * ```
 */
export function fitResidual(record: EquationFit): {
  readonly x: readonly number[]
  readonly values: readonly number[]
} {
  if (record.truth !== undefined) return { x: record.x, values: record.fit.map((f, i) => f - (record.truth?.[i] ?? 0)) }
  const o = record.observed
  if (o === undefined) return { x: record.x, values: record.fit.map(() => 0) }
  const at = (x: number): number => {
    const j = record.x.findIndex((v) => v >= x)
    if (j <= 0) return record.fit[0] ?? 0
    const a = record.x[j - 1] ?? 0
    const b = record.x[j] ?? a + 1
    return (record.fit[j - 1] ?? 0) + ((x - a) / (b - a)) * ((record.fit[j] ?? 0) - (record.fit[j - 1] ?? 0))
  }
  return { x: o.x, values: o.x.map((x, i) => at(x) - (o.y[i] ?? 0)) }
}
