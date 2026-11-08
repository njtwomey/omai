/**
 * The slot of evidence maximisation that varies as an option table: the update that turns the posterior at the current
 * precisions into the next precisions, either MacKay's fixed point (through the effective number of well-determined
 * weights) or the EM update of the type-II likelihood (through the posterior's second moments). Both climb the same
 * log evidence to the same stationary points; they differ in speed and path. The table is one `union` space for
 * controls, grids and tuners, and builds the update from a configuration.
 */

import { ddot } from '../numerics/linalg/index.ts'
import { dict, type Option, type OptionTable, optionTable } from '../space/index.ts'

/** What an evidence update sees: the posterior at the current precisions and the data's summaries. */
export interface EvidenceInputs {
  /** The rows and the weights (features). */
  readonly n: number
  readonly p: number
  /** The posterior mean of the weights and its covariance `[p][p]`. */
  readonly mean: readonly number[]
  readonly covariance: readonly (readonly number[])[]
  /** The Gram matrix X^T X, row-major `[p * p]`. */
  readonly gram: ArrayLike<number>
  /** The squared residual of the posterior mean, `|y - X m|^2`. */
  readonly residual: number
  /** The effective number of well-determined weights at the current precisions. */
  readonly gamma: number
}

/** The next precisions from the current posterior. */
export type EvidenceUpdate = (inputs: EvidenceInputs) => { readonly alpha: number; readonly beta: number }

/** What an update does, for validity rules and selectors. */
export interface EvidenceUpdateProperties {
  /** Whether it uses the effective number of parameters (MacKay) rather than the posterior's second moments (EM). */
  readonly usesGamma: boolean
}

function update(
  description: string,
  properties: EvidenceUpdateProperties,
  make: () => EvidenceUpdate,
): Option<EvidenceUpdate, EvidenceUpdateProperties> {
  return { description, properties, space: dict({}), make }
}

/**
 * The evidence updates as one slot: `mackay` (Bishop 3.5.2: alpha = gamma / |m|^2, beta = (n - gamma) / |y - X m|^2) and
 * `em` (alpha = p / (|m|^2 + tr S), beta = n / (|y - X m|^2 + tr(S X^T X)), the EM step of the marginal likelihood).
 *
 * @concept bayesian-linear-regression
 * @example
 * ```ts
 * import { evidenceUpdates } from 'omai-engine/linear'
 * const next = evidenceUpdates.make({ kind: 'em' })({ n: 4, p: 1, mean: [2], covariance: [[0.25]], gram: [4], residual: 1, gamma: 0.8 })
 * [next.alpha, next.beta].map((v) => Math.round(v * 1000) / 1000) // => [0.235, 2]
 * ```
 */
export const evidenceUpdates: OptionTable<EvidenceUpdate, EvidenceUpdateProperties> = optionTable(
  {
    mackay: update(
      'MacKay’s fixed point through the effective number of well-determined weights',
      { usesGamma: true },
      () => (i) => ({
        alpha: i.gamma / Math.max(ddot(i.mean, i.mean), Number.MIN_VALUE),
        beta: Math.max(i.n - i.gamma, Number.MIN_VALUE) / Math.max(i.residual, Number.MIN_VALUE),
      }),
    ),
    em: update('the EM update through the posterior’s second moments', { usesGamma: false }, () => (i) => {
      let trace = 0
      let traceGram = 0
      for (let a = 0; a < i.p; a++) {
        trace += i.covariance[a]?.[a] ?? 0
        for (let b = 0; b < i.p; b++) traceGram += (i.covariance[a]?.[b] ?? 0) * (i.gram[a * i.p + b] ?? 0)
      }
      return {
        alpha: i.p / Math.max(ddot(i.mean, i.mean) + trace, Number.MIN_VALUE),
        beta: i.n / Math.max(i.residual + traceGram, Number.MIN_VALUE),
      }
    }),
  },
  { default: 'mackay', label: 'evidence update', description: 'How the prior and noise precisions are updated.' },
)
