/**
 * The effective sample size as curves, for the diagnostic charts of Markov chains (ArviZ's `plot_ess` kinds `quantile` and
 * `evolution`): `chainEssCurves` takes draws by quantity and chain and returns, per quantity, the ESS of each quantile's
 * indicator and the bulk and tail ESS as the draws accumulate, in the `EssCurves` shape the `chainDiagnostics` record
 * takes (`exposition` cannot compute them: it may not import this folder).
 */

import type { EssCurves } from '../../exposition/index.ts'
import { ess } from './convergence.ts'

/** Options of {@link chainEssCurves}. */
export interface ChainEssCurvesOptions {
  /** The probabilities of the quantile curve (default 0.05 to 0.95 in steps of 0.05). */
  readonly probs?: readonly number[]
  /** The points of the evolution curve (default 20, evenly spaced up to all the draws). */
  readonly points?: number
  /** The fewest draws per chain at which the evolution starts (default 20). */
  readonly minDraws?: number
}

const finite = (v: number): boolean => Number.isFinite(v)

/**
 * The ESS curves of every quantity: the ESS of the indicator `x <= q_p` for each probability `p` (a low value says the
 * chains explore that part of the distribution badly: the tails, in a funnel), and the bulk and tail ESS using the first
 * `m` draws of every chain for `m` growing to all of them (a healthy run grows in line with `m`; a flat curve is a run
 * that has stopped adding information). Points where the ESS is not defined (too few draws, constant draws) are left out.
 *
 * @param draws - Quantity name to its draws by chain (`draws['mu'][c][i]`), at least four draws per chain.
 * @param options - The probabilities, the points of the evolution and where it starts.
 * @returns The curves by quantity.
 * @index function ESS by quantile and as the draws accumulate, per quantity, for the chain diagnostics record.
 * @example
 * ```ts
 * import { chainEssCurves } from 'omai-engine/sampling'
 * const x = Array.from({ length: 200 }, (_, i) => Math.sin(i * 12.9898) * 43758.5453 % 1)
 * const y = Array.from({ length: 200 }, (_, i) => Math.sin(i * 78.233) * 12345.678 % 1)
 * const curves = chainEssCurves({ mu: [x, y] }, { probs: [0.5], points: 4 })
 * [curves['mu']?.quantile.probs, curves['mu']?.evolution.draws.at(-1)] // => [[0.5], 200]
 * ```
 */
export function chainEssCurves(
  draws: Readonly<Record<string, readonly (readonly number[])[]>>,
  options: ChainEssCurvesOptions = {},
): Record<string, EssCurves> {
  const probs = options.probs ?? Array.from({ length: 19 }, (_, i) => (i + 1) / 20)
  const points = Math.max(1, options.points ?? 20)
  const minDraws = Math.max(4, options.minDraws ?? 20)
  const out: Record<string, EssCurves> = {}
  for (const [name, chains] of Object.entries(draws)) {
    const n = chains[0]?.length ?? 0
    if (n < 4 || chains.length === 0) continue
    const rows = chains.map((c) => [...c])
    const quantile = probs
      .map((prob) => ({ prob, value: ess(rows, { method: 'quantile', prob }) }))
      .filter((q) => finite(q.value))
    const first = Math.min(n, minDraws)
    const lengths = [
      ...new Set(
        Array.from({ length: points }, (_, i) => Math.round(first + ((n - first) * i) / Math.max(1, points - 1))),
      ),
    ]
    const evolution = lengths
      .map((m) => {
        const part = rows.map((c) => c.slice(0, m))
        return { m, bulk: ess(part, { method: 'bulk' }), tail: ess(part, { method: 'tail' }) }
      })
      .filter((e) => finite(e.bulk) && finite(e.tail))
    out[name] = {
      quantile: { probs: quantile.map((q) => q.prob), ess: quantile.map((q) => q.value) },
      evolution: {
        draws: evolution.map((e) => e.m),
        bulk: evolution.map((e) => e.bulk),
        tail: evolution.map((e) => e.tail),
      },
    }
  }
  return out
}
