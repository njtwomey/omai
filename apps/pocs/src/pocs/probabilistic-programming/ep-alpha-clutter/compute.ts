import {
  alphaEpMethod,
  type ClutterPosterior,
  clutterGraph,
  clutterPosterior,
  clutterProblem,
  marginalsOf,
} from 'omai-engine/message-passing'
import { key } from 'omai-engine/random'
import type { AnytimeRunSpec } from '../../../shared/useAnytimeRuns.ts'

/** The alphas fitted, from expectation propagation (1) toward the variational direction. */
export const ALPHAS: readonly number[] = [1, 0.6, 0.35, 0.2, 0.1, 0.05]

/** What one alpha's run has reached: the Gaussian it holds for the mean, and how far it is from the exact posterior. */
export interface Fit {
  readonly mean: number
  readonly sd: number
  /** KL(exact || Gaussian): large when the Gaussian misses mass the posterior has. */
  readonly forward: number
  /** KL(Gaussian || exact): large when the Gaussian has mass the posterior does not. */
  readonly reverse: number
}

/** A clutter problem with its exact posterior and factor graph, made once per seed and number of points. */
export function clutterSetup(
  seed: number,
  n: number,
): {
  readonly problem: ReturnType<typeof clutterProblem>
  readonly exact: ClutterPosterior
  readonly graph: ReturnType<typeof clutterGraph>
} {
  const problem = clutterProblem(key(seed), { n })
  return { problem, exact: clutterPosterior(problem), graph: clutterGraph(problem) }
}

/** One run per alpha, each reading the belief of `x` and its divergences from the exact posterior at a snapshot. */
export function clutterRuns(graph: ReturnType<typeof clutterGraph>, exact: ClutterPosterior): AnytimeRunSpec<Fit>[] {
  const method = alphaEpMethod(graph)
  return ALPHAS.map((alpha) => ({
    id: String(alpha),
    label: alpha === 1 ? 'alpha 1 (EP)' : `alpha ${alpha}`,
    method,
    config: { alpha, tol: 1e-7 },
    derive: (_s, x) => {
      const m = marginalsOf(graph, x.state as never)['x'] as { mean: number; variance: number }
      const d = exact.divergences(m.mean, m.variance)
      return { mean: m.mean, sd: Math.sqrt(m.variance), forward: d.forward, reverse: d.reverse }
    },
  }))
}
