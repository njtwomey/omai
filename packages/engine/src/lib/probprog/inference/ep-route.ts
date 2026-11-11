/**
 * The one place that chooses how expectation propagation runs a model: plain EP on the model as it is when every factor has a
 * message rule (`ep`), else EP inside a grid over the scalar hyperparameters a custom function reads (`ep-grid`), else
 * nothing (the verdict of plain EP says why). `epRoute` returns the choice with the verdict and the method, so the suite, the
 * coverage and a page that shows the code that ran agree on what ran. `epMultiStartMethod` is not on the route: it answers a
 * different question (which fixed points exist, for a model with discrete latents) and is chosen by the person.
 */

import type { BoundModel } from '../model/index.ts'
import { type EpGridMethod, epGridMethod } from './ep-grid.ts'
import type { Verdict } from './method.ts'
import { analysisOf } from './model-input.ts'
import { type EpSpecMethod, epSpecMethod } from './mp.ts'

/** How EP runs a model: `ep` (as it is), `ep-grid` (inside a grid over hyperparameters) or `none` (refused). */
export type EpVia = 'ep' | 'ep-grid' | 'none'

/** The route a model takes: how, the verdict that decided it (the refusal's reason when `none`) and the method to run. */
export type EpRoute =
  | { readonly via: 'ep'; readonly verdict: Verdict; readonly method: EpSpecMethod }
  | { readonly via: 'ep-grid'; readonly verdict: Verdict; readonly method: EpGridMethod }
  | { readonly via: 'none'; readonly verdict: Verdict; readonly method?: undefined }

/**
 * How expectation propagation runs a bound model, without running anything: plain EP when the compiler accepts the model and
 * has a rule for every operator, else EP inside a grid over the hyperparameters when that applies, else a refusal that carries
 * the verdict of plain EP.
 *
 * @param bound - The model with its data bound.
 * @returns The route: `via`, the verdict and the method.
 * @concept expectation-propagation
 * @index function How EP runs a bound model: plain, inside a grid over hyperparameters, or refused.
 * @example
 * ```ts
 * import { epRoute } from 'omai-engine/probprog/inference'
 * import { gaussianAtom, atomBound } from 'omai-engine/probprog/ep'
 * epRoute(atomBound(gaussianAtom.problem(gaussianAtom.defaults))).via // => 'ep'
 * ```
 */
export function epRoute(bound: BoundModel): EpRoute {
  const analysis = analysisOf(bound)
  const plain = epSpecMethod(bound)
  const verdict = plain.check?.(analysis) ?? {
    applies: false,
    reason: 'the method has no check',
    missing: [] as string[],
  }
  if (verdict.applies) return { via: 'ep', verdict, method: plain }
  // A model whose graph exists once some scalar latents are known runs by EP inside a grid over them.
  const grid = epGridMethod(bound)
  const inGrid = grid.check?.(analysis)
  if (inGrid?.applies === true) return { via: 'ep-grid', verdict: inGrid, method: grid }
  return { via: 'none', verdict }
}
