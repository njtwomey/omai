/**
 * A mixture-of-Gaussians guide over the flat unconstrained latent vector, for posteriors with several modes that no single
 * Gaussian covers. The component is a discrete site of the guide (`auto_component`), so the bound it optimises is that of the
 * joint of the component and the latent; with `traceEnumElbo` the component is enumerated and its weights, the components'
 * means and scales get exact gradients in the weights and pathwise gradients in the rest. The augmented bound is below the
 * ELBO of the mixture's marginal by the information the latent carries about its component, and exact when the components
 * do not overlap.
 */

import { categorical, compose, normal, type Support } from '../dists/index.ts'
import { foldIn, uniform } from '../random/index.ts'
import { full, mean, mul, primalOf, stack, sub, take, toFlat, type Value, zeros } from '../tensor/index.ts'
import { type AutoGuide, type AutoGuideOptions, jointGuide } from './guides.ts'
import { type Model, param, sample } from './primitives.ts'

/** Options of {@link autoMixtureNormal}. */
export interface AutoMixtureOptions extends AutoGuideOptions {
  /** The number of components (default 2). */
  readonly components?: number
  /**
   * The components start at independent uniform draws on (-1, 1), centred on their mean and multiplied by this (default 4),
   * so that they begin apart on opposite sides of their centre: components that start close fall into the same mode.
   */
  readonly spread?: number
}

/**
 * A mixture-of-Gaussians guide (diagonal components) over the unconstrained latent vector, with the component drawn at a
 * guide site `auto_component` marked for enumeration: fit it with `traceEnumElbo`, which sums the components out exactly
 * (a component's probability multiplies its bound), or with any estimator, which samples the component with a
 * score-function gradient. Parameters: `auto_logits` (`[K]`), `auto_locs` (`[K, D]`), `auto_scales` (`[K, D]`). The
 * components start at different random points.
 *
 * @param model - The model.
 * @param options - The number of components, initialisation and initial scale.
 * @returns The guide.
 * @concept variational-inference
 * @index function Mixture-of-Gaussians guide whose component is an enumerated guide site, for multimodal posteriors.
 * @example
 * ```ts
 * import { autoMixtureNormal, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * const t = traceOf(seed(autoMixtureNormal(() => { sample('x', normal(0, 1)) }, { components: 3 }), key(0)), undefined)
 * t.sites.get('auto_locs')?.value.shape // => [3, 1]
 * ```
 */
export function autoMixtureNormal<D = void>(model: Model<D>, options: AutoMixtureOptions = {}): AutoGuide<D> {
  const K = options.components ?? 2
  const s0 = options.initScale ?? 0.1
  const spread = options.spread ?? 4
  const positive: Support = { kind: 'positive' }
  return jointGuide(
    model,
    'autoMixtureNormal',
    {
      latent: (D) => {
        const logits = param('auto_logits', zeros([K]))
        const locs = param('auto_locs', (k) => {
          const u = stack(
            // banned-ok: non-numeric: one initial tensor per component, each from a key of its own, stacked
            Array.from({ length: K }, (_, i) => primalOf(uniform(foldIn(k, i), [D], { low: -1, high: 1 }))),
          )
          // Centred on the components' mean, so that they start apart, on opposite sides of their centre.
          return mul(sub(u, mean(u, 0)), spread)
        })
        const scales = param('auto_scales', full([K, D], s0), { support: positive })
        const c = sample('auto_component', categorical({ logits }), { infer: { enumerate: 'parallel' } })
        return compose.independent(normal(take(locs, c), take(scales, c)), 1)
      },
      fitted: (params) => {
        const locs = params['auto_locs'] as Value
        const scales = params['auto_scales'] as Value
        return compose.mixtureSameFamily(
          { logits: params['auto_logits'] as Value },
          compose.independent(normal(locs, scales), 1),
        )
      },
      centre: (params) => {
        const locs = params['auto_locs'] as Value
        const logits = params['auto_logits'] as Value
        // The location of the heaviest component.
        const w = Array.from(toFlat(primalOf(logits)))
        const best = w.indexOf(Math.max(...w))
        return take(locs, best)
      },
    },
    options,
  )
}
