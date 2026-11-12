/**
 * The exact posterior of a switch between Gaussian components observed through noise: the posterior of the switch is the ratio
 * of the branches' evidence, each branch is the conjugate update of its component, and what is selected is the mixture of the
 * branches weighted by that posterior. The spike and slab coefficient is the switch between a narrow and a wide zero-mean
 * component.
 */

import { type ExactMarginal, mixtureMarginal } from './marginal.ts'
import { normalPdf } from './special.ts'

/** A Gaussian: its mean and sd. */
export interface NormalComponent {
  readonly mean: number
  readonly sd: number
}

/** The conjugate update of a Gaussian component observed through Gaussian noise `s` at `y`. */
function update(prior: NormalComponent, y: number, s: number): NormalComponent {
  const prec = 1 / prior.sd ** 2 + 1 / s ** 2
  return { mean: (prior.mean / prior.sd ** 2 + y / s ** 2) / prec, sd: Math.sqrt(1 / prec) }
}

/** What a switch between two Gaussian components, observed through noise, has as its exact posterior. */
export interface TwoComponentGate {
  /** P(z = 1): the switch selects component 1. */
  readonly p1: number
  /** The components' conjugate updates (what each is if it is the selected one). */
  readonly updated: readonly [NormalComponent, NormalComponent]
  /** What is selected: the mixture of the two updates weighted by the switch. */
  readonly selected: ExactMarginal
  /** Each component's marginal: its update if selected, its prior otherwise. */
  readonly marginals: readonly [ExactMarginal, ExactMarginal]
}

/**
 * The exact posterior of a switch `z` (P(z = 1) = `pi`) that selects one of two independent Gaussian components, with `y`
 * observed through Gaussian noise `s`: the posterior of the switch is the ratio of the branches' evidence, each branch is the
 * conjugate update of its component, the selected value is their mixture and each component's marginal is its update if it was
 * selected and its prior otherwise.
 *
 * @param pi - The prior probability of component 1.
 * @param components - The two components' priors.
 * @param y - The observation.
 * @param s - The observation noise.
 * @returns The posterior.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { twoComponentGate } from 'omai-engine/probprog/validation'
 * const g = twoComponentGate(0.5, [{ mean: 0, sd: 1 }, { mean: 0, sd: 1 }], 0.3, 1)
 * g.p1 // => 0.5
 * ```
 */
export function twoComponentGate(
  pi: number,
  components: readonly [NormalComponent, NormalComponent],
  y: number,
  s: number,
): TwoComponentGate {
  const e = components.map((c) => normalPdf(y, c.mean, Math.sqrt(c.sd ** 2 + s ** 2)))
  const e1 = pi * (e[1] ?? 0)
  const e0 = (1 - pi) * (e[0] ?? 0)
  const p1 = e1 / (e1 + e0)
  const updated: [NormalComponent, NormalComponent] = [update(components[0], y, s), update(components[1], y, s)]
  return {
    p1,
    updated,
    selected: mixtureMarginal([
      { weight: 1 - p1, ...updated[0] },
      { weight: p1, ...updated[1] },
    ]),
    marginals: [
      mixtureMarginal([
        { weight: 1 - p1, ...updated[0] },
        { weight: p1, ...components[0] },
      ]),
      mixtureMarginal([
        { weight: p1, ...updated[1] },
        { weight: 1 - p1, ...components[1] },
      ]),
    ],
  }
}

/**
 * The exact posterior of a coefficient with a spike and slab prior (inclusion probability `pi`, slab sd, spike sd) observed
 * through noise: a switch between the spike and the slab, so the same closed form as a two-component gate with zero means.
 *
 * @param pi - The prior probability of the slab.
 * @param slab - The slab's sd.
 * @param spike - The spike's sd.
 * @param y - The observation.
 * @param s - The noise sd.
 * @returns The posterior: `p1` is the posterior probability of the slab, `updated[0]` the spike branch and `updated[1]` the slab branch.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { spikeSlabPosterior } from 'omai-engine/probprog/validation'
 * spikeSlabPosterior(0.5, 2, 0.1, 0, 0.5).p1 < 0.2 // => true
 * ```
 */
export function spikeSlabPosterior(pi: number, slab: number, spike: number, y: number, s: number): TwoComponentGate {
  return twoComponentGate(
    pi,
    [
      { mean: 0, sd: spike },
      { mean: 0, sd: slab },
    ],
    y,
    s,
  )
}
