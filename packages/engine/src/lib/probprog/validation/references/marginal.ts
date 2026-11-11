/**
 * The shape of a reference answer for one number, `ExactMarginal` (moments, density, the range to draw it on), and the closed
 * forms that make one: a finite mixture of normals, the moments of a mixture, the branch a switch selects, a coin's probability
 * and the marginal of posterior draws.
 */

import { normalPdf } from './special.ts'

/** The closed-form marginal of a quantity: its moments and its density. */
export interface ExactMarginal {
  readonly mean: number
  readonly sd: number
  readonly pdf: (x: number) => number
  /** The interval the curve is drawn on (default the mean plus or minus eight standard deviations, kept inside the support). */
  readonly range?: readonly [number, number]
}

/** One component of a mixture of normals. */
export interface MixtureComponent {
  readonly weight: number
  readonly mean: number
  readonly sd: number
}

/**
 * The marginal of a finite mixture of normals, exactly: its moments from the components and its density as their weighted sum.
 *
 * @param components - The weighted components (the weights are normalised).
 * @returns The marginal.
 * @example
 * ```ts
 * import { mixtureMarginal } from 'omai-engine/probprog/validation'
 * mixtureMarginal([{ weight: 1, mean: -1, sd: 1 }, { weight: 1, mean: 1, sd: 1 }]).mean // => 0
 * ```
 */
export function mixtureMarginal(components: readonly MixtureComponent[]): ExactMarginal {
  const total = components.reduce((a, c) => a + c.weight, 0)
  const parts = components.map((c) => ({ ...c, weight: c.weight / total }))
  const mean = parts.reduce((a, c) => a + c.weight * c.mean, 0)
  const second = parts.reduce((a, c) => a + c.weight * (c.sd ** 2 + c.mean ** 2), 0)
  const sd = Math.sqrt(Math.max(second - mean * mean, 0))
  const lo = Math.min(...parts.map((c) => c.mean - 5 * c.sd))
  const hi = Math.max(...parts.map((c) => c.mean + 5 * c.sd))
  return { mean, sd, pdf: (x) => parts.reduce((a, c) => a + c.weight * normalPdf(x, c.mean, c.sd), 0), range: [lo, hi] }
}

/**
 * The branch of a mixture that a switch selects, from the marginal belief of a component that is the selected branch with
 * probability `p` and its prior otherwise: if `b` is that marginal's mean and sd, the selected branch's moments follow from
 * the mixture's (`E[x] = p m1 + (1 - p) m0`, `E[x²] = p (s1² + m1²) + (1 - p)(s0² + m0²)`). Exact when `b` has the mixture's
 * own moments, as an EP message through a gate has.
 *
 * @param b - The component's marginal mean and sd.
 * @param p - The probability that the component is the selected one.
 * @param prior - The mean and sd the component keeps when it is not selected.
 * @returns The selected branch's mean and sd, or undefined when `p` is zero or the moments do not make a variance.
 * @example
 * ```ts
 * import { selectedBranch } from 'omai-engine/probprog/validation'
 * selectedBranch({ mean: 0.5, sd: 1 }, 0.5, { mean: 0, sd: 1 })?.mean // => 1
 * ```
 */
export function selectedBranch(
  b: { readonly mean: number; readonly sd: number },
  p: number,
  prior: { readonly mean: number; readonly sd: number },
): { readonly mean: number; readonly sd: number } | undefined {
  if (!(p > 1e-12)) return undefined
  const m1 = (b.mean - (1 - p) * prior.mean) / p
  const second = (b.sd ** 2 + b.mean ** 2 - (1 - p) * (prior.sd ** 2 + prior.mean ** 2)) / p
  const v = second - m1 * m1
  return v > 0 ? { mean: m1, sd: Math.sqrt(v) } : undefined
}

/**
 * The mean and standard deviation of a mixture of Gaussians: what a single Gaussian that matches its moments has.
 *
 * @param branches - The weighted branches (the weights are normalised).
 * @returns The mean and sd.
 * @example
 * ```ts
 * import { momentMatch } from 'omai-engine/probprog/validation'
 * momentMatch([{ weight: 1, mean: -1, sd: 0 }, { weight: 1, mean: 1, sd: 0 }]).sd // => 1
 * ```
 */
export function momentMatch(branches: readonly { weight: number; mean: number; sd: number }[]): {
  mean: number
  sd: number
} {
  const total = branches.reduce((a, c) => a + c.weight, 0)
  const mean = branches.reduce((a, c) => a + (c.weight / total) * c.mean, 0)
  const second = branches.reduce((a, c) => a + (c.weight / total) * (c.sd ** 2 + c.mean ** 2), 0)
  return { mean, sd: Math.sqrt(Math.max(second - mean * mean, 0)) }
}

/**
 * The marginal of a probability (the exact probability that a switch is on): its value and its spread as a coin, with no density
 * (a probability is checked on its mean, not drawn).
 *
 * @param p - The probability.
 * @returns The marginal.
 * @example
 * ```ts
 * import { probabilityMarginal } from 'omai-engine/probprog/validation'
 * probabilityMarginal(0.5).sd // => 0.5
 * ```
 */
export const probabilityMarginal = (p: number): ExactMarginal => ({ mean: p, sd: Math.sqrt(p * (1 - p)), pdf: () => 0 })

/**
 * The marginal of a quantity from posterior draws: the sample mean and sd, and a Gaussian kernel density estimate (Silverman's
 * bandwidth), over the range the draws cover.
 *
 * @param draws - The draws.
 * @returns The marginal.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { samplesMarginal } from 'omai-engine/probprog/validation'
 * samplesMarginal([0, 1, 2, 3, 4]).mean // => 2
 * ```
 */
export function samplesMarginal(draws: readonly number[]): ExactMarginal {
  const n = draws.length
  const mean = draws.reduce((a, b) => a + b, 0) / n
  const sd = Math.sqrt(draws.reduce((a, b) => a + (b - mean) ** 2, 0) / n)
  const h = Math.max(1.06 * sd * n ** -0.2, 1e-9)
  // Thinned, so the density costs a few hundred kernels, not thousands.
  const stride = Math.max(1, Math.floor(n / 400))
  const kept = draws.filter((_, i) => i % stride === 0)
  return {
    mean,
    sd,
    pdf: (x) => kept.reduce((a, c) => a + normalPdf(x, c, h), 0) / kept.length,
    range: [Math.min(...draws) - 2 * h, Math.max(...draws) + 2 * h],
  }
}
