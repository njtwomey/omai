/**
 * The numeric prior site: a prior whose density has no message family (a half-normal, half-Cauchy, uniform or truncated normal)
 * kept as the factor it is. Its EP message is the moment match of the Gaussian cavity times the exact density, found by the
 * quadrature of the likelihood sites and redone against the current cavity at every sweep, so the graph holds the prior itself
 * and not a projection of it made once (`projectedPrior` in `priors.ts` is that one-off projection). A variable with no
 * information yet is shown wide by the engine (`needsDensity`), so the first message is the density's own Gaussian projection.
 * The density is of the variable itself, or of its log (`log: 1`: the variable is `log x` for a positive `x`, the density
 * includes the Jacobian), which is how the compiler carries a positive scale.
 */

import { FLAT_CAVITY_PRECISION, messageArg, numberParam } from '../../core/args.ts'
import { gaussian } from '../../core/families.ts'
import { param } from '../../core/params.ts'
import type { Eta, FactorOperator, Params } from '../../core/types.ts'
import { logDiffPhi } from '../../numerics/_scalar.ts'
import type { TiltHint } from '../../numerics/_tilted.ts'
import { type Likelihood, likelihoodSite } from './likelihoods.ts'

const BASES = ['half-normal', 'half-cauchy', 'uniform', 'truncated-normal'] as const

/** The density of the base on the value scale, normalised, with its support. */
function valueDensity(p: Params): { readonly g: (x: number) => number; readonly lo: number; readonly hi: number } {
  const base = String(p['base'])
  const lo = numberParam(p, 'low', base === 'half-normal' || base === 'half-cauchy' ? 0 : Number.NEGATIVE_INFINITY)
  const hi = numberParam(p, 'high', Number.POSITIVE_INFINITY)
  if (base === 'half-normal') {
    const s = numberParam(p, 'scale')
    const c = Math.log(2) - 0.5 * Math.log(2 * Math.PI) - Math.log(s)
    return { g: (x) => c - (x * x) / (2 * s * s), lo, hi }
  }
  if (base === 'half-cauchy') {
    const s = numberParam(p, 'scale')
    const c = Math.log(2 / (Math.PI * s))
    return { g: (x) => c - Math.log1p((x / s) ** 2), lo, hi }
  }
  if (base === 'uniform') {
    const c = -Math.log(hi - lo)
    return { g: () => c, lo, hi }
  }
  const s = numberParam(p, 'scale')
  const loc = numberParam(p, 'loc')
  const c = -Math.log(s) - 0.5 * Math.log(2 * Math.PI) - logDiffPhi((lo - loc) / s, (hi - loc) / s)
  return { g: (x) => c - 0.5 * ((x - loc) / s) ** 2, lo, hi }
}

/**
 * What the density says of itself to the integral: the kinks at its finite support (on the value scale or, for a density of the log, the
 * log of it), and for a half-normal or half-Cauchy of the log (and a normal near zero on the log) that its whole window is kept, as a
 * zoom onto the mass would cut the variance of the exponential left tail (`e^l`).
 */
function locate(p: Params): TiltHint {
  const base = String(p['base'])
  const log = numberParam(p, 'log', 0) === 1
  const { lo, hi } = valueDensity(p)
  const finite = (v: number): number[] => (Number.isFinite(v) ? [log ? Math.log(v) : v] : [])
  const kinks = [...(lo > 0 || !log ? finite(lo) : []), ...finite(hi)]
  if (base === 'half-normal' || base === 'half-cauchy') return log ? { kinks, zoom: false } : { kinks }
  // On the log scale a uniform on (0, hi] has an exponential left tail too.
  if (base === 'uniform') return log && !(lo > 0) ? { kinks, zoom: false } : { kinks }
  const s = numberParam(p, 'scale')
  const loc = Math.min(Math.max(numberParam(p, 'loc'), lo), hi)
  // On the log scale a normal near zero (or at most a scale away) looks like a half-normal; one far from zero is narrow in the log.
  return log && loc <= s ? { kinks, zoom: false } : { kinks }
}

/** The one entry of the registry: the base density (parameter `base`), on the value or its log. */
const prior: Likelihood = {
  name: 'prior',
  description:
    'a prior density on a Gaussian variable, or on its log: half-normal, half-Cauchy, uniform, truncated normal',
  params: ['base', 'log', 'scale', 'loc', 'low', 'high'],
  spec: {
    base: param.text({ choices: BASES }),
    log: param.number({ optional: true }),
    scale: param.positive({ optional: true }),
    loc: param.number({ optional: true }),
    low: param.number({ optional: true, infinite: true }),
    high: param.number({ optional: true, infinite: true }),
  },
  check: (_y, p) => {
    const base = p['base']
    if (!(typeof base === 'string' && (BASES as readonly string[]).includes(base)))
      return `base must be one of ${BASES.join(', ')}`
    const { lo, hi } = valueDensity(p)
    if (!(lo < hi)) return 'low must be below high'
    if (numberParam(p, 'log', 0) === 1 && !(lo >= 0)) return 'a density of the log needs a positive support'
    if (
      (base === 'half-normal' || base === 'half-cauchy' || base === 'truncated-normal') &&
      !(numberParam(p, 'scale') > 0)
    )
      return 'scale must be positive'
    return undefined
  },
  density: (_y, p) => {
    const { g, lo, hi } = valueDensity(p)
    if (numberParam(p, 'log', 0) !== 1) return (x) => (x >= lo && x <= hi ? g(x) : Number.NEGATIVE_INFINITY)
    return (l) => {
      const x = Math.exp(l)
      return x >= lo && x <= hi ? g(x) + l : Number.NEGATIVE_INFINITY
    }
  },
  locate: (_y, p) => locate(p),
}

/**
 * A prior on one Gaussian variable that has no message family, kept exact: a half-normal or half-Cauchy (`scale`), a uniform
 * (`low`, `high`) or a truncated normal (`loc`, `scale`, `low`, `high`) of the variable (`log: 0`) or, for a positive one carried on
 * its log, of its log (`log: 1`, the Jacobian included), chosen by the parameter `base`, with the outcome parameter `y` zero. EP
 * matches the moments of the cavity times the density by quadrature at every sweep; the evidence is that integral, exact for the
 * model with the real prior.
 *
 * @concept factor-operators
 * @index structure An exact prior with no message family (half-normal, half-Cauchy, uniform, truncated): quadrature EP.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, priorSite } from 'omai-engine/message-passing'
 * const args = [messageArg(gaussian, gaussianEta(1, 0.25))]
 * const params = { y: 0, base: 'uniform', log: 0, low: 0, high: 2 }
 * priorSite.ep?.({ mode: 'ep', to: 0, args, params, power: 1 }).message.map((v) => v.toFixed(2)) // => ['1.17', '1.17']
 * ```
 */
export const priorSite: FactorOperator = (() => {
  const site = likelihoodSite('prior-site', { prior }, 'prior', { needsDensity: true })
  return {
    ...site,
    // The mean field has no cavity: the message of a prior that depends on nothing else is the Gaussian projection of the density
    // alone (the message the one-off projection gave), whatever the belief of the variable is.
    vmp: (c) => {
      const wide: Eta = [0, FLAT_CAVITY_PRECISION]
      const m = (site.ep as NonNullable<FactorOperator['ep']>)({
        ...c,
        mode: 'ep',
        args: [messageArg(gaussian, wide)],
        power: 1,
      }).message
      return { message: m.map((v, i) => v + (wide[i] ?? 0)) }
    },
  }
})()
