/**
 * The Gibbs sampler that is the reference of the gate atoms whose posterior has no closed form and is too wide for a grid: a
 * mixture of Gaussians with a normal prior on each mean, a gamma prior on each precision and a Dirichlet prior on the weights
 * (every conditional is conjugate, the ordered form is the relabelled chain).
 */

import { OmaiError } from '../../../base/index.ts'
import { foldIn, gammaVariates, type Key, key, uniformStream } from '../../../random/index.ts'
import { arange, full, toArray, toFlat, zeros } from '../../../tensor/index.ts'

/** A seeded stream of standard normals (Box-Muller on the library's counter-based uniforms) and gamma variates. */
function streams(k: Key): {
  readonly uniform: () => number
  readonly normal: () => number
  readonly gamma: (a: number) => number
} {
  const uniform = uniformStream(k)
  let spare: number | undefined
  let calls = 0
  return {
    uniform,
    normal: () => {
      if (spare !== undefined) {
        const s = spare
        spare = undefined
        return s
      }
      const r = Math.sqrt(-2 * Math.log(1 - uniform()))
      const t = 2 * Math.PI * uniform()
      spare = r * Math.sin(t)
      return r * Math.cos(t)
    },
    gamma: (a) => gammaVariates(foldIn(k, 1 + calls++), [a])[0] as number,
  }
}

/** What a Gibbs run of a mixture is told. */
export interface MixtureGibbsSpec {
  readonly x: readonly number[]
  /** The prior of each component's mean (one per component; their number is the number of components). */
  readonly means: readonly { readonly mean: number; readonly sd: number }[]
  /** The gamma prior of every component's precision. */
  readonly precision: { readonly shape: number; readonly rate: number }
  /** The Dirichlet prior of the weights, one entry per component. */
  readonly weights: readonly number[]
  readonly seed: number
  readonly burn: number
  readonly draws: number
  /** Relabel every draw so that the means ascend: the ordered form of the model, whose prior is exchangeable. */
  readonly ordered: boolean
}

/** The draws of a mixture's parameters (by component) and each observation's posterior probability of each component. */
export interface MixtureGibbs {
  readonly mean: readonly (readonly number[])[]
  readonly precision: readonly (readonly number[])[]
  readonly weight: readonly (readonly number[])[]
  /** `responsibility[i][k]`: the average over the draws of P(observation i is from component k | the draw's parameters). */
  readonly responsibility: readonly (readonly number[])[]
}

/**
 * A Gibbs sampler for a mixture of Gaussians with component means, precisions and weights unknown (normal, gamma and Dirichlet
 * priors): the switches, the means, the precisions and the weights in turn, every conditional in closed form. With `ordered`, each
 * kept draw is relabelled so that the means ascend, which is exact for a prior that is the same for every component.
 *
 * @param spec - The data, the priors, the seed and the lengths.
 * @returns The draws and the responsibilities.
 * @throws OmaiError with code `value` when the priors do not have one entry per component.
 * @concept posterior-validation
 * @index function Gibbs sampler for a Gaussian mixture, the gate atoms' reference.
 * @example
 * ```ts
 * import { mixtureGibbs } from 'omai-engine/probprog/validation'
 * const x = [-2, -2.2, -1.8, 2, 2.1, 1.9]
 * const g = mixtureGibbs({ x, means: [{ mean: 0, sd: 3 }, { mean: 0, sd: 3 }], precision: { shape: 2, rate: 2 }, weights: [2, 2], seed: 1, burn: 50, draws: 100, ordered: true })
 * g.mean[0]?.length // => 100
 * ```
 */
export function mixtureGibbs(spec: MixtureGibbsSpec): MixtureGibbs {
  const K = spec.means.length
  if (spec.weights.length !== K) throw new OmaiError('value', 'mixtureGibbs: one weight prior per component')
  const rng = streams(key(spec.seed))
  const { x } = spec
  const n = x.length
  // Start from the sorted data cut into K groups.
  const order = [...x.keys()].sort((a, b) => (x[a] as number) - (x[b] as number))
  const z = toFlat(zeros([n]))
  order.forEach((i, r) => {
    z[i] = Math.min(K - 1, Math.floor((r * K) / n))
  })
  let mu = toFlat(zeros([K]))
  let tau = toFlat(full([K], spec.precision.shape / spec.precision.rate))
  let pi = toFlat(full([K], 1 / K))
  const out = {
    // banned-ok: reference: an exact answer written out in plain arithmetic to check the library against
    mean: Array.from({ length: K }, () => [] as number[]),
    precision: Array.from({ length: K }, () => [] as number[]),
    weight: Array.from({ length: K }, () => [] as number[]),
  }
  const resp = toArray(zeros([n, K])) as number[][]
  for (let sweep = 0; sweep < spec.burn + spec.draws; sweep++) {
    // The means and precisions given the switches.
    const count = toFlat(zeros([K]))
    const total = toFlat(zeros([K]))
    for (let i = 0; i < n; i++) {
      const k = z[i] as number
      count[k] = (count[k] as number) + 1
      total[k] = (total[k] as number) + (x[i] as number)
    }
    mu = mu.map((_, k) => {
      const prior = spec.means[k] as { mean: number; sd: number }
      const precision = 1 / prior.sd ** 2 + (count[k] as number) * (tau[k] as number)
      const centre = (prior.mean / prior.sd ** 2 + (tau[k] as number) * (total[k] as number)) / precision
      return centre + rng.normal() / Math.sqrt(precision)
    })
    tau = tau.map((_, k) => {
      let ss = 0
      for (let i = 0; i < n; i++) if (z[i] === k) ss += ((x[i] as number) - (mu[k] as number)) ** 2
      return rng.gamma(spec.precision.shape + (count[k] as number) / 2) / (spec.precision.rate + ss / 2)
    })
    const g = pi.map((_, k) => rng.gamma((spec.weights[k] as number) + (count[k] as number)))
    const gs = g.reduce((a, b) => a + b, 0)
    pi = g.map((v) => v / gs)
    // The switches given the parameters.
    const rows: number[][] = []
    for (let i = 0; i < n; i++) {
      const lp = pi.map(
        (p, k) =>
          Math.log(p) +
          0.5 * Math.log(tau[k] as number) -
          0.5 * (tau[k] as number) * ((x[i] as number) - (mu[k] as number)) ** 2,
      )
      const top = Math.max(...lp)
      const w = lp.map((v) => Math.exp(v - top))
      const s = w.reduce((a, b) => a + b, 0)
      rows.push(w.map((v) => v / s))
      let u = rng.uniform() * s
      let pick = K - 1
      for (let k = 0; k < K; k++) {
        u -= w[k] as number
        if (u <= 0) {
          pick = k
          break
        }
      }
      z[i] = pick
    }
    if (sweep < spec.burn) continue
    const perm = toFlat(arange(K))
    if (spec.ordered) perm.sort((a, b) => (mu[a] as number) - (mu[b] as number))
    perm.forEach((from, to) => {
      ;(out.mean[to] as number[]).push(mu[from] as number)
      ;(out.precision[to] as number[]).push(tau[from] as number)
      ;(out.weight[to] as number[]).push(pi[from] as number)
      for (let i = 0; i < n; i++)
        (resp[i] as number[])[to] = ((resp[i] as number[])[to] as number) + ((rows[i] as number[])[from] as number)
    })
  }
  return { ...out, responsibility: resp.map((r) => r.map((v) => v / spec.draws)) }
}
