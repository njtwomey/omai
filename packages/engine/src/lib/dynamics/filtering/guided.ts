/**
 * Guided proposals for the particle filters: the locally optimal Gaussian proposal of a Gaussian state-space model, which
 * moves each particle towards the observation instead of blindly by the transition. The proposal is the posterior of the
 * next state given the particle and the observation after linearising the observation function at the particle's predicted
 * state (exact for a linear observation, an extended-Kalman step otherwise), and the filters weight by the transition
 * over the proposal density (`ParticleGuide`), so the evidence estimate stays unbiased whatever the linearisation.
 */

import { jacfwd } from '../../autodiff/index.ts'
import { OmaiError } from '../../base/index.ts'
import { dpotrf, dpotrs } from '../../numerics/linalg/index.ts'
import { normal } from '../../random/index.ts'
import { asTensor, reshape, sliceAxis, type TensorLike, type Value } from '../../tensor/index.ts'
import { fromNumbers, numbersOf } from '../ode/_vec.ts'
import type { NonlinearModel } from './nonlinear.ts'
import type { ParticleGuide, ParticleModel } from './particle.ts'
import { gaussianParticleModel } from './particle-options.ts'

/** Options of {@link gaussianGuidedModel}. */
export interface GuidedOptions {
  /**
   * The observation Jacobian `[p, n]` when it is constant (a linear observation): the proposal's covariance is then
   * computed once. Default: the Jacobian of `h` at each particle's predicted state, by forward-mode autodiff.
   */
  readonly observationMatrix?: TensorLike
}

const GUIDE_JITTER = 1e-12

function inverseOf(a: Float64Array, n: number, name: string): Float64Array {
  const { factor, info } = dpotrf(a, n)
  if (info !== 0) throw new OmaiError('value', `${name}: the matrix is not positive definite`)
  const out = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    const e = new Float64Array(n)
    e[j] = 1
    const col = dpotrs(factor, e, n)
    for (let i = 0; i < n; i++) out[i * n + j] = col[i] ?? 0
  }
  return out
}

/**
 * The particle form of a Gaussian state-space model `x_t = f(x_{t-1}) + w_t`, `z_t = h(x_t) + v_t` with the locally optimal
 * Gaussian proposal: particle `x` moves to `N(m, S)` with `S^-1 = Q^-1 + H' R^-1 H` and `m = f(x) + S H' R^-1 (z - h(f(x)))`,
 * `H` the Jacobian of `h` at `f(x)`. For a linear observation this is the optimal proposal (the weights no longer depend on
 * the draw, only on the previous particle), and the evidence estimate has far lower variance than the bootstrap filter's when
 * the observation is informative. The transition, likelihood, transition density and point prediction are those of
 * `gaussianParticleModel`, so the same model also feeds the bootstrap and auxiliary filters.
 *
 * @param model - The nonlinear Gaussian model.
 * @param options - A constant observation Jacobian, if the observation is linear.
 * @returns The particle model with a `guide`.
 * @throws OmaiError with code `value` when `Q` or `R` is not positive definite.
 * @concept particle-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key } from 'omai-engine/random'
 * const m = filtering.gaussianGuidedModel({ f: (x) => x, h: (x) => x, Q: [[0.1]], R: [[1]], x0: [0], P0: [[1]] })
 * m.guide?.propose(key(0), m.initial(key(1), 4), [0.5], 0).particles.shape // => [4, 1]
 * ```
 */
export function gaussianGuidedModel(model: NonlinearModel, options: GuidedOptions = {}): ParticleModel {
  const base = gaussianParticleModel(model)
  const Q = numbersOf(asTensor(model.Q))
  const R = numbersOf(asTensor(model.R))
  const n = Math.round(Math.sqrt(Q.length))
  const p = Math.round(Math.sqrt(R.length))
  const Qinv = inverseOf(Float64Array.from(Q), n, 'gaussianGuidedModel: Q')
  const Rinv = inverseOf(Float64Array.from(R), p, 'gaussianGuidedModel: R')
  const logDetQ = (() => {
    const { factor } = dpotrf(Q, n)
    let s = 0
    for (let i = 0; i < n; i++) s += 2 * Math.log(factor[i * n + i] ?? 1)
    return s
  })()
  const constantH = options.observationMatrix === undefined ? undefined : numbersOf(asTensor(options.observationMatrix))
  // The proposal covariance S, its upper Cholesky factor U (S = U'U) and log determinant for an observation Jacobian H [p, n].
  const proposalOf = (
    H: ArrayLike<number>,
  ): { S: Float64Array; U: Float64Array; logDetS: number; HtRinv: Float64Array } => {
    // HtRinv = H' R^-1 [n, p].
    const HtRinv = new Float64Array(n * p)
    for (let i = 0; i < n; i++)
      for (let k = 0; k < p; k++) {
        let s = 0
        for (let j = 0; j < p; j++) s += (H[j * n + i] ?? 0) * (Rinv[j * p + k] ?? 0)
        HtRinv[i * p + k] = s
      }
    const P = Float64Array.from(Qinv)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let s = 0
        for (let k = 0; k < p; k++) s += (HtRinv[i * p + k] ?? 0) * (H[k * n + j] ?? 0)
        P[i * n + j] = (P[i * n + j] ?? 0) + s
      }
    const S = inverseOf(P, n, 'gaussianGuidedModel: the proposal precision')
    // Symmetrise against rounding and jitter the diagonal, then factor.
    for (let i = 0; i < n; i++)
      for (let j = 0; j < i; j++) {
        const m = 0.5 * ((S[i * n + j] ?? 0) + (S[j * n + i] ?? 0))
        S[i * n + j] = m
        S[j * n + i] = m
      }
    const jittered = Float64Array.from(S)
    for (let i = 0; i < n; i++) jittered[i * n + i] = (jittered[i * n + i] ?? 0) + GUIDE_JITTER
    const { factor: U } = dpotrf(jittered, n)
    let logDetS = 0
    for (let i = 0; i < n; i++) logDetS += 2 * Math.log(U[i * n + i] ?? 1)
    return { S, U, logDetS, HtRinv }
  }
  const fixed = constantH === undefined ? undefined : proposalOf(constantH)
  const guide: ParticleGuide = {
    propose: (key, particles, z, t) => {
      void t
      const N = particles.shape[0] ?? 0
      const zs = numbersOf(reshape(z, [-1]))
      const eps = numbersOf(normal(key, [N, n]))
      const out = new Float64Array(N * n)
      const logRatio = new Float64Array(N)
      for (let i = 0; i < N; i++) {
        const row = reshape(sliceAxis(particles, 0, i, i + 1), [-1])
        const mu = numbersOf(model.f(row))
        const muValue: Value = fromNumbers(mu, [n])
        const H = constantH ?? numbersOf(reshape(jacfwd(model.h)(muValue), [p, n]))
        const prop = fixed ?? proposalOf(H)
        const hmu = numbersOf(reshape(model.h(muValue), [-1]))
        // m = mu + S H' R^-1 (z - h(mu)).
        const innov = new Float64Array(p)
        for (let k = 0; k < p; k++) innov[k] = (zs[k] ?? 0) - (hmu[k] ?? 0)
        const gain = new Float64Array(n)
        for (let a = 0; a < n; a++) {
          let s = 0
          for (let k = 0; k < p; k++) s += (prop.HtRinv[a * p + k] ?? 0) * (innov[k] ?? 0)
          gain[a] = s
        }
        const m = new Float64Array(n)
        for (let a = 0; a < n; a++) {
          let s = 0
          for (let b = 0; b < n; b++) s += (prop.S[a * n + b] ?? 0) * (gain[b] ?? 0)
          m[a] = (mu[a] ?? 0) + s
        }
        // x = m + U' eps; the proposal's quadratic form at x is eps' eps.
        let eps2 = 0
        const x = new Float64Array(n)
        for (let a = 0; a < n; a++) {
          let s = 0
          for (let b = 0; b <= a; b++) s += (prop.U[b * n + a] ?? 0) * (eps[i * n + b] ?? 0)
          x[a] = (m[a] ?? 0) + s
          eps2 += (eps[i * n + a] ?? 0) ** 2
        }
        // log N(x; mu, Q) - log N(x; m, S); the 2 pi terms cancel.
        const d = new Float64Array(n)
        for (let a = 0; a < n; a++) d[a] = (x[a] ?? 0) - (mu[a] ?? 0)
        let quadQ = 0
        for (let a = 0; a < n; a++)
          for (let b = 0; b < n; b++) quadQ += (d[a] ?? 0) * (Qinv[a * n + b] ?? 0) * (d[b] ?? 0)
        logRatio[i] = -0.5 * (logDetQ + quadQ) + 0.5 * (prop.logDetS + eps2)
        out.set(x, i * n)
      }
      return { particles: fromNumbers(out, [N, n]), logRatio: fromNumbers(logRatio, [N]) }
    },
  }
  return { ...base, guide }
}
