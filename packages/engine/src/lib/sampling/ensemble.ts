/**
 * Ensemble samplers that need no gradient and no tuning of a proposal's shape: a population of walkers moves
 * together, and each move is built from the positions of the other walkers, so the proposal adapts to the target's
 * scale and correlations by itself and the sampler is invariant under affine maps of the coordinates. The affine-invariant
 * stretch move and the differential evolution move of NumPyro's `AIES` (Goodman and Weare 2010, Foreman-Mackey et al.
 * 2013, ter Braak 2006), and ensemble slice sampling (Karamanis and Beutler 2021, NumPyro's `ESS`). Both are
 * `MCMCKernel`s on the product space of the walkers: the position holds every walker (row-major `[walkers, dim]`), the
 * target is the product of the walkers' densities, and a transition updates half the walkers given the other half and
 * then the second half given the first.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, normal, uniformStream } from '../random/index.ts'
import { lowerCholesky } from './_linear.ts'
import type { MCMCKernel, MCMCState } from './chains.ts'
import type { LogDensity } from './target.ts'

/** The state of an ensemble kernel: every walker's position (row-major) with the log density of each. */
export interface EnsembleState extends MCMCState {
  /** The log density of each walker (`logDensity` is their sum, the log density of the product target). */
  readonly walkerLogDensity: Float64Array
  /** Ensemble slice sampling: the scale of the directions, tuned during warm-up. */
  readonly mu: number
  /** Ensemble slice sampling: expansions and contractions since the scale was last tuned. */
  readonly expansions: number
  readonly contractions: number
}

function checkWalkers(numWalkers: number, name: string): void {
  if (!(Number.isInteger(numWalkers) && numWalkers >= 4)) {
    throw new OmaiError('value', `${name}: need an integer number of walkers of at least 4`)
  }
}

function initEnsemble(
  target: LogDensity,
  numWalkers: number,
  position: ArrayLike<number>,
  mu: number,
  name: string,
): EnsembleState {
  const dim = target.dim
  if (position.length !== numWalkers * dim) {
    throw new OmaiError(
      'shape',
      `${name}: the position has ${position.length} values, need ${numWalkers} walkers of ${dim}`,
    )
  }
  const x = Float64Array.from(position)
  const lds = Float64Array.from({ length: numWalkers }, (_, k) => target.logDensity(x.subarray(k * dim, (k + 1) * dim)))
  return {
    position: x,
    logDensity: lds.reduce((a, v) => a + v, 0),
    walkerLogDensity: lds,
    mu,
    expansions: 0,
    contractions: 0,
  }
}

/** The two halves a transition updates in turn: walker indices of the first and the second. */
function halves(numWalkers: number): readonly [readonly number[], readonly number[]] {
  const split = Math.floor(numWalkers / 2)
  const all = Array.from({ length: numWalkers }, (_, i) => i)
  return [all.slice(0, split), all.slice(split)]
}

/** Options of {@link stretchMoveKernel}. */
export interface StretchMoveOptions {
  /** The number of walkers (an integer, at least 4; more than twice the dimension is the usual advice). */
  readonly numWalkers: number
  /** `stretch`: the affine-invariant stretch move (default); `differential`: the differential evolution move. */
  readonly move?: 'stretch' | 'differential'
  /** The stretch scale a (default 2). */
  readonly a?: number
  /** The mean stretch of the differential move (default 2.38 / sqrt(2 dim)) and its relative spread (default 1e-5). */
  readonly gamma?: number
  readonly gammaSpread?: number
}

/**
 * The affine-invariant ensemble sampler (Goodman and Weare 2010, in the parallel form of Foreman-Mackey et al. 2013;
 * NumPyro `AIES` with the `StretchMove`): to move walker x_k, pick a walker y_j of the other half and propose
 * y_j + z (x_k - y_j) with z drawn with density proportional to 1 / sqrt(z) on [1/a, a], accepted with probability
 * min(1, z^(dim - 1) pi(proposal) / pi(x_k)). The `differential` move (ter Braak, as NumPyro's `DEMove`) proposes
 * x_k + gamma (y_i - y_j) for two walkers of the other half and accepts by the density ratio. Both leave the product of the
 * walkers' densities invariant, and the stretch move is invariant under affine transformations, so it needs no
 * preconditioner on correlated or badly scaled targets.
 *
 * @param target - The target (no gradient needed).
 * @param options - The number of walkers and the move.
 * @returns The kernel on the product space, with `dim` equal to walkers times the target's dimension.
 * @throws OmaiError with code `value` for fewer than 4 walkers or a non-positive stretch.
 * @concept ensemble-sampling
 * @reference J. Goodman and J. Weare, Ensemble samplers with affine invariance (CAMCoS 2010); D. Foreman-Mackey,
 *   D. W. Hogg, D. Lang and J. Goodman, emcee: the MCMC hammer (PASP 2013).
 * @index function Affine-invariant ensemble sampler (stretch and differential moves), gradient-free.
 * @example
 * ```ts
 * import { numericTarget, stretchMoveKernel } from 'omai-engine/sampling'
 * import { key, normal } from 'omai-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const k = stretchMoveKernel(t, { numWalkers: 6 })
 * const s = k.step(key(0), k.init(normal(key(1), [6]).data)).state
 * s.walkerLogDensity.length // => 6
 * ```
 */
export function stretchMoveKernel(target: LogDensity, options: StretchMoveOptions): MCMCKernel<EnsembleState> {
  const k = options.numWalkers
  checkWalkers(k, 'stretchMoveKernel')
  const dim = target.dim
  const move = options.move ?? 'stretch'
  const a = options.a ?? 2
  if (!(a > 1)) throw new OmaiError('value', 'stretchMoveKernel: the stretch scale must exceed 1')
  const gamma = options.gamma ?? 2.38 / Math.sqrt(2 * dim)
  const spread = options.gammaSpread ?? 1e-5
  const [first, second] = halves(k)
  return {
    name: move === 'stretch' ? 'stretch-move' : 'differential-move',
    dim: k * dim,
    config: { dim, numWalkers: k, move, a, gamma, gammaSpread: spread },
    init: (position) => initEnsemble(target, k, position, 1, 'stretchMoveKernel'),
    step: (key, state) => {
      const x = Float64Array.from(state.position)
      const lds = Float64Array.from(state.walkerLogDensity)
      let accepted = 0
      let rate = 0
      ;[first, second].forEach((active, half) => {
        const inactive = half === 0 ? second : first
        const kh = foldIn(key, half)
        const u = uniformStream(foldIn(kh, 0), 256)
        const z = move === 'differential' ? normal(foldIn(kh, 1), [active.length]).data : undefined
        for (const [n, walker] of active.entries()) {
          const xw = x.subarray(walker * dim, (walker + 1) * dim)
          const proposal = new Float64Array(dim)
          let logFactor = 0
          if (move === 'stretch') {
            const j = inactive[Math.min(inactive.length - 1, Math.floor(u() * inactive.length))] as number
            const zz = ((a - 1) * u() + 1) ** 2 / a
            logFactor = (dim - 1) * Math.log(zz)
            for (let d = 0; d < dim; d++) {
              const yj = x[j * dim + d] ?? 0
              proposal[d] = yj - (yj - (xw[d] ?? 0)) * zz
            }
          } else {
            const i = Math.min(inactive.length - 1, Math.floor(u() * inactive.length))
            let j = Math.min(inactive.length - 2, Math.floor(u() * (inactive.length - 1)))
            if (j >= i) j += 1
            const g = gamma * (1 + spread * (z?.[n] ?? 0))
            const wi = inactive[i] as number
            const wj = inactive[j] as number
            for (let d = 0; d < dim; d++)
              proposal[d] = (xw[d] ?? 0) + g * ((x[wj * dim + d] ?? 0) - (x[wi * dim + d] ?? 0))
          }
          const lp = target.logDensity(proposal)
          const logAlpha = logFactor + lp - (lds[walker] ?? 0)
          const alpha = Number.isNaN(logAlpha) ? 0 : logAlpha >= 0 ? 1 : Math.exp(logAlpha)
          rate += alpha
          if (u() < alpha) {
            x.set(proposal, walker * dim)
            lds[walker] = lp
            accepted += 1
          }
        }
      })
      return {
        state: { ...state, position: x, walkerLogDensity: lds, logDensity: lds.reduce((s, v) => s + v, 0) },
        info: { acceptanceRate: rate / k, accepted: accepted > 0 },
      }
    },
  }
}

/** Options of {@link ensembleSliceKernel}. */
export interface EnsembleSliceOptions {
  /** The number of walkers (an integer, at least 4). */
  readonly numWalkers: number
  /** The directions: `differential` (the difference of two walkers of the other half, default) or `gaussian` (from its covariance). */
  readonly move?: 'differential' | 'gaussian'
  /** The scale of the directions (default 1); tuned during warm-up when `tuneMu`. */
  readonly mu?: number
  /** Re-tune the scale after every full iteration from the stepping out and shrinking counts (warm-up only: it is not Markov). */
  readonly tuneMu?: boolean
  /** The most stepping-out expansions per walker (default 1000). */
  readonly maxExpansions?: number
  /** The most shrinkage proposals per walker before it stays (default 1000). */
  readonly maxContractions?: number
}

/**
 * Ensemble slice sampling (Karamanis and Beutler 2021; NumPyro `ESS`): each walker of one half does a slice-sampling
 * step along a direction that is built from the other half, the difference of two of its walkers (`differential`) or a
 * draw from the Gaussian fitted to it (`gaussian`), scaled by 2 mu: draw a level under the density, step out a bracket
 * along the line, and sample inside it by shrinking. Every walker moves each transition, there is no acceptance
 * step, and the only parameter, the scale mu, is tuned by the ratio of expansions to contractions (`tuneMu`, in warm-up;
 * the tuned kernel with a fixed mu is Markov).
 *
 * @param target - The target (no gradient needed).
 * @param options - Walkers, move, the scale, whether to tune it and the caps.
 * @returns The kernel on the product space; its state carries the scale and the counts.
 * @throws OmaiError with code `value` for fewer than 4 walkers or a non-positive scale.
 * @concept ensemble-sampling
 * @reference M. Karamanis and F. Beutler, Ensemble slice sampling (Statistics and Computing 2022).
 * @index function Ensemble slice sampler: slice steps along directions built from the other walkers.
 * @example
 * ```ts
 * import { ensembleSliceKernel, numericTarget } from 'omai-engine/sampling'
 * import { key, normal } from 'omai-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const k = ensembleSliceKernel(t, { numWalkers: 6 })
 * k.step(key(0), k.init(normal(key(1), [6]).data)).info.accepted // => true
 * ```
 */
export function ensembleSliceKernel(target: LogDensity, options: EnsembleSliceOptions): MCMCKernel<EnsembleState> {
  const k = options.numWalkers
  checkWalkers(k, 'ensembleSliceKernel')
  const dim = target.dim
  const move = options.move ?? 'differential'
  const mu0 = options.mu ?? 1
  if (!(mu0 > 0)) throw new OmaiError('value', 'ensembleSliceKernel: the scale must be positive')
  const tune = options.tuneMu ?? false
  const maxExpansions = options.maxExpansions ?? 1000
  const maxContractions = options.maxContractions ?? 1000
  const [first, second] = halves(k)
  return {
    name: 'ensemble-slice',
    dim: k * dim,
    config: { dim, numWalkers: k, move, mu: mu0, tuneMu: tune },
    init: (position) => initEnsemble(target, k, position, mu0, 'ensembleSliceKernel'),
    step: (key, state) => {
      const x = Float64Array.from(state.position)
      const lds = Float64Array.from(state.walkerLogDensity)
      const mu = state.mu
      let expansions = state.expansions
      let contractions = state.contractions
      ;[first, second].forEach((active, half) => {
        const inactive = half === 0 ? second : first
        const kh = foldIn(key, half)
        const u = uniformStream(foldIn(kh, 0), 512)
        let factor: Float64Array | undefined
        if (move === 'gaussian') {
          // The Cholesky factor of the other half's covariance (with a little jitter on the diagonal).
          const mean = new Float64Array(dim)
          for (const w of inactive)
            for (let d = 0; d < dim; d++) mean[d] = (mean[d] ?? 0) + (x[w * dim + d] ?? 0) / inactive.length
          const cov = new Float64Array(dim * dim)
          for (const w of inactive)
            for (let p = 0; p < dim; p++)
              for (let q = 0; q < dim; q++)
                cov[p * dim + q] =
                  (cov[p * dim + q] ?? 0) +
                  (((x[w * dim + p] ?? 0) - (mean[p] ?? 0)) * ((x[w * dim + q] ?? 0) - (mean[q] ?? 0))) /
                    (inactive.length - 1)
          for (let d = 0; d < dim; d++) cov[d * dim + d] = (cov[d * dim + d] ?? 0) + 1e-10
          factor = lowerCholesky(cov, dim, 'ensembleSliceKernel')
        }
        const normals = move === 'gaussian' ? normal(foldIn(kh, 1), [active.length, dim]).data : undefined
        for (const [n, walker] of active.entries()) {
          const x0 = Float64Array.from(x.subarray(walker * dim, (walker + 1) * dim))
          const direction = new Float64Array(dim)
          if (move === 'differential') {
            const i = Math.min(inactive.length - 1, Math.floor(u() * inactive.length))
            let j = Math.min(inactive.length - 2, Math.floor(u() * (inactive.length - 1)))
            if (j >= i) j += 1
            const wi = inactive[i] as number
            const wj = inactive[j] as number
            for (let d = 0; d < dim; d++) direction[d] = 2 * mu * ((x[wj * dim + d] ?? 0) - (x[wi * dim + d] ?? 0))
          } else {
            for (let p = 0; p < dim; p++) {
              let s = 0
              for (let q = 0; q <= p; q++) s += (factor?.[p * dim + q] ?? 0) * (normals?.[n * dim + q] ?? 0)
              direction[p] = 2 * mu * s
            }
          }
          const along = (t: number): number => target.logDensity(x0.map((v, d) => v + t * (direction[d] ?? 0)))
          const level = (lds[walker] ?? 0) + Math.log(1 - u()) // log y = log pi(x) - Exp(1)
          let left = -u()
          let right = left + 1
          // Stepping out, the expansion budget split at random between the two sides (Neal 2003).
          let budgetLeft = Math.floor(maxExpansions * u())
          let budgetRight = maxExpansions - 1 - budgetLeft
          while (budgetLeft > 0 && along(left) > level) {
            left -= 1
            budgetLeft -= 1
            expansions += 1
          }
          while (budgetRight > 0 && along(right) > level) {
            right += 1
            budgetRight -= 1
            expansions += 1
          }
          for (let s = 0; s < maxContractions; s++) {
            const t = left + u() * (right - left)
            const lp = along(t)
            if (lp > level) {
              x.set(
                x0.map((v, d) => v + t * (direction[d] ?? 0)),
                walker * dim,
              )
              lds[walker] = lp
              break
            }
            if (t < 0) left = t
            else right = t
            contractions += 1
          }
        }
      })
      let nextMu = mu
      let nextExp = expansions
      let nextCon = contractions
      if (tune) {
        nextMu = (2 * Math.max(1, expansions)) / (Math.max(1, expansions) + contractions)
        nextExp = 0
        nextCon = 0
      }
      return {
        state: {
          position: x,
          logDensity: lds.reduce((s, v) => s + v, 0),
          walkerLogDensity: lds,
          mu: nextMu,
          expansions: nextExp,
          contractions: nextCon,
        },
        info: { acceptanceRate: 1, accepted: true, stepSize: mu },
      }
    },
  }
}
