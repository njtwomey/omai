/**
 * Microcanonical Langevin Monte Carlo (Robnik, De Luca, Silverstein and Seljak 2023; BlackJAX `mclmc`): the unadjusted
 * sampler of the isokinetic (ESH) dynamics, where the momentum is a unit vector that the gradient rotates instead of a
 * Gaussian vector it accelerates, a partial refresh of its direction every step, and no Metropolis step. It is cheap (one
 * gradient per step), has no accept-reject cost and, tuned, matches NUTS on many targets, at the price of a bias that
 * shrinks as the step size does. Also its three-phase tuning (step size from the energy error, momentum decoherence
 * length and a diagonal preconditioner from the chain's own variance) as a stepped algorithm, as BlackJAX's
 * `mclmc_find_L_and_step_size`.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { fromStorage } from '../tensor/index.ts'
import { type Algorithm, channels, numberFromJson, numberToJson, stateFromJson, stateToJson } from '../trace/index.ts'
import { vectorOf } from './_linear.ts'
import { checkStepSize } from './_mcmc.ts'
import { type AdaptPath, chainFromJson, chainToJson } from './adaptation-contract.ts'
import type { MCMCKernel, MCMCState } from './chains.ts'
import { ess } from './diagnostics/index.ts'
import { INTEGRATOR_COEFFICIENTS, type IntegratorScheme } from './hamiltonian/integrators.ts'
import type { LogDensity } from './target.ts'

/** The state of {@link mclmcKernel}: the position with its log density and gradient, the unit momentum and the last energy error. */
export interface MclmcState extends MCMCState {
  readonly grad: Float64Array
  /** The direction of motion, a unit vector in the preconditioned coordinates (all zeros before the first step). */
  readonly momentum: Float64Array
  /** The energy change of the last transition (the kinetic energy change minus the log density change). */
  readonly energyChange: number
}

/** Options of {@link mclmcKernel}. */
export interface MclmcOptions {
  /** The integrator step size (positive). */
  readonly stepSize: number
  /** The momentum decoherence length L: the distance over which the direction is forgotten (positive; `Infinity` never refreshes). */
  readonly momentumDecoherenceLength: number
  /** The diagonal inverse mass matrix, as a number or a vector (default 1): the posterior variances, when known. */
  readonly inverseMassMatrix?: TensorLike | number
  /** The splitting scheme (default `velocityVerlet`; `mcLachlan` is BlackJAX's default). */
  readonly integrator?: IntegratorScheme
}

function unitVector(key: Key, dim: number): Float64Array {
  const z = normal(key, [dim]).data
  const n = dnrm2(z)
  return z.map((v) => v / n)
}

/** One ESH momentum update (BlackJAX `esh_dynamics_momentum_update_one_step`): the rotated momentum and the kinetic energy change. */
function eshMomentumUpdate(
  u: Float64Array,
  gradient: Float64Array,
  sqrtInverse: Float64Array,
  stepSize: number,
  coef: number,
): { readonly momentum: Float64Array; readonly kineticChange: number } {
  const dim = u.length
  const g = gradient.map((v, i) => v * (sqrtInverse[i] ?? 1))
  const norm = dnrm2(g)
  const e = norm > 1e-13 ? g.map((v) => v / norm) : g
  let proj = 0
  for (let i = 0; i < dim; i++) proj += (u[i] ?? 0) * (e[i] ?? 0)
  const delta = (stepSize * coef * norm) / (dim - 1)
  const zeta = Math.exp(-delta)
  const raw = u.map((v, i) => (e[i] ?? 0) * (1 - zeta) * (1 + zeta + proj * (1 - zeta)) + 2 * zeta * v)
  const rn = dnrm2(raw)
  const next = rn > 1e-13 ? raw.map((v) => v / rn) : raw
  let sPlus = 0
  let sMinus = 0
  for (let i = 0; i < dim; i++) {
    sPlus += 0.5 * ((u[i] ?? 0) + (e[i] ?? 0)) ** 2
    sMinus += 0.5 * ((u[i] ?? 0) - (e[i] ?? 0)) ** 2
  }
  const z2 = zeta * zeta
  const logA = Math.log(z2 <= 1 ? sPlus * (1 - z2) + 2 * z2 : sMinus * (z2 - 1) + 2)
  return { momentum: next, kineticChange: (delta - Math.LN2 + logA) * (dim - 1) }
}

/** The partial refresh of the direction: add Gaussian noise scaled by the decoherence length and renormalise. */
function refresh(u: Float64Array, key: Key, stepSize: number, length: number): Float64Array {
  if (!Number.isFinite(length)) return u
  const dim = u.length
  const nu = Math.sqrt((Math.exp((2 * stepSize) / length) - 1) / dim)
  const z = normal(key, [dim]).data
  const m = u.map((v, i) => v + nu * (z[i] ?? 0))
  const n = dnrm2(m)
  return m.map((v) => v / n)
}

/**
 * Microcanonical Langevin Monte Carlo: per transition the direction is partly refreshed, one step of the isokinetic
 * (ESH) dynamics is taken with a symplectic splitting (a position update x += eps sqrt(M^-1) u between rotations of u
 * towards the gradient), and the direction is partly refreshed again. There is no Metropolis correction, so the chain
 * samples a distribution that differs from the target by an error that vanishes with the step size (BlackJAX's
 * `mclmc`); the energy change of each step is the diagnostic that tuning controls. A step that leaves the support or
 * produces a non-finite energy is undone and the direction is redrawn. The target needs at least two coordinates.
 *
 * @param target - The target (with a gradient, at least two coordinates).
 * @param options - Step size, momentum decoherence length, diagonal inverse mass matrix and splitting scheme.
 * @returns The kernel; its state carries the gradient, the direction and the energy change of the last step.
 * @throws OmaiError with code `value` for a target without a gradient or with one coordinate, a non-positive step size
 *   or length, or non-positive inverse masses.
 * @concept microcanonical-langevin-monte-carlo
 * @reference J. Robnik, G. B. De Luca, E. Silverstein and U. Seljak, Microcanonical Hamiltonian Monte Carlo (JMLR 2023);
 *   J. Robnik and U. Seljak, Microcanonical Langevin Monte Carlo (arXiv 2023).
 * @index function Microcanonical Langevin Monte Carlo kernel: isokinetic dynamics, no accept step.
 * @example
 * ```ts
 * import { mclmcKernel, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const t = numericTarget((x) => -0.5 * ((x[0] ?? 0) ** 2 + (x[1] ?? 0) ** 2), 2, { grad: (x) => [-(x[0] ?? 0), -(x[1] ?? 0)] })
 * const k = mclmcKernel(t, { stepSize: 0.5, momentumDecoherenceLength: 2 })
 * Math.abs(k.step(key(0), k.init([0.3, -0.2])).state.momentum.reduce((a, v) => a + v * v, 0) - 1) < 1e-12 // => true
 * ```
 */
export function mclmcKernel(target: LogDensity, options: MclmcOptions): MCMCKernel<MclmcState> {
  const eps = options.stepSize
  const length = options.momentumDecoherenceLength
  checkStepSize(eps, 'mclmcKernel')
  if (!(length > 0)) throw new OmaiError('value', 'mclmcKernel: the momentum decoherence length must be positive')
  if (!target.hasGradient) throw new OmaiError('value', 'mclmcKernel: the target has no gradient')
  const dim = target.dim
  if (dim < 2) throw new OmaiError('value', 'mclmcKernel: the target needs at least two coordinates')
  const inverse = vectorOf(options.inverseMassMatrix ?? 1, dim, 'mclmcKernel')
  if (!inverse.every((v) => v > 0 && Number.isFinite(v))) {
    throw new OmaiError('value', 'mclmcKernel: the inverse mass matrix must be positive and finite')
  }
  const sqrtInverse = inverse.map(Math.sqrt)
  const scheme = options.integrator ?? 'velocityVerlet'
  const coefficients = typeof scheme === 'string' ? INTEGRATOR_COEFFICIENTS[scheme] : scheme
  if (coefficients.length % 2 !== 1) throw new OmaiError('value', 'mclmcKernel: need an odd number of coefficients')
  return {
    name: 'mclmc',
    dim,
    config: { dim, stepSize: eps, momentumDecoherenceLength: Number.isFinite(length) ? length : 'infinite' },
    init: (position) => {
      if (position.length !== dim) {
        throw new OmaiError('shape', `mclmcKernel: the position has ${position.length} values, need ${dim}`)
      }
      const x = Float64Array.from(position)
      const r = target.valueAndGrad(x)
      return { position: x, logDensity: r.value, grad: r.grad, momentum: new Float64Array(dim), energyChange: 0 }
    },
    step: (key, state) => {
      // The first step draws the direction (an initial state has none); later steps keep it.
      const direction = dnrm2(state.momentum) > 0 ? state.momentum : unitVector(foldIn(key, 9), dim)
      let u = refresh(direction, foldIn(key, 0), eps * 0.5, length)
      let x = state.position
      let logDensity = state.logDensity
      let grad = state.grad
      let kinetic = 0
      let evaluations = 0
      const last = coefficients.length - 1
      for (let i = 0; i <= last; i++) {
        const c = coefficients[i] ?? 0
        if (i % 2 === 0) {
          const m = eshMomentumUpdate(u, grad, sqrtInverse, eps, c)
          u = m.momentum
          kinetic += m.kineticChange
        } else {
          const uu = u
          const xx = x
          x = Float64Array.from(xx, (v, j) => v + eps * c * (sqrtInverse[j] ?? 1) * (uu[j] ?? 0))
          const r = target.valueAndGrad(x)
          logDensity = r.value
          grad = r.grad
          evaluations += 1
        }
      }
      const energyChange = kinetic - logDensity + state.logDensity
      const finite = Number.isFinite(logDensity) && Number.isFinite(energyChange) && u.every(Number.isFinite)
      if (!finite) {
        return {
          state: { ...state, momentum: unitVector(foldIn(key, 1), dim), energyChange: 0 },
          info: { acceptanceRate: 0, accepted: false, numSteps: evaluations, stepSize: eps },
        }
      }
      return {
        state: {
          position: x,
          logDensity,
          grad,
          momentum: refresh(u, foldIn(key, 2), eps * 0.5, length),
          energyChange,
        },
        info: { acceptanceRate: 1, accepted: true, numSteps: evaluations, stepSize: eps },
      }
    },
  }
}

/** Options of {@link mclmcAdaptationAlgorithm}. */
export interface MclmcAdaptationOptions {
  /** The warm-up length: the three phases split it (30% to the step size, 30% to the variance and 10% more to refine it, the rest to L). */
  readonly numWarmup: number
  /** The energy variance per dimension the step size aims for (default 5e-4, BlackJAX's). */
  readonly desiredEnergyVar?: number
  /** Learn a diagonal inverse mass matrix from the second phase (default true). */
  readonly diagonalPreconditioning?: boolean
  /** The factor from the estimated autocorrelation length to L (default 0.4). */
  readonly lFactor?: number
  /** How much to trust the one-step estimate of the step size (default 1.5). */
  readonly trustInEstimate?: number
  /** The effective number of samples the running estimates average over (default 150). */
  readonly numEffectiveSamples?: number
  readonly integrator?: IntegratorScheme
}

/** The state of {@link mclmcAdaptationAlgorithm}. */
export interface MclmcWarmupState {
  readonly chain: MclmcState
  readonly stepSize: number
  readonly length: number
  readonly inverseMassMatrix: Float64Array
  /** The weighted running averages of the step-size estimate: the normaliser and the weighted sum. */
  readonly time: number
  readonly xAverage: number
  /** The step size above which a transition failed (infinite until one does). */
  readonly stepSizeMax: number
  /** The step-size weighted running mean of x and x^2 (`[2 dim]`) and its total weight. */
  readonly moments: Float64Array
  readonly momentsWeight: number
  /** The positions of the last phase, row-major (for the autocorrelation length). */
  readonly samples: Float64Array
}

/** The schedule of the three phases, from the warm-up length. */
function mclmcSchedule(numWarmup: number, preconditioning: boolean): readonly [number, number, number, number] {
  const n1 = Math.round(numWarmup * 0.3)
  const n2 = Math.round(numWarmup * 0.3)
  const n2b = preconditioning ? Math.round(n2 / 3) : 0
  const n3 = Math.max(0, numWarmup - n1 - n2 - n2b)
  return [n1, n2, n2b, n3]
}

/**
 * The tuning of MCLMC as a stepped algorithm (BlackJAX `mclmc_find_L_and_step_size`), one transition per step in three
 * phases of the warm-up. In the first two, a running estimate of the step size that gives the desired energy variance
 * (the energy error scales as the step size to the sixth power, so each transition proposes the step size its energy
 * error implies, down-weighted when far from the target, and a transition that fails caps the step size at 80% of
 * itself) and, in the second, the chain's step-size weighted mean and variance, from which L becomes the square root of
 * the summed variances and the diagonal inverse mass matrix the variances (then L is the square root of the dimension and
 * a short refinement of the step size follows). The last phase runs with those parameters and sets L from the
 * autocorrelation of the positions: `lFactor` times the step size times the mean integrated autocorrelation time. The
 * tuned parameters are read with {@link mclmcAdaptationResult}; the state serialises.
 *
 * @param target - The target.
 * @param position - The starting position.
 * @param options - Warm-up length, desired energy variance, preconditioning, L factor and the trust and averaging knobs.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for fewer than 10 warm-up steps.
 * @concept microcanonical-langevin-monte-carlo
 * @example
 * ```ts
 * import { mclmcAdaptationAlgorithm, mclmcAdaptationResult, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const t = numericTarget((x) => -0.5 * ((x[0] ?? 0) ** 2 + (x[1] ?? 0) ** 2), 2, { grad: (x) => [-(x[0] ?? 0), -(x[1] ?? 0)] })
 * const alg = mclmcAdaptationAlgorithm(t, Float64Array.of(0.5, -0.5), { numWarmup: 200 })
 * const run = runAlgorithm(alg, { key: key(0), maxSteps: 200 })
 * mclmcAdaptationResult(run.state).stepSize > 0 // => true
 * ```
 */
export function mclmcAdaptationAlgorithm(
  target: LogDensity,
  position: Float64Array,
  options: MclmcAdaptationOptions,
): Algorithm<MclmcWarmupState, AdaptPath> {
  const dim = target.dim
  if (options.numWarmup < 10) throw new OmaiError('value', 'mclmcAdaptationAlgorithm: at least 10 warm-up steps')
  const desired = options.desiredEnergyVar ?? 5e-4
  const preconditioning = options.diagonalPreconditioning ?? true
  const lFactor = options.lFactor ?? 0.4
  const trust = options.trustInEstimate ?? 1.5
  const decay = ((options.numEffectiveSamples ?? 150) - 1) / ((options.numEffectiveSamples ?? 150) + 1)
  const [n1, n2, n2b] = mclmcSchedule(options.numWarmup, preconditioning)
  const phase2End = n1 + n2
  const refineEnd = phase2End + n2b
  const kernelFor = (s: {
    stepSize: number
    length: number
    inverseMassMatrix: Float64Array
  }): MCMCKernel<MclmcState> =>
    mclmcKernel(target, {
      stepSize: s.stepSize,
      momentumDecoherenceLength: s.length,
      inverseMassMatrix: s.inverseMassMatrix,
      ...(options.integrator === undefined ? {} : { integrator: options.integrator }),
    })
  return {
    name: 'sampling/mclmc-adaptation',
    config: { dim, numWarmup: options.numWarmup, desiredEnergyVar: desired, diagonalPreconditioning: preconditioning },
    schema: channels({
      metrics: {
        stepSize: { description: 'step size in force' },
        energyChange: { description: 'energy change of the transition' },
        logDensity: { description: 'log density at the current position' },
      },
    }),
    init: () => {
      const inverseMassMatrix = new Float64Array(dim).fill(1)
      const stepSize = Math.sqrt(dim) * 0.25
      const length = Math.sqrt(dim)
      return {
        chain: kernelFor({ stepSize, length, inverseMassMatrix }).init(position),
        stepSize,
        length,
        inverseMassMatrix,
        time: 0,
        xAverage: 0,
        stepSizeMax: Number.POSITIVE_INFINITY,
        moments: new Float64Array(2 * dim),
        momentsWeight: 0,
        samples: new Float64Array(0),
      }
    },
    step: (s, ctx) => {
      const t = ctx.t
      const tuning = t < refineEnd
      const kernel = kernelFor(s)
      const { state, info } = kernel.step(ctx.key, s.chain)
      ctx.log('adapt/stepSize', s.stepSize)
      ctx.log('adapt/accept', info.acceptanceRate)
      ctx.log('adapt/logDensity', state.logDensity)
      if (!tuning) {
        // The last phase: the parameters stand, the positions are kept for the autocorrelation length.
        const grown = new Float64Array(s.samples.length + dim)
        grown.set(s.samples)
        grown.set(state.position, s.samples.length)
        const next = { ...s, chain: state, samples: grown }
        if (t + 1 < options.numWarmup) return next
        const rows = grown.length / dim
        if (rows < 4) return next
        let sum = 0
        let count = 0
        for (let j = 0; j < dim; j++) {
          const column = Float64Array.from({ length: rows }, (_, i) => grown[i * dim + j] ?? 0)
          const e = ess(fromStorage(column, [1, rows], 'float64'))
          if (Number.isFinite(e) && e > 0) {
            sum += rows / e
            count += 1
          }
        }
        return count === 0
          ? next
          : { ...next, length: lFactor * s.stepSize * (sum / count), samples: new Float64Array(0) }
      }
      const ok = info.accepted && Number.isFinite(state.energyChange)
      let stepSizeMax = s.stepSizeMax
      let stepSize = s.stepSize
      let time = s.time
      let xAverage = s.xAverage
      if (!ok) {
        // A failed transition: no estimate from it, and the step size shrinks below where it failed.
        stepSizeMax = s.stepSize * 0.8
        stepSize = Math.min(s.stepSize, stepSizeMax)
      } else {
        const xi = (state.energyChange * state.energyChange) / (dim * desired) + 1e-8
        const weight = Math.exp(-0.5 * (Math.log(xi) / (6 * trust)) ** 2)
        xAverage = decay * xAverage + weight * (xi / s.stepSize ** 6)
        time = decay * time + weight
        stepSize = Math.min((xAverage / time) ** (-1 / 6), stepSizeMax)
      }
      let moments = s.moments
      let momentsWeight = s.momentsWeight
      // The second phase (and the refinement) average x and x^2 with the step size as weight.
      if (t >= n1 && t < phase2End && ok) {
        const w = stepSize
        const total = momentsWeight + w
        moments = Float64Array.from(s.moments, (m, k) => {
          const xv = state.position[k % dim] ?? 0
          const value = k < dim ? xv : xv * xv
          return (momentsWeight * m + w * value) / total
        })
        momentsWeight = total
      }
      let next: MclmcWarmupState = { ...s, chain: state, stepSize, time, xAverage, stepSizeMax, moments, momentsWeight }
      if (t + 1 === phase2End && n2 > 1) {
        const variances = Float64Array.from({ length: dim }, (_, j) =>
          Math.max(1e-12, (moments[dim + j] ?? 0) - (moments[j] ?? 0) ** 2),
        )
        let total = 0
        for (const v of variances) total += v
        next = preconditioning
          ? { ...next, length: Math.sqrt(dim), inverseMassMatrix: variances }
          : { ...next, length: Math.sqrt(total) }
        // The mass matrix changed: the gradient and energy at the current state stand, the step size is re-estimated.
        next = { ...next, time: 0, xAverage: 0, stepSizeMax: Number.POSITIVE_INFINITY }
      }
      return next
    },
    observe: (s) => ({
      metrics: { stepSize: s.stepSize, logDensity: s.chain.logDensity, energyChange: s.chain.energyChange },
    }),
    done: (_, t) => (t >= options.numWarmup ? 'completed' : undefined),
    serialise: (s) => ({
      chain: chainToJson(s.chain),
      stepSize: numberToJson(s.stepSize),
      length: numberToJson(s.length),
      inverseMassMatrix: stateToJson(s.inverseMassMatrix),
      time: s.time,
      xAverage: numberToJson(s.xAverage),
      stepSizeMax: numberToJson(s.stepSizeMax),
      moments: stateToJson(s.moments),
      momentsWeight: s.momentsWeight,
      samples: stateToJson(s.samples),
    }),
    deserialise: (j) => {
      const o = j as unknown as Record<string, Json>
      return {
        chain: chainFromJson<MclmcState>(o['chain'] as Json),
        stepSize: numberFromJson(o['stepSize']),
        length: numberFromJson(o['length']),
        inverseMassMatrix: stateFromJson(o['inverseMassMatrix'] as Json) as Float64Array,
        time: o['time'] as number,
        xAverage: numberFromJson(o['xAverage']),
        stepSizeMax: numberFromJson(o['stepSizeMax']),
        moments: stateFromJson(o['moments'] as Json) as Float64Array,
        momentsWeight: o['momentsWeight'] as number,
        samples: stateFromJson(o['samples'] as Json) as Float64Array,
      }
    },
  }
}

/** The tuned parameters of a warm-up state: what {@link mclmcKernel} takes. */
export interface MclmcParameters {
  readonly stepSize: number
  readonly momentumDecoherenceLength: number
  readonly inverseMassMatrix: Float64Array
}

/**
 * The parameters a warm-up state ends on.
 *
 * @param state - A state of {@link mclmcAdaptationAlgorithm}.
 * @returns The step size, the momentum decoherence length and the diagonal inverse mass matrix.
 * @concept microcanonical-langevin-monte-carlo
 * @example
 * ```ts
 * import { mclmcAdaptationAlgorithm, mclmcAdaptationResult, numericTarget } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const t = numericTarget((x) => -0.5 * ((x[0] ?? 0) ** 2 + (x[1] ?? 0) ** 2), 2, { grad: (x) => [-(x[0] ?? 0), -(x[1] ?? 0)] })
 * const run = runAlgorithm(mclmcAdaptationAlgorithm(t, Float64Array.of(0, 0), { numWarmup: 100 }), { key: key(1), maxSteps: 100 })
 * mclmcAdaptationResult(run.state).inverseMassMatrix.length // => 2
 * ```
 */
export function mclmcAdaptationResult(state: MclmcWarmupState): MclmcParameters {
  return {
    stepSize: state.stepSize,
    momentumDecoherenceLength: state.length,
    inverseMassMatrix: state.inverseMassMatrix,
  }
}
