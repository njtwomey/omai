/**
 * The kernels of the sampling library as inference methods on programs and bound specs, one method each with its own
 * knobs and its own tape: random-walk Metropolis with an adapted proposal (`rwmMethod`), the Metropolis-adjusted Langevin
 * algorithm (`malaMethod`), Barker's proposal (`barkerMethod`), Hamiltonian Monte Carlo with a fixed trajectory
 * (`hmcMethod`), Neal's slice sampler (`sliceMethod`), elliptical slice sampling for Gaussian-prior latents
 * (`ellipticalSliceMethod`), the stochastic-gradient samplers SGLD and SGHMC on a program with subsampled plates
 * (`sgldMethod`, `sghmcMethod`) and microcanonical Langevin Monte Carlo (`mclmcMethod`). Every one runs the chains one
 * transition per step on the unconstrained scale of the model (the layout of `potentialEnergy`, bijectors and enumerated
 * discrete sites handled as for NUTS), warms up with the library's own adaptation where the kernel has one, cancels within
 * one iteration, resumes from a serialised state and publishes its draws so far as a posterior. `nutsMethod` stays the
 * method of NUTS.
 */

import { OmaiError } from '../../base/index.ts'
import { blockDiag } from '../../numerics/linalg/index.ts'
import { foldIn as foldInKey, subsample } from '../../random/index.ts'
import {
  barkerKernel,
  ellipticalSlice,
  type HamiltonianParameters,
  hmcKernel,
  type MCMCKernel,
  type MclmcParameters,
  type MclmcWarmupState,
  malaKernel,
  mclmcAdaptationAlgorithm,
  mclmcAdaptationResult,
  mclmcKernel,
  polynomialStepSize,
  type RwmWarmupState,
  randomWalkAdaptationAlgorithm,
  randomWalkMetropolis,
  type StochasticGradient,
  sghmcKernel,
  sgldKernel,
  sliceSampler,
  type TransitionInfo,
} from '../../sampling/index.ts'
import { bool, choice, dict, int, real } from '../../space/index.ts'
import {
  asTensor,
  concat,
  diag,
  item,
  matmul,
  mul,
  ones,
  reshape,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  transpose,
  zeros,
} from '../../tensor/index.ts'
import type { Config } from '../../trace/index.ts'
import { analyse, type ProgramAnalysis } from '../analysis.ts'
import { potentialEnergy } from '../density.ts'
import { substitute } from '../handlers.ts'
import type { Message } from '../primitives.ts'
import {
  type McmcBuild,
  type McmcPath,
  type McmcPlan,
  type McmcRunState,
  mcmcMethodOf,
  type Warmer,
} from './_mcmc-driver.ts'
import { fixedPlan, massDiagonal, massOf, windowKnobs, windowPlan } from './_mcmc-families.ts'
import { flag, isProgramAnalysis, type MethodModel, type ModelAnalysis, no, num, pick, yes } from './_support.ts'
import type { InferenceMethod } from './method.ts'

/** The type of an MCMC method on a model: its state, knobs, tape paths and analysis. */
export type McmcMethod = InferenceMethod<McmcRunState, Config, McmcPath, ModelAnalysis>

/**
 * Random-walk Metropolis as a method: Gaussian steps on the unconstrained scale, a proposal tuned in warm-up by the
 * library's Robbins-Monro adaptation of the scale towards `targetAcceptance` (0.234) and, in the second half of the warm-up,
 * of the shape to the covariance of the chain (`adaptCovariance`). One transition of every chain per step.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `rwm`).
 * @returns The method.
 * @concept random-walk-metropolis
 * @index estimator Random-walk Metropolis with an adapted covariance as a steppable method.
 * @example
 * ```ts
 * import { rwmMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const m = rwmMethod(model)
 * const s = session(m, { config: { numChains: 1, numWarmup: 40, numDraws: 30 }, steps: 500, mode: 'eager' })
 * [s.status, m.posterior?.(s.state, s.config).status] // => ['converged', 'completed']
 * ```
 */
export function rwmMethod(model: MethodModel, options: { readonly name?: string } = {}): McmcMethod {
  type T = { readonly scale: number; readonly shape: Float64Array }
  return mcmcMethodOf<T>(
    {
      name: 'rwm',
      description:
        'random-walk Metropolis with an adapted scale and covariance, one transition of every chain per step',
      knobs: dict({
        targetAcceptance: real(0.05, 0.9, { default: 0.234, label: 'target acceptance' }),
        adaptCovariance: bool({ default: true, label: 'adapt the proposal covariance' }),
      }),
      plan: (b) => {
        const dim = b.um.dim
        const target = b.um.logDensity
        const numWarmup = num(b.config, 'numWarmup')
        const goal = num(b.config, 'targetAcceptance')
        const adaptCovariance = flag(b.config, 'adaptCovariance')
        const kernel = (t: T): MCMCKernel => {
          if (t.shape.length === 0) return randomWalkMetropolis(target, { scale: t.scale })
          // The proposal's covariance is (scale L)(scale L)^T for the learned Cholesky factor L.
          const factor = reshape(asTensor(t.shape), [dim, dim])
          return randomWalkMetropolis(target, {
            covariance: mul(mul(matmul(factor, transpose(factor)), t.scale), t.scale),
          })
        }
        const cell: { info: TransitionInfo | undefined } = { info: undefined }
        const algorithmFor = (start: Float64Array) =>
          randomWalkAdaptationAlgorithm(target, start, {
            numWarmup,
            targetAcceptance: goal,
            adaptCovariance,
            onTransition: (_, info) => {
              cell.info = info
            },
          })
        const shared = algorithmFor(Float64Array.from(toFlat(zeros([dim]))))
        const warm: Warmer<T> | undefined =
          numWarmup > 0
            ? {
                init: (start, key) => algorithmFor(start).init(key),
                step: (w, ctx) => {
                  cell.info = undefined
                  return shared.step(w as RwmWarmupState, ctx)
                },
                chain: (w) => (w as RwmWarmupState).chain,
                info: () => cell.info,
                result: (w) => ({
                  scale: Math.exp((w as RwmWarmupState).logScale),
                  // banned-ok: sentinel: an empty initial array (no draws yet)
                  shape: (w as RwmWarmupState).shape ?? new Float64Array(0),
                }),
                toJson: (w) => shared.serialise?.(w as RwmWarmupState) ?? null,
                fromJson: (j) => shared.deserialise?.(j),
              }
            : undefined
        return {
          // banned-ok: sentinel: an empty initial array (no draws yet)
          ...fixedPlan<T>(b, { scale: 2.38 / Math.sqrt(dim), shape: new Float64Array(0) }, kernel, (t) => ({
            scale: t.scale,
            shape: Array.from(t.shape),
          })),
          ...(warm === undefined ? {} : { warm }),
        }
      },
    },
    model,
    options,
  )
}

/**
 * The Metropolis-adjusted Langevin algorithm as a method: a Langevin proposal accepted by the Metropolis-Hastings ratio,
 * with the step size tuned by dual averaging towards `targetAcceptance` (0.574, the optimum in high dimension) and the
 * preconditioner learned in Stan's slow windows (`dense` for a full covariance). One transition of every chain per step.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `mala`).
 * @returns The method.
 * @concept langevin-monte-carlo
 * @index estimator MALA with window-adapted step size and preconditioner as a steppable method.
 * @example
 * ```ts
 * import { malaMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const m = malaMethod(model)
 * const s = session(m, { config: { numChains: 1, numWarmup: 40, numDraws: 30 }, steps: 500, mode: 'eager' })
 * s.status // => 'converged'
 * ```
 */
export function malaMethod(model: MethodModel, options: { readonly name?: string } = {}): McmcMethod {
  return mcmcMethodOf<HamiltonianParameters>(
    {
      name: 'mala',
      description:
        'the Metropolis-adjusted Langevin algorithm with window adaptation, one transition of every chain per step',
      knobs: windowKnobs(0.574, true),
      plan: (b) =>
        windowPlan(
          b,
          (p) => malaKernel(b.um.logDensity, { stepSize: p.stepSize, preconditioner: massOf(p, b.um.dim) }),
          { targetAcceptance: 0.574 },
        ),
    },
    model,
    options,
  )
}

/**
 * Barker's proposal as a method: a gradient-skewed Metropolis-Hastings kernel that tolerates a wide range of step sizes,
 * with the step size tuned by dual averaging towards `targetAcceptance` (0.4) and a diagonal preconditioner learned in
 * Stan's slow windows. One transition of every chain per step.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `barker`).
 * @returns The method.
 * @concept barker-proposal
 * @index estimator Barker's proposal with window-adapted step size as a steppable method.
 * @example
 * ```ts
 * import { barkerMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const s = session(barkerMethod(model), { config: { numChains: 1, numWarmup: 40, numDraws: 30 }, steps: 500, mode: 'eager' })
 * s.status // => 'converged'
 * ```
 */
export function barkerMethod(model: MethodModel, options: { readonly name?: string } = {}): McmcMethod {
  return mcmcMethodOf<HamiltonianParameters>(
    {
      name: 'barker',
      description: "Barker's gradient-skewed proposal with window adaptation, one transition of every chain per step",
      knobs: windowKnobs(0.4, false),
      plan: (b) =>
        windowPlan(
          b,
          (p) =>
            barkerKernel(b.um.logDensity, {
              stepSize: p.stepSize,
              scale: massDiagonal(p, b.um.dim).map(Math.sqrt),
            }),
          { targetAcceptance: 0.4 },
        ),
    },
    model,
    options,
  )
}

/**
 * Hamiltonian Monte Carlo with a fixed number of leapfrog steps as a method (the fixed-trajectory sibling of NUTS): the
 * step size and mass matrix are tuned by Stan's window adaptation towards `targetAcceptance` (0.8), the trajectory is
 * `numSteps` steps of the tuned size. One transition of every chain per step.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `hmc`).
 * @returns The method.
 * @concept hamiltonian-monte-carlo
 * @index estimator HMC with a fixed trajectory and window adaptation as a steppable method.
 * @example
 * ```ts
 * import { hmcMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const s = session(hmcMethod(model), { config: { numChains: 1, numWarmup: 40, numDraws: 30, numSteps: 5 }, steps: 500, mode: 'eager' })
 * s.status // => 'converged'
 * ```
 */
export function hmcMethod(model: MethodModel, options: { readonly name?: string } = {}): McmcMethod {
  return mcmcMethodOf<HamiltonianParameters>(
    {
      name: 'hmc',
      description:
        'Hamiltonian Monte Carlo with a fixed trajectory and window adaptation, one transition of every chain per step',
      knobs: dict({
        ...windowKnobs(0.8, true).spaces,
        numSteps: int(1, 1000, { default: 10, label: 'leapfrog steps' }),
      }),
      plan: (b) => {
        const steps = num(b.config, 'numSteps')
        return windowPlan(
          b,
          (p) =>
            hmcKernel(b.um.logDensity, {
              stepSize: p.stepSize,
              inverseMassMatrix: p.inverseMassMatrix,
              numIntegrationSteps: steps,
            }),
          { targetAcceptance: 0.8 },
        )
      },
    },
    model,
    options,
  )
}

/**
 * Neal's slice sampler as a method: coordinate-wise stepping out and shrinking on the unconstrained scale, no gradient
 * and nothing to tune beyond the bracket `width`. One transition of every chain per step; warm-up is burn-in.
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `slice`).
 * @returns The method.
 * @concept slice-sampling
 * @index estimator Coordinate-wise slice sampling as a steppable method.
 * @example
 * ```ts
 * import { sliceMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const s = session(sliceMethod(model), { config: { numChains: 1, numWarmup: 10, numDraws: 30 }, steps: 500, mode: 'eager' })
 * s.status // => 'converged'
 * ```
 */
export function sliceMethod(model: MethodModel, options: { readonly name?: string } = {}): McmcMethod {
  type T = { readonly width: number; readonly maxStepsOut: number }
  return mcmcMethodOf<T>(
    {
      name: 'slice',
      description: 'coordinate-wise slice sampling on the unconstrained scale, one transition of every chain per step',
      knobs: dict({
        width: real(0.01, 100, { log: true, default: 1, label: 'bracket width' }),
        maxStepsOut: int(0, 1000, { default: 50, label: 'most stepping-out steps' }),
      }),
      plan: (b) =>
        fixedPlan<T>(
          b,
          { width: num(b.config, 'width'), maxStepsOut: num(b.config, 'maxStepsOut') },
          (t) => sliceSampler(b.um.logDensity, { width: t.width, maxStepsOut: t.maxStepsOut }),
          (t) => ({ ...t }),
        ),
    },
    model,
    options,
  )
}

// ---- elliptical slice sampling -------------------------------------------------------------------------------------

/** The Gaussian-prior structure of a program for elliptical slice sampling, read from the analysis. */
function gaussianPriors(a: ProgramAnalysis): { readonly ok: boolean; readonly reason: string } {
  if (a.latent.length === 0) return { ok: false, reason: 'no latent sites' }
  const latent = new Set(a.latent)
  for (const s of a.sites) {
    if (!latent.has(s.name)) continue
    const d = s.name
    const site = a.prototype.sites.get(d)
    if (site?.type !== 'sample') continue
    const family = site.fn.name
    if (family !== 'normal' && family !== 'multivariateNormal')
      return { ok: false, reason: `the prior of ${d} is ${family}, not normal or multivariateNormal` }
    const dependent = s.parents.filter((p) => latent.has(p))
    if (dependent.length > 0)
      return { ok: false, reason: `the prior of ${d} depends on the latent site ${dependent.join(', ')}` }
  }
  return { ok: true, reason: 'every latent site has a Gaussian prior that does not depend on another latent site' }
}

/**
 * Elliptical slice sampling as a method, for a model whose latent sites all have Gaussian priors (`normal` or
 * `multivariateNormal`) that do not depend on another latent site: each transition draws a level under the likelihood and
 * a point on the ellipse through the current value and a prior draw, shrinking the angle until it lands inside. There is no
 * step size and every transition moves; it is the sampler for Gaussian-process and other strongly correlated Gaussian
 * priors. One transition of every chain per step; warm-up is burn-in. The method does not apply to a model with another
 * kind of latent site (`check` says which).
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `elliptical-slice`).
 * @returns The method.
 * @concept elliptical-slice-sampling
 * @index estimator Elliptical slice sampling for Gaussian-prior latents as a steppable method.
 * @example
 * ```ts
 * import { ellipticalSliceMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const s = session(ellipticalSliceMethod(model), { config: { numChains: 1, numWarmup: 10, numDraws: 30 }, steps: 500, mode: 'eager' })
 * s.status // => 'converged'
 * ```
 */
export function ellipticalSliceMethod(model: MethodModel, options: { readonly name?: string } = {}): McmcMethod {
  type T = { readonly maxIter: number }
  return mcmcMethodOf<T>(
    {
      name: 'elliptical-slice',
      description: 'elliptical slice sampling for Gaussian-prior latents, one transition of every chain per step',
      knobs: dict({ maxIter: int(1, 10000, { default: 100, label: 'most angle proposals' }) }),
      check: (a) => {
        if (!isProgramAnalysis(a)) {
          for (const n of a.nodes.filter((x) => x.kind === 'latent')) {
            if (n.family !== 'normal' && n.family !== 'multivariateNormal')
              return no(
                `the prior of ${n.name} is ${n.family ?? 'unknown'}, not normal or multivariateNormal`,
                ['Gaussian priors'],
                'use another kernel',
              )
            if (!n.parents.every((p) => p.startsWith('data:')))
              return no(`the prior of ${n.name} depends on another node`, ['Gaussian priors'], 'use another kernel')
          }
          return yes('every latent node has a Gaussian prior that depends on data only')
        }
        const g = gaussianPriors(a)
        return g.ok ? yes(g.reason) : no(g.reason, ['Gaussian priors'], 'use another kernel')
      },
      plan: (b) => {
        const a = analyse(b.program, undefined)
        const g = gaussianPriors(a)
        if (!g.ok) throw new OmaiError('value', `elliptical-slice: ${g.reason}`)
        const um = b.um
        const dim = um.dim
        const priors = um.layout.map((l) => {
          const site = a.prototype.sites.get(l.name)
          if (site?.type !== 'sample') throw new OmaiError('value', `elliptical-slice: no site ${l.name}`)
          return { name: l.name, fn: site.fn, size: l.size, shape: l.unconstrainedShape }
        })
        // The prior's mean and covariance in layout order; the posterior is prior times likelihood, the likelihood being the
        // log joint without the priors of these sites.
        const means = priors.map((p) => reshape(asTensor(p.fn.mean() as TensorLike), [p.size]))
        const blocks = priors.map((p) =>
          p.fn.name === 'multivariateNormal'
            ? reshape(asTensor((p.fn as unknown as { covariance(): TensorLike }).covariance()), [p.size, p.size])
            : diag(reshape(asTensor(p.fn.variance() as TensorLike), [p.size])),
        )
        const mean = concat(means, 0)
        const cov = blockDiag(...blocks)
        const logPrior = (u: Float64Array): number => {
          let s = 0
          let k = 0
          for (const p of priors) {
            const x = asTensor(u.subarray(k, k + p.size)) as Tensor
            const lp = p.fn.logProb(p.shape.length === 0 ? x : reshape(x, [...p.shape]))
            s += item(sum(asTensor(lp as TensorLike)))
            k += p.size
          }
          return s
        }
        const logLikelihood = (u: Float64Array): number => um.logDensity.logDensity(u) - logPrior(u)
        return fixedPlan<T>(
          b,
          { maxIter: num(b.config, 'maxIter') },
          (t) => ellipticalSlice(logLikelihood, { mean, covariance: cov }, dim, { maxIter: t.maxIter }),
          (t) => ({ ...t }),
        )
      },
    },
    model,
    options,
  )
}

// ---- stochastic gradient samplers ------------------------------------------------------------------------------------

/**
 * The stochastic gradient of a program's log density: the program's subsampled plates draw a fresh subsample at every call
 * (the program applies the subsample to its data, and the plate scales the likelihood by size over subsample size, so the
 * gradient is unbiased); a program with no subsampled plate gives the exact gradient.
 */
function stochasticGradient(b: McmcBuild): {
  readonly gradient: StochasticGradient
  readonly subsampled: readonly string[]
} {
  const a = analyse(b.program, undefined)
  const plates = a.plates.filter((p) => p.subsampleSize < p.size)
  if (plates.length === 0) return { gradient: (_, x) => b.um.logDensity.valueAndGrad(x).grad, subsampled: [] }
  const cell: { subsamples: Readonly<Record<string, Tensor>> } = { subsamples: {} }
  const wrapped = (d: unknown): unknown =>
    substitute(b.program, (msg: Message) => (msg.type === 'plate' ? cell.subsamples[msg.name] : undefined))(d as never)
  const um = potentialEnergy(wrapped as never, undefined)
  return {
    subsampled: plates.map((p) => p.name),
    gradient: (key, x) => {
      cell.subsamples = Object.fromEntries(
        plates.map((p, i) => [p.name, subsample(foldInKey(key, i), p.size, p.subsampleSize) as Tensor]),
      )
      return um.logDensity.valueAndGrad(x).grad
    },
  }
}

const decayKnobs = {
  stepSize: real(1e-6, 10, { log: true, default: 0.01, label: 'step size' }),
  decay: real(0, 1, {
    default: 0,
    label: 'step-size decay',
    description: 'the exponent gamma of eps_t = a (1 + t)^-gamma; 0 keeps the step size constant',
  }),
  temperature: real(0.01, 100, { log: true, default: 1, label: 'temperature' }),
}

const scheduleOf = (c: Config): number | ((t: number) => number) =>
  num(c, 'decay') > 0 ? polynomialStepSize(num(c, 'stepSize'), 1, Math.max(0.5, num(c, 'decay'))) : num(c, 'stepSize')

/**
 * Stochastic-gradient Langevin dynamics as a method: unadjusted Langevin steps with a minibatch gradient, for a program
 * whose plates are subsampled (`plate(name, n, fn, { subsampleSize })`, the program taking the subsample of its data; with no
 * subsampled plate the gradient is exact and the method is the unadjusted Langevin algorithm). The step size is constant or
 * decays as `a (1 + t)^-gamma` (`decay`); warm-up is burn-in. `trackDensity` evaluates the full log density at each draw for the
 * tape and the diagnostics (a full pass over the data: turn it off for large data).
 *
 * @param model - A program closed over its data (a bound spec has no subsampled plates and gets the exact gradient).
 * @param options - The method's name (default `sgld`).
 * @returns The method.
 * @concept stochastic-gradient-mcmc
 * @index estimator Stochastic-gradient Langevin dynamics on a program with subsampled plates, as a steppable method.
 * @example
 * ```ts
 * import { sgldMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, plate, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * import { take, tensor } from 'omai-engine/tensor'
 * const y = tensor([1, 2, 3, 4, 5, 6])
 * const model = () => {
 *   const mu = sample('mu', normal(0, 5))
 *   plate('n', 6, (i) => { observe('y', normal(mu, 1), take(y, i)) }, { subsampleSize: 3 })
 * }
 * const s = session(sgldMethod(model), { config: { numChains: 1, numWarmup: 10, numDraws: 20, stepSize: 0.01 }, steps: 500, mode: 'eager' })
 * s.status // => 'converged'
 * ```
 */
export function sgldMethod(model: MethodModel, options: { readonly name?: string } = {}): McmcMethod {
  type T = { readonly stepSize: number }
  return mcmcMethodOf<T>(
    {
      name: 'sgld',
      minibatch: true,
      description:
        'stochastic-gradient Langevin dynamics with minibatch gradients from subsampled plates, one transition per step',
      knobs: dict({ ...decayKnobs, trackDensity: bool({ default: true, label: 'track the full log density' }) }),
      plan: (b) => {
        const { gradient } = stochasticGradient(b)
        const track = flag(b.config, 'trackDensity')
        return fixedPlan<T>(
          b,
          { stepSize: num(b.config, 'stepSize') },
          () =>
            sgldKernel(gradient, b.um.dim, {
              stepSize: scheduleOf(b.config),
              temperature: num(b.config, 'temperature'),
              ...(track ? { target: b.um.logDensity } : {}),
            }),
          (t) => ({ ...t }),
        )
      },
    },
    model,
    options,
  )
}

/**
 * Stochastic-gradient Hamiltonian Monte Carlo as a method: friction-damped Hamiltonian dynamics driven by a minibatch
 * gradient, a fresh momentum every transition and `numSteps` Euler steps of it (see `sgldMethod` for the program it needs and
 * the knobs it shares). The friction and the estimate of the gradient noise are knobs; warm-up is burn-in.
 *
 * @param model - A program closed over its data.
 * @param options - The method's name (default `sghmc`).
 * @returns The method.
 * @concept stochastic-gradient-mcmc
 * @index estimator Stochastic-gradient Hamiltonian Monte Carlo with friction as a steppable method.
 * @example
 * ```ts
 * import { sghmcMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * const model = () => { const mu = sample('mu', normal(0, 1)); observe('y', normal(mu, 1), 1) }
 * const s = session(sghmcMethod(model), { config: { numChains: 1, numWarmup: 10, numDraws: 20, stepSize: 0.05 }, steps: 500, mode: 'eager' })
 * s.status // => 'converged'
 * ```
 */
export function sghmcMethod(model: MethodModel, options: { readonly name?: string } = {}): McmcMethod {
  type T = { readonly stepSize: number }
  return mcmcMethodOf<T>(
    {
      name: 'sghmc',
      minibatch: true,
      description:
        'stochastic-gradient HMC with friction and minibatch gradients from subsampled plates, one transition per step',
      knobs: dict({
        ...decayKnobs,
        numSteps: int(1, 1000, { default: 10, label: 'steps per transition' }),
        friction: real(1e-4, 10, { log: true, default: 0.05, label: 'friction' }),
        noiseEstimate: real(0, 100, { default: 0, label: 'gradient noise estimate' }),
        trackDensity: bool({ default: true, label: 'track the full log density' }),
      }),
      plan: (b) => {
        const { gradient } = stochasticGradient(b)
        const track = flag(b.config, 'trackDensity')
        return fixedPlan<T>(
          b,
          { stepSize: num(b.config, 'stepSize') },
          () =>
            sghmcKernel(gradient, b.um.dim, {
              stepSize: scheduleOf(b.config),
              numSteps: num(b.config, 'numSteps'),
              friction: num(b.config, 'friction'),
              noiseEstimate: num(b.config, 'noiseEstimate'),
              temperature: num(b.config, 'temperature'),
              ...(track ? { target: b.um.logDensity } : {}),
            }),
          (t) => ({ ...t }),
        )
      },
    },
    model,
    options,
  )
}

// ---- microcanonical Langevin ------------------------------------------------------------------------------------------

/**
 * Microcanonical Langevin Monte Carlo as a method: the unadjusted isokinetic sampler, tuned in warm-up by the library's
 * three-phase procedure (the step size from the energy error towards `desiredEnergyVar`, the momentum decoherence length
 * and a diagonal preconditioner from the chain's own variance, and L from the autocorrelation). There is no accept step, so the
 * draws carry a small bias that shrinks with the step size, and the tape's `mcmc/accept` is 1 except for a transition that
 * failed. The model needs at least two unconstrained coordinates; warm-up needs at least 10 iterations (fewer is burn-in at the
 * default parameters).
 *
 * @param model - A bound spec, or a program closed over its data.
 * @param options - The method's name (default `mclmc`).
 * @returns The method.
 * @concept microcanonical-langevin-monte-carlo
 * @index estimator Microcanonical Langevin Monte Carlo with three-phase tuning as a steppable method.
 * @example
 * ```ts
 * import { mclmcMethod } from 'omai-engine/probprog/inference'
 * import { normal } from 'omai-engine/dists'
 * import { observe, sample } from 'omai-engine/probprog'
 * import { session } from 'omai-engine/trace'
 * import { add } from 'omai-engine/tensor'
 * const model = () => { const a = sample('a', normal(0, 1)); const b = sample('b', normal(0, 1)); observe('y', normal(add(a, b), 1), 1) }
 * const s = session(mclmcMethod(model), { config: { numChains: 1, numWarmup: 60, numDraws: 30 }, steps: 500, mode: 'eager' })
 * s.status // => 'converged'
 * ```
 */
export function mclmcMethod(model: MethodModel, options: { readonly name?: string } = {}): McmcMethod {
  return mcmcMethodOf<MclmcParameters>(
    {
      name: 'mclmc',
      description:
        'microcanonical Langevin Monte Carlo with three-phase tuning, one transition of every chain per step',
      knobs: dict({
        desiredEnergyVar: real(1e-5, 1, { log: true, default: 5e-4, label: 'desired energy variance' }),
        diagonalPreconditioning: bool({ default: true, label: 'diagonal preconditioner' }),
        lFactor: real(0.05, 2, { default: 0.4, label: 'L factor' }),
        integrator: choice(['velocityVerlet', 'mcLachlan'], { default: 'mcLachlan', label: 'integrator' }),
      }),
      plan: (b) => {
        const dim = b.um.dim
        if (dim < 2) throw new OmaiError('value', 'mclmc: the model needs at least two unconstrained coordinates')
        const target = b.um.logDensity
        const numWarmup = num(b.config, 'numWarmup')
        const integrator = pick(b.config, 'integrator') === 'velocityVerlet' ? 'velocityVerlet' : 'mcLachlan'
        const kernel = (p: MclmcParameters): MCMCKernel => mclmcKernel(target, { ...p, integrator })
        const algorithmOptions = {
          numWarmup,
          desiredEnergyVar: num(b.config, 'desiredEnergyVar'),
          diagonalPreconditioning: flag(b.config, 'diagonalPreconditioning'),
          lFactor: num(b.config, 'lFactor'),
          integrator,
        } as const
        const shared = mclmcAdaptationAlgorithm(target, Float64Array.from(toFlat(zeros([dim]))), algorithmOptions)
        const warm: Warmer<MclmcParameters> | undefined =
          numWarmup >= 10
            ? {
                init: (start, key) => mclmcAdaptationAlgorithm(target, start, algorithmOptions).init(key),
                step: (w, ctx) => shared.step(w as MclmcWarmupState, ctx),
                chain: (w) => (w as MclmcWarmupState).chain,
                info: (w) => ({ acceptanceRate: 1, accepted: true, stepSize: (w as MclmcWarmupState).stepSize }),
                result: (w) => mclmcAdaptationResult(w as MclmcWarmupState),
                toJson: (w) => shared.serialise?.(w as MclmcWarmupState) ?? null,
                fromJson: (j) => shared.deserialise?.(j),
              }
            : undefined
        const tuned0: MclmcParameters = {
          stepSize: Math.sqrt(dim) * 0.25,
          momentumDecoherenceLength: Math.sqrt(dim),
          inverseMassMatrix: Float64Array.from(toFlat(ones([dim]))),
        }
        return {
          ...fixedPlan<MclmcParameters>(b, tuned0, kernel, (p) => ({
            stepSize: p.stepSize,
            momentumDecoherenceLength: p.momentumDecoherenceLength,
            inverseMassMatrix: Array.from(p.inverseMassMatrix),
          })),
          ...(warm === undefined ? {} : { warm }),
        } satisfies McmcPlan<MclmcParameters>
      },
    },
    model,
    options,
  )
}
