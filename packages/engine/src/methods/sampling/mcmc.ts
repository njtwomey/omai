/**
 * Markov chain Monte Carlo on a two-dimensional target, as a method of nothing but its configuration: a target
 * (Haario's banana, Neal's funnel, a correlated Gaussian, a two-component mixture), a kernel (random-walk Metropolis or
 * Hamiltonian Monte Carlo) and its step size, run by the library's sampler algorithm. The chain's positions and the
 * transition statistics are on the tape, with a running acceptance rate, and the target's density over the plane is a
 * probe for the chain to be drawn over.
 */

import { type DecisionSurface, decisionSurface } from '../../lib/exposition/index.ts'
import { type Key, normal } from '../../lib/random/index.ts'
import {
  banana,
  hmcKernel,
  type LogDensity,
  type MCMCKernel,
  type MCMCState,
  nealsFunnel,
  randomWalkMetropolis,
  type SamplerStep,
  samplerAlgorithm,
  target,
} from '../../lib/sampling/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { add, exp, log, mul, split, square, sub, sum, type Value } from '../../lib/tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../lib/trace/index.ts'

/** The targets on offer. */
export type McmcTarget = 'banana' | 'funnel' | 'gaussian' | 'mixture'
/** The kernels on offer. */
export type McmcKernelName = 'random-walk' | 'hmc'

/** The paths of {@link mcmc}. */
export type McmcPath =
  | 'chain/position'
  | 'chain/logDensity'
  | 'chain/acceptanceRate'
  | 'chain/acceptance'
  | 'target/density'

/** The chain's state: the sampler's step plus the running count of transitions and acceptances. */
export interface McmcState extends SamplerStep<MCMCState> {
  readonly transitions: number
  readonly accepted: number
}

interface TargetSpec {
  readonly density: LogDensity
  readonly start: readonly number[]
  /** The plane drawn: `[[x low, x high], [y low, y high]]`. */
  readonly window: readonly [readonly [number, number], readonly [number, number]]
}

const coordinates = (x: Value): [Value, Value] => split(x, 2).map((v) => sum(v)) as [Value, Value]

function targetOf(name: McmcTarget): TargetSpec {
  switch (name) {
    case 'banana':
      return {
        density: banana({ curvature: 0.1, scale: 3 }),
        start: [0, 0],
        window: [
          [-10, 10],
          [-4, 10],
        ],
      }
    case 'funnel':
      return {
        density: nealsFunnel(2, { scale: 2 }),
        start: [0, 0],
        window: [
          [-8, 8],
          [-15, 15],
        ],
      }
    case 'gaussian': {
      // Correlation 0.8: log p = -(x1^2 + x2^2 - 2 rho x1 x2) / (2 (1 - rho^2)).
      const rho = 0.8
      const fn = (x: Value): Value => {
        const [x1, x2] = coordinates(x)
        const quad = sub(add(square(x1), square(x2)), mul(2 * rho, mul(x1, x2)))
        return mul(quad, -0.5 / (1 - rho * rho))
      }
      return {
        density: target(fn, 2),
        start: [2, -2],
        window: [
          [-4, 4],
          [-4, 4],
        ],
      }
    }
    case 'mixture': {
      // Equal Gaussians of unit variance at (-2.5, 0) and (2.5, 0).
      const fn = (x: Value): Value => {
        const [x1, x2] = coordinates(x)
        const a = mul(add(square(add(x1, 2.5)), square(x2)), -0.5)
        const b = mul(add(square(sub(x1, 2.5)), square(x2)), -0.5)
        return log(add(exp(a), exp(b)))
      }
      return {
        density: target(fn, 2),
        start: [-2.5, 0],
        window: [
          [-6, 6],
          [-4, 4],
        ],
      }
    }
  }
}

function kernelOf(density: LogDensity, c: Config): MCMCKernel<MCMCState> {
  const stepSize = c['stepSize'] as number
  if ((c['kernel'] as McmcKernelName) === 'hmc')
    return hmcKernel(density, { stepSize, numIntegrationSteps: c['leapfrogSteps'] as number })
  return randomWalkMetropolis(density, { scale: stepSize })
}

// The target's density over its window, normalised to a maximum of one; computed once per target.
const SURFACES = new Map<McmcTarget, DecisionSurface>()
function surfaceOf(name: McmcTarget): DecisionSurface {
  const cached = SURFACES.get(name)
  if (cached !== undefined) return cached
  const spec = targetOf(name)
  const res = 70
  const axis = ([lo, hi]: readonly [number, number]): number[] =>
    Array.from({ length: res }, (_, i) => lo + ((hi - lo) * i) / (res - 1))
  const xs = axis(spec.window[0])
  const ys = axis(spec.window[1])
  const logs = ys.map((y) => xs.map((x) => spec.density.logDensity([x, y])))
  const top = Math.max(...logs.flat().filter((v) => Number.isFinite(v)))
  const values = logs.map((row) => row.map((v) => (Number.isFinite(v) ? Math.exp(v - top) : 0)))
  const [f0, f1] = spec.density.names
  const surface = decisionSurface({
    features: [f0 ?? 'x[0]', f1 ?? 'x[1]'],
    x: xs,
    y: ys,
    values,
    field: { name: 'density', smooth: true },
  })
  SURFACES.set(name, surface)
  return surface
}

/**
 * A Markov chain on a two-dimensional target, one transition per step.
 *
 * Configuration: `target` (banana, funnel, gaussian, mixture), `kernel` (random-walk, hmc), `stepSize` (the proposal
 * scale, or the leapfrog step) and `leapfrogSteps` (HMC only). Paths: `chain/position` (full), `chain/logDensity`,
 * `chain/acceptanceRate` (this transition's probability), `chain/acceptance` (the running rate), and the probe
 * `target/density` (the target over its window as a `decisionSurface`).
 */
export const mcmc: Method<McmcState, Config, McmcPath> = defineMethod<McmcState, Config, McmcPath>({
  name: 'sampling/mcmc',
  config: dict({
    target: choice(['banana', 'funnel', 'gaussian', 'mixture'] as const, { default: 'banana' }),
    kernel: choice(['random-walk', 'hmc'] as const, { default: 'random-walk' }),
    stepSize: real(0.01, 5, { default: 1, log: true, label: 'step size' }),
    leapfrogSteps: int(1, 50, { default: 10, label: 'leapfrog steps (HMC)' }),
  }),
  make: (c, key: Key): Algorithm<McmcState, McmcPath> => {
    const spec = targetOf(c['target'] as McmcTarget)
    const kernel = kernelOf(spec.density, c)
    const jitter = normal(key, [2], { scale: 0.5 }).data
    const start = spec.start.map((v, i) => v + (jitter[i] ?? 0))
    const alg = samplerAlgorithm<MCMCState>(kernel, start)
    return {
      ...alg,
      init: (k) => ({ ...alg.init(k), transitions: 0, accepted: 0 }),
      step: (s, ctx) => {
        const next = alg.step(s, ctx)
        const transitions = s.transitions + 1
        const accepted = s.accepted + (next.info?.accepted === true ? 1 : 0)
        ctx.log('chain/acceptance', accepted / transitions)
        return { ...next, transitions, accepted }
      },
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as McmcState,
    }
  },
  contract: (c): Contract<McmcState, McmcPath> => ({
    'chain/position': tensor({ from: 'position', role: 'state', shape: [2], description: 'the current position' }),
    'chain/logDensity': scalar({ from: 'logDensity', description: 'the log density at the current position' }),
    'chain/acceptanceRate': scalar({
      from: 'acceptanceRate',
      description: 'the Metropolis acceptance probability of this transition',
      direction: 'higher',
    }),
    'chain/acceptance': scalar({ description: 'the share of transitions accepted so far', direction: 'higher' }),
    'target/density': derived<McmcState>({
      description: 'the target density over the plane (its maximum scaled to one)',
      cost: 'dear',
      compute: () => surfaceOf(c['target'] as McmcTarget),
    }),
  }),
  defaults: {
    'chain/position': Log({ detail: 'full' }),
    'chain/logDensity': Log(),
    'chain/acceptanceRate': Log(),
    'chain/acceptance': Log(),
  },
})
