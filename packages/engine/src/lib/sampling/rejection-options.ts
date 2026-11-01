/**
 * The exact sampler as one slot: an option table over rejection sampling under an envelope, adaptive rejection sampling
 * of a log-concave density, and the ratio of uniforms, each with the knobs of its own (the log of the envelope constant,
 * the hull size and starting spread, the search range for the rectangle) and what it needs of the problem (a proposal, a
 * gradient, a plain density). `rejectionSampler` builds the steppable algorithm a configuration names.
 */

import { MdmlError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import {
  type DictSpace,
  defaults,
  dict,
  int,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../space/index.ts'
import type { Algorithm } from '../trace/index.ts'
import {
  adaptiveRejectionAlgorithm,
  type RejectionBaseState,
  type RejectionPath,
  ratioOfUniformsAlgorithm,
  rejectionAlgorithm,
} from './rejection.ts'
import type { LogDensity, Proposal } from './target.ts'

/** What an exact sampler is given: the key and number of draws, and the target in the form the method needs. */
export interface RejectionProblem {
  readonly key: Key
  readonly n: number
  /** The target as a log density (rejection sampling; adaptive rejection also needs its gradient). */
  readonly target?: LogDensity
  /** The proposal (rejection sampling). */
  readonly proposal?: Proposal
  /** The unnormalised density itself (ratio of uniforms). */
  readonly density?: (x: number) => number
  /** The support [low, high] (adaptive rejection; default the real line). */
  readonly domain?: readonly [number, number]
}

/** An exact sampler built from a configuration: the algorithm for a problem. */
export type RejectionSampler = (problem: RejectionProblem) => Algorithm<RejectionBaseState, RejectionPath>

/** What an exact sampler needs and does, for validity rules and selectors. */
export interface RejectionSamplerProperties {
  /** Whether the envelope is fixed by the caller (and may be violated) rather than found by the method. */
  readonly fixedEnvelope: boolean
  /** Whether it needs the target to be log-concave. */
  readonly logConcave: boolean
  /** Whether it handles more than one dimension. */
  readonly multivariate: boolean
}

type Knobs = (name: string) => number

function sampler(
  description: string,
  properties: RejectionSamplerProperties,
  space: DictSpace,
  build: (k: Knobs) => RejectionSampler,
): Option<RejectionSampler, RejectionSamplerProperties> {
  return {
    description,
    properties,
    space,
    make: (config) => {
      const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }
      return build((name) => full[name] as number)
    },
  }
}

const needs = (what: string, has: boolean, name: string): void => {
  if (!has) throw new MdmlError('value', `${name}: the problem needs ${what}`)
}

const budget = int(1, 100_000_000, { log: true, default: 100_000, label: 'most proposals' })

/**
 * The exact samplers as one slot: `rejection` (a proposal q and a constant M with pi <= M q), `adaptive` (a log-concave
 * density in one dimension, whose envelope tightens as it goes) and `ratio-of-uniforms` (a plain density in one
 * dimension, sampled as a ratio of two uniforms). All three give independent exact draws; they differ in what they ask
 * of the target and how often they reject.
 *
 * @concept rejection-sampling
 * @example
 * ```ts
 * import { rejectionSamplers } from 'mdml-engine/sampling'
 * rejectionSamplers.names // => ['rejection', 'adaptive', 'ratio-of-uniforms']
 * ```
 */
export const rejectionSamplers: OptionTable<RejectionSampler, RejectionSamplerProperties> = optionTable(
  {
    rejection: sampler(
      'draw from a proposal q and keep the point with probability pi / (M q): exact when pi <= M q everywhere',
      { fixedEnvelope: true, logConcave: false, multivariate: true },
      dict({
        logEnvelope: real(-50, 50, { default: 2, label: 'log M, the envelope constant' }),
        maxProposals: budget,
      }),
      (k) => (p) => {
        needs('a target', p.target !== undefined, 'rejection')
        needs('a proposal', p.proposal !== undefined, 'rejection')
        return rejectionAlgorithm(p.target as LogDensity, p.proposal as Proposal, {
          key: p.key,
          n: p.n,
          logEnvelope: k('logEnvelope'),
          maxProposals: k('maxProposals'),
        })
      },
    ),
    adaptive: sampler(
      'tangents of log pi form an envelope that tightens at every evaluation: for log-concave densities in one dimension',
      { fixedEnvelope: false, logConcave: true, multivariate: false },
      dict({
        spread: real(0.01, 100, { log: true, default: 1, label: 'distance of the two starting points from 0' }),
        maxPoints: int(2, 500, { log: true, default: 50, label: 'most abscissae in the hull' }),
        maxProposals: budget,
      }),
      (k) => (p) => {
        needs('a target with a gradient', p.target?.hasGradient === true, 'adaptive')
        return adaptiveRejectionAlgorithm(p.target as LogDensity, {
          key: p.key,
          n: p.n,
          initialPoints: [-k('spread'), k('spread')],
          maxPoints: k('maxPoints'),
          maxProposals: k('maxProposals'),
          ...(p.domain === undefined ? {} : { domain: p.domain }),
        })
      },
    ),
    'ratio-of-uniforms': sampler(
      'x = v / u for (u, v) uniform on the region under sqrt f, found by rejection from its bounding rectangle',
      { fixedEnvelope: false, logConcave: false, multivariate: false },
      dict({
        range: real(1, 10_000, {
          log: true,
          default: 30,
          label: 'half-width of the interval searched for the rectangle',
        }),
        maxProposals: budget,
      }),
      (k) => (p) => {
        needs('a density', p.density !== undefined, 'ratio-of-uniforms')
        return ratioOfUniformsAlgorithm(p.density as (x: number) => number, {
          key: p.key,
          n: p.n,
          range: [-k('range'), k('range')],
          maxProposals: k('maxProposals'),
        })
      },
    ),
  },
  {
    default: 'rejection',
    label: 'exact sampler',
    description: 'How independent exact draws are made from an unnormalised density.',
  },
)

/**
 * The steppable algorithm a configuration of the `rejectionSamplers` slot builds for a problem.
 *
 * @param config - The option and its knobs, such as `{ kind: 'adaptive', maxPoints: 20 }`.
 * @param problem - The key, the number of draws, and the target in the form the method needs.
 * @returns The algorithm; one step is one proposal.
 * @throws MdmlError with code `value` when the problem lacks what the method needs.
 * @concept rejection-sampling
 * @example
 * ```ts
 * import { numericTarget, rejectionSampler } from 'mdml-engine/sampling'
 * import { key } from 'mdml-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * rejectionSampler({ kind: 'adaptive' }, { key: key(0), n: 5, target: t }).name // => 'adaptive-rejection'
 * ```
 */
export function rejectionSampler(
  config: SpaceValue,
  problem: RejectionProblem,
): Algorithm<RejectionBaseState, RejectionPath> {
  return rejectionSamplers.make(config)(problem)
}
