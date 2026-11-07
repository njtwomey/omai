/**
 * The point-process families as one slot: an option table over the homogeneous Poisson process, a Poisson process with a
 * sinusoidal intensity (by thinning), renewal processes with gamma or Weibull gaps and the self-exciting Hawkes process,
 * each with the knobs of its own. `pointProcess` builds the steppable simulator a configuration names, for a horizon.
 */

import { gamma, weibull } from '../dists/index.ts'
import {
  type DictSpace,
  dict,
  type Knobs,
  knobOption,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../space/index.ts'
import type { Algorithm } from '../trace/index.ts'
import {
  type EventState,
  hawkesProcessAlgorithm,
  type PointProcessPath,
  poissonProcessAlgorithm,
  renewalProcessAlgorithm,
  thinningAlgorithm,
} from './point-processes.ts'

/** A simulator built from a configuration: the algorithm for a horizon. */
export type PointProcessSimulator = (horizon: number) => Algorithm<EventState, PointProcessPath>

/** What a point process is like, for validity rules and selectors. */
export interface PointProcessProperties {
  /** Whether the gaps between events are independent of the past (Poisson and renewal processes). */
  readonly memoryless: boolean
  /** Whether events make further events likelier. */
  readonly selfExciting: boolean
  /** Whether it is simulated by thinning (some candidates are rejected). */
  readonly thinned: boolean
}

function process(
  description: string,
  properties: PointProcessProperties,
  space: DictSpace,
  build: (k: Knobs) => PointProcessSimulator,
): Option<PointProcessSimulator, PointProcessProperties> {
  return knobOption(description, properties, space, build)
}

const rate = (label: string, def: number): ReturnType<typeof real> =>
  real(0.001, 1000, { log: true, default: def, label })

/**
 * The point-process families as one slot: `poisson` (constant intensity), `sinusoidal` (a Poisson process whose
 * intensity oscillates, simulated by thinning), `renewal-gamma` and `renewal-weibull` (independent gaps; the shape
 * makes events regular above one and bursty below), `hawkes` (each event raises the intensity, which then decays).
 *
 * @concept poisson-process
 * @example
 * ```ts
 * import { pointProcesses } from 'mdml-engine/stochastic'
 * pointProcesses.names // => ['poisson', 'sinusoidal', 'renewal-gamma', 'renewal-weibull', 'hawkes']
 * ```
 */
export const pointProcesses: OptionTable<PointProcessSimulator, PointProcessProperties> = optionTable(
  {
    poisson: process(
      'events at a constant rate: exponential gaps',
      { memoryless: true, selfExciting: false, thinned: false },
      dict({ rate: rate('events per unit time', 2) }),
      (k) => (horizon) => poissonProcessAlgorithm(k('rate'), horizon),
    ),
    sinusoidal: process(
      'a Poisson process whose intensity is base + amplitude sin(2 pi t / period), by thinning',
      { memoryless: false, selfExciting: false, thinned: true },
      dict({
        base: rate('mean intensity', 2),
        amplitude: real(0, 1000, { default: 1.5, label: 'amplitude (at most the mean)' }),
        period: real(0.01, 1000, { log: true, default: 5, label: 'period' }),
      }),
      (k) => (horizon) => {
        const [base, amplitude, period] = [k('base'), Math.min(k('amplitude'), k('base')), k('period')]
        return thinningAlgorithm(
          (t) => base + amplitude * Math.sin((2 * Math.PI * t) / period),
          horizon,
          base + amplitude,
        )
      },
    ),
    'renewal-gamma': process(
      'independent gamma gaps: regular for shape above 1, bursty below, Poisson at 1',
      { memoryless: true, selfExciting: false, thinned: false },
      dict({ shape: real(0.1, 100, { log: true, default: 3, label: 'shape' }), rate: rate('rate of the gaps', 3) }),
      (k) => (horizon) => renewalProcessAlgorithm(gamma(k('shape'), k('rate')), horizon),
    ),
    'renewal-weibull': process(
      'independent Weibull gaps: regular for shape above 1, bursty below',
      { memoryless: true, selfExciting: false, thinned: false },
      dict({
        shape: real(0.1, 100, { log: true, default: 2, label: 'shape' }),
        scale: rate('scale (typical gap)', 0.5),
      }),
      (k) => (horizon) => renewalProcessAlgorithm(weibull(k('shape'), k('scale')), horizon),
    ),
    hawkes: process(
      'self-exciting: each event adds alpha to the intensity, which decays at rate beta back to mu',
      { memoryless: false, selfExciting: true, thinned: true },
      dict({
        mu: rate('baseline intensity', 1),
        alpha: real(0, 100, { default: 0.5, label: 'jump of the intensity at an event' }),
        beta: rate('decay rate', 1),
      }),
      (k) => (horizon) => hawkesProcessAlgorithm({ mu: k('mu'), alpha: k('alpha'), beta: k('beta') }, horizon),
    ),
  },
  {
    default: 'poisson',
    label: 'point process',
    description: 'The family of random event times: how the gap to the next event depends on the past.',
  },
)

/**
 * The steppable simulator a configuration of the `pointProcesses` slot builds for a horizon.
 *
 * @param config - The option and its knobs, such as `{ kind: 'hawkes', alpha: 0.8 }`.
 * @param horizon - T.
 * @returns The algorithm; one step is one candidate event, and the seed is the run's.
 * @throws MdmlError with code `value` for a horizon that is not positive.
 * @concept poisson-process
 * @example
 * ```ts
 * import { pointProcess } from 'mdml-engine/stochastic'
 * pointProcess({ kind: 'hawkes' }, 10).name // => 'hawkes-ogata'
 * ```
 */
export function pointProcess(config: SpaceValue, horizon: number): Algorithm<EventState, PointProcessPath> {
  return pointProcesses.make(config)(horizon)
}
