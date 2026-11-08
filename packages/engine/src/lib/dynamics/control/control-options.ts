/**
 * The slots of the control loops as option tables: the anti-windup of a PID controller (conditional integration,
 * back-calculation, none) with its gain, the quadratic-program method of a linear MPC (active set, interior point), and
 * the setpoint and the disturbance as functions of time (a step, a square wave, a ramp, a sine, a pulse), each with the
 * knobs of its own. Each table is one `union` space for controls, grids and tuners.
 */

import {
  type DictSpace,
  dict,
  fillKnobs,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../../space/index.ts'
import type { PidOptions } from './pid.ts'
import type { LinearMpcOptions } from './predictive.ts'

/** What an anti-windup does, for validity rules and selectors. */
export interface AntiWindupProperties {
  /** Whether it needs a gain. */
  readonly hasGain: boolean
  /** Whether the integral can still grow while the output is saturated. */
  readonly integratesWhileSaturated: boolean
}

function windup(
  description: string,
  properties: AntiWindupProperties,
  space: DictSpace,
  make: (knobs: Readonly<Record<string, SpaceValue>>) => Pick<PidOptions, 'antiWindup' | 'kb'>,
): Option<Pick<PidOptions, 'antiWindup' | 'kb'>, AntiWindupProperties> {
  return {
    description,
    properties,
    space,
    make: (config) => make(fillKnobs(space, config)),
  }
}

/**
 * The anti-windup schemes as one slot: `clamping` (stop integrating while saturated), `back-calculation` (feed the
 * saturation back into the integral at a gain) and `none`. A value spreads into the options of `pid`.
 *
 * @concept pid-control
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * control.antiWindups.make({ kind: 'back-calculation', kb: 2 }) // => { antiWindup: 'back-calculation', kb: 2 }
 * ```
 */
export const antiWindups: OptionTable<Pick<PidOptions, 'antiWindup' | 'kb'>, AntiWindupProperties> = optionTable(
  {
    clamping: windup(
      'stop integrating while the output is saturated and the error pushes further',
      { hasGain: false, integratesWhileSaturated: false },
      dict({}),
      () => ({ antiWindup: 'clamping' }),
    ),
    'back-calculation': windup(
      'feed the difference between the saturated and the raw output back into the integral',
      { hasGain: true, integratesWhileSaturated: true },
      dict({ kb: real(0.01, 100, { log: true, default: 1, label: 'back-calculation gain' }) }),
      (k) => ({ antiWindup: 'back-calculation', kb: Number(k['kb']) }),
    ),
    none: windup('let the integral wind up', { hasGain: false, integratesWhileSaturated: true }, dict({}), () => ({
      antiWindup: 'none',
    })),
  },
  {
    default: 'clamping',
    label: 'anti-windup',
    description: 'What stops the integral growing while the actuator saturates.',
  },
)

/** What a QP method does, for validity rules and selectors. */
export interface QpMethodProperties {
  /** Whether it follows the boundary of the feasible set (active set) rather than its interior. */
  readonly activeSet: boolean
}

/**
 * The quadratic-program methods of a linear MPC as one slot: `active-set` and `interior-point`. The value is the
 * `method` option of `linearMpc`.
 *
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * control.qpMethods.make({ kind: 'interior-point' }) // => 'interior-point'
 * ```
 */
export const qpMethods: OptionTable<NonNullable<LinearMpcOptions['method']>, QpMethodProperties> = optionTable(
  {
    'active-set': {
      description: 'walk the faces of the feasible set, solving equality problems',
      properties: { activeSet: true },
      space: dict({}),
      make: () => 'active-set',
    },
    'interior-point': {
      description: 'follow a central path through the interior with a barrier',
      properties: { activeSet: false },
      space: dict({}),
      make: () => 'interior-point',
    },
  },
  { default: 'active-set', label: 'QP method', description: 'How each planning step solves its quadratic program.' },
)

/** A signal of time. */
export type Signal = (t: number) => number

/** What a signal is, for validity rules and selectors. */
export interface SignalProperties {
  /** Whether it jumps. */
  readonly discontinuous: boolean
  /** Whether it repeats. */
  readonly periodic: boolean
}

function signal(
  description: string,
  properties: SignalProperties,
  space: DictSpace,
  build: (k: (name: string) => number) => Signal,
): Option<Signal, SignalProperties> {
  return {
    description,
    properties,
    space,
    make: (config) => {
      const full = fillKnobs(space, config)
      return build((name) => Number(full[name]))
    },
  }
}

const time = (label: string, def: number) => real(0, 10_000, { default: def, label })
const level = (label: string, def: number) => real(-1000, 1000, { default: def, label })

/**
 * Setpoints as one slot: `step` (a level from a time on), `square` (alternating between two levels), `ramp` (a constant
 * rate from a time on) and `sine`. The value is a function of time for the loops of `pid`.
 *
 * @concept pid-control
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * control.setpoints.make({ kind: 'square', low: 0, high: 2, period: 10 })(7) // => 0
 * ```
 */
export const setpoints: OptionTable<Signal, SignalProperties> = optionTable(
  {
    step: signal(
      'a level from a given time on, zero before',
      { discontinuous: true, periodic: false },
      dict({ level: level('level', 1), at: time('time of the step', 0) }),
      (k) => (t) => (t >= k('at') ? k('level') : 0),
    ),
    square: signal(
      'alternating between a high and a low level, high first',
      { discontinuous: true, periodic: true },
      dict({
        low: level('low', 0),
        high: level('high', 1),
        period: real(0.01, 10_000, { log: true, default: 20, label: 'period' }),
      }),
      (k) => (t) => (t % k('period') < k('period') / 2 ? k('high') : k('low')),
    ),
    ramp: signal(
      'a constant rate of change from a given time on',
      { discontinuous: false, periodic: false },
      dict({ rate: level('rate', 0.1), at: time('start time', 0) }),
      (k) => (t) => (t >= k('at') ? k('rate') * (t - k('at')) : 0),
    ),
    sine: signal(
      'a sine around a mean',
      { discontinuous: false, periodic: true },
      dict({
        mean: level('mean', 0),
        amplitude: level('amplitude', 1),
        period: real(0.01, 10_000, { log: true, default: 20, label: 'period' }),
      }),
      (k) => (t) => k('mean') + k('amplitude') * Math.sin((2 * Math.PI * t) / k('period')),
    ),
  },
  { default: 'step', label: 'setpoint', description: 'The value the loop is asked to hold, as a function of time.' },
)

/**
 * Output disturbances as one slot: `none`, `step` (a constant offset from a time on), `pulse` (an offset for a while)
 * and `sine`. The value is a function of time for the loops of `pid`.
 *
 * @concept pid-control
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * control.disturbances.make({ kind: 'pulse', size: 0.5, at: 2, width: 1 })(2.5) // => 0.5
 * ```
 */
export const disturbances: OptionTable<Signal, SignalProperties> = optionTable(
  {
    none: signal('no disturbance', { discontinuous: false, periodic: false }, dict({}), () => () => 0),
    step: signal(
      'a constant offset from a given time on',
      { discontinuous: true, periodic: false },
      dict({ size: level('size', 0.2), at: time('time of the step', 10) }),
      (k) => (t) => (t >= k('at') ? k('size') : 0),
    ),
    pulse: signal(
      'an offset for a while',
      { discontinuous: true, periodic: false },
      dict({
        size: level('size', 0.5),
        at: time('start', 10),
        width: real(0.01, 10_000, { log: true, default: 2, label: 'width' }),
      }),
      (k) => (t) => (t >= k('at') && t < k('at') + k('width') ? k('size') : 0),
    ),
    sine: signal(
      'a sinusoidal offset',
      { discontinuous: false, periodic: true },
      dict({
        amplitude: level('amplitude', 0.1),
        period: real(0.01, 10_000, { log: true, default: 5, label: 'period' }),
      }),
      (k) => (t) => k('amplitude') * Math.sin((2 * Math.PI * t) / k('period')),
    ),
  },
  { default: 'none', label: 'disturbance', description: 'What pushes on the plant output besides the controller.' },
)
