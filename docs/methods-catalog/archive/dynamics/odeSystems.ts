/**
 * Named ODE systems (predator and prey, Lorenz, Van der Pol, a damped pendulum, an SIR epidemic) integrated by the
 * library's adaptive Runge-Kutta steppers, one accepted step per step of the run: a method of nothing but its
 * configuration. Each step puts the time, the state, the step taken and the field evaluations spent on the tape.
 */

import type { Json } from '../../lib/base/index.ts'
import { MdmlError } from '../../lib/base/index.ts'
import { type NamedSystemName, namedSystems } from '../../lib/dynamics/index.ts'
import { stepperFor } from '../../lib/dynamics/ode/index.ts'
import { choice, dict, real } from '../../lib/space/index.ts'
import { fromStorage, toFlat } from '../../lib/tensor/index.ts'
import {
  type Config,
  defineMethod,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../lib/trace/index.ts'

/**
 * The state of {@link odeSystems}: plain numbers, so that any step can be checkpointed and seeked. The stepper is
 * rebuilt from them at each step, with `h` as its first step, so the sequence of steps is the continuous solve's.
 */
export interface OdeSystemsState {
  /** The time reached. */
  readonly t: number
  /** The state at `t`. */
  readonly y: number[]
  /** The size of the next step to try. */
  readonly h: number
  /** Field evaluations spent so far. */
  readonly evaluations: number
  /** Whether the stepper gave up (a step that would not converge). */
  readonly failed: boolean
}

const SYSTEMS = Object.keys(namedSystems) as NamedSystemName[]

/**
 * The paths of {@link odeSystems}: the time reached, the state, the step taken and the evaluations of the field so far.
 *
 * @example
 * ```ts
 * import type { OdeSystemsPath } from 'mdml-engine/methods/dynamics'
 * const path: OdeSystemsPath = 'y'
 * path // => 'y'
 * ```
 */
export type OdeSystemsPath = 't' | 'y' | 'h' | 'evaluations'

/**
 * Integrate a named system with an adaptive Runge-Kutta method (`RK23`, `RK45` or `DOP853`) at a relative tolerance
 * (the absolute one follows it at a thousandth). A step of the run is one accepted step of the method; the run ends
 * when the system's time span does. Each step starts by evaluating the field where the last one ended, which an
 * uninterrupted solve would reuse, so `evaluations` counts one more per step than the library's own `nfev`.
 *
 * @example
 * ```ts
 * import { odeSystems } from 'mdml-engine/methods/dynamics'
 * import { session } from 'mdml-engine/trace'
 * const s = session(odeSystems, { mode: 'eager', steps: 10, config: { system: 'lorenz' } })
 * s.state.y.length // => 3
 * ```
 */
export const odeSystems: Method<OdeSystemsState, Config, OdeSystemsPath> = defineMethod<
  OdeSystemsState,
  Config,
  OdeSystemsPath
>({
  name: 'dynamics/ode-systems',
  config: dict({
    system: choice(SYSTEMS, { default: 'lotka-volterra', label: 'system' }),
    method: choice(['RK45', 'RK23', 'DOP853'] as const, { default: 'RK45', label: 'method' }),
    rtol: real(1e-10, 1e-2, { log: true, default: 1e-5, label: 'relative tolerance' }),
  }),
  make: (c) => {
    const name = c['system'] as NamedSystemName
    const system = namedSystems[name]
    if (system === undefined) throw new MdmlError('value', `odeSystems: unknown system ${String(name)}`)
    const method = c['method'] as 'RK45' | 'RK23' | 'DOP853'
    const rtol = c['rtol'] as number
    const atol = Math.max(1e-14, rtol * 1e-3)
    const [t0, tEnd] = system.span
    const vector = (y: readonly number[]) => fromStorage(Float64Array.from(y), [y.length], 'float64')
    return {
      name: 'ode-systems',
      config: { system: name, method, rtol } as Json,
      init: () => {
        const start = stepperFor(method, { rtol, atol }).init(system.field, t0, vector(system.y0), tEnd)
        return { t: t0, y: [...system.y0], h: start.hAbs, evaluations: start.counts.nfev, failed: false }
      },
      step: (s, ctx) => {
        // The stepper starts from the plain state with the step the last one proposed.
        const remaining = tEnd - s.t
        const first = s.h > 0 && s.h < remaining ? s.h : remaining
        const stepper = stepperFor(method, { rtol, atol, firstStep: first })
        const next = stepper.step(stepper.init(system.field, s.t, vector(s.y), tEnd))
        if (next.status === 'failed') return { ...s, failed: true }
        const y = Array.from(toFlat(next.y))
        const evaluations = s.evaluations + next.counts.nfev
        ctx.log('t', next.t)
        ctx.log('y', y)
        ctx.log('h', next.t - s.t)
        ctx.log('evaluations', evaluations)
        return { t: next.t, y, h: next.hAbs, evaluations, failed: false }
      },
      done: (s) => (s.failed ? 'failed' : s.t >= tEnd ? 'converged' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as OdeSystemsState,
    }
  },
  contract: {
    t: scalar({ description: 'the time reached', unit: 'time' }),
    y: tensor({ role: 'state', shape: ['n'], description: 'the state at that time' }),
    h: scalar({ description: 'the size of the step taken (the controller’s choice)', unit: 'time' }),
    evaluations: scalar({ description: 'evaluations of the vector field so far' }),
  },
  // The state is drawn element by element, so it is kept in full (`all` keeps summaries).
  defaults: {
    t: Log(),
    y: Log({ detail: 'full' }),
    h: Log(),
    evaluations: Log(),
  },
})
