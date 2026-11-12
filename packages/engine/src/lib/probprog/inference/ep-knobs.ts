/**
 * The knobs of expectation propagation, declared once. Every EP method of this folder (`epSpecMethod`, `epGridMethod`,
 * `epMultiStartMethod`) takes the same run knobs (schedule, damping, improper policy, tolerance, sweep cap, restarts and the
 * power of fractional EP) from `epRunKnobs(method)`, which differs between the methods only in the defaults (`ep` stops at
 * 1e-8 over 200 sweeps; a node of a grid and a start of a multi-start run at 1e-9, over 200 and 300 sweeps) and in
 * `granularity`, which only a method whose step is a sweep has. `EpKnobs` is the typed record of them (what an atom's `ep`
 * is), `checkKnobKeys` rejects a key the space does not have (a misspelled knob would otherwise run silently on the
 * defaults), and `atPower` puts the power of fractional EP on the factors that take one.
 */

import { OmaiError } from '../../base/index.ts'
import {
  type CompiledGraph,
  epDefaults,
  type Granularity,
  type ImproperPolicy,
  type MessagePassingOptions,
  messagePassingOptions,
  messagePassingSpace,
  takesPower,
  withPowers,
} from '../../message-passing/index.ts'
import { choice, type DictSpace, dict, real } from '../../space/index.ts'
import type { Config } from '../../trace/index.ts'

/** The EP methods of this folder, which share the run knobs. */
export type EpMethodName = 'ep' | 'ep-grid' | 'ep-multistart'

/** The knobs of an EP run: what a method is configured with, or an atom opens on, over the method's defaults. */
export type EpKnobs = {
  /** The order of a sweep. */
  readonly schedule?: 'serial' | 'flooding' | 'tree' | 'residual'
  /** How much of a sweep a step does (plain `ep` only: a step of the grid and the multi-start methods is a whole EP run). */
  readonly granularity?: Granularity
  /** The weight of the old message in the new. */
  readonly damping?: number
  /** The repair of an update that makes a belief improper. */
  readonly improper?: ImproperPolicy
  /** Stop when no belief moves more than this over a sweep. */
  readonly tol?: number
  /** The sweep cap. */
  readonly maxSweeps?: number
  /** Automatic restarts with a stronger damping when the run stalls. */
  readonly restarts?: number
  /** Fractional EP: the power of every factor that supports one (1 is EP). */
  readonly power?: number
  /** Damped first sweeps (`auto`: on a graph with exchangeable labels, the components of a mixture; plain `ep` and `ep-multistart` only). */
  readonly warmup?: 'auto' | 'off' | 'on'
}

type RunDefaults = Required<Omit<MessagePassingOptions, 'order' | 'initial'>>

/** The defaults of each method: the engine's, but a node of a grid and a start of a multi-start run tighter and a start gets more sweeps. */
const DEFAULTS: Readonly<Record<EpMethodName, RunDefaults>> = {
  ep: epDefaults,
  'ep-grid': { ...epDefaults, tol: 1e-9, maxSweeps: 200 },
  'ep-multistart': { ...epDefaults, tol: 1e-9, maxSweeps: 300 },
}

/**
 * The run knobs of an EP method as a `Space`, with the method's defaults.
 *
 * @param method - `ep` (the default), `ep-grid` or `ep-multistart`.
 * @returns The space: schedule, [granularity for `ep`], damping, improper, tol, maxSweeps, restarts, [warmup, not for `ep-grid`] and power.
 * @example
 * ```ts
 * import { defaults } from 'omai-engine/space'
 * import { epRunKnobs } from 'omai-engine/probprog/inference'
 * [defaults(epRunKnobs('ep'))['maxSweeps'], defaults(epRunKnobs('ep-multistart'))['maxSweeps']] // => [200, 300]
 * ```
 */
export function epRunKnobs(method: EpMethodName = 'ep'): DictSpace {
  const shared = Object.entries(messagePassingSpace(DEFAULTS[method]).spaces).filter(
    ([k]) => method === 'ep' || k !== 'granularity',
  )
  return dict({
    ...Object.fromEntries(shared),
    ...(method === 'ep-grid'
      ? {}
      : {
          warmup: choice(['auto', 'off', 'on'] as const, {
            default: 'auto',
            description:
              'the first sweeps damped by a half, then the run at its own damping: auto does it where the labels of a mixture are exchangeable (a start from flat messages can fall into the symmetric fixed point, every component the same), on always, off never',
          }),
        }),
    power: real(0.05, 2, {
      default: 1,
      label: 'power',
      description:
        'fractional EP: the power of every factor that supports one (1 is EP). Below 1 each factor steps less (the site is divided by the power), which can settle a run that oscillates; the fixed point is the fractional-EP one, so the spreads move a little, not to the exact ones of variables that many factors share (a joint site does that)',
    }),
  })
}

/**
 * Reject the keys of a configuration that a space does not have: a misspelled knob (`dampng`) would otherwise be dropped and
 * the run would go on with the defaults, reporting success.
 *
 * @param who - What is being configured (for the message).
 * @param config - The configuration.
 * @param space - The space of the method.
 * @throws OmaiError with code `value` naming the unknown keys and the known ones.
 * @example
 * ```ts
 * import { checkKnobKeys, epRunKnobs } from 'omai-engine/probprog/inference'
 * checkKnobKeys('my atom', { damping: 0.5 }, epRunKnobs('ep')) // => undefined
 * ```
 */
export function checkKnobKeys(who: string, config: Config, space: DictSpace): void {
  const known = Object.keys(space.spaces)
  const unknown = Object.keys(config).filter((k) => !known.includes(k))
  if (unknown.length > 0)
    throw new OmaiError(
      'value',
      `${who}: unknown EP knob${unknown.length > 1 ? 's' : ''} ${unknown.map((k) => `'${k}'`).join(', ')} (the knobs are ${known.join(', ')})`,
    )
}

/**
 * The graph with the power of fractional EP on every factor that takes one (the others stay at 1); the graph itself at power 1.
 *
 * @param graph - The compiled graph.
 * @param power - The power.
 * @returns The graph at that power.
 * @throws OmaiError with code `value` for a power that is not positive.
 */
export function atPower(graph: CompiledGraph, power: number): CompiledGraph {
  return power === 1 ? graph : withPowers(graph, (f) => (takesPower(f.operator, f.params) ? power : 1))
}

/** The power a configuration asks for (1 when it gives none). */
export function powerOf(config: Config): number {
  return typeof config['power'] === 'number' ? config['power'] : 1
}

/** The settings of one EP run of a method's configuration: the engine's options, from the configuration over the engine's defaults. */
export type EpRunSettings = Required<
  Pick<MessagePassingOptions, 'schedule' | 'damping' | 'improper' | 'tol' | 'maxSweeps' | 'restarts'>
>

/**
 * The settings of an EP run from a configuration of the shared knobs (a grid node and a start of a multi-start are each one).
 *
 * @param config - The configuration of the method.
 * @returns The schedule, damping, improper policy, tolerance, sweep cap and restarts.
 */
export function epRunSettings(config: Config): EpRunSettings {
  const o = messagePassingOptions(config)
  return {
    schedule: o.schedule ?? epDefaults.schedule,
    damping: o.damping ?? epDefaults.damping,
    improper: o.improper ?? epDefaults.improper,
    tol: o.tol ?? epDefaults.tol,
    maxSweeps: o.maxSweeps ?? epDefaults.maxSweeps,
    restarts: o.restarts ?? epDefaults.restarts,
  }
}
