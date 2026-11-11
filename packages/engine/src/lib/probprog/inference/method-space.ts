/**
 * Every inference method of the library that can run on a model, as one choice with each method's own knobs: for a bound model,
 * `methodSpace` asks every kind of `methodKinds` whether it applies (the method's own capability verdict over the analysis of
 * the model) and returns the union space of the ones that do, with the budget and snapshot knobs left out (the caller's
 * budget), together with the kinds that do not apply and the reason the library gives. A page offers the first in a drop-down
 * and draws the second disabled beside them with the reason, so nothing is hand-picked and nothing is silently absent. It
 * chooses nothing for the user: a method that applies but is slow or poor on the model stays in the list (the budget is the
 * guard, and `notes` can say what is known of it).
 */

import { OmaiError } from '../../base/index.ts'
import { type DictSpace, dict, type Space, type UnionSpace, union } from '../../space/index.ts'
import type { Config } from '../../trace/index.ts'
import { analysisOf, type MethodModel } from './_support.ts'
import { anytimeKnobs } from './method.ts'
import { methodKinds } from './models.ts'

/** Options of {@link methodSpace}. */
export interface MethodSpaceOptions {
  /** The kinds to consider (default every kind of `methodKinds`). */
  readonly kinds?: readonly string[]
  /** Starting values of knobs, by kind (`{ nuts: { maxTreeDepth: 5 } }`): the defaults the controls open with, nothing is removed. */
  readonly starting?: Readonly<Record<string, Readonly<Record<string, number | string | boolean>>>>
  /** Knobs that mean nothing for the model, left out of every option (`['inferDiscrete']` for a model with no discrete sites). */
  readonly omit?: readonly string[]
  /** The kind the union opens on when it applies (default the first that does). */
  readonly first?: string
}

/** What {@link methodSpace} returns. */
export interface MethodSpace {
  /** The union of the kinds that apply (tagged `method`), each option the method's own knobs; those that do not apply are in it too, with no knobs, so a control can draw them disabled. */
  readonly space: UnionSpace
  /** The kinds that apply, in the order of the table. */
  readonly applicable: readonly string[]
  /** The kinds that do not apply, with the library's own reason (and what would help, when it says). */
  readonly unavailable: Readonly<Record<string, string>>
}

/** The knobs of a configuration value of the union: the value without its tag. */
export function methodConfig(value: Readonly<Record<string, unknown>>): { kind: string; config: Config } {
  const { method, ...config } = value
  return { kind: String(method), config: config as Config }
}

/**
 * Which methods run on a model, as one union space of their knobs and a list of the ones that do not apply with the reason.
 *
 * @param model - A bound spec or a program.
 * @param options - The kinds to consider, the starting knobs of each, knobs to leave out.
 * @returns The union (tag `method`), the applicable kinds and the reason for each other.
 * @throws OmaiError with code `value` for a kind that is not in `methodKinds`.
 * @concept inference-engines
 * @index function Every inference method that runs on a model, as one union of knobs.
 * @example
 * ```ts
 * import { methodSpace } from 'omai-engine/probprog/inference'
 * import { defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 1)), 1), b.data('y', { dims: ['row'] })) })
 * const s = methodSpace(m.bind({ y: [1, 2] }), { kinds: ['nuts', 'enumeration'] })
 * [s.applicable, Object.keys(s.unavailable)] // => [['nuts'], ['enumeration']]
 * ```
 */
export function methodSpace(model: MethodModel, options: MethodSpaceOptions = {}): MethodSpace {
  const kinds = options.kinds ?? Object.keys(methodKinds)
  const analysis = analysisOf(model)
  const omit = new Set([...(options.omit ?? []), ...Object.keys(anytimeKnobs.spaces)])
  const optionsOf: Record<string, DictSpace> = {}
  const applicable: string[] = []
  const unavailable: Record<string, string> = {}
  for (const kind of kinds) {
    const make = methodKinds[kind]
    if (make === undefined) throw new OmaiError('value', `methodSpace: no inference method of kind '${kind}'`)
    let method: ReturnType<typeof make>
    try {
      method = make(model, { name: kind })
    } catch (e) {
      unavailable[kind] = e instanceof Error ? e.message : String(e)
      optionsOf[kind] = dict({})
      continue
    }
    const verdict = method.check?.(analysis)
    if (verdict !== undefined && !verdict.applies) {
      unavailable[kind] = verdict.hint === undefined ? verdict.reason : `${verdict.reason} (${verdict.hint})`
      optionsOf[kind] = dict({})
      continue
    }
    const starting = options.starting?.[kind] ?? {}
    const spaces: Record<string, Space> = {}
    for (const [knob, space] of Object.entries(method.config.spaces)) {
      if (omit.has(knob)) continue
      spaces[knob] = knob in starting ? ({ ...space, default: starting[knob] } as Space) : space
    }
    optionsOf[kind] = dict(spaces)
    applicable.push(kind)
  }
  const first = options.first !== undefined && applicable.includes(options.first) ? options.first : applicable[0]
  return {
    space: union(optionsOf, { tag: 'method', ...(first === undefined ? {} : { default: first }) }),
    applicable,
    unavailable,
  }
}
