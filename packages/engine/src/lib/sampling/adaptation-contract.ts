/**
 * What the warm-up algorithms log and how their chain states serialise: the `adapt/` paths shared by the step-size
 * search, Stan-style window adaptation and random-walk adaptation, and a codec for the chain state they carry (a
 * position, its log density and whatever else a kernel keeps).
 */

import type { Json } from '../base/index.ts'
import {
  type Contract,
  defineContract,
  message,
  numberFromJson,
  numberToJson,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import type { MCMCState } from './chains.ts'

/** The paths the warm-up algorithms log, as a type. */
export type AdaptPath =
  | 'adapt/stepSize'
  | 'adapt/window'
  | 'adapt/accept'
  | 'adapt/logDensity'
  | 'adapt/divergent'
  | 'adapt/treeDepth'
  | 'adapt/massMatrix'
  | 'adapt/windowEnd'
  | 'adapt/direction'

/**
 * What the warm-up algorithms log. The step-size search logs each size tried, whether the trial accepted above the
 * target (its direction, +1 to double, -1 to halve); window adaptation logs the step size in force, the acceptance of
 * the transition, the slow window it is in (0 in a fast buffer), divergences, tree depth, the inverse mass matrix and a
 * `windowEnd` message when a slow window closes; random-walk adaptation logs the proposal scale as `adapt/stepSize`.
 */
export const adaptContract: Contract<unknown, AdaptPath> = defineContract({
  'adapt/stepSize': scalar({ description: 'the step size (or proposal scale) of the transition' }),
  'adapt/window': scalar({ description: 'the slow window the iteration is in, 0 in a fast buffer' }),
  'adapt/accept': scalar({ description: 'the acceptance probability of the transition', direction: 'higher' }),
  'adapt/logDensity': scalar({ description: 'the log density at the current position' }),
  'adapt/divergent': scalar({ description: '1 when the transition diverged', direction: 'lower' }),
  'adapt/treeDepth': scalar({ description: 'the NUTS tree depth' }),
  'adapt/massMatrix': tensor({ role: 'state', shape: ['mass'], description: 'the inverse mass matrix in force' }),
  'adapt/windowEnd': message({
    description: 'a slow window ended: the new step size and inverse mass matrix',
    payload: { stepSize: 'number', inverseMassMatrix: 'number[]' },
  }),
  'adapt/direction': scalar({ description: 'the step-size search direction: +1 doubles, -1 halves' }),
})

/** A chain state as JSON (the log density may be infinite, which a plain round trip would lose). */
export function chainToJson(s: MCMCState): Json {
  return { rest: stateToJson({ ...s, logDensity: undefined }), logDensity: numberToJson(s.logDensity) }
}

/** The inverse of {@link chainToJson}. */
export function chainFromJson<S extends MCMCState>(j: Json): S {
  const o = j as { rest: Json; logDensity: Json }
  return { ...(stateFromJson(o.rest) as object), logDensity: numberFromJson(o.logDensity) } as S
}
