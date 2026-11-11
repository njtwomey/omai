/**
 * Beliefs and what is read off them: the belief of a variable (the product of the messages of its edges), the marginal of
 * every variable by id and the snapshot of a run. Functions of the messages alone, so any state of a run has them.
 */

import type { Json } from '../../base/index.ts'
import { FLAT_CAVITY_PRECISION } from '../core/args.ts'
import { ParameterError } from '../core/errors.ts'
import type { CompiledGraph } from '../core/graph.ts'
import type { Eta } from '../core/types.ts'
import type { MpSnapshot, MpState } from './state.ts'

/** The belief of a variable: the product of the messages of its edges (flat for none). */
export function beliefOf(g: CompiledGraph, messages: readonly (readonly number[])[], variable: number): Eta {
  const v = g.variables[variable]
  if (v === undefined) throw new ParameterError(`no variable ${variable}`)
  let acc: Eta = v.family.flat
  for (const e of v.edges) acc = v.family.product(acc, messages[e] as Eta)
  return acc
}

export { FLAT_CAVITY_PRECISION }

/**
 * The current marginal of every variable, by variable id: its family and the summary of its belief (`mean` and
 * `variance`, `probs`, `alpha`, ...), `improper: true` where the belief has no density yet, and for an observed
 * variable its value. This is what a view colours and annotates a node with.
 *
 * @param g - The compiled graph.
 * @param state - A state of a run.
 * @returns The marginals keyed by variable id.
 */
export function marginalsOf(g: CompiledGraph, state: MpState): Record<string, Record<string, Json>> {
  const out: Record<string, Record<string, Json>> = {}
  for (const [i, v] of g.variables.entries()) {
    if (v.observed !== undefined) {
      out[v.name] = { family: v.family.kind, observed: v.observed as Json }
      continue
    }
    out[v.name] = { family: v.family.kind, ...v.family.summary(beliefOf(g, state.messages, i)) }
  }
  return out
}

/**
 * A snapshot of a run: where it is (sweeps, messages sent, the last change, the improper counts) and the current
 * marginal of every variable by id. The anytime result: a function of the state alone, so any state of a run (the
 * live one, a checkpoint) has one.
 *
 * @param g - The compiled graph.
 * @param state - A state of a run.
 * @returns The snapshot.
 */
export function mpSnapshot(g: CompiledGraph, state: MpState): MpSnapshot {
  return {
    sweeps: state.sweeps,
    updates: state.updates,
    change: state.change,
    improper: state.improper,
    unresolved: state.unresolved,
    marginals: marginalsOf(g, state),
  }
}
