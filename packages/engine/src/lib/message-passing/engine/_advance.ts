/**
 * The advance of a run by one step: the messages of one group of the plan sent from the current ones (a sweep, a factor or a
 * message; flooding from the messages frozen at the start of the sweep), or, for the residual schedule, the one message whose
 * recomputed candidate has moved most, with the candidates of the edges it touches recomputed. Internal to message passing.
 */

import type { CompiledGraph } from '../core/graph.ts'
import type { Eta } from '../core/types.ts'
import { refreshDefined } from './_arguments.ts'
import { isPulled } from './_cavities.ts'
import { propose, resolve, sweepChange } from './_update.ts'
import type { Plan } from './schedules.ts'
import type { EngineSettings, ImproperCounts, MpState } from './state.ts'

/** The step function of a run of `g` under `settings` along `plan`, with the candidate message of an edge (what the residual schedule ranks). */
export function advancerOf(
  g: CompiledGraph,
  settings: EngineSettings,
  plan: Plan,
): {
  readonly candidateOf: (messages: readonly (readonly number[])[], e: number) => { eta: number[] | null; r: number }
  readonly advance: (state: MpState) => MpState
} {
  const residual = settings.schedule === 'residual'
  const flooding = settings.schedule === 'flooding'
  const candidateOf = (messages: readonly (readonly number[])[], e: number): { eta: number[] | null; r: number } => {
    const p = propose(g, settings, messages, e)
    if (p.kind === 'blocked' || p.kind === 'failed') return { eta: null, r: 0 }
    const edge = g.edges[e] as CompiledGraph['edges'][number]
    const family = (g.variables[edge.variable] as CompiledGraph['variables'][number]).family
    const raw = p.kind === 'flat' ? family.flat : p.eta
    const old = messages[e] as Eta
    return { eta: [...raw], r: family.change(old, raw) }
  }

  const endOfSweep = (
    state: MpState,
    messages: readonly (readonly number[])[],
    counts: ImproperCounts,
    sent: readonly number[],
    updates: number,
    candidates: MpState['candidates'],
    residuals: readonly number[],
    pending: number,
  ): MpState => {
    const { change, beliefs } = sweepChange(g, state.start, messages)
    return {
      messages,
      start: beliefs,
      frozen: null,
      sweeps: state.sweeps + 1,
      cursor: 0,
      updates,
      change,
      improper: counts,
      sent,
      candidates,
      residuals,
      unresolved: pending,
      pending: 0,
      damping: state.damping,
      restarts: state.restarts,
      restartAt: state.restartAt,
    }
  }

  const advance = (state: MpState): MpState => {
    const counts = { ...state.improper }
    const s = state.damping === settings.damping ? settings : { ...settings, damping: state.damping }
    if (residual) {
      const messages = state.messages.slice()
      let best = -1
      let bestR = 0
      for (let e = 0; e < state.residuals.length; e++) {
        const r = state.residuals[e] ?? 0
        if (r > bestR) {
          bestR = r
          best = e
        }
      }
      const candidates = state.candidates.slice()
      const residuals = state.residuals.slice()
      const sent: number[] = []
      if (best >= 0) {
        const c = state.candidates[best]
        messages[best] = resolve(
          g,
          s,
          messages,
          best,
          c === null || c === undefined ? { kind: 'failed' } : { kind: 'message', eta: c },
          counts,
        )
        sent.push(best)
        refreshDefined(g, settings.projection, messages, [best])
        const edge = g.edges[best] as CompiledGraph['edges'][number]
        const touched = new Set<number>()
        for (const e2 of (g.variables[edge.variable] as CompiledGraph['variables'][number]).edges) {
          for (const p of g.factors[(g.edges[e2] as CompiledGraph['edges'][number]).factor]?.ports ?? [])
            if (p.edge >= 0) touched.add(p.edge)
        }
        for (const e2 of touched) {
          const c2 = candidateOf(messages, e2)
          candidates[e2] = c2.eta
          residuals[e2] = c2.r
        }
        candidates[best] = candidates[best] ?? null
      }
      const cursor = state.cursor + 1
      const updates = state.updates + sent.length
      if (cursor >= plan.size) return endOfSweep(state, messages, counts, sent, updates, candidates, residuals, 0)
      return { ...state, messages, cursor, updates, improper: counts, sent, candidates, residuals }
    }
    const group = plan.groups[state.cursor] ?? []
    const frozen = flooding ? (state.frozen ?? state.messages) : null
    const messages = state.messages.slice()
    let unresolved = 0
    for (const e of group) {
      const proposal = propose(g, s, frozen ?? messages, e)
      const dropped = counts.skipped + counts.reset
      messages[e] = resolve(g, s, messages, e, proposal, counts)
      // An update that could not be computed, or that the improper policy dropped or reset, leaves the message as it was or as nothing: it has not been recomputed.
      if (
        proposal.kind === 'failed' ||
        (proposal.kind === 'blocked' && !isPulled(g, e, settings.projection)) ||
        counts.skipped + counts.reset > dropped
      )
        unresolved++
    }
    refreshDefined(g, settings.projection, messages, group)
    const cursor = state.cursor + 1
    const updates = state.updates + group.length
    const pending = state.pending + unresolved
    if (cursor >= plan.groups.length) return endOfSweep(state, messages, counts, group, updates, [], [], pending)
    return { ...state, messages, frozen, cursor, updates, improper: counts, sent: group, pending }
  }
  return { candidateOf, advance }
}
