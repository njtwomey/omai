/**
 * The message-passing engine: one implementation that runs a factor graph as a steppable, resumable `Algorithm`, with
 * the choices that distinguish expectation propagation, variational message passing and belief propagation as slots
 * of its settings (the projection, the schedule and its granularity, damping, what to do with an improper message).
 * The state is plain numbers (the message on every edge), so a run serialises, resumes, and can be stopped between any
 * two steps; the beliefs, the marginals and the evidence or bound are functions of the state, computed when asked.
 *
 * @index module The message-passing engine: one algorithm, slots for projection, schedule, damping, improper handling.
 */

import type { Json } from '../../base/index.ts'
import { type Algorithm, channels, type RunStatus, stateFromJson, stateToJson } from '../../trace/index.ts'
import { assertDamping } from '../core/_messages.ts'
import { ParameterError } from '../core/errors.ts'
import type { CompiledGraph } from '../core/graph.ts'
import type { Eta } from '../core/types.ts'
import { advancerOf } from './_advance.ts'
import { isStalled, restartOf } from './_restart.ts'
import { beliefOf, marginalsOf } from './cavity.ts'
import { evidenceOf } from './evidence.ts'
import { type Plan, planOf } from './schedules.ts'
import type { EngineSettings, ImproperCounts, MpState } from './state.ts'
import { mpVerdict } from './verdict.ts'

const NO_IMPROPER: ImproperCounts = { events: 0, skipped: 0, damped: 0, reset: 0 }

function configOf(s: EngineSettings): Json {
  return {
    projection: s.projection.name,
    schedule: s.schedule,
    granularity: s.granularity,
    damping: s.damping,
    improper: s.improper,
    tol: s.tol,
    maxSweeps: s.maxSweeps,
    restarts: s.restarts ?? 0,
  }
}

/**
 * The engine: a factor graph run as an `Algorithm`. A step sends the messages of one group of the plan (the whole
 * sweep, one factor or one message, by `granularity`); at the end of a sweep the change of the beliefs is measured
 * and the run is done when it falls below `tol` (`converged`), when a belief or message stops being finite
 * (`diverged`), or at `maxSweeps` (`completed`). Improper updates are repaired by the `improper` policy and counted.
 * The tape carries, per observation, the largest belief change, the sweeps and messages so far, the improper counts,
 * the evidence or bound, the messages of the last step (from the factor's id to the variable's, ids as in {@link graphStructure}) and the
 * marginals of every variable by id.
 *
 * Expectation propagation, variational message passing and belief propagation are settings of this one (see
 * `epAlgorithm`, `vmpAlgorithm` and `bpAlgorithm`, which are algorithms of their own with their own knobs).
 *
 * @param g - The compiled graph.
 * @param settings - The slots.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for damping outside [0, 1) or a schedule that does not fit the graph.
 * @concept message-passing
 * @index algorithm The message-passing engine on a factor graph: projection, schedule, damping slots.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gaussianPrior, messagePassingAlgorithm, momentProjection } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = compileGraph(
 *   { variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 1, 4)] },
 *   defaultOperators(),
 * )
 * const alg = messagePassingAlgorithm(g, { projection: momentProjection, schedule: 'serial', granularity: 'sweep', damping: 0, improper: 'skip', tol: 1e-9, maxSweeps: 10 })
 * runAlgorithm(alg, { maxSteps: 10 }).status // => 'converged'
 * ```
 */
export function messagePassingAlgorithm(g: CompiledGraph, settings: EngineSettings): Algorithm<MpState> {
  assertDamping(settings.damping, 'message passing')
  if (g.edges.length === 0) throw new ParameterError('message passing: the graph has no edge to send a message on')
  const plan: Plan = planOf(g, settings.schedule, settings.granularity, settings.order)
  const residual = settings.schedule === 'residual'
  const emptyBeliefs = (messages: readonly (readonly number[])[]): number[][] =>
    g.variables.map((_, i) => [...beliefOf(g, messages, i)])

  const { candidateOf, advance } = advancerOf(g, settings, plan)

  const init = (): MpState => {
    const messages: number[][] = g.edges.map((edge) => [
      ...(g.variables[edge.variable] as CompiledGraph['variables'][number]).family.flat,
    ])
    // The start messages the graph carries (a vague proper message where a factor needs a cavity before anything is known), then the run's own.
    for (const f of g.factors)
      for (const [pi, m] of f.start.entries()) {
        const e = f.ports[pi]?.edge
        if (m !== null && e !== undefined && e >= 0) messages[e] = [...m]
      }
    for (const m of settings.initial ?? []) {
      const e = g.factors[m.factor]?.ports[m.port]?.edge
      if (e === undefined || e < 0)
        throw new ParameterError(`message passing: no edge at port ${m.port} of factor ${m.factor}`)
      messages[e] = [...m.eta]
    }
    const candidates: (number[] | null)[] = []
    const residuals: number[] = []
    if (residual) {
      for (let e = 0; e < g.edges.length; e++) {
        const c = candidateOf(messages, e)
        candidates.push(c.eta)
        residuals.push(c.r)
      }
    }
    return {
      messages,
      start: emptyBeliefs(messages),
      frozen: null,
      sweeps: 0,
      cursor: 0,
      updates: 0,
      change: Number.POSITIVE_INFINITY,
      improper: NO_IMPROPER,
      sent: [],
      candidates,
      residuals,
      unresolved: 0,
      pending: 0,
      damping: settings.damping,
      restarts: 0,
      restartAt: 0,
    }
  }

  const step = (state0: MpState): MpState => {
    const next = advance(state0)
    return isStalled(g, settings, next) ? restartOf(settings, init, next) : next
  }

  const evidenceNow = (state: MpState): number => evidenceOf(g, settings, state)
  const evidenceName = settings.projection.evidence === 'elbo' ? 'elbo' : 'evidence'

  return {
    name: settings.projection.hook,
    config: configOf(settings),
    schema: channels({
      metrics: {
        change: { description: 'the largest change of a belief over the last completed sweep' },
        sweeps: { description: 'completed sweeps' },
        updates: { description: 'messages sent so far' },
        improper: { description: 'updates that would have made a belief improper, or failed' },
        skipped: { description: 'updates dropped by the improper policy' },
        damped: { description: 'updates shortened until the belief stayed proper' },
        reset: { description: 'messages reset to flat' },
        unresolved: {
          description:
            'updates of the last completed sweep that could not be computed (those messages were not recomputed)',
        },
        restarts: { description: 'automatic restarts from the start messages with a stronger damping' },
        [evidenceName]: {
          description:
            evidenceName === 'elbo' ? 'the variational lower bound' : 'the EP (or Bethe) estimate of the log evidence',
          direction: 'higher',
        },
      },
      messages: {
        message: { description: 'a message from a factor to a variable (ids: the factor id and the variable name)' },
        marginals: { description: 'the current marginal of every variable, by variable id' },
        verdict: {
          description: 'why a run that has ended did not converge (stalled, degenerate, diverged, completed)',
        },
      },
      frames: true,
    }),
    init,
    step,
    done: (state): RunStatus | undefined => mpVerdict(g, state, settings).status,
    observe: (state) => {
      const ev = evidenceNow(state)
      const metrics: Record<string, number> = {
        sweeps: state.sweeps,
        updates: state.updates,
        improper: state.improper.events,
        skipped: state.improper.skipped,
        damped: state.improper.damped,
        reset: state.improper.reset,
        unresolved: state.unresolved,
        restarts: state.restarts,
        ...(Number.isFinite(state.change) ? { change: state.change } : {}),
        ...(Number.isFinite(ev) ? { [evidenceName]: ev } : {}),
      }
      const marginals = marginalsOf(g, state)
      const verdict = mpVerdict(g, state, settings)
      return {
        metrics,
        messages: [
          ...(verdict.reason === undefined
            ? []
            : [{ kind: 'verdict', payload: { status: verdict.status ?? 'running', reason: verdict.reason } as Json }]),
          ...state.sent.map((e) => {
            const edge = g.edges[e] as CompiledGraph['edges'][number]
            const name = (g.variables[edge.variable] as CompiledGraph['variables'][number]).name
            return {
              kind: 'message',
              from: (g.factors[edge.factor] as CompiledGraph['factors'][number]).id,
              to: name,
              payload: {
                factor: edge.factor,
                variable: name,
                operator: (g.factors[edge.factor] as CompiledGraph['factors'][number]).operator.name,
                eta: [...(state.messages[e] as Eta)],
              } as Json,
            }
          }),
          { kind: 'marginals', payload: marginals as Json },
        ],
        frame: {
          marginals: marginals as Json,
          sent: state.sent.map((e) => {
            const edge = g.edges[e] as CompiledGraph['edges'][number]
            return {
              kind: 'factorToVariable',
              from: (g.factors[edge.factor] as CompiledGraph['factors'][number]).id,
              to: (g.variables[edge.variable] as CompiledGraph['variables'][number]).name,
            }
          }),
        },
      }
    },
    serialise: (state) => stateToJson(state),
    // A state saved before the run counted its unresolved updates has no such fields: they start at their initial values.
    deserialise: (json) => {
      const saved = stateFromJson(json) as unknown as Partial<MpState> & MpState
      return {
        ...saved,
        unresolved: saved.unresolved ?? 0,
        pending: saved.pending ?? 0,
        damping: saved.damping ?? settings.damping,
        restarts: saved.restarts ?? 0,
        restartAt: saved.restartAt ?? 0,
      }
    },
  }
}
