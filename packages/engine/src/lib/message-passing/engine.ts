/**
 * The message-passing engine: one implementation that runs a factor graph as a steppable, resumable `Algorithm`, with
 * the choices that distinguish expectation propagation, variational message passing and belief propagation as slots
 * of its settings (the projection, the schedule and its granularity, damping, what to do with an improper message).
 * The state is plain numbers (the message on every edge), so a run serialises, resumes, and can be stopped between any
 * two steps; the beliefs, the marginals and the evidence or bound are functions of the state, computed when asked.
 *
 * @index module The message-passing engine: one algorithm, slots for projection, schedule, damping, improper handling.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { type Algorithm, channels, type RunStatus, stateFromJson, stateToJson } from '../trace/index.ts'
import { type CompiledGraph, factorId } from './graph.ts'
import { ImproperMessageError } from './operators/args.ts'
import type { Projection } from './projections.ts'
import { type CustomOrder, type Granularity, type Plan, planOf, type ScheduleKind } from './schedules.ts'
import type { Arg, Eta, OperatorContext, OperatorResult } from './types.ts'

/** What to do when an update would make a proper belief improper (a negative precision: the usual failure of EP). */
export type ImproperPolicy = 'skip' | 'damp' | 'reset' | 'error'

/** A message to start from instead of the flat one. */
export interface InitialMessage {
  /** The factor's index. */
  readonly factor: number
  /** The port. */
  readonly port: number
  readonly eta: Eta
}

/** The slots of the engine. */
export interface EngineSettings {
  /** The message rule: moment matching, expected log, sum-product. */
  readonly projection: Projection
  /** The order of a sweep. */
  readonly schedule: ScheduleKind
  /** The order of a `custom` schedule. */
  readonly order?: CustomOrder
  /** How much of a sweep a step does. */
  readonly granularity: Granularity
  /** Damping in [0, 1): the new message is `(1 - damping) new + damping old` in natural parameters. */
  readonly damping: number
  /** The repair of an update that makes a proper belief improper. */
  readonly improper: ImproperPolicy
  /** Stop when no belief moves by more than this in a sweep (in the units of the family's change measure). */
  readonly tol: number
  /** Stop, uncompleted, after this many sweeps. */
  readonly maxSweeps: number
  /** Messages to start from. */
  readonly initial?: readonly InitialMessage[]
}

/** How often the improper-message repair acted. */
export interface ImproperCounts {
  /** Updates that would have made a belief improper, or failed on an improper quantity. */
  readonly events: number
  /** Updates dropped (the old message kept). */
  readonly skipped: number
  /** Updates shortened by halving the step until the belief stayed proper. */
  readonly damped: number
  /** Messages reset to flat. */
  readonly reset: number
}

/** The state of a run: a message on every edge, and where the run is in its sweep. */
export interface MpState {
  /** The message of every edge (factor to variable), natural parameters. */
  readonly messages: readonly (readonly number[])[]
  /** The beliefs at the start of the sweep, for the change over it. */
  readonly start: readonly (readonly number[])[]
  /** Flooding: the messages at the start of the sweep that every message of it is computed from. */
  readonly frozen: readonly (readonly number[])[] | null
  /** Completed sweeps. */
  readonly sweeps: number
  /** The step within the sweep (the group of edges to send next); 0 between sweeps. */
  readonly cursor: number
  /** Messages sent so far. */
  readonly updates: number
  /** The largest change of a belief over the last completed sweep (Infinity before the first). */
  readonly change: number
  readonly improper: ImproperCounts
  /** The edges sent by the last step. */
  readonly sent: readonly number[]
  /** The residual schedule: the message each edge would send, and how far it is from the one it has. */
  readonly candidates: readonly (readonly number[] | null)[]
  readonly residuals: readonly number[]
}

/** A picture of a run at one moment: where it is, and what it currently believes of every variable, by variable id. */
export interface MpSnapshot {
  readonly sweeps: number
  readonly updates: number
  readonly change: number
  readonly improper: ImproperCounts
  /** The current marginal of every variable by id (see {@link marginalsOf}). */
  readonly marginals: Readonly<Record<string, Readonly<Record<string, Json>>>>
}

const NO_IMPROPER: ImproperCounts = { events: 0, skipped: 0, damped: 0, reset: 0 }

const isFlat = (eta: Eta | undefined): boolean => eta !== undefined && eta.every((v) => v === 0)

/** The belief of a variable: the product of the messages of its edges (flat for none). */
export function beliefOf(g: CompiledGraph, messages: readonly (readonly number[])[], variable: number): Eta {
  const v = g.variables[variable]
  if (v === undefined) throw new OmaiError('value', `no variable ${variable}`)
  let acc: Eta = v.family.flat
  for (const e of v.edges) acc = v.family.product(acc, messages[e] as Eta)
  return acc
}

/** The belief of a variable without the message of one edge. */
function beliefWithout(
  g: CompiledGraph,
  messages: readonly (readonly number[])[],
  variable: number,
  edge: number,
): Eta {
  const v = g.variables[variable] as CompiledGraph['variables'][number]
  let acc: Eta = v.family.flat
  for (const e of v.edges) if (e !== edge) acc = v.family.product(acc, messages[e] as Eta)
  return acc
}

/** The arguments of a factor's operator: the cavity or the belief of each variable, the value of each observed one. */
function argsOf(
  g: CompiledGraph,
  messages: readonly (readonly number[])[],
  factor: number,
  projection: Projection,
): Arg[] {
  const f = g.factors[factor] as CompiledGraph['factors'][number]
  return f.ports.map((p): Arg => {
    if (p.edge < 0) return { family: p.family, eta: undefined, value: p.value }
    const b = beliefOf(g, messages, p.variable as number)
    if (projection.arguments === 'belief') return { family: p.family, eta: b, value: undefined }
    const own = messages[p.edge] as Eta
    const eta = f.power === 1 ? p.family.quotient(b, own).eta : p.family.product(b, p.family.power(own, -f.power))
    return { family: p.family, eta, value: undefined }
  })
}

/** The cavity as a raw difference, without the renormalisation of a discrete family: what the evidence's masses are taken of. */
function rawCavity(g: CompiledGraph, messages: readonly (readonly number[])[], factor: number): Arg[] {
  const f = g.factors[factor] as CompiledGraph['factors'][number]
  return f.ports.map((p): Arg => {
    if (p.edge < 0) return { family: p.family, eta: undefined, value: p.value }
    const b = beliefOf(g, messages, p.variable as number)
    const own = messages[p.edge] as Eta
    return { family: p.family, eta: b.map((v, i) => v - f.power * (own[i] ?? 0)), value: undefined }
  })
}

/** What an update proposes: a message, no information (a flat one), nothing (the old message stays), or a failure. */
type Proposal =
  | { readonly kind: 'message'; readonly eta: Eta }
  | { readonly kind: 'flat' }
  | { readonly kind: 'blocked' }
  | { readonly kind: 'failed' }

function propose(g: CompiledGraph, s: EngineSettings, source: readonly (readonly number[])[], e: number): Proposal {
  const edge = g.edges[e] as CompiledGraph['edges'][number]
  const f = g.factors[edge.factor] as CompiledGraph['factors'][number]
  const op = f.operator
  const to = edge.port
  const args = argsOf(g, source, edge.factor, s.projection)
  const target = args[to] as Arg
  const reads = typeof op.readsTarget === 'function' ? op.readsTarget(to) : op.readsTarget === true
  if (reads && target.eta !== undefined && !target.family.isProper(target.eta)) return { kind: 'blocked' }
  const ctx: OperatorContext = { mode: s.projection.hook, to, args, params: f.params, power: f.power }
  const hook = s.projection.hook === 'ep' ? op.ep : s.projection.hook === 'vmp' ? op.vmp : op.bp
  try {
    let result: OperatorResult
    if (hook !== undefined) {
      result = hook.call(op, ctx)
    } else if (s.projection.hook === 'ep' && op.tilted !== undefined) {
      const t = op.tilted(ctx)
      const own = target.eta as Eta
      const projected = target.family.fromMoments(t.moments)
      result = { message: projected.map((v, i) => (v - (own[i] ?? 0)) / f.power) }
    } else {
      throw new OmaiError('not-implemented', `${op.name} has no ${s.projection.hook} message rule`)
    }
    return { kind: 'message', eta: result.message }
  } catch (err) {
    if (err instanceof ImproperMessageError || (err instanceof OmaiError && err.code === 'value')) {
      const others = args.filter((a, q) => q !== to && a.eta !== undefined)
      return others.some((a) => isFlat(a.eta)) ? { kind: 'flat' } : { kind: 'failed' }
    }
    throw err
  }
}

/** The new message of an edge from a proposal: damped, and repaired if it would make a proper belief improper. */
function resolve(
  g: CompiledGraph,
  s: EngineSettings,
  current: readonly (readonly number[])[],
  e: number,
  proposal: Proposal,
  counts: { events: number; skipped: number; damped: number; reset: number },
): Eta {
  const edge = g.edges[e] as CompiledGraph['edges'][number]
  const family = (g.variables[edge.variable] as CompiledGraph['variables'][number]).family
  const old = current[e] as Eta
  if (proposal.kind === 'blocked') return old
  if (proposal.kind === 'failed') {
    counts.events++
    counts.skipped++
    return old
  }
  const raw = proposal.kind === 'flat' ? family.flat : proposal.eta
  const d = s.damping
  const next = d === 0 ? raw : family.product(family.power(old, d), family.power(raw, 1 - d))
  const base = beliefWithout(g, current, edge.variable, e)
  if (!family.isProper(family.product(base, old)) || family.isProper(family.product(base, next))) return next
  counts.events++
  if (s.improper === 'error') {
    throw new OmaiError(
      'convergence',
      `message passing: an update of ${factorId(edge.factor)} makes the belief of ${(g.variables[edge.variable] as CompiledGraph['variables'][number]).name} improper`,
    )
  }
  if (s.improper === 'reset') {
    counts.reset++
    return family.flat
  }
  if (s.improper === 'damp') {
    let step = 1
    for (let k = 0; k < 10; k++) {
      step /= 2
      const trial = old.map((v, i) => v + step * ((next[i] ?? 0) - v))
      if (family.isProper(family.product(base, trial))) {
        counts.damped++
        return trial
      }
    }
  }
  counts.skipped++
  return old
}

function sweepChange(
  g: CompiledGraph,
  start: readonly (readonly number[])[],
  messages: readonly (readonly number[])[],
): {
  readonly change: number
  readonly beliefs: number[][]
} {
  let change = 0
  const beliefs: number[][] = []
  for (const [i, v] of g.variables.entries()) {
    const b = [...beliefOf(g, messages, i)]
    beliefs.push(b)
    if (v.edges.length > 0) change = Math.max(change, v.family.change(start[i] as Eta, b))
  }
  return { change, beliefs }
}

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
    marginals: marginalsOf(g, state),
  }
}

/**
 * The evidence estimate or variational bound of a state: for moment matching and sum-product the EP free energy
 * (`sum_f log Z_f + sum_v (1 - degree_v) A(belief_v)`, which on a tree is the log evidence and for loopy BP the
 * Bethe approximation), for the expected-log rule the variational lower bound (`sum_f E_q[log f] + sum_v H(q_v)`).
 *
 * @param g - The compiled graph.
 * @param s - The settings that fix the projection.
 * @param state - A state of a run.
 * @returns The estimate, or NaN when some factor has no way to give its part or a belief has no density yet.
 */
export function evidenceOf(g: CompiledGraph, s: EngineSettings, state: MpState): number {
  let total = 0
  if (s.projection.evidence === 'ep-energy') {
    for (const [fi, f] of g.factors.entries()) {
      if (f.operator.logEvidence === undefined) return Number.NaN
      const args = rawCavity(g, state.messages, fi)
      total += f.operator.logEvidence({ mode: s.projection.hook, to: 0, args, params: f.params, power: f.power })
    }
    for (const [i, v] of g.variables.entries()) {
      if (v.observed !== undefined) continue
      total += (1 - v.edges.length) * v.family.logPartition(beliefOf(g, state.messages, i))
    }
    return total
  }
  for (const [fi, f] of g.factors.entries()) {
    if (f.operator.expectedLog === undefined) return Number.NaN
    const args = argsOf(g, state.messages, fi, s.projection)
    total += f.operator.expectedLog({ mode: 'vmp', to: 0, args, params: f.params, power: f.power })
  }
  for (const [i, v] of g.variables.entries()) {
    if (v.observed !== undefined) continue
    const b = beliefOf(g, state.messages, i)
    if (!v.family.isProper(b)) return Number.NaN
    total += v.family.entropy(b)
  }
  return total
}

function configOf(s: EngineSettings): Json {
  return {
    projection: s.projection.name,
    schedule: s.schedule,
    granularity: s.granularity,
    damping: s.damping,
    improper: s.improper,
    tol: s.tol,
    maxSweeps: s.maxSweeps,
  }
}

/**
 * The engine: a factor graph run as an `Algorithm`. A step sends the messages of one group of the plan (the whole
 * sweep, one factor or one message, by `granularity`); at the end of a sweep the change of the beliefs is measured
 * and the run is done when it falls below `tol` (`converged`), when a belief or message stops being finite
 * (`diverged`), or at `maxSweeps` (`completed`). Improper updates are repaired by the `improper` policy and counted.
 * The tape carries, per observation, the largest belief change, the sweeps and messages so far, the improper counts,
 * the evidence or bound, the messages of the last step (from `f<i>` to the variable, ids as in {@link graphStructure}) and the
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
  if (!(settings.damping >= 0 && settings.damping < 1))
    throw new OmaiError('value', `message passing: damping must be in [0, 1), got ${settings.damping}`)
  if (g.edges.length === 0) throw new OmaiError('value', 'message passing: the graph has no edge to send a message on')
  const plan: Plan = planOf(g, settings.schedule, settings.granularity, settings.order)
  const residual = settings.schedule === 'residual'
  const flooding = settings.schedule === 'flooding'
  const emptyBeliefs = (messages: readonly (readonly number[])[]): number[][] =>
    g.variables.map((_, i) => [...beliefOf(g, messages, i)])

  const candidateOf = (messages: readonly (readonly number[])[], e: number): { eta: number[] | null; r: number } => {
    const p = propose(g, settings, messages, e)
    if (p.kind === 'blocked' || p.kind === 'failed') return { eta: null, r: 0 }
    const edge = g.edges[e] as CompiledGraph['edges'][number]
    const family = (g.variables[edge.variable] as CompiledGraph['variables'][number]).family
    const raw = p.kind === 'flat' ? family.flat : p.eta
    const old = messages[e] as Eta
    return { eta: [...raw], r: family.change(old, raw) }
  }

  const init = (): MpState => {
    const messages: number[][] = g.edges.map((edge) => [
      ...(g.variables[edge.variable] as CompiledGraph['variables'][number]).family.flat,
    ])
    for (const m of settings.initial ?? []) {
      const e = g.factors[m.factor]?.ports[m.port]?.edge
      if (e === undefined || e < 0)
        throw new OmaiError('value', `message passing: no edge at port ${m.port} of factor ${m.factor}`)
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
    }
  }

  const endOfSweep = (
    state: MpState,
    messages: readonly (readonly number[])[],
    counts: ImproperCounts,
    sent: readonly number[],
    updates: number,
    candidates: MpState['candidates'],
    residuals: readonly number[],
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
    }
  }

  const step = (state: MpState): MpState => {
    const counts = { ...state.improper }
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
          settings,
          messages,
          best,
          c === null || c === undefined ? { kind: 'failed' } : { kind: 'message', eta: c },
          counts,
        )
        sent.push(best)
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
      if (cursor >= plan.size) return endOfSweep(state, messages, counts, sent, updates, candidates, residuals)
      return { ...state, messages, cursor, updates, improper: counts, sent, candidates, residuals }
    }
    const group = plan.groups[state.cursor] ?? []
    const frozen = flooding ? (state.frozen ?? state.messages) : null
    const messages = state.messages.slice()
    for (const e of group)
      messages[e] = resolve(g, settings, messages, e, propose(g, settings, frozen ?? messages, e), counts)
    const cursor = state.cursor + 1
    const updates = state.updates + group.length
    if (cursor >= plan.groups.length) return endOfSweep(state, messages, counts, group, updates, [], [])
    return { ...state, messages, frozen, cursor, updates, improper: counts, sent: group }
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
        [evidenceName]: {
          description:
            evidenceName === 'elbo' ? 'the variational lower bound' : 'the EP (or Bethe) estimate of the log evidence',
          direction: 'higher',
        },
      },
      messages: {
        message: { description: 'a message from a factor to a variable (ids: f<i> and the variable name)' },
        marginals: { description: 'the current marginal of every variable, by variable id' },
      },
      frames: true,
    }),
    init,
    step,
    done: (state): RunStatus | undefined => {
      if (state.messages.some((m) => m.some((v) => Number.isNaN(v) || v === Number.POSITIVE_INFINITY)))
        return 'diverged'
      if (residual) {
        if (state.updates > 0 && (state.residuals.length === 0 || Math.max(...state.residuals) < settings.tol))
          return 'converged'
      } else if (state.sweeps >= 1 && state.cursor === 0 && state.change < settings.tol) return 'converged'
      return state.sweeps >= settings.maxSweeps ? 'completed' : undefined
    },
    observe: (state) => {
      const ev = evidenceNow(state)
      const metrics: Record<string, number> = {
        sweeps: state.sweeps,
        updates: state.updates,
        improper: state.improper.events,
        skipped: state.improper.skipped,
        damped: state.improper.damped,
        reset: state.improper.reset,
        ...(Number.isFinite(state.change) ? { change: state.change } : {}),
        ...(Number.isFinite(ev) ? { [evidenceName]: ev } : {}),
      }
      const marginals = marginalsOf(g, state)
      return {
        metrics,
        messages: [
          ...state.sent.map((e) => {
            const edge = g.edges[e] as CompiledGraph['edges'][number]
            const name = (g.variables[edge.variable] as CompiledGraph['variables'][number]).name
            return {
              kind: 'message',
              from: factorId(edge.factor),
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
              from: factorId(edge.factor),
              to: (g.variables[edge.variable] as CompiledGraph['variables'][number]).name,
            }
          }),
        },
      }
    },
    serialise: (state) => stateToJson(state),
    deserialise: (json) => stateFromJson(json) as MpState,
  }
}
