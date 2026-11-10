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
import type { CompiledGraph } from './graph.ts'
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

const isFlat = (eta: Eta | undefined): boolean => eta?.every((v) => v === 0) === true

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

/**
 * What the messages of a variable's other factors add up to, when the variable is defined by a deterministic factor and
 * the rule is the variational one: the messages those factors would send now (`pull`), summed. A defined variable keeps
 * no message from them (its belief is its definer's pushforward alone); the definer reads their sum as the cavity of its
 * output and passes it to its inputs.
 */
function pulledCavity(
  g: CompiledGraph,
  messages: readonly (readonly number[])[],
  variable: number,
  definer: number,
  projection: Projection,
): Eta {
  const v = g.variables[variable] as CompiledGraph['variables'][number]
  let acc: Eta = v.family.flat
  for (const e of v.edges) {
    if (e === definer) continue
    const edge = g.edges[e] as CompiledGraph['edges'][number]
    const f = g.factors[edge.factor] as CompiledGraph['factors'][number]
    const p = operatorMessage(f, projection, edge.port, argsOf(g, messages, edge.factor, projection))
    if (p.kind === 'message') acc = v.family.product(acc, p.eta)
  }
  return acc
}

/** Whether an edge's message is never stored: under the variational rule, a message to a defined variable from a factor that does not define it. */
function isPulled(g: CompiledGraph, e: number, projection: Projection): boolean {
  if (projection.hook !== 'vmp') return false
  const definer = (
    g.variables[(g.edges[e] as CompiledGraph['edges'][number]).variable] as CompiledGraph['variables'][number]
  ).definer
  return definer !== undefined && definer !== e
}

/**
 * Re-send the output message of every deterministic factor whose inputs moved (and, transitively, the factors reading
 * the outputs that changed), so that under the variational rule a defined variable's belief is at all times the
 * pushforward of the current beliefs of its inputs, whatever the schedule. The message to a defined output depends on the
 * inputs' beliefs alone, so the output port is shown no cavity.
 */
function refreshDefined(
  g: CompiledGraph,
  projection: Projection,
  messages: (readonly number[])[],
  edges: readonly number[],
): void {
  if (projection.hook !== 'vmp' || !g.hasDefined) return
  const moved = new Set(edges.map((e) => (g.edges[e] as CompiledGraph['edges'][number]).variable))
  for (const f of g.factors) {
    const k = f.operator.defines
    const out = k === undefined ? undefined : f.ports[k]
    if (k === undefined || out === undefined || out.edge < 0) continue
    if (!f.ports.some((p, i) => i !== k && p.variable !== undefined && moved.has(p.variable))) continue
    const args = argsOf(g, messages, f.index, projection, true)
    const proposal = operatorMessage(f, projection, k, args)
    if (proposal.kind === 'message') messages[out.edge] = proposal.eta
    moved.add(out.variable as number)
  }
}

/** The arguments of a factor's operator: the cavity or the belief of each variable, the value of each observed one. */
function argsOf(
  g: CompiledGraph,
  messages: readonly (readonly number[])[],
  factor: number,
  projection: Projection,
  forward = false,
): Arg[] {
  const f = g.factors[factor] as CompiledGraph['factors'][number]
  return f.ports.map((p, k): Arg => {
    if (p.edge < 0) return { family: p.family, eta: undefined, value: p.value }
    if (projection.hook === 'vmp' && f.operator.defines === k)
      return {
        family: p.family,
        eta: forward ? p.family.flat : pulledCavity(g, messages, p.variable as number, p.edge, projection),
        value: undefined,
      }
    const b = beliefOf(g, messages, p.variable as number)
    if (projection.arguments === 'belief') return { family: p.family, eta: b, value: undefined }
    const own = messages[p.edge] as Eta
    const eta = f.power === 1 ? p.family.quotient(b, own).eta : p.family.product(b, p.family.power(own, -f.power))
    return { family: p.family, eta, value: undefined }
  })
}

/** The belief of a variable as the plain sum of its messages: no renormalisation, so the masses of the evidence keep their scale. */
function rawBelief(g: CompiledGraph, messages: readonly (readonly number[])[], variable: number): number[] {
  const v = g.variables[variable] as CompiledGraph['variables'][number]
  const acc = new Array<number>(v.family.size).fill(0)
  for (const e of v.edges) {
    const m = messages[e] as Eta
    for (let i = 0; i < acc.length; i++) acc[i] = (acc[i] ?? 0) + (m[i] ?? 0)
  }
  return acc
}

/** The cavity as a raw difference of raw sums, without the renormalisation of a discrete family: what the evidence's masses are taken of. */
function rawCavity(g: CompiledGraph, messages: readonly (readonly number[])[], factor: number): Arg[] {
  const f = g.factors[factor] as CompiledGraph['factors'][number]
  return f.ports.map((p): Arg => {
    if (p.edge < 0) return { family: p.family, eta: undefined, value: p.value }
    const b = rawBelief(g, messages, p.variable as number)
    const own = messages[p.edge] as Eta
    return { family: p.family, eta: b.map((v, i) => v - f.power * (own[i] ?? 0)), value: undefined }
  })
}

/** What an update proposes: a message, no information (a flat one), nothing (the old message stays), or a failure. */
export type FactorProposal =
  | { readonly kind: 'message'; readonly eta: Eta }
  | { readonly kind: 'flat' }
  | { readonly kind: 'blocked' }
  | { readonly kind: 'failed' }

/**
 * The message a factor's operator sends to one port under a projection, from the arguments it is shown: the closed-form
 * rule of the projection's hook when the operator has one, else (for expectation propagation) the moments of its
 * `tilted` function projected onto the family and divided by the factor's power. A port whose own incoming message has
 * no density is `blocked` for an operator that reads it, and an operator that fails for an improper quantity gives
 * `flat` (when some other message is still flat: nothing is known yet) or `failed`.
 *
 * @param f - The compiled factor.
 * @param projection - The projection (its hook is the rule asked for).
 * @param to - The port to send to.
 * @param args - The arguments of every port: the cavities (or the beliefs) and the observed values.
 * @param power - The exponent of the factor in this call (the factor's own power, or the exponent of an alpha update).
 * @returns The proposal.
 */
export function operatorMessage(
  f: CompiledGraph['factors'][number],
  projection: Projection,
  to: number,
  args: readonly Arg[],
  power: number = f.power,
): FactorProposal {
  const op = f.operator
  const target = args[to] as Arg
  const reads = typeof op.readsTarget === 'function' ? op.readsTarget(to) : op.readsTarget === true
  if (reads && target.eta !== undefined && !target.family.isProper(target.eta)) return { kind: 'blocked' }
  const ctx: OperatorContext = {
    mode: projection.hook,
    to,
    args,
    params: f.params,
    power,
    ...(projection.numeric === undefined ? {} : { rule: projection.numeric }),
  }
  const hook = projection.hook === 'ep' ? op.ep : projection.hook === 'vmp' ? op.vmp : op.bp
  try {
    let result: OperatorResult
    if (hook !== undefined) {
      result = hook.call(op, ctx)
    } else if (projection.hook === 'ep' && op.tilted !== undefined) {
      const t = op.tilted(ctx)
      const own = target.eta as Eta
      const projected = target.family.fromMoments(t.moments)
      result = { message: projected.map((v, i) => (v - (own[i] ?? 0)) / power) }
    } else {
      throw new OmaiError('not-implemented', `${op.name} has no ${projection.hook} message rule`)
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

/** The largest number of fixed-point iterations of one alpha update. */
const ALPHA_ITERATIONS = 200

/**
 * The update of an alpha-divergence projection (Minka 2005) for one edge: the member of the family that minimises the
 * alpha-divergence D_alpha(f q_cavity || q) of the tilted distribution, found as the fixed point of "project the
 * distribution proportional to f^alpha cavity^alpha q^(1 - alpha) onto the family by moment matching" (the stationarity
 * condition of D_alpha for an exponential family). The first iteration at alpha = 1 is the EP update.
 */
function alphaProposal(
  g: CompiledGraph,
  s: EngineSettings,
  source: readonly (readonly number[])[],
  e: number,
  alpha: number,
): FactorProposal {
  const edge = g.edges[e] as CompiledGraph['edges'][number]
  const f = g.factors[edge.factor] as CompiledGraph['factors'][number]
  const to = edge.port
  const cavities = argsOf(g, source, edge.factor, s.projection)
  const target = cavities[to] as Arg
  const family = target.family
  const cavity = target.eta as Eta
  // The other ports see the cavity with the (1 - alpha) share of their own current message: cavity^alpha q^(1 - alpha).
  const others = f.ports.map((p, q): Arg => {
    const a = cavities[q] as Arg
    if (q === to || p.edge < 0 || a.eta === undefined) return a
    return { ...a, eta: a.family.product(a.eta, a.family.power(source[p.edge] as Eta, 1 - alpha)) }
  })
  let own: Eta = source[e] as Eta
  for (let k = 0; k < ALPHA_ITERATIONS; k++) {
    const base = family.product(cavity, family.power(own, 1 - alpha))
    const args = others.map((a, q) => (q === to ? { ...a, eta: base } : a))
    const p = operatorMessage(f, s.projection, to, args, alpha)
    if (p.kind !== 'message') return p
    // The operator returns (projection - base) / alpha: the projected member of the family is base + alpha * message.
    const projected = base.map((v, i) => v + alpha * (p.eta[i] ?? 0))
    const next = projected.map((v, i) => v - (cavity[i] ?? 0))
    const moved = family.change(own, next)
    own = next
    if (!(moved > 1e-12)) break
  }
  return { kind: 'message', eta: own }
}

function propose(
  g: CompiledGraph,
  s: EngineSettings,
  source: readonly (readonly number[])[],
  e: number,
): FactorProposal {
  const edge = g.edges[e] as CompiledGraph['edges'][number]
  const f = g.factors[edge.factor] as CompiledGraph['factors'][number]
  if (isPulled(g, e, s.projection)) return { kind: 'blocked' }
  const alpha = s.projection.alpha
  if (alpha !== undefined && alpha !== 1 && f.operator.power === true && f.power === 1)
    return alphaProposal(g, s, source, e, alpha)
  return operatorMessage(f, s.projection, edge.port, argsOf(g, source, edge.factor, s.projection))
}

/** The new message of an edge from a proposal: damped, and repaired if it would make a proper belief improper. */
function resolve(
  g: CompiledGraph,
  s: EngineSettings,
  current: readonly (readonly number[])[],
  e: number,
  proposal: FactorProposal,
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
      `message passing: an update of ${(g.factors[edge.factor] as CompiledGraph['factors'][number]).id} makes the belief of ${(g.variables[edge.variable] as CompiledGraph['variables'][number]).name} improper`,
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
 * Factors with a power p contribute `log Z_a / p` and the variable terms use the sum of the inverse powers (power EP).
 *
 * @param g - The compiled graph.
 * @param s - What fixes the estimate: the engine settings of a run, or just `{ projection }` (the estimate depends on nothing else).
 * @param state - A state of a run.
 * @returns The estimate, or NaN when some factor has no way to give its part, a belief has no density yet (a state with flat
 * messages, under any projection), or the projection is an alpha-divergence one.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, evidenceOf, gaussianPrior, klqpProjection, vmpAlgorithm } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1)] }, defaultOperators())
 * Number.isNaN(evidenceOf(g, { projection: klqpProjection }, runAlgorithm(vmpAlgorithm(g), { maxSteps: 0 }).state)) // => true
 * ```
 */
export function evidenceOf(g: CompiledGraph, s: Pick<EngineSettings, 'projection'>, state: MpState): number {
  let total = 0
  // An alpha-divergence projection has no evidence estimate here (its free energy is not the EP one).
  if (s.projection.alpha !== undefined && s.projection.alpha !== 1) return Number.NaN
  if (s.projection.evidence === 'ep-energy') {
    // The power EP energy: sum_a (1 / p_a) log Z_a + sum_v (1 - sum_{a at v} 1 / p_a) A(belief_v), with Z_a the mass of
    // the factor to the power p_a against its cavity (the usual EP energy when every power is 1).
    const inversePowers = new Array<number>(g.variables.length).fill(0)
    for (const [fi, f] of g.factors.entries()) {
      if (f.operator.logEvidence === undefined) return Number.NaN
      const args = rawCavity(g, state.messages, fi)
      total +=
        f.operator.logEvidence({
          mode: s.projection.hook,
          to: 0,
          args,
          params: f.params,
          power: f.power,
          ...(s.projection.numeric === undefined ? {} : { rule: s.projection.numeric }),
        }) / f.power
      for (const p of f.ports) {
        if (p.edge >= 0) inversePowers[p.variable as number] = (inversePowers[p.variable as number] ?? 0) + 1 / f.power
      }
    }
    for (const [i, v] of g.variables.entries()) {
      if (v.observed !== undefined) continue
      total += (1 - (inversePowers[i] ?? 0)) * v.family.logPartition(rawBelief(g, state.messages, i))
    }
    return total
  }
  for (const [fi, f] of g.factors.entries()) {
    if (f.operator.expectedLog === undefined) return Number.NaN
    const args = argsOf(g, state.messages, fi, s.projection)
    try {
      total += f.operator.expectedLog({ mode: 'vmp', to: 0, args, params: f.params, power: f.power })
    } catch (err) {
      // A belief with no density yet (flat messages at the start of a run) has no expectation: no bound, as for EP.
      if (err instanceof ImproperMessageError || (err instanceof OmaiError && err.code === 'value')) return Number.NaN
      throw err
    }
  }
  for (const [i, v] of g.variables.entries()) {
    // A defined variable is a function of the others under the mean field: its entropy is not a term of the bound.
    if (v.observed !== undefined || v.definer !== undefined) continue
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
      if (cursor >= plan.size) return endOfSweep(state, messages, counts, sent, updates, candidates, residuals)
      return { ...state, messages, cursor, updates, improper: counts, sent, candidates, residuals }
    }
    const group = plan.groups[state.cursor] ?? []
    const frozen = flooding ? (state.frozen ?? state.messages) : null
    const messages = state.messages.slice()
    for (const e of group)
      messages[e] = resolve(g, settings, messages, e, propose(g, settings, frozen ?? messages, e), counts)
    refreshDefined(g, settings.projection, messages, group)
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
        message: { description: 'a message from a factor to a variable (ids: the factor id and the variable name)' },
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
    deserialise: (json) => stateFromJson(json) as MpState,
  }
}
