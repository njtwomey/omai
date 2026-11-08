/**
 * Belief propagation (message passing) on the factor graph of any discrete model: sum-product for marginals or
 * max-product for max-marginals, with flooding, sequential or residual schedules and damping, exact on trees and an
 * approximation (loopy BP) otherwise. A traced algorithm, the function returning beliefs with the Bethe approximation
 * of log Z, and the Bethe free energy.
 */

import { heap, type Json, OmaiError } from '../../base/index.ts'
import {
  type Algorithm,
  channels,
  type RunStatus,
  runAlgorithm,
  type TracePolicy,
  type Tracer,
} from '../../trace/index.ts'
import { type Factor, logFactor } from '../factor.ts'
import type { DiscreteModel } from '../models/factor-graph.ts'
import {
  type Compiled,
  compile,
  damp,
  factorBeliefLogs,
  marginalFactor,
  messageChange,
  normaliseLog,
  observedMarginals,
  projectFactor,
} from './_compiled.ts'

/** Message schedules of {@link beliefPropagation}. */
export type BpSchedule = 'flooding' | 'sequential' | 'residual' | 'tree'

/** Options of {@link beliefPropagation}. */
export interface BeliefPropagationOptions {
  /** Observed states by variable (names or indices). */
  readonly evidence?: Readonly<Record<string, string | number>>
  /** Sum-product (marginals) or max-product (max-marginals). */
  readonly mode?: 'sum' | 'max'
  /**
   * `flooding` (every message from the previous iteration's, default), `sequential` (factor by factor, using the
   * newest messages), `residual` (the message that would change most first, Elidan et al. 2006) or `tree` (one
   * leaves-to-root-to-leaves pass, exact on a factor tree).
   */
  readonly schedule?: BpSchedule
  /**
   * How much one step does. `sweep` (default): a whole iteration, as above. `factor`: one factor's messages (for
   * flooding the first step of an iteration computes every variable-to-factor message, then each step is one factor's
   * outgoing messages; the tree and residual schedules treat it as `message`). `message`: a single directed message
   * (for the residual schedule, the pending message that would change most). A run takes more, smaller steps and ends
   * in the same beliefs; the convergence test is made at the end of an iteration.
   */
  readonly granularity?: 'sweep' | 'factor' | 'message'
  /** Damping in [0, 1): each new factor-to-variable message is (1 - damping) new + damping old. */
  readonly damping?: number
  /** Stop when no message changes by more than this (in probability, default 1e-10). */
  readonly tol?: number
  /** Iteration cap (default 200). */
  readonly maxIter?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of a belief-propagation run: every message on every edge, in log space and normalised. */
export interface BeliefPropagationState {
  /** Factor-to-variable messages, by edge. */
  readonly f2v: readonly Float64Array[]
  /** Variable-to-factor messages, by edge. */
  readonly v2f: readonly Float64Array[]
  /** The largest message change in the last completed iteration (the pending change under the residual schedule). */
  readonly residual: number
  /** Message updates so far. */
  readonly updates: number
  /** Where a finer-grained run is in its iteration: the index of the next step of the plan (0 between iterations). */
  readonly cursor: number
  /** The largest message change so far in the iteration under way (finer granularity only). */
  readonly running: number
}

/** Result of {@link beliefPropagation}. */
export interface BeliefPropagationResult {
  /** The belief (approximate marginal, or max-marginal) of every variable; observed ones are point masses. */
  readonly marginals: Record<string, Factor>
  /** The belief of each factor with unobserved variables, over its reduced scope. */
  readonly factorBeliefs: readonly Factor[]
  /** The Bethe free energy of the beliefs (sum-product only; NaN under max-product). */
  readonly betheFreeEnergy: number
  /** The Bethe approximation of log Z (exact on trees): minus the free energy, plus constants from the evidence. */
  readonly logPartition: number
  readonly converged: boolean
  readonly status: RunStatus
  readonly iterations: number
  readonly residual: number
}

const uniform = (k: number): Float64Array => new Float64Array(k).fill(-Math.log(k))

/** Variable-to-factor message on edge e: the sum of the variable's other incoming factor messages. */
function variableMessage(c: Compiled, f2v: readonly Float64Array[], e: number, mode: 'sum' | 'max'): Float64Array {
  const { variable } = c.edges[e] ?? { variable: 0 }
  const out = new Float64Array(c.card[variable] ?? 1)
  for (const e2 of c.variableEdges[variable] ?? []) {
    if (e2 === e) continue
    const m = f2v[e2]
    if (m !== undefined) for (let x = 0; x < out.length; x++) out[x] = (out[x] ?? 0) + (m[x] ?? 0)
  }
  normaliseLog(out, mode)
  return out
}

/** Factor-to-variable message on edge e from the current variable-to-factor messages. */
function factorMessage(c: Compiled, v2f: readonly Float64Array[], e: number, mode: 'sum' | 'max'): Float64Array {
  const { factor, variable, position } = c.edges[e] ?? { factor: 0, variable: 0, position: 0 }
  const f = c.factors[factor]
  if (f === undefined) throw new OmaiError('value', 'belief propagation: unknown factor')
  const incoming = (c.factorEdges[factor] ?? []).map((e2) => (e2 === e ? null : (v2f[e2] ?? null)))
  const out = projectFactor(f, c.card[variable] ?? 1, position, incoming, mode)
  normaliseLog(out, mode)
  return out
}

/** The directed edge updates of one leaves-to-root-to-leaves pass over a factor forest. */
function treeSchedule(c: Compiled): { kind: 'v2f' | 'f2v'; edge: number }[] {
  const n = c.names.length
  const node = (e: number, side: 'v' | 'f'): number =>
    side === 'v' ? (c.edges[e]?.variable ?? 0) : n + (c.edges[e]?.factor ?? 0)
  const parentEdge = new Map<number, number>()
  const seen = new Set<number>()
  const order: number[] = []
  const total = n + c.factors.length
  for (let root = 0; root < total; root++) {
    if (seen.has(root)) continue
    seen.add(root)
    const queue = [root]
    while (queue.length > 0) {
      const u = queue.shift() as number
      order.push(u)
      const es = u < n ? (c.variableEdges[u] ?? []) : (c.factorEdges[u - n] ?? [])
      for (const e of es) {
        const w = u < n ? node(e, 'f') : node(e, 'v')
        if (seen.has(w)) {
          if (parentEdge.get(u) !== e)
            throw new OmaiError('value', 'belief propagation: the tree schedule needs a factor tree')
          continue
        }
        seen.add(w)
        parentEdge.set(w, e)
        queue.push(w)
      }
    }
  }
  // A message from u along edge e: v2f when u is a variable, f2v when u is a factor.
  const send = (u: number, e: number): { kind: 'v2f' | 'f2v'; edge: number } => ({
    kind: u < n ? 'v2f' : 'f2v',
    edge: e,
  })
  const collect = [...order].reverse().flatMap((u) => {
    const e = parentEdge.get(u)
    return e === undefined ? [] : [send(u, e)]
  })
  const distribute = order.flatMap((u) => {
    const es = u < n ? (c.variableEdges[u] ?? []) : (c.factorEdges[u - n] ?? [])
    return es.filter((e) => parentEdge.get(u) !== e).map((e) => send(u, e))
  })
  return [...collect, ...distribute]
}

/** One directed message update of a plan. */
type BpUnit = { readonly kind: 'v2f' | 'f2v'; readonly edge: number }

/**
 * The steps of one iteration of a finer-grained run: each step is a group of directed updates. Flooding computes every
 * variable-to-factor message first (from the messages the iteration began with, which do not change until the
 * factor-to-variable ones start), sequential goes factor by factor with the newest messages, and the tree schedule
 * follows its leaves-to-root-to-leaves order.
 */
function stepPlan(
  c: Compiled,
  schedule: BpSchedule,
  granularity: 'factor' | 'message',
  tree: readonly BpUnit[],
): BpUnit[][] {
  const v2f = (e: number): BpUnit => ({ kind: 'v2f', edge: e })
  const f2v = (e: number): BpUnit => ({ kind: 'f2v', edge: e })
  const each = (units: readonly BpUnit[]): BpUnit[][] => units.map((u) => [u])
  if (schedule === 'tree') return each(tree)
  if (schedule === 'flooding') {
    const all = c.edges.map((_, e) => e)
    if (granularity === 'message') return each([...all.map(v2f), ...all.map(f2v)])
    return [all.map(v2f), ...c.factorEdges.map((es) => es.map(f2v))]
  }
  // sequential: a factor's variable messages, then its factor messages.
  if (granularity === 'message') return each(c.factorEdges.flatMap((es) => [...es.map(v2f), ...es.map(f2v)]))
  return c.factorEdges.map((es) => [...es.map(v2f), ...es.map(f2v)])
}

/**
 * Belief propagation on the factor graph of any discrete model, as an {@link Algorithm}: sum-product for marginals or
 * max-product for max-marginals, with flooding, sequential, residual or tree schedules and optional damping. On a
 * factor tree it is exact; on a loopy graph it iterates to a fixed point of the Bethe free energy (loopy belief
 * propagation) or stops at the iteration cap. Each step is one iteration (one pass of updates, or for the residual
 * schedule as many single updates as there are edges), or with `granularity` one factor's messages or a single message;
 * observed states record the residual and the Bethe free energy, every variable's belief as the frame, and the
 * factor-to-variable messages as `message` events.
 *
 * @param model - The model.
 * @param options - Evidence, mode, schedule, damping and stopping rule.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for the tree schedule on a loopy graph, or inconsistent evidence.
 * @concept belief-propagation
 * @example
 * ```ts
 * import { isingModel, beliefPropagationAlgorithm } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(beliefPropagationAlgorithm(isingModel({ rows: 3, cols: 3, coupling: 0.2 })), { maxSteps: 200 }).status // => 'converged'
 * ```
 */
export function beliefPropagationAlgorithm(
  model: DiscreteModel,
  options: BeliefPropagationOptions = {},
): Algorithm<BeliefPropagationState> {
  const c = compile(model, options.evidence ?? {})
  const mode = options.mode ?? 'sum'
  const schedule = options.schedule ?? 'flooding'
  const damping = options.damping ?? 0
  const tol = options.tol ?? 1e-10
  if (!(damping >= 0 && damping < 1)) throw new OmaiError('value', 'belief propagation: damping must be in [0, 1)')
  const tree = schedule === 'tree' ? treeSchedule(c) : []
  const nEdges = c.edges.length
  const granularity = options.granularity ?? 'sweep'
  const plan = granularity === 'sweep' || schedule === 'residual' ? [] : stepPlan(c, schedule, granularity, tree)
  return {
    name: 'beliefPropagation',
    config: { mode, schedule, damping, tol, granularity } as Json,
    schema: channels({
      metrics: {
        residual: { description: 'largest message change in the iteration', direction: 'lower' },
        betheFreeEnergy: { description: 'Bethe free energy of the current beliefs' },
      },
      messages: { factorToVariable: { description: 'a factor-to-variable message (log values)' } },
    }),
    init: () => {
      const f2v = c.edges.map((e) => uniform(c.card[e.variable] ?? 1))
      const v2f = c.edges.map((e) => uniform(c.card[e.variable] ?? 1))
      return { f2v, v2f, residual: Number.POSITIVE_INFINITY, updates: 0, cursor: 0, running: 0 }
    },
    step: (state) => {
      const f2v = [...state.f2v]
      const v2f = [...state.v2f]
      let residual = 0
      const setF2V = (e: number, next: Float64Array): void => {
        const old = f2v[e] as Float64Array
        const m = damp(next, old, damping, mode)
        residual = Math.max(residual, messageChange(m, old))
        f2v[e] = m
      }
      if (granularity !== 'sweep' && schedule === 'residual') {
        // One message: the pending one that would change most (the same choice the sweep's queue makes).
        for (let e = 0; e < nEdges; e++) v2f[e] = variableMessage(c, f2v, e, mode)
        let best = -1
        let change = -1
        let pick: Float64Array | undefined
        for (let e = 0; e < nEdges; e++) {
          const next = factorMessage(c, v2f, e, mode)
          const r = messageChange(next, f2v[e] as Float64Array)
          if (r > change) {
            best = e
            change = r
            pick = next
          }
        }
        if (best >= 0 && pick !== undefined) setF2V(best, pick)
        for (let e = 0; e < nEdges; e++) v2f[e] = variableMessage(c, f2v, e, mode)
        residual = 0
        for (let e = 0; e < nEdges; e++)
          residual = Math.max(residual, messageChange(factorMessage(c, v2f, e, mode), f2v[e] as Float64Array))
        return { f2v, v2f, residual, updates: state.updates + 1, cursor: 0, running: 0 }
      }
      if (granularity !== 'sweep') {
        const group = plan[state.cursor] ?? []
        for (const u of group) {
          if (u.kind === 'v2f') v2f[u.edge] = variableMessage(c, f2v, u.edge, mode)
          else setF2V(u.edge, factorMessage(c, v2f, u.edge, mode))
        }
        const running = Math.max(state.running, residual)
        const cursor = state.cursor + 1
        const updates = state.updates + group.length
        if (cursor < plan.length) return { f2v, v2f, residual: state.residual, updates, cursor, running }
        return { f2v, v2f, residual: schedule === 'tree' ? 0 : running, updates, cursor: 0, running: 0 }
      }
      if (schedule === 'flooding') {
        for (let e = 0; e < nEdges; e++) v2f[e] = variableMessage(c, state.f2v, e, mode)
        for (let e = 0; e < nEdges; e++) setF2V(e, factorMessage(c, v2f, e, mode))
      } else if (schedule === 'sequential') {
        for (const es of c.factorEdges) {
          for (const e of es) v2f[e] = variableMessage(c, f2v, e, mode)
          for (const e of es) setF2V(e, factorMessage(c, v2f, e, mode))
        }
      } else if (schedule === 'tree') {
        for (const u of tree) {
          if (u.kind === 'v2f') v2f[u.edge] = variableMessage(c, f2v, u.edge, mode)
          else setF2V(u.edge, factorMessage(c, v2f, u.edge, mode))
        }
        residual = 0
      } else {
        for (let e = 0; e < nEdges; e++) v2f[e] = variableMessage(c, f2v, e, mode)
        const candidate = c.edges.map((_, e) => factorMessage(c, v2f, e, mode))
        const version = new Int32Array(nEdges)
        const queue = heap<{ r: number; e: number; v: number }>(
          (a, b) => b.r - a.r || a.e - b.e,
          candidate.map((m, e) => ({ r: messageChange(m, f2v[e] as Float64Array), e, v: 0 })),
        )
        for (let k = 0; k < nEdges && queue.size > 0; k++) {
          const top = queue.pop()
          if (top === undefined) break
          if (top.v !== version[top.e]) {
            k--
            continue
          }
          setF2V(top.e, candidate[top.e] as Float64Array)
          const { variable } = c.edges[top.e] ?? { variable: 0 }
          for (const e2 of c.variableEdges[variable] ?? []) {
            if (e2 === top.e) continue
            v2f[e2] = variableMessage(c, f2v, e2, mode)
            for (const e3 of c.factorEdges[c.edges[e2]?.factor ?? 0] ?? []) {
              if (e3 === e2) continue
              candidate[e3] = factorMessage(c, v2f, e3, mode)
              version[e3] = (version[e3] ?? 0) + 1
              queue.push({
                r: messageChange(candidate[e3] as Float64Array, f2v[e3] as Float64Array),
                e: e3,
                v: version[e3] ?? 0,
              })
            }
          }
        }
        // The residual of the state is the largest pending change.
        residual = 0
        for (let e = 0; e < nEdges; e++)
          residual = Math.max(residual, messageChange(candidate[e] as Float64Array, f2v[e] as Float64Array))
      }
      return {
        f2v,
        v2f,
        residual,
        updates: state.updates + (schedule === 'tree' ? tree.length : nEdges),
        cursor: 0,
        running: 0,
      }
    },
    done: (state, t) =>
      t > 0 && state.cursor === 0 && (schedule === 'tree' || state.residual <= tol) ? 'converged' : undefined,
    observe: (state) => {
      const metrics: Record<string, number> = { residual: Number.isFinite(state.residual) ? state.residual : -1 }
      const current = bethe(c, state, mode)
      if (mode === 'sum') metrics['betheFreeEnergy'] = current.free
      // The current belief of every variable (probabilities, or max-marginals scaled to 1 at the largest), so a view
      // need not rebuild them from the messages.
      const beliefs: Record<string, number[]> = {}
      for (const [i, b] of current.variables.entries())
        beliefs[c.names[i] ?? String(i)] = Array.from(b, (lb) => Math.exp(lb))
      return {
        metrics,
        frame: { beliefs },
        messages: c.edges.map((e, i) => ({
          kind: 'factorToVariable',
          payload: { factor: e.factor, variable: c.names[e.variable] ?? '', values: Array.from(state.f2v[i] ?? []) },
          from: `f${e.factor}`,
          to: c.names[e.variable] ?? '',
        })),
      }
    },
  }
}

/** Beliefs and the Bethe free energy of a message state. */
function bethe(
  c: Compiled,
  state: BeliefPropagationState,
  mode: 'sum' | 'max' = 'sum',
): { variables: Float64Array[]; factors: Float64Array[]; free: number } {
  const variables = c.names.map((_, i) => {
    const b = new Float64Array(c.card[i] ?? 1)
    for (const e of c.variableEdges[i] ?? []) {
      const m = state.f2v[e]
      if (m !== undefined) for (let x = 0; x < b.length; x++) b[x] = (b[x] ?? 0) + (m[x] ?? 0)
    }
    normaliseLog(b, mode)
    return b
  })
  const factors = c.factors.map((f, a) =>
    factorBeliefLogs(
      f,
      (c.factorEdges[a] ?? []).map((e) => state.v2f[e] ?? new Float64Array(0)),
    ),
  )
  let free = 0
  for (const [a, f] of c.factors.entries()) {
    const b = factors[a] ?? new Float64Array(0)
    for (let e = 0; e < b.length; e++) {
      const lb = b[e] ?? Number.NEGATIVE_INFINITY
      if (lb === Number.NEGATIVE_INFINITY) continue
      free += Math.exp(lb) * (lb - (f.logValues[e] ?? 0))
    }
  }
  for (const [i, b] of variables.entries()) {
    const degree = c.variableEdges[i]?.length ?? 0
    let negEntropy = 0
    for (const lb of b) if (lb > Number.NEGATIVE_INFINITY) negEntropy += Math.exp(lb) * lb
    free -= (degree - 1) * negEntropy
  }
  return { variables, factors, free }
}

/**
 * Run belief propagation ({@link beliefPropagationAlgorithm}) to convergence or the iteration cap, and return the
 * beliefs and the Bethe approximation of log Z.
 *
 * @param model - The model.
 * @param options - Evidence, mode, schedule, damping, stopping rule and tracer.
 * @returns Variable and factor beliefs, the Bethe free energy, log Z and convergence information.
 * @concept belief-propagation
 * @index function Loopy and tree belief propagation.
 * @example
 * ```ts
 * import { bayesNet, beliefPropagation, factorValues } from 'omai-engine/pgm'
 * const bn = bayesNet([
 *   { name: 'rain', states: ['no', 'yes'], parents: [], cpt: [0.8, 0.2] },
 *   { name: 'wet', states: ['no', 'yes'], parents: ['rain'], cpt: [0.9, 0.1, 0.2, 0.8] },
 * ])
 * const r = beliefPropagation(bn, { evidence: { wet: 'yes' }, schedule: 'tree' })
 * Object.values(r.marginals).map((m) => factorValues(m).map((p) => Math.round(p * 1000) / 1000)) // => [[0.333, 0.667], [0, 1]]
 * Math.round(Math.exp(r.logPartition) * 1e6) / 1e6 // => 0.24
 * ```
 */
export function beliefPropagation(
  model: DiscreteModel,
  options: BeliefPropagationOptions = {},
): BeliefPropagationResult {
  const c = compile(model, options.evidence ?? {})
  const mode = options.mode ?? 'sum'
  const result = runAlgorithm(beliefPropagationAlgorithm(model, options), {
    maxSteps: options.maxIter ?? 200,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const b = bethe(c, result.state, mode)
  const marginals: Record<string, Factor> = observedMarginals(c)
  for (const [i, name] of c.names.entries()) marginals[name] = marginalFactor(name, b.variables[i] as Float64Array)
  const ordered = Object.fromEntries(c.all.map((v) => [v.name, marginals[v.name] as Factor]))
  const factorBeliefs = c.factors.map((f, a) =>
    logFactor(
      f.vars.map((i) => c.names[i] ?? ''),
      f.vars.map((i) => c.card[i] ?? 1),
      b.factors[a] as Float64Array,
    ),
  )
  return {
    marginals: ordered,
    factorBeliefs,
    betheFreeEnergy: mode === 'sum' ? b.free : Number.NaN,
    logPartition: mode === 'sum' ? c.constant - b.free : Number.NaN,
    converged: result.status === 'converged',
    status: result.status,
    iterations: result.steps,
    residual: result.state.residual,
  }
}

/**
 * The Bethe free energy of given variable and factor beliefs on a model's factor graph:
 * F = sum_a sum b_a log(b_a / f_a) - sum_i (d_i - 1) sum b_i log b_i, whose minimum over the local polytope
 * approximates -log Z (exactly on trees).
 *
 * @param model - The model (no evidence: reduce it first, or pass the beliefs of a run on it).
 * @param variableBeliefs - Normalised beliefs by variable.
 * @param factorBeliefs - Normalised beliefs, one per model factor, over each factor's scope (in its variable order).
 * @returns The free energy.
 * @concept belief-propagation
 * @example
 * ```ts
 * import { factor, factorGraph, betheFreeEnergy } from 'omai-engine/pgm'
 * const fg = factorGraph([{ name: 'a', states: ['0', '1'] }], [factor(['a'], [2], [1, 3])])
 * const b = factor(['a'], [2], [0.25, 0.75])
 * Math.round(-betheFreeEnergy(fg, { a: b }, [b]) * 1e12) / 1e12 === Math.round(Math.log(4) * 1e12) / 1e12 // => true
 * ```
 */
export function betheFreeEnergy(
  model: DiscreteModel,
  variableBeliefs: Readonly<Record<string, Factor>>,
  factorBeliefs: readonly Factor[],
): number {
  const c = compile(model)
  let free = 0
  for (const [a, f] of c.factors.entries()) {
    const b = factorBeliefs[a]
    if (b === undefined || b.logValues.length !== f.logValues.length) {
      throw new OmaiError('shape', `betheFreeEnergy: factor belief ${a} does not match its factor`)
    }
    for (let e = 0; e < f.logValues.length; e++) {
      const lb = b.logValues[e] ?? Number.NEGATIVE_INFINITY
      if (lb > Number.NEGATIVE_INFINITY) free += Math.exp(lb) * (lb - (f.logValues[e] ?? 0))
    }
  }
  for (const [i, name] of c.names.entries()) {
    const b = variableBeliefs[name]
    if (b === undefined) throw new OmaiError('value', `betheFreeEnergy: no belief for ${name}`)
    let negEntropy = 0
    for (const lb of b.logValues) if (lb > Number.NEGATIVE_INFINITY) negEntropy += Math.exp(lb) * lb
    free -= ((c.variableEdges[i]?.length ?? 0) - 1) * negEntropy
  }
  return free - c.constant
}
