/**
 * FOIL (Quinlan 1990), a top-down inductive logic programming learner: it learns clauses for a target relation from
 * positive and negative tuples and background relations by repeatedly adding the body literal with the highest
 * information gain until no negatives are covered. The gain function, a traced covering loop, and the learnt theory as
 * Prolog clauses.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../../trace/index.ts'
import { atom, type Clause, compound, type Term, variable } from '../terms.ts'

/** A tuple of constants. */
export type Tuple = readonly string[]

/**
 * A FOIL learning task: the target relation's positive (and optionally negative) tuples and extensional background
 * relations. Without negatives the closed-world assumption supplies every other tuple over the constants.
 *
 * @concept inductive-logic-programming
 */
export interface FoilTask {
  readonly target: string
  readonly arity: number
  readonly positive: readonly Tuple[]
  readonly negative?: readonly Tuple[]
  readonly background: Readonly<Record<string, readonly Tuple[]>>
}

/** Options of {@link foil}. */
export interface FoilOptions extends Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> {
  /** Body literals per clause (default 4). */
  readonly maxBodyLiterals?: number
  /** Clauses in the theory (default 8). */
  readonly maxClauses?: number
  /** New variables one literal may introduce (default 1). */
  readonly maxNewVariables?: number
  /**
   * Allow the target relation in bodies, over bound variables only and evaluated against the positive tuples
   * (default false).
   */
  readonly recursion?: boolean
  /** Allow negated literals over bound variables (default false). */
  readonly negation?: boolean
}

/** A body literal of a FOIL clause over variable indices (0 .. arity-1 are the head's). */
export interface FoilLiteral {
  readonly predicate: string
  readonly args: readonly number[]
  readonly negated: boolean
}

/** A candidate literal with its FOIL gain and the tuple counts after adding it. */
export interface FoilCandidate {
  readonly literal: FoilLiteral
  readonly gain: number
  readonly positive: number
  readonly negative: number
}

/** The observable state of FOIL. */
export interface FoilState {
  readonly status: 'running' | 'done' | 'stuck'
  readonly clauses: readonly (readonly FoilLiteral[])[]
  /** The clause being grown. */
  readonly body: readonly FoilLiteral[]
  readonly variables: number
  readonly positiveTuples: number
  readonly negativeTuples: number
  /** Positive examples not yet covered. */
  readonly uncovered: number
  /** What the last step did, with the best candidates it weighed. */
  readonly event: { readonly kind: 'literal' | 'clause' | 'stuck'; readonly best: readonly FoilCandidate[] } | null
}

interface FoilMachine extends FoilState {
  status: 'running' | 'done' | 'stuck'
  clauses: FoilLiteral[][]
  body: FoilLiteral[]
  variables: number
  positiveTuples: number
  negativeTuples: number
  uncovered: number
  event: { kind: 'literal' | 'clause' | 'stuck'; best: FoilCandidate[] } | null
  remaining: Tuple[]
  pos: Tuple[]
  neg: Tuple[]
}

const FOIL_SCHEMA: TraceSchema = channels({
  metrics: {
    positiveTuples: { description: 'positive tuples of the clause being grown' },
    negativeTuples: { description: 'negative tuples still covered' },
    uncovered: { description: 'positive examples not yet covered by the theory' },
    clauses: { description: 'clauses learned' },
  },
  messages: {
    literal: { description: 'the literal added and the best candidates by gain', payload: { best: 'candidates' } },
    clause: { description: 'a clause completed', payload: { best: '[]' } },
    stuck: { description: 'no literal has positive gain', payload: { best: 'candidates' } },
  },
})

const tupleKey = (t: Tuple): string => t.join('\u0000')

function letters(i: number): string {
  return i < 26 ? String.fromCharCode(65 + i) : `V${i}`
}

/**
 * A FOIL clause as a Prolog clause, with variables named A, B, C, ... by index.
 *
 * @param target - The head predicate.
 * @param arity - The head's arity.
 * @param body - The body literals.
 * @returns The clause.
 * @example
 * ```ts
 * import { foilClause } from 'mdml-engine/logic/ilp'
 * import { clauseToString } from 'mdml-engine/logic'
 * clauseToString(foilClause('p', 1, [{ predicate: 'q', args: [0, 1], negated: false }, { predicate: 'r', args: [1], negated: true }])) // => 'p(A) :- q(A,B), \\+r(B).'
 * ```
 */
export function foilClause(target: string, arity: number, body: readonly FoilLiteral[]): Clause {
  const v = (i: number): Term => variable(letters(i))
  const head = compound(
    target,
    Array.from({ length: arity }, (_, i) => v(i)),
  )
  return {
    head,
    body: body.map((l) => {
      const a = l.args.length === 0 ? atom(l.predicate) : compound(l.predicate, l.args.map(v))
      return l.negated ? compound('\\+', [a]) : a
    }),
  }
}

function constantsOf(task: FoilTask): string[] {
  const out = new Set<string>()
  for (const t of task.positive) for (const c of t) out.add(c)
  for (const t of task.negative ?? []) for (const c of t) out.add(c)
  for (const rel of Object.values(task.background)) for (const t of rel) for (const c of t) out.add(c)
  return [...out].sort()
}

function closedWorld(task: FoilTask): Tuple[] {
  const consts = constantsOf(task)
  const pos = new Set(task.positive.map(tupleKey))
  const out: Tuple[] = []
  const total = consts.length ** task.arity
  for (let m = 0; m < total; m++) {
    let k = m
    const t: string[] = []
    for (let i = 0; i < task.arity; i++) {
      t.unshift(consts[k % consts.length] as string)
      k = Math.floor(k / consts.length)
    }
    if (!pos.has(tupleKey(t))) out.push(t)
  }
  return out
}

/** Index a relation by the values at bound argument positions. */
class Relation {
  private readonly index = new Map<string, Tuple[]>()

  constructor(readonly tuples: readonly Tuple[]) {}

  matching(bound: readonly (string | undefined)[]): readonly Tuple[] {
    const positions = bound.flatMap((b, i) => (b === undefined ? [] : [i]))
    const k = `${positions.join(',')}|${positions.map((i) => bound[i]).join('\u0000')}`
    let hit = this.index.get(k)
    if (hit === undefined) {
      hit = this.tuples.filter((t) => positions.every((i) => t[i] === bound[i]))
      this.index.set(k, hit)
    }
    return hit
  }
}

/** Extend a tuple by a literal: every binding of its new variables (none, or the tuple itself for a negation). */
function extend(rel: Relation, tuple: Tuple, lit: FoilLiteral): Tuple[] {
  const bound = lit.args.map((a) => tuple[a])
  const matches = rel.matching(bound)
  if (lit.negated) return matches.length === 0 ? [tuple] : []
  const out: Tuple[] = []
  for (const m of matches) {
    const next = [...tuple]
    let ok = true
    lit.args.forEach((a, i) => {
      if (a < tuple.length) return
      // A new variable: the first occurrence binds, a repeat must agree.
      if (next[a] === undefined) next[a] = m[i] as string
      else if (next[a] !== m[i]) ok = false
    })
    if (ok) out.push(next)
  }
  return out
}

/** Candidate literals: each predicate with arguments from the bound variables and up to `maxNew` new ones. */
function candidates(
  preds: readonly [string, number][],
  vars: number,
  maxNew: number,
  negation: boolean,
  target: string,
  body: readonly FoilLiteral[],
): FoilLiteral[] {
  const out: FoilLiteral[] = []
  const seen = new Set(body.map((l) => JSON.stringify(l)))
  for (const [p, m] of preds) {
    const rec = (prefix: number[], next: number): void => {
      if (prefix.length === m) {
        if (!prefix.some((a) => a < vars)) return
        // A recursive literal uses bound variables only (so the definition can terminate) and differs from the head.
        if (p === target && (prefix.some((a) => a >= vars) || prefix.every((a, i) => a === i))) return
        const lit = { predicate: p, args: [...prefix], negated: false }
        if (!seen.has(JSON.stringify(lit))) out.push(lit)
        if (negation && prefix.every((a) => a < vars)) {
          const neg = { predicate: p, args: [...prefix], negated: true }
          if (!seen.has(JSON.stringify(neg))) out.push(neg)
        }
        return
      }
      for (let a = 0; a < vars; a++) rec([...prefix, a], next)
      if (next - vars < maxNew) rec([...prefix, next], next + 1)
    }
    rec([], vars)
  }
  return out
}

/**
 * The FOIL information gain of adding a literal (Quinlan 1990): `t * (log2(p1 / (p1 + n1)) - log2(p0 / (p0 + n0)))`,
 * where `p0, n0` count positive and negative tuples before, `p1, n1` after, and `t` is the number of positive tuples
 * that have at least one extension.
 *
 * @param p0 - Positive tuples before.
 * @param n0 - Negative tuples before.
 * @param p1 - Positive tuples after.
 * @param n1 - Negative tuples after.
 * @param t - Positive tuples before that are still covered.
 * @returns The gain (0 when no positive tuple remains).
 * @concept inductive-logic-programming
 * @example
 * ```ts
 * import { foilGain } from 'mdml-engine/logic/ilp'
 * foilGain(2, 2, 2, 0, 2) // => 2
 * ```
 */
export function foilGain(p0: number, n0: number, p1: number, n1: number, t: number): number {
  if (p1 === 0 || p0 === 0) return 0
  return t * (Math.log2(p1 / (p1 + n1)) - Math.log2(p0 / (p0 + n0)))
}

function foilMachine(task: FoilTask, options: FoilOptions): Algorithm<FoilMachine> {
  const maxBody = options.maxBodyLiterals ?? 4
  const maxClauses = options.maxClauses ?? 8
  const maxNew = options.maxNewVariables ?? 1
  const recursion = options.recursion ?? false
  const negation = options.negation ?? false
  for (const t of [...task.positive, ...(task.negative ?? [])]) {
    if (t.length !== task.arity) throw new MdmlError('value', `foil: tuple (${t.join(', ')}) has the wrong arity`)
  }
  const relations = new Map<string, Relation>()
  const preds: [string, number][] = []
  for (const [p, tuples] of Object.entries(task.background).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
    relations.set(p, new Relation(tuples))
    preds.push([p, tuples[0]?.length ?? 0])
  }
  if (recursion) {
    relations.set(task.target, new Relation(task.positive))
    preds.push([task.target, task.arity])
  }
  const negatives = task.negative ?? closedWorld(task)
  const startClause = (s: FoilMachine): void => {
    s.body = []
    s.variables = task.arity
    s.pos = [...s.remaining]
    s.neg = [...negatives]
    s.positiveTuples = s.pos.length
    s.negativeTuples = s.neg.length
  }
  return {
    name: 'foil',
    config: { maxBodyLiterals: maxBody, maxNewVariables: maxNew, recursion, negation },
    schema: FOIL_SCHEMA,
    init: () => {
      const s: FoilMachine = {
        status: task.positive.length === 0 ? 'done' : 'running',
        clauses: [],
        body: [],
        variables: task.arity,
        positiveTuples: 0,
        negativeTuples: 0,
        uncovered: task.positive.length,
        event: null,
        remaining: [...task.positive],
        pos: [],
        neg: [],
      }
      startClause(s)
      return s
    },
    step: (s) => {
      if (s.neg.length === 0 || s.body.length >= maxBody) {
        if (s.neg.length > 0) {
          s.status = 'stuck'
          s.event = { kind: 'stuck', best: [] }
          return s
        }
        // The clause is complete: its positive tuples' head values are the examples it covers.
        const covered = new Set(s.pos.map((t) => tupleKey(t.slice(0, task.arity))))
        s.clauses.push(s.body)
        s.remaining = s.remaining.filter((t) => !covered.has(tupleKey(t)))
        s.uncovered = s.remaining.length
        s.event = { kind: 'clause', best: [] }
        if (s.remaining.length === 0) s.status = 'done'
        else if (s.clauses.length >= maxClauses) s.status = 'stuck'
        else startClause(s)
        return s
      }
      const p0 = s.pos.length
      const n0 = s.neg.length
      const scored: (FoilCandidate & { pos: Tuple[]; neg: Tuple[]; fresh: number; order: number })[] = []
      candidates(preds, s.variables, maxNew, negation, task.target, s.body).forEach((lit, order) => {
        const rel = relations.get(lit.predicate) as Relation
        let t = 0
        const pos: Tuple[] = []
        for (const tuple of s.pos) {
          const ext = extend(rel, tuple, lit)
          if (ext.length > 0) t++
          pos.push(...ext)
        }
        const neg = s.neg.flatMap((tuple) => extend(rel, tuple, lit))
        const fresh = lit.args.filter((a) => a >= s.variables).length
        const gain = foilGain(p0, n0, pos.length, neg.length, t)
        scored.push({ literal: lit, gain, positive: pos.length, negative: neg.length, pos, neg, fresh, order })
      })
      // Highest gain; ties to fewer new variables, then generation order.
      scored.sort((a, b) => b.gain - a.gain || a.fresh - b.fresh || a.order - b.order)
      const best = scored[0]
      const top = scored
        .slice(0, 5)
        .map(({ literal, gain, positive, negative }) => ({ literal, gain, positive, negative }))
      if (best === undefined || !(best.gain > 1e-12)) {
        s.status = 'stuck'
        s.event = { kind: 'stuck', best: top }
        return s
      }
      s.body = [...s.body, best.literal]
      s.variables += best.fresh
      s.pos = best.pos
      s.neg = best.neg
      s.positiveTuples = s.pos.length
      s.negativeTuples = s.neg.length
      s.event = { kind: 'literal', best: top }
      return s
    },
    done: (s) => (s.status === 'running' ? undefined : s.status === 'done' ? 'converged' : 'stopped'),
    observe: (s) => ({
      metrics: {
        positiveTuples: s.positiveTuples,
        negativeTuples: s.negativeTuples,
        uncovered: s.uncovered,
        clauses: s.clauses.length,
      },
      messages: s.event === null ? [] : [{ kind: s.event.kind, payload: { best: s.event.best } as unknown as Json }],
    }),
  }
}

/**
 * FOIL as a traced {@link Algorithm}: a covering loop over the positive examples, each clause grown general to
 * specific by adding the literal of highest information gain until it covers no negative tuple. Each step adds one
 * literal or completes a clause, and the trace records the best candidates weighed.
 *
 * @param task - The target, examples and background relations.
 * @param options - Clause length, theory size, new variables, recursion and negation.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a tuple of the wrong arity.
 * @concept inductive-logic-programming
 * @example
 * ```ts
 * import { foilAlgorithm } from 'mdml-engine/logic/ilp'
 * foilAlgorithm({ target: 'p', arity: 1, positive: [['a']], background: { q: [['a']] } }).name // => 'foil'
 * ```
 */
export function foilAlgorithm(task: FoilTask, options: FoilOptions = {}): Algorithm<FoilState> {
  return foilMachine(task, options)
}

/** The outcome of {@link foil}. */
export interface FoilResult {
  /** `complete` when every positive is covered (and no negative), else `incomplete`. */
  readonly status: 'complete' | 'incomplete'
  readonly clauses: readonly Clause[]
  readonly literals: readonly (readonly FoilLiteral[])[]
  readonly uncovered: readonly Tuple[]
}

/**
 * Learn a theory with FOIL.
 *
 * @param task - The target, examples and background relations.
 * @param options - Clause length, theory size, new variables, recursion, negation and tracing.
 * @returns The status, the clauses (variables A, B, C, ...) and any positives left uncovered.
 * @throws MdmlError with code `value` for a tuple of the wrong arity.
 * @concept inductive-logic-programming
 * @reference J. R. Quinlan, Learning logical definitions from relations, Machine Learning 5 (1990).
 * @example
 * ```ts
 * import { foil } from 'mdml-engine/logic/ilp'
 * import { clauseToString } from 'mdml-engine/logic'
 * const parent = [['tom', 'bob'], ['tom', 'liz'], ['bob', 'ann'], ['bob', 'pat'], ['pat', 'jim']]
 * const r = foil({ target: 'grandparent', arity: 2, positive: [['tom', 'ann'], ['tom', 'pat'], ['bob', 'jim']], background: { parent } })
 * r.clauses.map((c) => clauseToString(c)) // => ['grandparent(A,B) :- parent(A,C), parent(C,B).']
 * ```
 */
export function foil(task: FoilTask, options: FoilOptions = {}): FoilResult {
  const { maxBodyLiterals: _b, maxClauses: _c, maxNewVariables: _n, recursion: _r, negation: _g, ...run } = options
  const { state } = runAlgorithm(foilMachine(task, options), { ...run, maxSteps: 10_000 })
  return {
    status: state.status === 'done' ? 'complete' : 'incomplete',
    clauses: state.clauses.map((b) => foilClause(task.target, task.arity, b)),
    literals: state.clauses,
    uncovered: state.remaining,
  }
}
