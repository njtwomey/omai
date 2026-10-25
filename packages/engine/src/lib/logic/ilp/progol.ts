/**
 * Progol-style inverse entailment for inductive logic programming: mode declarations describe which literals may
 * appear, each seed example is saturated into its most specific bottom clause relative to the background knowledge, and
 * a covering loop searches the clauses that generalise it for the best-scoring one, as in Progol and Aleph.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../../trace/index.ts'
import { parseTerm } from '../parse.ts'
import { consult, type Database, solve } from '../resolution.ts'
import {
  atom,
  type Clause,
  clauseToString,
  compareTerms,
  compound,
  type OperatorDef,
  standardOperators,
  type Term,
  termToString,
  termVariables,
  variable,
} from '../terms.ts'
import { renameTerm } from '../unify.ts'

/** One argument of a mode declaration: an input (`+type`), an output (`-type`) or a constant (`#type`). */
export interface ModeArgument {
  readonly io: '+' | '-' | '#'
  readonly type: string
}

/**
 * A mode declaration (Progol's `modeh` and `modeb`): a predicate, the mode and type of each argument, and the recall
 * (how many answers one call may contribute; `*` for all).
 *
 * @concept inductive-logic-programming
 */
export interface ModeDeclaration {
  readonly kind: 'modeh' | 'modeb'
  readonly recall: number
  readonly predicate: string
  readonly args: readonly ModeArgument[]
}

/**
 * Read a mode declaration, `modeh(1, eastbound(+train))` or `modeb(*, has_car(+train, -car))`.
 *
 * @param text - The declaration.
 * @returns The mode.
 * @throws MdmlError with code `value` for a malformed declaration.
 * @concept inductive-logic-programming
 * @example
 * ```ts
 * import { parseMode } from 'mdml-engine/logic/ilp'
 * parseMode('modeb(*, has_car(+train, -car))') // => { kind: 'modeb', recall: Infinity, predicate: 'has_car', args: [{ io: '+', type: 'train' }, { io: '-', type: 'car' }] }
 * ```
 */
export function parseMode(text: string): ModeDeclaration {
  const ops = standardOperators()
  ;(ops.prefix as Map<string, OperatorDef>).set('#', { priority: 200, type: 'fy' })
  const t = parseTerm(text.replace(/\*/g, "'*'"), { operators: ops })
  const fail = (): never => {
    throw new MdmlError('value', `parseMode: ${text} is not modeh(Recall, p(+t, -t, #t))`)
  }
  if (t.kind !== 'compound' || (t.functor !== 'modeh' && t.functor !== 'modeb') || t.args.length !== 2) return fail()
  const [r, a] = t.args as [Term, Term]
  const recall = r.kind === 'num' ? r.value : r.kind === 'atom' && r.name === '*' ? Number.POSITIVE_INFINITY : fail()
  if (a.kind === 'atom') return { kind: t.functor, recall, predicate: a.name, args: [] }
  if (a.kind !== 'compound') return fail()
  const args = a.args.map((x): ModeArgument => {
    if (x.kind === 'compound' && x.args.length === 1 && ['+', '-', '#'].includes(x.functor)) {
      const ty = x.args[0] as Term
      if (ty.kind === 'atom') return { io: x.functor as '+' | '-' | '#', type: ty.name }
    }
    return fail()
  })
  return { kind: t.functor, recall, predicate: a.functor, args }
}

/** Options of {@link bottomClause}. */
export interface BottomClauseOptions {
  /** Variable depth: rounds of adding body literals whose inputs are already known (default 2). */
  readonly depth?: number
  /** Step budget of each background query (default 10000). */
  readonly queryBudget?: number
}

/** A bottom clause with, for each body literal, its input and output variables (for mode-conformant refinement). */
export interface BottomClause {
  readonly clause: Clause
  readonly inputs: readonly (readonly string[])[]
  readonly outputs: readonly (readonly string[])[]
  /** The head's input variables. */
  readonly headInputs: readonly string[]
}

/**
 * The most specific clause (bottom clause) of an example relative to background knowledge and mode declarations
 * (Muggleton's saturation): the head is the example with typed constants replaced by variables, and for each depth
 * every body mode is called with all combinations of known terms of the right types as inputs; each answer adds a
 * literal, and its output terms become known for the next depth.
 *
 * @param example - A ground positive example.
 * @param db - The background knowledge.
 * @param modeh - The head mode.
 * @param modebs - The body modes, in order.
 * @param options - Variable depth and the query budget.
 * @returns The bottom clause with its literals' input and output variables.
 * @throws MdmlError with code `value` when the example does not fit the head mode.
 * @concept inductive-logic-programming
 * @example
 * ```ts
 * import { bottomClause, parseMode } from 'mdml-engine/logic/ilp'
 * import { clauseToString, consult, parseTerm } from 'mdml-engine/logic'
 * const db = consult('parent(ann, mary). parent(ann, tom). female(mary). female(ann).')
 * const b = bottomClause(parseTerm('daughter(mary, ann)'), db, parseMode('modeh(1, daughter(+person, +person))'), [parseMode('modeb(*, parent(+person, -person))'), parseMode('modeb(1, female(+person))')])
 * clauseToString(b.clause) // => 'daughter(A,B) :- parent(B,A), parent(B,C), female(A), female(B).'
 * ```
 */
export function bottomClause(
  example: Term,
  db: Database,
  modeh: ModeDeclaration,
  modebs: readonly ModeDeclaration[],
  options: BottomClauseOptions = {},
): BottomClause {
  const depth = options.depth ?? 2
  const budget = options.queryBudget ?? 10_000
  if (example.kind !== 'compound' || example.functor !== modeh.predicate || example.args.length !== modeh.args.length) {
    if (!(example.kind === 'atom' && example.name === modeh.predicate && modeh.args.length === 0)) {
      throw new MdmlError('value', `bottomClause: ${termToString(example)} does not fit the head mode`)
    }
  }
  const names = new Map<string, string>()
  const vars = (t: Term): Term => {
    const k = termToString(t)
    let v = names.get(k)
    if (v === undefined) {
      v = letter(names.size)
      names.set(k, v)
    }
    return variable(v)
  }
  // Known terms by type, with the depth at which they appeared.
  const known: { term: Term; type: string }[] = []
  const know = (t: Term, type: string): void => {
    if (!known.some((k) => k.type === type && compareTerms(k.term, t) === 0)) known.push({ term: t, type })
  }
  const exArgs = example.kind === 'compound' ? example.args : []
  const headInputs: string[] = []
  const head =
    modeh.args.length === 0
      ? example
      : compound(
          modeh.predicate,
          modeh.args.map((m, i) => {
            const t = exArgs[i] as Term
            if (m.io === '#') return t
            const v = vars(t)
            if (m.io === '+') {
              know(t, m.type)
              if (v.kind === 'var') headInputs.push(v.name)
            }
            return v
          }),
        )
  const body: Term[] = []
  const inputs: string[][] = []
  const outputs: string[][] = []
  const seen = new Set<string>()
  for (let d = 0; d < depth; d++) {
    const snapshot = [...known]
    for (const m of modebs) {
      const inPositions = m.args.flatMap((a, i) => (a.io === '+' ? [i] : []))
      const pools = inPositions.map((i) => snapshot.filter((k) => k.type === m.args[i]?.type).map((k) => k.term))
      const combos = cartesian(pools)
      for (const combo of combos) {
        const queryArgs = m.args.map((_, i) => {
          const j = inPositions.indexOf(i)
          return j >= 0 ? (combo[j] as Term) : variable(`Q${i}`)
        })
        const goal = queryArgs.length === 0 ? atom(m.predicate) : compound(m.predicate, queryArgs)
        const r = solve(db, goal, {
          maxSolutions: m.recall,
          maxSteps: budget,
          unknown: 'fail',
        })
        for (const ans of r.answers) {
          const ground = m.args.map((_, i) => ans.bindings[`Q${i}`] ?? (queryArgs[i] as Term))
          const lit =
            ground.length === 0
              ? atom(m.predicate)
              : compound(
                  m.predicate,
                  ground.map((g, i) => ((m.args[i] as ModeArgument).io === '#' ? g : vars(g))),
                )
          const k = termToString(lit)
          if (seen.has(k)) continue
          seen.add(k)
          body.push(lit)
          const ins: string[] = []
          const outs: string[] = []
          ground.forEach((g, i) => {
            const a = m.args[i] as ModeArgument
            if (a.io === '#') return
            const v = vars(g)
            if (v.kind !== 'var') return
            if (a.io === '+') ins.push(v.name)
            else {
              outs.push(v.name)
              know(g, a.type)
            }
          })
          inputs.push(ins)
          outputs.push(outs)
        }
      }
    }
  }
  return { clause: { head, body }, inputs, outputs, headInputs }
}

function letter(i: number): string {
  return i < 26 ? String.fromCharCode(65 + i) : `V${i}`
}

function cartesian<T>(pools: readonly (readonly T[])[]): T[][] {
  let out: T[][] = [[]]
  for (const p of pools) out = out.flatMap((prefix) => p.map((x) => [...prefix, x]))
  return out
}

/** A Progol-style learning task. */
export interface ProgolTask {
  /** Background knowledge (Prolog text or a database). */
  readonly background: string | Database
  readonly positive: readonly (Term | string)[]
  readonly negative: readonly (Term | string)[]
  /** The head mode and the body modes (text or parsed). */
  readonly modeh: ModeDeclaration | string
  readonly modeb: readonly (ModeDeclaration | string)[]
}

/** Options of {@link progol}. */
export interface ProgolOptions extends Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> {
  /** Variable depth of bottom clauses (default 2). */
  readonly depth?: number
  /** Body literals of a hypothesis clause (default 4). */
  readonly maxBodyLiterals?: number
  /** Negative examples a clause may cover (default 0). */
  readonly noise?: number
  /** Candidate clauses evaluated per search (default 2000). */
  readonly maxNodes?: number
  /** Step budget for proving one example with a candidate (default 2000). */
  readonly queryBudget?: number
}

/** A scored candidate clause. */
export interface ScoredClause {
  readonly clause: Clause
  readonly positive: number
  readonly negative: number
  /** Compression: positives - negatives - body length. */
  readonly score: number
}

/** The observable state of the covering loop. */
export interface ProgolState {
  readonly status: 'running' | 'done'
  readonly theory: readonly Clause[]
  readonly uncovered: number
  readonly bottom: Clause | null
  readonly evaluated: number
  readonly last: ScoredClause | null
  readonly best: ScoredClause | null
}

interface Node {
  readonly literals: readonly number[]
}

interface ProgolMachine extends ProgolState {
  status: 'running' | 'done'
  theory: Clause[]
  uncovered: number
  bottom: Clause | null
  evaluated: number
  last: ScoredClause | null
  best: ScoredClause | null
  remaining: Term[]
  bc: BottomClause | null
  queue: Node[]
  searched: number
  event: 'bottom' | 'evaluate' | 'accept' | null
}

const PROGOL_SCHEMA: TraceSchema = channels({
  metrics: {
    uncovered: { description: 'positive examples not yet covered' },
    evaluated: { description: 'candidate clauses evaluated' },
    theory: { description: 'clauses in the theory' },
  },
  messages: {
    bottom: { description: 'the bottom clause of the next seed example', payload: { clause: 'text' } },
    evaluate: {
      description: 'a candidate clause and its coverage',
      payload: { clause: 'text', positive: 'P', negative: 'N', score: 'P - N - L' },
    },
    accept: {
      description: 'the best clause added to the theory',
      payload: { clause: 'text', positive: 'P', negative: 'N', score: 'score' },
    },
  },
})

function asTerm(x: Term | string): Term {
  return typeof x === 'string' ? parseTerm(x) : x
}

function progolMachine(task: ProgolTask, options: ProgolOptions): Algorithm<ProgolMachine> {
  const db = typeof task.background === 'string' ? consult(task.background) : task.background
  const modeh = typeof task.modeh === 'string' ? parseMode(task.modeh) : task.modeh
  const modebs = task.modeb.map((m) => (typeof m === 'string' ? parseMode(m) : m))
  const pos = task.positive.map(asTerm)
  const neg = task.negative.map(asTerm)
  const maxBody = options.maxBodyLiterals ?? 4
  const noise = options.noise ?? 0
  const maxNodes = options.maxNodes ?? 2000
  const budget = options.queryBudget ?? 2000
  const covers = (clauses: readonly Clause[], e: Term): boolean => {
    const r = solve(consult(clauses, { base: db }), e, { maxSolutions: 1, maxSteps: budget, unknown: 'fail' })
    return r.answers.length > 0
  }
  const clauseOf = (bc: BottomClause, lits: readonly number[]): Clause => ({
    head: bc.clause.head,
    body: lits.map((i) => bc.clause.body[i] as Term),
  })
  const score = (s: ProgolMachine, c: Clause): ScoredClause => {
    const p = s.remaining.filter((e) => covers([c], e)).length
    const n = neg.filter((e) => covers([c], e)).length
    return { clause: c, positive: p, negative: n, score: p - n - c.body.length }
  }
  // Mode-conformant refinements: add a later bottom-clause literal whose inputs are all bound.
  const refine = (bc: BottomClause, node: Node): Node[] => {
    if (node.literals.length >= maxBody) return []
    const bound = new Set(bc.headInputs)
    for (const i of node.literals) for (const v of bc.outputs[i] ?? []) bound.add(v)
    const last = node.literals[node.literals.length - 1] ?? -1
    const out: Node[] = []
    for (let i = last + 1; i < bc.clause.body.length; i++) {
      if ((bc.inputs[i] ?? []).every((v) => bound.has(v))) out.push({ literals: [...node.literals, i] })
    }
    return out
  }
  const better = (a: ScoredClause, b: ScoredClause | null): boolean =>
    b === null || a.score > b.score || (a.score === b.score && a.clause.body.length < b.clause.body.length)
  const nextSeed = (s: ProgolMachine): void => {
    const seed = s.remaining[0]
    if (seed === undefined) {
      s.status = 'done'
      s.bc = null
      return
    }
    s.bc = bottomClause(seed, db, modeh, modebs, { depth: options.depth ?? 2 })
    s.bottom = s.bc.clause
    s.queue = [{ literals: [] }]
    s.best = null
    s.searched = 0
    s.event = 'bottom'
  }
  return {
    name: 'progol',
    config: { depth: options.depth ?? 2, maxBodyLiterals: maxBody, noise, maxNodes },
    schema: PROGOL_SCHEMA,
    init: () => {
      const s: ProgolMachine = {
        status: 'running',
        theory: [],
        uncovered: pos.length,
        bottom: null,
        evaluated: 0,
        last: null,
        best: null,
        remaining: [...pos],
        bc: null,
        queue: [],
        searched: 0,
        event: null,
      }
      nextSeed(s)
      return s
    },
    step: (s) => {
      s.event = null
      const bc = s.bc
      if (bc === null) {
        s.status = 'done'
        return s
      }
      const node = s.queue.shift()
      if (node === undefined || s.searched >= maxNodes) {
        // Search over: accept the best consistent clause, or the seed itself when none generalises it.
        const seed = s.remaining[0] as Term
        const chosen: ScoredClause =
          s.best !== null && s.best.positive > 0
            ? s.best
            : { clause: { head: seed, body: [] }, positive: 1, negative: 0, score: 0 }
        s.theory.push(chosen.clause)
        s.remaining = s.remaining.filter((e) => !covers(s.theory, e))
        if (s.remaining.length > 0 && compareTerms(s.remaining[0] as Term, seed) === 0) s.remaining.shift()
        s.uncovered = s.remaining.length
        s.last = chosen
        s.event = 'accept'
        nextSeed(s)
        return s
      }
      s.searched++
      s.evaluated++
      const c = clauseOf(bc, node.literals)
      const sc = score(s, c)
      s.last = sc
      s.event = 'evaluate'
      if (sc.negative <= noise && sc.positive > 0 && better(sc, s.best)) s.best = sc
      // Refine only while negatives remain and the clause could still beat the best (refinements cover no more).
      const bound = sc.positive - c.body.length - 1
      if (sc.negative > noise && sc.positive > 0 && (s.best === null || bound > s.best.score)) {
        s.queue.push(...refine(bc, node))
      }
      return s
    },
    done: (s) => (s.status === 'done' ? 'converged' : undefined),
    observe: (s) => {
      const messages: { kind: string; payload: Json }[] = []
      if (s.event === 'bottom' && s.bottom !== null)
        messages.push({ kind: 'bottom', payload: { clause: clauseToString(s.bottom) } })
      if ((s.event === 'evaluate' || s.event === 'accept') && s.last !== null) {
        messages.push({
          kind: s.event,
          payload: {
            clause: clauseToString(s.last.clause),
            positive: s.last.positive,
            negative: s.last.negative,
            score: s.last.score,
          },
        })
      }
      return { metrics: { uncovered: s.uncovered, evaluated: s.evaluated, theory: s.theory.length }, messages }
    },
  }
}

/**
 * Progol-style inverse entailment as a traced {@link Algorithm}: a covering loop that takes the first uncovered
 * positive as a seed, builds its bottom clause, and searches breadth first over mode-conformant subsets of the bottom
 * clause's literals (general to specific) for the clause of best compression (positives minus negatives minus length)
 * covering at most `noise` negatives. Each step evaluates one candidate clause.
 *
 * @param task - Background, examples and modes.
 * @param options - Depth, clause length, noise and budgets.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for malformed modes or examples.
 * @concept inductive-logic-programming
 * @example
 * ```ts
 * import { progolAlgorithm } from 'mdml-engine/logic/ilp'
 * progolAlgorithm({ background: 'q(a).', positive: ['p(a)'], negative: [], modeh: 'modeh(1, p(+t))', modeb: ['modeb(1, q(+t))'] }).name // => 'progol'
 * ```
 */
export function progolAlgorithm(task: ProgolTask, options: ProgolOptions = {}): Algorithm<ProgolState> {
  return progolMachine(task, options)
}

/** The outcome of {@link progol}. */
export interface ProgolResult {
  readonly theory: readonly Clause[]
  readonly evaluated: number
}

/**
 * Learn a theory by inverse entailment (Progol and Aleph's basic algorithm).
 *
 * @param task - Background, examples and modes.
 * @param options - Depth, clause length, noise, budgets and tracing.
 * @returns The theory and the number of candidates evaluated.
 * @throws MdmlError with code `value` for malformed modes or examples.
 * @concept inductive-logic-programming
 * @reference S. Muggleton, Inverse entailment and Progol, New Generation Computing 13 (1995).
 * @example
 * ```ts
 * import { progol } from 'mdml-engine/logic/ilp'
 * import { clauseToString } from 'mdml-engine/logic'
 * const background = 'parent(ann, mary). parent(ann, tom). parent(tom, eve). parent(tom, ian). female(ann). female(mary). female(eve).'
 * const r = progol({ background, positive: ['daughter(mary, ann)', 'daughter(eve, tom)'], negative: ['daughter(tom, ann)', 'daughter(eve, ann)', 'daughter(ian, tom)'], modeh: 'modeh(1, daughter(+person, +person))', modeb: ['modeb(*, parent(+person, -person))', 'modeb(*, parent(-person, +person))', 'modeb(1, female(+person))'] })
 * r.theory.map((c) => clauseToString(c)) // => ['daughter(A,B) :- parent(B,A), female(A).']
 * ```
 */
export function progol(task: ProgolTask, options: ProgolOptions = {}): ProgolResult {
  const { depth: _d, maxBodyLiterals: _b, noise: _n, maxNodes: _m, queryBudget: _q, ...run } = options
  const { state } = runAlgorithm(progolMachine(task, options), { ...run, maxSteps: Number.MAX_SAFE_INTEGER })
  return { theory: state.theory, evaluated: state.evaluated }
}

/**
 * Rename a clause's variables A, B, C, ... in order of first occurrence (head first), for comparing learned clauses.
 *
 * @param c - The clause.
 * @returns The renamed clause.
 * @example
 * ```ts
 * import { canonicalClause } from 'mdml-engine/logic/ilp'
 * import { clauseToString, parseClause } from 'mdml-engine/logic'
 * clauseToString(canonicalClause(parseClause('p(X, Y) :- q(Y, Z).'))) // => 'p(A,B) :- q(B,C).'
 * ```
 */
export function canonicalClause(c: Clause): Clause {
  const names = new Map<string, string>()
  for (const v of termVariables([c.head, ...c.body])) names.set(v.name, letter(names.size))
  const r = (n: string): string => names.get(n) ?? n
  return { head: renameTerm(c.head, r), body: c.body.map((b) => renameTerm(b, r)) }
}
