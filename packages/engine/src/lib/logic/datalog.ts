/**
 * Datalog: function-free logic programs evaluated bottom-up to the set of all facts they imply, the basis of deductive
 * databases. Reading programs in Prolog syntax with safety checks and stratified negation, naive and semi-naive
 * evaluation as a traced algorithm, least models, queries against a model, and derivation trees showing how each fact
 * was obtained.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { condensation, digraph, topologicalSort } from '../graph/index.ts'
import { type Structure, type StructureEdge, type StructureNode, structure } from '../scene/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { type Bindings, evaluationOrder, factKey, instantiate, joinRule, Store } from './_relations.ts'
import { parseProgram, parseQuery } from './parse.ts'
import {
  type Clause,
  compareTerms,
  compound,
  indicator,
  type OperatorDef,
  type Operators,
  standardOperators,
  type Term,
  termToString,
  termVariables,
} from './terms.ts'

/** A body literal of a Datalog rule: a positive or negated atom, or a built-in comparison of terms. */
export type DatalogLiteral =
  | { readonly kind: 'pos'; readonly atom: Term }
  | { readonly kind: 'neg'; readonly atom: Term }
  | { readonly kind: 'cmp'; readonly op: string; readonly left: Term; readonly right: Term }

/** A Datalog rule `head :- body` (a fact when the body is empty and the head ground). */
export interface DatalogRule {
  readonly head: Term
  readonly body: readonly DatalogLiteral[]
}

/**
 * A checked Datalog program: function-free rules, each safe (every variable of the head, of a negated literal and of
 * a comparison occurs in a positive body literal), with negation stratified. `strata` lists the predicates of each
 * stratum, lowest first.
 *
 * @concept datalog
 */
export interface DatalogProgram {
  readonly rules: readonly DatalogRule[]
  readonly strata: readonly (readonly string[])[]
  /** Predicates that only occur as facts or in bodies (extensional). */
  readonly edb: readonly string[]
  /** Predicates defined by rules (intensional). */
  readonly idb: readonly string[]
}

const COMPARISONS = new Set(['=', '\\=', '==', '\\==', '<', '>', '=<', '>=', '@<', '@>', '@=<', '@>='])

/**
 * The operators of logic programs: the standard table plus `not` as a prefix operator (as in answer set syntax).
 *
 * @returns A fresh operator table.
 * @example
 * ```ts
 * import { logicProgramOperators } from 'omai-engine/logic'
 * logicProgramOperators().prefix.get('not') // => { priority: 900, type: 'fy' }
 * ```
 */
export function logicProgramOperators(): Operators {
  const ops = standardOperators()
  ;(ops.prefix as Map<string, OperatorDef>).set('not', { priority: 900, type: 'fy' })
  return ops
}

function literalOf(g: Term): DatalogLiteral {
  if (g.kind === 'compound' && (g.functor === '\\+' || g.functor === 'not') && g.args.length === 1) {
    return { kind: 'neg', atom: g.args[0] as Term }
  }
  if (g.kind === 'compound' && g.args.length === 2 && COMPARISONS.has(g.functor)) {
    return { kind: 'cmp', op: g.functor, left: g.args[0] as Term, right: g.args[1] as Term }
  }
  return { kind: 'pos', atom: g }
}

function checkAtom(t: Term, where: string): void {
  if (t.kind !== 'atom' && t.kind !== 'compound') {
    throw new OmaiError('value', `${where}: ${termToString(t)} is not an atom`)
  }
  if (t.kind === 'compound') {
    for (const a of t.args) {
      if (a.kind === 'compound') {
        throw new OmaiError('value', `${where}: function symbols are not allowed in Datalog (${termToString(t)})`)
      }
    }
  }
}

/**
 * Datalog rules from parsed clauses: body goals become positive literals, negations (`not`, `\+`) or comparisons.
 *
 * @param clauses - The clauses.
 * @param where - The caller's name, for error messages.
 * @returns The rules.
 * @throws OmaiError with code `value` for a head or literal that is not an atom, or a function symbol.
 * @example
 * ```ts
 * import { parseClause, rulesFromClauses } from 'omai-engine/logic'
 * rulesFromClauses([parseClause('p(X) :- q(X), X > 1, \\+ r(X).')], 'example')[0].body.map((l) => l.kind) // => ['pos', 'cmp', 'neg']
 * ```
 */
export function rulesFromClauses(clauses: readonly Clause[], where: string): DatalogRule[] {
  return clauses.map((c) => {
    checkAtom(c.head, where)
    const body = c.body.map(literalOf)
    for (const l of body) {
      if (l.kind !== 'cmp') checkAtom(l.atom, where)
      else if (l.left.kind === 'compound' || l.right.kind === 'compound') {
        throw new OmaiError('value', `${where}: function symbols are not allowed in Datalog`)
      }
    }
    return { head: c.head, body }
  })
}

/**
 * Check that a rule is safe (range restricted): every variable of its head, its negated literals and its comparisons
 * occurs in a positive body literal (anonymous variables excepted).
 *
 * @param rule - The rule.
 * @param where - The caller's name, for error messages.
 * @returns Nothing.
 * @throws OmaiError with code `value` naming the first unsafe variable.
 * @example
 * ```ts
 * import { checkSafety, parseClause, rulesFromClauses } from 'omai-engine/logic'
 * const [unsafe] = rulesFromClauses([parseClause('p(X, Y) :- q(X).')], 'example')
 * (() => { try { checkSafety(unsafe, 'example') } catch (e) { return e.message } })() // => 'example: unsafe rule p(X,Y) :- q(X). (variable Y)'
 * ```
 */
export function checkSafety(rule: DatalogRule, where: string): void {
  const bound = new Set(rule.body.flatMap((l) => (l.kind === 'pos' ? termVariables(l.atom).map((v) => v.name) : [])))
  const need = [
    ...termVariables(rule.head),
    ...rule.body.flatMap((l) =>
      l.kind === 'neg' ? termVariables(l.atom) : l.kind === 'cmp' ? termVariables([l.left, l.right]) : [],
    ),
  ]
  for (const v of need) {
    if (!bound.has(v.name) && !v.name.startsWith('_')) {
      throw new OmaiError('value', `${where}: unsafe rule ${ruleToString(rule)} (variable ${v.name})`)
    }
  }
}

/**
 * Write a rule as text.
 *
 * @param rule - The rule.
 * @returns `head :- body.` or `head.`.
 * @example
 * ```ts
 * import { datalog, ruleToString } from 'omai-engine/logic'
 * ruleToString(datalog('p(X) :- q(X), not r(X). q(1).').rules[0]) // => 'p(X) :- q(X), not r(X).'
 * ```
 */
export function ruleToString(rule: DatalogRule): string {
  const lit = (l: DatalogLiteral): string =>
    l.kind === 'pos'
      ? termToString(l.atom)
      : l.kind === 'neg'
        ? `not ${termToString(l.atom)}`
        : termToString(compound(l.op, [l.left, l.right]))
  return rule.body.length === 0
    ? `${termToString(rule.head)}.`
    : `${termToString(rule.head)} :- ${rule.body.map(lit).join(', ')}.`
}

/**
 * Stratify predicates by their dependencies: a predicate's stratum is at least that of every predicate it depends on
 * positively, and greater than that of every predicate it depends on through negation. Strongly connected components
 * of the dependency graph (via `graph`'s condensation) share a stratum; a negative edge inside one makes the program
 * unstratifiable.
 *
 * @param rules - The rules.
 * @returns The predicates of each stratum, lowest first (each sorted).
 * @throws OmaiError with code `value` when negation occurs in a recursive cycle.
 * @concept datalog
 * @example
 * ```ts
 * import { datalog, stratify } from 'omai-engine/logic'
 * stratify(datalog('r(X) :- e(X, _). u(X) :- r(X), not s(X). s(X) :- e(_, X).').rules) // => [['e/2', 'r/1', 's/1'], ['u/1']]
 * ```
 */
export function stratify(rules: readonly DatalogRule[]): string[][] {
  const preds = new Set<string>()
  const edges: [string, string, boolean][] = []
  for (const r of rules) {
    const h = indicator(r.head)
    preds.add(h)
    for (const l of r.body) {
      if (l.kind === 'cmp') continue
      const b = indicator(l.atom)
      preds.add(b)
      edges.push([b, h, l.kind === 'neg'])
    }
  }
  const names = [...preds].sort()
  const index = new Map(names.map((n, i) => [n, i]))
  const g = digraph(
    edges.map(([a, b]) => [index.get(a) ?? 0, index.get(b) ?? 0] as [number, number]),
    { nodes: names.length },
  )
  const cond = condensation(g)
  for (const [a, b, neg] of edges) {
    if (neg && cond.mapping[index.get(a) ?? 0] === cond.mapping[index.get(b) ?? 0]) {
      throw new OmaiError('value', `stratify: ${b} depends negatively on ${a} in a recursive cycle`)
    }
  }
  const order = topologicalSort(cond.graph)
  const level = new Array<number>(cond.components.length).fill(0)
  for (const c of order) {
    for (const [a, b, neg] of edges) {
      const ca = cond.mapping[index.get(a) ?? 0] ?? 0
      const cb = cond.mapping[index.get(b) ?? 0] ?? 0
      if (ca === c && cb !== c) level[cb] = Math.max(level[cb] ?? 0, (level[ca] ?? 0) + (neg ? 1 : 0))
    }
  }
  const strata: string[][] = []
  names.forEach((n, i) => {
    const l = level[cond.mapping[i] ?? 0] ?? 0
    while (strata.length <= l) strata.push([])
    strata[l]?.push(n)
  })
  return strata.filter((s) => s.length > 0).map((s) => s.sort())
}

/**
 * Read and check a Datalog program (Prolog syntax; negation as `not p(X)` or `\+ p(X)`; comparisons `=`, `\=`, `<`,
 * `>`, `=<`, `>=` and the standard-order ones).
 *
 * @param source - Program text or clauses.
 * @returns The checked program with its strata.
 * @throws OmaiError with code `value` for function symbols, unsafe rules, unstratifiable negation or a syntax
 *   error.
 * @concept datalog
 * @example
 * ```ts
 * import { datalog } from 'omai-engine/logic'
 * const p = datalog('edge(a, b). edge(b, c). path(X, Y) :- edge(X, Y). path(X, Z) :- edge(X, Y), path(Y, Z).')
 * [p.edb, p.idb] // => [['edge/2'], ['path/2']]
 * ```
 */
export function datalog(source: string | readonly Clause[]): DatalogProgram {
  const clauses =
    typeof source === 'string' ? parseProgram(source, { operators: logicProgramOperators() }).clauses : source
  const rules = rulesFromClauses(clauses, 'datalog')
  for (const r of rules) checkSafety(r, 'datalog')
  const strata = stratify(rules)
  const defined = new Set(rules.filter((r) => r.body.length > 0).map((r) => indicator(r.head)))
  const all = strata.flat()
  return { rules, strata, edb: all.filter((p) => !defined.has(p)), idb: all.filter((p) => defined.has(p)) }
}

// ------------------------------------------------------------------------------------------------ evaluation

/** A ground fact: a {@link Term} with no variables. */
export type Fact = Term

/** How a fact was first derived: the rule (index into the program) and the positive body facts it used. */
export interface Derivation {
  readonly rule: number
  readonly premises: readonly Fact[]
  readonly round: number
}

/** A set of ground facts by predicate indicator (each relation in insertion order). */
export type Interpretation = ReadonlyMap<string, readonly Fact[]>

/** Options of {@link evaluateDatalog}. */
export interface DatalogOptions extends Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> {
  /** Naive (every rule over every fact each round) or semi-naive (only joins touching last round's new facts; default). */
  readonly method?: 'naive' | 'semi-naive'
  /** Stop with status `budget` after this many facts (default 1000000). */
  readonly maxFacts?: number
}

/** New facts in one round of one stratum. */
export interface DatalogRound {
  readonly stratum: number
  readonly round: number
  readonly derived: number
}

/** The observable state of a bottom-up evaluation. */
export interface DatalogState {
  readonly status: 'running' | 'fixpoint'
  readonly stratum: number
  /** Rounds completed in the current stratum. */
  readonly round: number
  readonly facts: number
  /** Facts derived in the last round. */
  readonly delta: readonly Fact[]
  /** New facts per round, over all strata in order (each stratum ends with a round that derives nothing). */
  readonly history: readonly DatalogRound[]
}

interface DatalogMachine extends DatalogState {
  status: 'running' | 'fixpoint'
  stratum: number
  round: number
  facts: number
  delta: Fact[]
  history: DatalogRound[]
  readonly store: Store
  /** The current stratum's facts before the last round's additions (semi-naive's "old" relations). */
  old: Store
}

const DATALOG_SCHEMA: TraceSchema = channels({
  metrics: {
    facts: { description: 'facts derived so far (including the given ones)' },
    delta: { description: 'new facts in the last round' },
    stratum: { description: 'the stratum being evaluated' },
    round: { description: 'round within the stratum' },
  },
  messages: {
    round: {
      description: 'a round of bottom-up evaluation',
      payload: { stratum: 'stratum', round: 'round', derived: 'facts derived in it', facts: 'the new facts' },
    },
  },
})

function datalogMachine(p: DatalogProgram, options: DatalogOptions): Algorithm<DatalogMachine> {
  const method = options.method ?? 'semi-naive'
  const maxFacts = options.maxFacts ?? 1_000_000
  const stratumOf = new Map<string, number>()
  p.strata.forEach((preds, i) => {
    for (const q of preds) stratumOf.set(q, i)
  })
  const rules = p.rules
    .map((rule, index) => ({ rule, index, order: evaluationOrder(rule), k: stratumOf.get(indicator(rule.head)) ?? 0 }))
    .filter((r) => r.rule.body.length > 0)
  const round = (s: DatalogMachine): Fact[] => {
    const k = s.stratum
    const inStratum = (pred: string): boolean => stratumOf.get(pred) === k
    const derived: { fact: Fact; d: Derivation }[] = []
    const seen = new Set<string>()
    const holds = (f: Fact): boolean => s.store.has(f)
    const delta = new Store()
    for (const f of s.delta) delta.add(f, null)
    for (const { rule, index, order } of rules.filter((r) => r.k === k)) {
      const recursive = rule.body.flatMap((l, i) => (l.kind === 'pos' && inStratum(indicator(l.atom)) ? [i] : []))
      const emit = (b: Bindings, premises: readonly Fact[]): void => {
        const fact = instantiate(rule.head, b)
        const key = factKey(fact)
        if (s.store.has(fact) || seen.has(key)) return
        seen.add(key)
        derived.push({ fact, d: { rule: index, premises: [...premises], round: s.round + 1 } })
      }
      if (method === 'naive' || s.round === 0) joinRule(rule, order, (_, pred) => s.store.get(pred), holds, emit)
      else {
        // Semi-naive: one join per recursive literal j, which reads last round's new facts; recursive literals before
        // it read the full relations and those after it the relations as they were before the last round.
        for (const j of recursive) {
          const source = (i: number, pred: string): readonly Fact[] =>
            !inStratum(pred)
              ? s.store.get(pred)
              : i === j
                ? delta.get(pred)
                : i < j
                  ? s.store.get(pred)
                  : s.old.get(pred)
          joinRule(rule, order, source, holds, emit)
        }
      }
    }
    const old = new Store()
    for (const pred of p.strata[k] ?? []) for (const f of s.store.get(pred)) old.add(f, null)
    s.old = old
    for (const { fact, d } of derived) s.store.add(fact, d)
    return derived.map((x) => x.fact)
  }
  return {
    name: method === 'naive' ? 'naive-evaluation' : 'semi-naive-evaluation',
    config: { method, strata: p.strata.length },
    schema: DATALOG_SCHEMA,
    init: () => {
      const store = new Store()
      for (const r of p.rules) {
        if (r.body.length > 0) continue
        if (termVariables(r.head).length > 0) {
          throw new OmaiError('value', `datalog: fact ${termToString(r.head)} is not ground`)
        }
        store.add(r.head, null)
      }
      return {
        status: p.strata.length === 0 ? 'fixpoint' : 'running',
        stratum: 0,
        round: 0,
        facts: store.size,
        delta: [],
        history: [],
        store,
        old: new Store(),
      }
    },
    step: (s) => {
      s.delta = round(s)
      s.round++
      s.facts = s.store.size
      s.history.push({ stratum: s.stratum, round: s.round, derived: s.delta.length })
      if (s.delta.length === 0) {
        s.stratum++
        s.round = 0
        if (s.stratum >= p.strata.length) s.status = 'fixpoint'
      }
      return s
    },
    done: (s) => (s.status === 'fixpoint' ? 'converged' : s.facts >= maxFacts ? 'stopped' : undefined),
    observe: (s) => {
      const last = s.history[s.history.length - 1]
      return {
        metrics: { facts: s.facts, delta: s.delta.length, stratum: s.stratum, round: s.round },
        messages:
          last === undefined
            ? []
            : [{ kind: 'round', payload: { ...last, facts: s.delta.map((f) => termToString(f)) } as Json }],
      }
    },
  }
}

/**
 * Bottom-up evaluation as a traced {@link Algorithm}: strata in order, each iterated to its fixpoint one round per
 * step (the immediate consequence operator), naive or semi-naive.
 *
 * @param p - The program.
 * @param options - The method and a fact budget.
 * @returns The algorithm.
 * @concept datalog
 * @example
 * ```ts
 * import { datalog, datalogAlgorithm } from 'omai-engine/logic'
 * datalogAlgorithm(datalog('p(a).'), { method: 'naive' }).name // => 'naive-evaluation'
 * ```
 */
export function datalogAlgorithm(p: DatalogProgram, options: DatalogOptions = {}): Algorithm<DatalogState> {
  return datalogMachine(p, options)
}

/** The outcome of {@link evaluateDatalog}. */
export interface DatalogResult {
  readonly status: 'fixpoint' | 'budget'
  /** The model: the least Herbrand model of a definite program, the perfect (stratified) model with negation. */
  readonly model: Interpretation
  readonly facts: number
  /** Rounds over all strata. */
  readonly rounds: number
  readonly history: readonly DatalogRound[]
  /** The first derivation of every derived fact, by fact text. */
  readonly derivations: ReadonlyMap<string, Derivation>
  readonly program: DatalogProgram
}

/**
 * Evaluate a Datalog program bottom-up to its model.
 *
 * @param program - A checked program or program text.
 * @param options - The method (semi-naive by default), a fact budget and tracing.
 * @returns The model, the rounds and the derivations.
 * @throws OmaiError with code `value` for an invalid program.
 * @concept datalog
 * @example
 * ```ts
 * import { evaluateDatalog } from 'omai-engine/logic'
 * const r = evaluateDatalog('edge(1, 2). edge(2, 3). edge(3, 4). path(X, Y) :- edge(X, Y). path(X, Z) :- path(X, Y), edge(Y, Z).')
 * [r.model.get('path/2').length, r.rounds] // => [6, 4]
 * ```
 */
export function evaluateDatalog(program: DatalogProgram | string, options: DatalogOptions = {}): DatalogResult {
  const p = typeof program === 'string' ? datalog(program) : program
  const { method: _m, maxFacts: _f, ...run } = options
  const { state } = runAlgorithm(datalogMachine(p, options), { ...run, maxSteps: Number.MAX_SAFE_INTEGER })
  return {
    status: state.status === 'fixpoint' ? 'fixpoint' : 'budget',
    model: state.store.byPred,
    facts: state.facts,
    rounds: state.history.length,
    history: state.history,
    derivations: state.store.derivations,
    program: p,
  }
}

/**
 * The least Herbrand model of a definite program (or the perfect model of a stratified one) as sorted fact texts.
 *
 * @param program - A program or its text.
 * @returns Every fact of the model, written out and sorted.
 * @concept datalog
 * @example
 * ```ts
 * import { leastModel } from 'omai-engine/logic'
 * leastModel('p(a). q(X) :- p(X). r(X) :- q(X), not p(X).') // => ['p(a)', 'q(a)']
 * ```
 */
export function leastModel(program: DatalogProgram | string): string[] {
  const r = evaluateDatalog(program)
  return [...r.model.values()].flatMap((fs) => fs.map((f) => termToString(f))).sort()
}

/**
 * Answer a query (a conjunction of atoms, negations and comparisons) against a model: every binding of the query's
 * variables, in the order found.
 *
 * @param model - The model (from {@link evaluateDatalog}).
 * @param query - The query text, e.g. `path(a, X), X \= b`.
 * @returns The bindings, each a map from variable name to value text.
 * @throws OmaiError with code `value` for an unsafe query.
 * @example
 * ```ts
 * import { evaluateDatalog, queryModel } from 'omai-engine/logic'
 * const r = evaluateDatalog('parent(tom, bob). parent(bob, ann). anc(X, Y) :- parent(X, Y). anc(X, Z) :- parent(X, Y), anc(Y, Z).')
 * queryModel(r.model, 'anc(tom, X)') // => [{ X: 'bob' }, { X: 'ann' }]
 * ```
 */
export function queryModel(model: Interpretation, query: string): Record<string, string>[] {
  const q = parseQuery(query, { operators: logicProgramOperators() })
  const [rule] = rulesFromClauses(
    [
      {
        head: compound(
          '$query',
          q.variables.map((name) => ({ kind: 'var', name })),
        ),
        body: q.goal.kind === 'compound' && q.goal.functor === ',' ? flattenConj(q.goal) : [q.goal],
      },
    ],
    'queryModel',
  )
  if (rule === undefined) return []
  checkSafety(rule, 'queryModel')
  const out: Record<string, string>[] = []
  const has = (f: Fact): boolean => (model.get(indicator(f)) ?? []).some((g) => compareTerms(f, g) === 0)
  joinRule(
    rule,
    evaluationOrder(rule),
    (_, pred) => model.get(pred) ?? [],
    has,
    (b) => {
      const row: Record<string, string> = {}
      for (const name of q.variables) {
        const v = b.get(name)
        if (v !== undefined) row[name] = termToString(v)
      }
      out.push(row)
    },
  )
  return out
}

function flattenConj(t: Term): Term[] {
  if (t.kind === 'compound' && t.functor === ',' && t.args.length === 2) {
    return [...flattenConj(t.args[0] as Term), ...flattenConj(t.args[1] as Term)]
  }
  return [t]
}

/**
 * The derivation tree of a fact as a {@link Structure}: the fact at the root, each derived fact's children being the
 * body facts of the rule instance that first derived it (given facts are leaves).
 *
 * @param result - An evaluation result.
 * @param fact - The fact, as text or a term.
 * @returns The structure.
 * @throws OmaiError with code `value` when the fact is not in the model.
 * @concept datalog
 * @example
 * ```ts
 * import { derivationTree, evaluateDatalog } from 'omai-engine/logic'
 * const r = evaluateDatalog('e(1, 2). e(2, 3). p(X, Y) :- e(X, Y). p(X, Z) :- e(X, Y), p(Y, Z).')
 * derivationTree(r, 'p(1, 3)').nodes.map((n) => n.label) // => ['p(1,3)', 'e(1,2)', 'p(2,3)', 'e(2,3)']
 * ```
 */
export function derivationTree(result: DatalogResult, fact: string | Fact): Structure {
  const root = typeof fact === 'string' ? parseQuery(fact).goal : fact
  const key = factKey(root)
  const present = (result.model.get(indicator(root)) ?? []).some((f) => factKey(f) === key)
  if (!present) throw new OmaiError('value', `derivationTree: ${key} is not in the model`)
  const nodes: StructureNode[] = []
  const edges: StructureEdge[] = []
  const visit = (f: Fact, parent: string | null, depth: number): void => {
    const id = `n${nodes.length}`
    const k = factKey(f)
    const d = result.derivations.get(k)
    nodes.push({
      id,
      label: k,
      role: d === undefined ? 'leaf' : parent === null ? 'output' : 'internal',
      ...(d === undefined ? {} : { detail: ruleToString(result.program.rules[d.rule] as DatalogRule) }),
      badges: d === undefined ? { given: 1 } : { round: d.round },
    })
    if (parent !== null) edges.push({ from: parent, to: id, directed: true })
    if (d !== undefined && depth < 64) for (const premise of d.premises) visit(premise, id, depth + 1)
  }
  visit(root, null, 0)
  return structure('derivation-tree', nodes, edges, { direction: 'down' })
}
