import { type Json, WhiteboxError } from '../../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../../trace/index.ts'
import { parseTerm } from '../parse.ts'
import { consult, type Database, solve } from '../resolution.ts'
import { type Clause, clauseToString, compound, indicator, type Term, termVariables, variable } from '../terms.ts'
import { applySubstitution, renameTerm, type Substitution, unify } from '../unify.ts'
import { thetaSubsumes } from './generalisation.ts'
import { canonicalClause } from './progol.ts'

// ------------------------------------------------------------------------------------------------ metarules

/**
 * A metarule: a clause template whose predicate symbols are second-order variables. Atoms are lists
 * `[predicate, arg1, ...]`; `symbols` names the second-order variables (the head's predicate first).
 *
 * @concept meta-interpretive-learning
 */
export interface Metarule {
  readonly name: string
  readonly symbols: readonly string[]
  readonly head: readonly string[]
  readonly body: readonly (readonly string[])[]
}

/**
 * The standard metarules of Metagol (identity, inverse, precon, postcon, chain, tail recursion).
 *
 * @example
 * ```ts
 * import { METARULES } from 'whitebox-engine/logic/ilp'
 * Object.keys(METARULES) // => ['ident', 'inverse', 'precon', 'postcon', 'chain', 'tailrec']
 * ```
 */
export const METARULES: Readonly<Record<string, Metarule>> = {
  ident: { name: 'ident', symbols: ['P', 'Q'], head: ['P', 'A', 'B'], body: [['Q', 'A', 'B']] },
  inverse: { name: 'inverse', symbols: ['P', 'Q'], head: ['P', 'A', 'B'], body: [['Q', 'B', 'A']] },
  precon: {
    name: 'precon',
    symbols: ['P', 'Q', 'R'],
    head: ['P', 'A', 'B'],
    body: [
      ['Q', 'A'],
      ['R', 'A', 'B'],
    ],
  },
  postcon: {
    name: 'postcon',
    symbols: ['P', 'Q', 'R'],
    head: ['P', 'A', 'B'],
    body: [
      ['Q', 'A', 'B'],
      ['R', 'B'],
    ],
  },
  chain: {
    name: 'chain',
    symbols: ['P', 'Q', 'R'],
    head: ['P', 'A', 'B'],
    body: [
      ['Q', 'A', 'C'],
      ['R', 'C', 'B'],
    ],
  },
  tailrec: {
    name: 'tailrec',
    symbols: ['P', 'Q'],
    head: ['P', 'A', 'B'],
    body: [
      ['Q', 'A', 'C'],
      ['P', 'C', 'B'],
    ],
  },
}

/** A metasubstitution: a metarule with its second-order variables bound to predicate symbols. */
export interface Metasub {
  readonly metarule: string
  readonly symbols: Readonly<Record<string, string>>
}

/**
 * The clause a metasubstitution stands for.
 *
 * @param m - The metarule.
 * @param symbols - The predicate symbol of each second-order variable.
 * @returns The first-order clause.
 * @example
 * ```ts
 * import { METARULES, metaruleClause } from 'whitebox-engine/logic/ilp'
 * import { clauseToString } from 'whitebox-engine/logic'
 * clauseToString(metaruleClause(METARULES.chain, { P: 'grandparent', Q: 'parent', R: 'parent' })) // => 'grandparent(A,B) :- parent(A,C), parent(C,B).'
 * ```
 */
export function metaruleClause(m: Metarule, symbols: Readonly<Record<string, string>>): Clause {
  const lit = (a: readonly string[]): Term => {
    const [p, ...args] = a
    const name = symbols[p ?? ''] ?? p ?? ''
    return compound(
      name,
      args.map((x) => variable(x)),
    )
  }
  return { head: lit(m.head), body: m.body.map(lit) }
}

// ------------------------------------------------------------------------------------------------ Metagol

/** A meta-interpretive learning task. */
export interface MilTask {
  /** Background knowledge: Prolog text or a database. */
  readonly background: string | Database
  /** The background predicates the learner may use, as `name/arity`. */
  readonly predicates: readonly string[]
  readonly positive: readonly (Term | string)[]
  readonly negative: readonly (Term | string)[]
  /** The metarules to use, in the order they are tried (names of {@link METARULES} or metarules). */
  readonly metarules: readonly (Metarule | string)[]
}

/** Options of {@link metagol}. */
export interface MetagolOptions extends Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> {
  /** The largest program tried, in clauses (default 4). */
  readonly maxClauses?: number
  /** Invented predicate symbols allowed (default 1). */
  readonly maxInvented?: number
  /** Proof depth bound (default 12). */
  readonly maxDepth?: number
  /** Meta-interpreter steps before giving up on a program size (default 200000). */
  readonly maxSteps?: number
}

/** The observable state of Metagol's iterative deepening over program sizes. */
export interface MetagolState {
  readonly status: 'running' | 'learned' | 'failed'
  /** The program size bound being searched. */
  readonly bound: number
  readonly candidates: number
  readonly program: readonly Metasub[] | null
  readonly last: { readonly program: readonly Metasub[]; readonly consistent: boolean } | null
}

interface MetagolMachine extends MetagolState {
  status: 'running' | 'learned' | 'failed'
  bound: number
  candidates: number
  program: Metasub[] | null
  last: { program: Metasub[]; consistent: boolean } | null
  generator: Iterator<Metasub[]> | null
}

const MIL_SCHEMA: TraceSchema = channels({
  metrics: {
    bound: { description: 'program size bound (clauses)' },
    candidates: { description: 'programs proving every positive' },
  },
  messages: {
    candidate: {
      description: 'a program proving every positive example',
      payload: { program: 'clauses', consistent: 'proves no negative' },
    },
    bound: { description: 'the size bound was raised', payload: { bound: 'clauses' } },
  },
})

function asAtom(x: Term | string): Term {
  return typeof x === 'string' ? parseTerm(x) : x
}

function metaruleOf(m: Metarule | string): Metarule {
  if (typeof m !== 'string') return m
  const r = METARULES[m]
  if (r === undefined) throw new WhiteboxError('value', `metagol: unknown metarule ${m}`)
  return r
}

/** Every program (within the size bound) under which every positive example is provable, in Metagol's order. */
function* provePositives(
  examples: readonly Term[],
  db: Database,
  background: ReadonlySet<string>,
  arities: ReadonlyMap<string, number>,
  metarules: readonly Metarule[],
  target: string,
  targetArity: number,
  bound: number,
  maxInvented: number,
  maxDepth: number,
  budget: { steps: number },
): Generator<Metasub[]> {
  let fresh = 0
  const rename = (t: Term): Term => renameTerm(t, (n) => `${n}_${fresh}`)
  const invented = (prog: readonly Metasub[]): string[] => {
    const out: string[] = []
    for (const m of prog)
      for (const s of Object.values(m.symbols)) if (s.startsWith(`${target}_`) && !out.includes(s)) out.push(s)
    return out
  }
  type Goal = { readonly atom: Term; readonly depth: number }
  function* prove(goals: readonly Goal[], prog: Metasub[], s: Substitution): Generator<[Metasub[], Substitution]> {
    if (budget.steps <= 0) return
    budget.steps--
    const [g, ...rest] = goals
    if (g === undefined) {
      yield [prog, s]
      return
    }
    const a = applySubstitution(g.atom, s)
    if (a.kind !== 'compound' && a.kind !== 'atom') return
    const key = indicator(a)
    const name = a.kind === 'compound' ? a.functor : a.name
    const args = a.kind === 'compound' ? a.args : []
    if (background.has(key)) {
      const r = solve(db, a, { maxSteps: 20_000, unknown: 'fail' })
      for (const ans of r.answers) {
        let s2: Substitution | null = s
        for (const v of termVariables(a)) {
          const val = ans.bindings[v.name]
          if (val !== undefined) s2 = s2 === null ? null : unify(v, val, s2)
        }
        if (s2 !== null) yield* prove(rest, prog, s2)
      }
      return
    }
    if (g.depth >= maxDepth) return
    // Clauses already in the program first, then new metasubstitutions while the bound allows.
    const tryMetasub = function* (
      m: Metarule,
      symbols: Record<string, string>,
      p: Metasub[],
    ): Generator<[Metasub[], Substitution]> {
      fresh++
      const c = metaruleClause(m, symbols)
      const head = rename(c.head)
      const s2 = unify(head, a, s)
      if (s2 === null) return
      yield* prove([...c.body.map((b) => ({ atom: rename(b), depth: g.depth + 1 })), ...rest], p, s2)
    }
    for (const sub of prog) {
      const m = metarules.find((x) => x.name === sub.metarule)
      if (m === undefined || sub.symbols[m.symbols[0] ?? ''] !== name) continue
      yield* tryMetasub(m, { ...sub.symbols }, prog)
    }
    if (prog.length >= bound) return
    for (const m of metarules) {
      if (m.head.length - 1 !== args.length) continue
      const others = m.symbols.slice(1)
      // Candidate symbols for each body second-order variable: background, then the target and invented predicates.
      const pool = (arity: number, p: readonly Metasub[]): string[] => {
        const inv = invented(p)
        const out = [...background].filter((k) => arities.get(k) === arity).map((k) => k.slice(0, k.lastIndexOf('/')))
        if (arity === targetArity) out.push(target)
        // Invented predicates are dyadic, as in Metagol.
        if (arity === 2) {
          out.push(...inv)
          if (inv.length < maxInvented) out.push(`${target}_${inv.length + 1}`)
        }
        return [...new Set(out)]
      }
      const assign = function* (i: number, symbols: Record<string, string>): Generator<Record<string, string>> {
        if (i === others.length) {
          yield symbols
          return
        }
        const v = others[i] as string
        const arity = (m.body.find((b) => b[0] === v)?.length ?? 1) - 1
        for (const sym of pool(arity, prog)) yield* assign(i + 1, { ...symbols, [v]: sym })
      }
      for (const symbols of assign(0, { [m.symbols[0] as string]: name })) {
        // No left recursion: the first body atom may not call the head's predicate (it would never terminate).
        const firstBody = m.body[0]?.[0]
        if (firstBody !== undefined && symbols[firstBody] === name) continue
        const sub: Metasub = { metarule: m.name, symbols }
        if (prog.some((x) => JSON.stringify(x) === JSON.stringify(sub))) continue
        yield* tryMetasub(m, { ...symbols }, [...prog, sub])
      }
    }
  }
  function* all(i: number, prog: Metasub[]): Generator<Metasub[]> {
    const e = examples[i]
    if (e === undefined) {
      yield prog
      return
    }
    for (const [p] of prove([{ atom: e, depth: 0 }], prog, new Map())) yield* all(i + 1, p)
  }
  yield* all(0, [])
}

/**
 * The clauses of a program of metasubstitutions, with canonical variable names.
 *
 * @param program - The metasubstitutions.
 * @param metarules - The metarules they refer to (default the standard ones).
 * @returns The clauses.
 * @example
 * ```ts
 * import { metagolProgram } from 'whitebox-engine/logic/ilp'
 * import { clauseToString } from 'whitebox-engine/logic'
 * metagolProgram([{ metarule: 'inverse', symbols: { P: 'child', Q: 'parent' } }]).map((c) => clauseToString(c)) // => ['child(A,B) :- parent(B,A).']
 * ```
 */
export function metagolProgram(
  program: readonly Metasub[],
  metarules: readonly (Metarule | string)[] = Object.keys(METARULES),
): Clause[] {
  const rules = metarules.map(metaruleOf)
  return program.map((sub) => {
    const m = rules.find((r) => r.name === sub.metarule) ?? metaruleOf(sub.metarule)
    return canonicalClause(metaruleClause(m, sub.symbols))
  })
}

function metagolMachine(task: MilTask, options: MetagolOptions): Algorithm<MetagolMachine> {
  const db = typeof task.background === 'string' ? consult(task.background) : task.background
  const pos = task.positive.map(asAtom)
  const neg = task.negative.map(asAtom)
  const metarules = task.metarules.map(metaruleOf)
  const maxClauses = options.maxClauses ?? 4
  const maxInvented = options.maxInvented ?? 1
  const maxDepth = options.maxDepth ?? 12
  const maxSteps = options.maxSteps ?? 200_000
  const first = pos[0]
  if (first === undefined) throw new WhiteboxError('value', 'metagol: no positive examples')
  const target = first.kind === 'compound' ? first.functor : first.kind === 'atom' ? first.name : ''
  const targetArity = first.kind === 'compound' ? first.args.length : 0
  const arities = new Map<string, number>()
  for (const p of task.predicates) arities.set(p, Number(p.slice(p.lastIndexOf('/') + 1)))
  const background = new Set(task.predicates)
  const consistent = (prog: readonly Metasub[]): boolean => {
    const clauses = prog.map((sub) =>
      metaruleClause(metarules.find((m) => m.name === sub.metarule) ?? metaruleOf(sub.metarule), sub.symbols),
    )
    const theory = consult(clauses, { base: db })
    return neg.every(
      (e) =>
        solve(theory, e, { maxSolutions: 1, maxSteps: 20_000, maxDepth: 50, unknown: 'fail' }).answers.length === 0,
    )
  }
  const start = (bound: number): Iterator<Metasub[]> =>
    provePositives(pos, db, background, arities, metarules, target, targetArity, bound, maxInvented, maxDepth, {
      steps: maxSteps,
    })
  return {
    name: 'metagol',
    config: { maxClauses, maxInvented, maxDepth, metarules: metarules.map((m) => m.name) },
    schema: MIL_SCHEMA,
    init: () => ({ status: 'running', bound: 1, candidates: 0, program: null, last: null, generator: start(1) }),
    step: (s) => {
      s.last = null
      const gen = s.generator
      if (gen === null) return s
      const r = gen.next()
      if (r.done === true) {
        if (s.bound >= maxClauses) {
          s.status = 'failed'
          s.generator = null
        } else {
          s.bound++
          s.generator = start(s.bound)
        }
        return s
      }
      s.candidates++
      const ok = consistent(r.value)
      s.last = { program: r.value, consistent: ok }
      if (ok) {
        s.program = r.value
        s.status = 'learned'
      }
      return s
    },
    done: (s) => (s.status === 'running' ? undefined : s.status === 'learned' ? 'converged' : 'stopped'),
    observe: (s) => ({
      metrics: { bound: s.bound, candidates: s.candidates },
      messages:
        s.last === null
          ? []
          : [
              {
                kind: 'candidate',
                payload: {
                  program: metagolProgram(s.last.program, metarules).map((c) => clauseToString(c)),
                  consistent: s.last.consistent,
                } as Json,
              },
            ],
    }),
  }
}

/**
 * Metagol's meta-interpretive learning as a traced {@link Algorithm}: a meta-interpreter proves every positive example,
 * resolving background atoms against the background knowledge and other atoms against the program's clauses or, while
 * the size bound allows, against a metarule whose second-order variables it binds to background, target or invented
 * predicate symbols (abduction; left-recursive instances are excluded, as Metagol's ordering constraints do). Bounds grow from one clause (iterative deepening), so the first program that also
 * proves no negative example is a smallest one. Each step tries one program that proves every positive.
 *
 * @param task - Background, its predicates, examples and metarules.
 * @param options - Program size, invented predicates and budgets.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` for no positives or an unknown metarule.
 * @concept meta-interpretive-learning
 * @example
 * ```ts
 * import { metagolAlgorithm } from 'whitebox-engine/logic/ilp'
 * metagolAlgorithm({ background: 'q(a, b).', predicates: ['q/2'], positive: ['p(a, b)'], negative: [], metarules: ['ident'] }).name // => 'metagol'
 * ```
 */
export function metagolAlgorithm(task: MilTask, options: MetagolOptions = {}): Algorithm<MetagolState> {
  return metagolMachine(task, options)
}

/** The outcome of {@link metagol} and {@link learnFromFailures}. */
export interface MilResult {
  readonly status: 'learned' | 'failed'
  readonly program: readonly Clause[]
  /** Hypotheses tried (programs proving every positive for Metagol; tested hypotheses for learning from failures). */
  readonly candidates: number
}

/**
 * Learn a smallest program by meta-interpretive learning (Metagol).
 *
 * @param task - Background, its predicates, examples and metarules.
 * @param options - Program size, invented predicates, budgets and tracing.
 * @returns The status, the program (canonical variable names) and the candidates tried.
 * @throws WhiteboxError with code `value` for no positives or an unknown metarule.
 * @concept meta-interpretive-learning
 * @reference S. H. Muggleton, D. Lin and A. Tamaddoni-Nezhad, Meta-interpretive learning of higher-order dyadic Datalog,
 *   Machine Learning 100 (2015).
 * @example
 * ```ts
 * import { metagol } from 'whitebox-engine/logic/ilp'
 * import { clauseToString } from 'whitebox-engine/logic'
 * const background = 'parent(ann, amy). parent(ann, andy). parent(amy, amelia). parent(andy, spongebob).'
 * const r = metagol({ background, predicates: ['parent/2'], positive: ['grandparent(ann, amelia)', 'grandparent(ann, spongebob)'], negative: ['grandparent(amy, amelia)'], metarules: ['ident', 'chain'] })
 * r.program.map((c) => clauseToString(c)) // => ['grandparent(A,B) :- parent(A,C), parent(C,B).']
 * ```
 */
export function metagol(task: MilTask, options: MetagolOptions = {}): MilResult {
  const { maxClauses: _c, maxInvented: _i, maxDepth: _d, maxSteps: _s, ...run } = options
  const { state } = runAlgorithm(metagolMachine(task, options), { ...run, maxSteps: Number.MAX_SAFE_INTEGER })
  return {
    status: state.status === 'learned' ? 'learned' : 'failed',
    program: state.program === null ? [] : metagolProgram(state.program, task.metarules),
    candidates: state.candidates,
  }
}

// ------------------------------------------------------------------------------------------------ learning from failures

/** A hypothesis space for {@link learnFromFailures}: the head and body predicates and size limits. */
export interface FailureTask {
  readonly background: string | Database
  /** The target predicate as `name/arity`. */
  readonly head: string
  /** Body predicates as `name/arity` (include the head's for recursion). */
  readonly body: readonly string[]
  readonly positive: readonly (Term | string)[]
  readonly negative: readonly (Term | string)[]
}

/** Options of {@link learnFromFailures}. */
export interface FailureOptions extends Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> {
  /** Body literals per clause (default 3). */
  readonly maxBody?: number
  /** Variables per clause (default 4). */
  readonly maxVars?: number
  /** Clauses per program (default 2). */
  readonly maxClauses?: number
  /** Hypotheses tested before giving up (default 20000). */
  readonly maxHypotheses?: number
  /** Step budget for proving one example (default 5000). */
  readonly queryBudget?: number
}

/** The observable state of learning from failures. */
export interface FailureState {
  readonly status: 'running' | 'learned' | 'failed'
  readonly tested: number
  readonly pruned: number
  readonly constraints: number
  readonly last: {
    readonly program: readonly string[]
    readonly truePositives: number
    readonly falsePositives: number
    readonly outcome: 'solution' | 'too general' | 'too specific' | 'both'
  } | null
  readonly program: readonly Clause[] | null
}

interface FailureMachine extends FailureState {
  status: 'running' | 'learned' | 'failed'
  tested: number
  pruned: number
  constraints: number
  last: FailureState['last']
  program: Clause[] | null
  readonly generalOf: Clause[][]
  readonly specialOf: Clause[][]
  readonly programs: Iterator<Clause[]>
}

const LFF_SCHEMA: TraceSchema = channels({
  metrics: {
    tested: { description: 'hypotheses tested' },
    pruned: { description: 'hypotheses skipped by constraints' },
    constraints: { description: 'constraints learned from failures' },
  },
  messages: {
    test: {
      description: 'a hypothesis tested and what its failure teaches',
      payload: {
        program: 'clauses',
        truePositives: 'positives covered',
        falsePositives: 'negatives covered',
        outcome: 'outcome',
      },
    },
  },
})

function parsePredicate(p: string): [string, number] {
  const i = p.lastIndexOf('/')
  const arity = Number(p.slice(i + 1))
  if (i <= 0 || !Number.isInteger(arity)) throw new WhiteboxError('value', `learnFromFailures: ${p} is not name/arity`)
  return [p.slice(0, i), arity]
}

/** Every clause of the hypothesis space, smallest first, with canonical variables and no singleton variables. */
function clauseSpace(
  head: [string, number],
  body: readonly [string, number][],
  maxBody: number,
  maxVars: number,
): Clause[] {
  const out: Clause[] = []
  const v = (i: number): Term => variable(String.fromCharCode(65 + i))
  const headTerm = compound(
    head[0],
    Array.from({ length: head[1] }, (_, i) => v(i)),
  )
  // Literals as (predicate index, argument variable indices); bodies are non-decreasing sequences of literals.
  const literalsFor = (vars: number): { p: number; args: number[] }[] => {
    const lits: { p: number; args: number[] }[] = []
    body.forEach(([, arity], p) => {
      const rec = (prefix: number[]): void => {
        if (prefix.length === arity) {
          lits.push({ p, args: prefix })
          return
        }
        for (let a = 0; a < vars; a++) rec([...prefix, a])
      }
      rec([])
    })
    return lits
  }
  const all = literalsFor(maxVars)
  for (let n = 1; n <= maxBody; n++) {
    const rec = (prefix: { p: number; args: number[] }[], start: number): void => {
      if (prefix.length === n) {
        // Canonical variables: new ones introduced in order after the head's.
        let next = head[1]
        const seen = new Set(Array.from({ length: head[1] }, (_, i) => i))
        for (const l of prefix) {
          for (const a of l.args) {
            if (seen.has(a)) continue
            if (a !== next) return
            seen.add(a)
            next++
          }
        }
        const counts = new Map<number, number>()
        for (let i = 0; i < head[1]; i++) counts.set(i, 1)
        for (const l of prefix) for (const a of l.args) counts.set(a, (counts.get(a) ?? 0) + 1)
        if ([...counts.values()].some((c) => c < 2)) return
        const bodyTerms = prefix.map((l) => compound((body[l.p] as [string, number])[0], l.args.map(v)))
        // Skip a clause that is just its own head (p(A, B) :- p(A, B)).
        if (
          bodyTerms.some((b) => clauseToString({ head: b, body: [] }) === clauseToString({ head: headTerm, body: [] }))
        )
          return
        out.push({ head: headTerm, body: bodyTerms })
        return
      }
      for (let i = start; i < all.length; i++) rec([...prefix, all[i] as { p: number; args: number[] }], i + 1)
    }
    rec([], 0)
  }
  return out
}

/** Programs of up to `maxClauses` distinct clauses, in order of total size (literals), then clause order. */
function* programSpace(clauses: readonly Clause[], maxClauses: number, maxBody: number): Generator<Clause[]> {
  const size = (c: Clause): number => 1 + c.body.length
  const maxSize = maxClauses * (1 + maxBody)
  for (let total = 2; total <= maxSize; total++) {
    const rec = function* (start: number, left: number, k: number, chosen: Clause[]): Generator<Clause[]> {
      if (left === 0) {
        yield chosen
        return
      }
      if (k === 0) return
      for (let i = start; i < clauses.length; i++) {
        const c = clauses[i] as Clause
        if (size(c) > left) continue
        yield* rec(i + 1, left - size(c), k - 1, [...chosen, c])
      }
    }
    yield* rec(0, total, maxClauses, [])
  }
}

/** Whether every clause of `h` is subsumed by a clause of `g` (so `g` entails `h`). */
function subsumedBy(h: readonly Clause[], g: readonly Clause[]): boolean {
  return h.every((c) => g.some((d) => thetaSubsumes(d, c) !== null))
}

function failureMachine(task: FailureTask, options: FailureOptions): Algorithm<FailureMachine> {
  const db = typeof task.background === 'string' ? consult(task.background) : task.background
  const pos = task.positive.map(asAtom)
  const neg = task.negative.map(asAtom)
  const head = parsePredicate(task.head)
  const body = task.body.map(parsePredicate)
  const maxHypotheses = options.maxHypotheses ?? 20_000
  const budget = options.queryBudget ?? 5000
  const clauses = clauseSpace(head, body, options.maxBody ?? 3, options.maxVars ?? 4)
  const covered = (prog: readonly Clause[], e: Term): boolean =>
    solve(consult(prog, { base: db }), e, { maxSolutions: 1, maxSteps: budget, maxDepth: 30, unknown: 'fail' }).answers
      .length > 0
  return {
    name: 'learning-from-failures',
    config: { clauses: clauses.length, maxClauses: options.maxClauses ?? 2 },
    schema: LFF_SCHEMA,
    init: () => ({
      status: 'running',
      tested: 0,
      pruned: 0,
      constraints: 0,
      last: null,
      program: null,
      generalOf: [],
      specialOf: [],
      programs: programSpace(clauses, options.maxClauses ?? 2, options.maxBody ?? 3),
    }),
    step: (s) => {
      for (;;) {
        const r = s.programs.next()
        if (r.done === true) {
          s.status = 'failed'
          return s
        }
        const h = r.value
        // Generalisations of a too-general hypothesis and specialisations of a too-specific one are pruned.
        if (s.generalOf.some((g) => subsumedBy(g, h)) || s.specialOf.some((sp) => subsumedBy(h, sp))) {
          s.pruned++
          continue
        }
        s.tested++
        const tp = pos.filter((e) => covered(h, e)).length
        const fp = neg.filter((e) => covered(h, e)).length
        const general = fp > 0
        const specific = tp < pos.length
        const outcome = general && specific ? 'both' : general ? 'too general' : specific ? 'too specific' : 'solution'
        s.last = { program: h.map((c) => clauseToString(c)), truePositives: tp, falsePositives: fp, outcome }
        if (outcome === 'solution') {
          s.program = h
          s.status = 'learned'
          return s
        }
        if (general) {
          s.generalOf.push(h)
          s.constraints++
        }
        if (specific) {
          s.specialOf.push(h)
          s.constraints++
        }
        return s
      }
    },
    done: (s) =>
      s.status === 'learned' ? 'converged' : s.status === 'failed' || s.tested >= maxHypotheses ? 'stopped' : undefined,
    observe: (s) => ({
      metrics: { tested: s.tested, pruned: s.pruned, constraints: s.constraints },
      messages: s.last === null ? [] : [{ kind: 'test', payload: s.last as unknown as Json }],
    }),
  }
}

/**
 * Learning from failures (Popper's loop) as a traced {@link Algorithm}: hypotheses (sets of clauses over the given
 * predicates, without singleton variables) are generated smallest first and tested on the examples. A hypothesis that
 * proves a negative example is too general, so every hypothesis that entails it (clause-wise subsumes it) is pruned;
 * one that misses a positive is too specific, so every hypothesis it entails is pruned. The first hypothesis that
 * proves every positive and no negative is returned, and it is of minimal size. Each step tests one hypothesis.
 *
 * @param task - Background, head and body predicates, and examples.
 * @param options - Clause and program size limits and budgets.
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` for a predicate not written `name/arity`.
 * @concept meta-interpretive-learning
 * @example
 * ```ts
 * import { learnFromFailuresAlgorithm } from 'whitebox-engine/logic/ilp'
 * learnFromFailuresAlgorithm({ background: 'q(a).', head: 'p/1', body: ['q/1'], positive: ['p(a)'], negative: [] }).name // => 'learning-from-failures'
 * ```
 */
export function learnFromFailuresAlgorithm(task: FailureTask, options: FailureOptions = {}): Algorithm<FailureState> {
  return failureMachine(task, options)
}

/**
 * Learn a smallest program by learning from failures.
 *
 * @param task - Background, head and body predicates, and examples.
 * @param options - Clause and program size limits, budgets and tracing.
 * @returns The status, the program (canonical variables) and the hypotheses tested.
 * @throws WhiteboxError with code `value` for a predicate not written `name/arity`.
 * @concept meta-interpretive-learning
 * @reference A. Cropper and R. Morel, Learning programs by learning from failures, Machine Learning 110 (2021).
 * @example
 * ```ts
 * import { learnFromFailures } from 'whitebox-engine/logic/ilp'
 * import { clauseToString } from 'whitebox-engine/logic'
 * const background = 'parent(ann, amy). parent(ann, andy). parent(amy, amelia). parent(andy, spongebob).'
 * const r = learnFromFailures({ background, head: 'grandparent/2', body: ['parent/2'], positive: ['grandparent(ann, amelia)', 'grandparent(ann, spongebob)'], negative: ['grandparent(amy, amelia)', 'grandparent(ann, amy)'] })
 * r.program.map((c) => clauseToString(c)) // => ['grandparent(A,B) :- parent(A,C), parent(C,B).']
 * ```
 */
export function learnFromFailures(task: FailureTask, options: FailureOptions = {}): MilResult {
  const { maxBody: _b, maxVars: _v, maxClauses: _c, maxHypotheses: _h, queryBudget: _q, ...run } = options
  const { state } = runAlgorithm(failureMachine(task, options), { ...run, maxSteps: Number.MAX_SAFE_INTEGER })
  return {
    status: state.status === 'learned' ? 'learned' : 'failed',
    program: (state.program ?? []).map(canonicalClause),
    candidates: state.tested,
  }
}
