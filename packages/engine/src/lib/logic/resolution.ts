import { type Json, WhiteboxError } from '../base/index.ts'
import { type Structure, type StructureNode, structure } from '../scene/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { addArgs, BUILTINS, copyTerm, type Engine, type MutableDb } from './_builtins.ts'
import { existenceError, instantiationError, PrologThrow, typeError } from './_errors.ts'
import { LIBRARY_SOURCE } from './_library.ts'
import { type ParsedProgram, parseProgram, parseQuery } from './parse.ts'
import {
  atom,
  type Clause,
  compareTerms,
  compound,
  indicator,
  list,
  num,
  type Operators,
  standardOperators,
  type Term,
  termToString,
  termVariables,
  type Var,
} from './terms.ts'
import { Bindings, isVariant, renameApart } from './unify.ts'

/**
 * A Prolog database: clauses by predicate indicator (`name/arity`), the predicates declared dynamic, and the operator
 * table the program was read with. Build one with {@link consult}.
 *
 * @concept logic-programming
 */
export interface Database {
  readonly predicates: ReadonlyMap<string, readonly Clause[]>
  readonly dynamic: ReadonlySet<string>
  readonly operators: Operators
  /** Directives other than `dynamic`, `discontiguous` and `op`, in order (they are recorded, not run). */
  readonly directives: readonly Term[]
}

const CONTROL: ReadonlySet<string> = new Set([
  'true/0',
  'fail/0',
  'false/0',
  '!/0',
  ',/2',
  ';/2',
  '->/2',
  '*->/2',
  '\\+/1',
  'not/1',
  'call/1',
  'call/2',
  'call/3',
  'call/4',
  'call/5',
  'call/6',
  'call/7',
  'call/8',
  'findall/3',
  'findall/4',
  'bagof/3',
  'setof/3',
  'catch/3',
  'throw/1',
])

const isBuiltinKey = (key: string): boolean => CONTROL.has(key) || BUILTINS.has(key)

const LIBRARY: ParsedProgram = parseProgram(LIBRARY_SOURCE)

/**
 * Names of the built-in predicates and control constructs, sorted.
 *
 * @example
 * ```ts
 * import { BUILTIN_PREDICATES } from 'whitebox-engine/logic'
 * [BUILTIN_PREDICATES.includes('findall/3'), BUILTIN_PREDICATES.includes('append/3')] // => [true, false]
 * ```
 */
export const BUILTIN_PREDICATES: readonly string[] = [...CONTROL, ...BUILTINS.keys()].sort()

/**
 * Names of the standard library predicates (defined in Prolog), sorted.
 *
 * @example
 * ```ts
 * import { LIBRARY_PREDICATES } from 'whitebox-engine/logic'
 * LIBRARY_PREDICATES.includes('append/3') // => true
 * ```
 */
export const LIBRARY_PREDICATES: readonly string[] = [
  ...new Set(LIBRARY.clauses.map((c) => indicator(c.head)).filter((k) => !k.startsWith('$'))),
].sort()

/** Options of {@link consult}. */
export interface ConsultOptions {
  /** Include the standard library (lists, apply, aggregate_all, phrase; default true). */
  readonly library?: boolean
  /** The initial operator table. */
  readonly operators?: Operators
  /** Add to an existing database instead of starting afresh. */
  readonly base?: Database
}

function predicateIndicators(t: Term): string[] {
  if (t.kind === 'compound' && (t.functor === ',' || t.functor === '.') && t.args.length === 2) {
    return [...predicateIndicators(t.args[0] as Term), ...predicateIndicators(t.args[1] as Term)]
  }
  if (t.kind === 'compound' && t.functor === '/' && t.args.length === 2) {
    const [n, a] = t.args as [Term, Term]
    if (n.kind === 'atom' && a.kind === 'num') return [`${n.name}/${a.value}`]
  }
  return []
}

/**
 * Load a Prolog program into a database. Clauses for a predicate defined by the standard library replace the library
 * definition (as user definitions override autoloaded ones in SWI-Prolog); `:- dynamic p/1.` declares a predicate that
 * may have no clauses yet.
 *
 * @param source - Program text, or parsed clauses.
 * @param options - Whether to include the library, the operator table, and a database to extend.
 * @returns The database.
 * @throws WhiteboxError with code `value` on a syntax error or a clause for a built-in predicate.
 * @concept logic-programming
 * @example
 * ```ts
 * import { consult } from 'whitebox-engine/logic'
 * const db = consult('parent(tom, bob). parent(bob, ann).')
 * db.predicates.get('parent/2').length // => 2
 * db.predicates.has('append/3') // => true
 * ```
 */
export function consult(source: string | readonly Clause[], options: ConsultOptions = {}): Database {
  const base = options.base
  const ops = options.operators ?? base?.operators ?? standardOperators()
  const parsed: ParsedProgram =
    typeof source === 'string'
      ? parseProgram(source, { operators: ops })
      : { clauses: source, directives: [], operators: ops }
  const preds = new Map<string, readonly Clause[]>()
  if (base !== undefined) for (const [k, v] of base.predicates) preds.set(k, v)
  else if (options.library ?? true) {
    for (const c of LIBRARY.clauses) {
      const k = indicator(c.head)
      preds.set(k, [...(preds.get(k) ?? []), c])
    }
  }
  const dynamic = new Set(base?.dynamic ?? [])
  const directives = [...(base?.directives ?? [])]
  const fresh = new Set<string>()
  for (const d of parsed.directives) {
    if (d.kind === 'compound' && d.functor === 'dynamic' && d.args.length === 1) {
      for (const k of predicateIndicators(d.args[0] as Term)) {
        dynamic.add(k)
        if (!preds.has(k)) preds.set(k, [])
      }
    } else if (!(d.kind === 'compound' && (d.functor === 'discontiguous' || d.functor === 'op'))) directives.push(d)
  }
  for (const c of parsed.clauses) {
    if (c.head.kind === 'var' || c.head.kind === 'num') {
      throw new WhiteboxError('value', `consult: clause head ${termToString(c.head)} is not callable`)
    }
    const k = indicator(c.head)
    if (isBuiltinKey(k)) throw new WhiteboxError('value', `consult: cannot redefine the built-in ${k}`)
    // The first user clause of a library predicate replaces the library definition.
    const old = fresh.has(k) ? (preds.get(k) ?? []) : base !== undefined ? (preds.get(k) ?? []) : []
    fresh.add(k)
    preds.set(k, [...old, c])
  }
  return { predicates: preds, dynamic, operators: parsed.operators, directives }
}

// ------------------------------------------------------------------------------------------------ the machine

/** A continuation item: a goal to run, or engine bookkeeping. */
type Item =
  | { readonly kind: 'goal'; readonly goal: Term; readonly cut: number; readonly depth: number }
  | { readonly kind: 'cut'; readonly to: number }
  | { readonly kind: 'softcut'; readonly cp: AltPoint }
  | { readonly kind: 'exit'; readonly goal: Term; readonly depth: number }
  | { readonly kind: 'collect'; readonly frame: Collector; readonly template: Term }
  | { readonly kind: 'catchexit'; readonly cp: CatchPoint }

interface Cont {
  readonly item: Item
  readonly next: Cont | null
}

interface Collector {
  readonly results: Term[]
}

interface ClausesPoint {
  readonly kind: 'clauses'
  readonly goal: Term
  readonly clauses: readonly Clause[]
  readonly index: number
  readonly next: Cont | null
  readonly trail: number
  readonly depth: number
  readonly node: number
}
interface AltPoint {
  readonly kind: 'alt'
  readonly cont: Cont | null
  readonly trail: number
  readonly node: number
  disabled: boolean
}
interface GenPoint {
  readonly kind: 'gen'
  readonly iter: Iterator<() => boolean>
  readonly goal: Term
  readonly next: Cont | null
  readonly trail: number
  readonly node: number
}
interface CatchPoint {
  readonly kind: 'catch'
  readonly catcher: Term
  readonly recovery: Term
  readonly next: Cont | null
  readonly trail: number
  readonly depth: number
  readonly node: number
  active: boolean
}
interface FindallPoint {
  readonly kind: 'findall'
  readonly frame: Collector
  readonly finish: (results: readonly Term[]) => Iterable<() => boolean>
  readonly goal: Term
  readonly next: Cont | null
  readonly trail: number
  readonly node: number
}
type ChoicePoint = ClausesPoint | AltPoint | GenPoint | CatchPoint | FindallPoint

/** A node of a recorded SLD tree: the resolvent reached, how, and whether it is a success or failure leaf. */
export interface SldNode {
  readonly id: number
  readonly parent: number | null
  /** The goals still to prove, written out (`[]` for the empty resolvent). */
  readonly goals: string
  /** What produced this node from its parent: a clause (`name/arity #k`), a built-in, or `fail`. */
  readonly via: string
  kind: 'goal' | 'success' | 'failure' | 'error'
}

/** An answer: the bindings of the query's variables and their SWI-style text (`X = a, Y = b`, or `true`). */
export interface Answer {
  readonly bindings: Readonly<Record<string, Term>>
  readonly text: string
}

/** What a step did, for the trace (Byrd's box ports plus solutions, cuts and exceptions). */
export interface SldEvent {
  readonly kind: 'call' | 'exit' | 'redo' | 'fail' | 'solution' | 'cut' | 'exception' | 'depth'
  readonly goal: Term | null
  readonly depth: number
  readonly via?: string
}

/** The state of a resolution run, as seen by observers. */
export interface SldState {
  readonly status: 'running' | 'exhausted' | 'stopped' | 'error'
  readonly answers: readonly Answer[]
  /** Calls resolved against clauses or built-ins. */
  readonly inferences: number
  readonly choicepoints: number
  readonly depth: number
  readonly depthLimited: boolean
  readonly error: Term | null
  readonly output: string
  /** What the last step did (several events when a failure backtracks into a choice point in the same step). */
  readonly events: readonly SldEvent[]
  /** The SLD tree so far, when recording. */
  readonly tree: readonly SldNode[] | undefined
  /** The database as modified by assert and retract. */
  readonly database: Database
}

interface Machine extends SldState {
  status: 'running' | 'exhausted' | 'stopped' | 'error'
  answers: Answer[]
  inferences: number
  choicepoints: number
  depth: number
  depthLimited: boolean
  error: Term | null
  output: string
  events: SldEvent[]
  tree: SldNode[] | undefined
  database: Database
  readonly b: Bindings
  readonly cps: ChoicePoint[]
  readonly db: MutableDb
  cont: Cont | null
  counter: number
  node: number
  backtrackNext: boolean
}

/** Options of {@link solve} and {@link sldAlgorithm}. */
export interface SolveOptions extends Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> {
  /** Stop after this many answers (default unlimited). */
  readonly maxSolutions?: number
  /** The step budget: the run stops with status `budget` after this many steps (default 1000000). */
  readonly maxSteps?: number
  /** Calls deeper than this fail, and the result records that the search was depth-limited (default 100000). */
  readonly maxDepth?: number
  /** Unify with the occurs check (default false, as in standard Prolog). */
  readonly occursCheck?: boolean
  /** What calling an undefined predicate does: raise an existence error (default) or fail. */
  readonly unknown?: 'error' | 'fail'
  /** Record the SLD tree (default false). */
  readonly recordTree?: boolean
}

const SLD_SCHEMA: TraceSchema = channels({
  metrics: {
    inferences: { description: 'calls resolved against clauses or built-ins' },
    choicepoints: { description: 'open choice points (alternatives left to try)' },
    depth: { description: 'depth of the current goal in the proof tree' },
    answers: { description: 'answers found so far' },
  },
  messages: {
    call: {
      description: 'a goal called (resolved against a clause or a built-in)',
      payload: { goal: 'goal', depth: 'depth', via: 'clause' },
    },
    exit: { description: 'a goal proved', payload: { goal: 'goal', depth: 'depth' } },
    redo: {
      description: 'backtracking into a goal for another clause',
      payload: { goal: 'goal', depth: 'depth', via: 'clause' },
    },
    fail: { description: 'a goal with no (more) matching clauses', payload: { goal: 'goal', depth: 'depth' } },
    solution: { description: 'an answer', payload: { answer: 'bindings' } },
    cut: { description: 'a cut removed choice points', payload: { removed: 'count' } },
    exception: { description: 'an exception raised or caught', payload: { ball: 'term', caught: 'boolean' } },
    depth: { description: 'a call beyond the depth limit failed', payload: { goal: 'goal', depth: 'depth' } },
  },
})

const FAIL: Term = atom('fail')

function pushGoals(goals: readonly Term[], cut: number, depth: number, tail: Cont | null): Cont | null {
  let c = tail
  for (let i = goals.length - 1; i >= 0; i--)
    c = { item: { kind: 'goal', goal: goals[i] as Term, cut, depth }, next: c }
  return c
}

function principal(t: Term): string {
  if (t.kind === 'compound') return `${t.functor}/${t.args.length}`
  if (t.kind === 'atom') return `a:${t.name}`
  if (t.kind === 'num') return `n:${t.value}:${t.float}`
  return ''
}

function snapshot(db: MutableDb, operators: Operators, directives: readonly Term[]): Database {
  return { predicates: new Map(db.preds), dynamic: new Set(db.dynamic), operators, directives }
}

function answerText(names: readonly string[], bindings: Readonly<Record<string, Term>>, ops: Operators): string {
  const parts = names
    .filter((n) => bindings[n] !== undefined)
    .map((n) => `${n} = ${termToString(bindings[n] as Term, { operators: ops })}`)
  return parts.length === 0 ? 'true' : parts.join(', ')
}

function queryOf(query: string | Term, ops: Operators): { goal: Term; names: string[] } {
  if (typeof query === 'string') {
    const q = parseQuery(query, { operators: ops })
    return { goal: q.goal, names: [...q.variables] }
  }
  return {
    goal: query,
    names: termVariables(query)
      .map((v) => v.name)
      .filter((n) => !/^_(G?\d+)$/.test(n)),
  }
}

function machine(db0: Database, query: string | Term, options: SolveOptions): Algorithm<Machine> {
  const maxSolutions = options.maxSolutions ?? Number.POSITIVE_INFINITY
  const maxDepth = options.maxDepth ?? 100_000
  const occursCheck = options.occursCheck ?? false
  const unknown = options.unknown ?? 'error'
  const record = options.recordTree ?? false
  const ops = db0.operators
  const { goal, names } = queryOf(query, ops)

  // Events carry the goal as instantiated when they happen.
  const emit = (s: Machine, e: SldEvent): void => {
    s.events.push(e.goal === null ? e : { ...e, goal: s.b.resolve(e.goal) })
  }
  const resolvent = (s: Machine): string => {
    const goals: string[] = []
    for (let c = s.cont; c !== null && goals.length <= 6; c = c.next) {
      if (c.item.kind === 'goal') goals.push(termToString(s.b.resolve(c.item.goal), { operators: ops }))
    }
    if (goals.length === 0) return '[]'
    return goals.length > 6 ? `${goals.slice(0, 6).join(', ')}, ...` : goals.join(', ')
  }
  const addNode = (s: Machine, parent: number, via: string, kind: SldNode['kind'] = 'goal'): void => {
    if (s.tree === undefined) return
    const id = s.tree.length
    s.tree.push({ id, parent, goals: kind === 'failure' ? 'fail' : resolvent(s), via, kind })
    if (kind === 'goal') s.node = id
  }

  const engine = (s: Machine): Engine => ({
    bindings: s.b,
    db: s.db,
    operators: ops,
    occursCheck,
    unify: (a, b) => s.b.unify(a, b, occursCheck),
    deref: (t) => s.b.deref(t),
    resolve: (t) => s.b.resolve(t),
    fresh: () => ({ kind: 'var', name: `_${s.counter++}` }),
    write: (text) => {
      s.output += text
    },
    isBuiltin: isBuiltinKey,
  })

  const mayMatch = (s: Machine, head: Term, g: Term): boolean => {
    if (head.kind !== 'compound' || g.kind !== 'compound') return true
    const h0 = head.args[0] as Term
    if (h0.kind === 'var') return true
    const g0 = s.b.deref(g.args[0] as Term)
    return g0.kind === 'var' || principal(h0) === principal(g0)
  }

  // Resolve a call against clauses from `start`; on success the continuation is the clause body.
  const tryClauses = (
    s: Machine,
    g: Term,
    clauses: readonly Clause[],
    start: number,
    next: Cont | null,
    depth: number,
    node: number,
    redo: boolean,
  ): boolean => {
    for (let i = start; i < clauses.length; i++) {
      const c = clauses[i] as Clause
      if (!mayMatch(s, c.head, g)) continue
      // The next candidate clause, found before unification binds the goal's arguments.
      let more = -1
      for (let j = i + 1; j < clauses.length; j++) {
        if (mayMatch(s, (clauses[j] as Clause).head, g)) {
          more = j
          break
        }
      }
      const mark = s.b.mark()
      const renamed = renameApart(c, s.counter)
      s.counter = renamed.next
      if (!s.b.unify(renamed.clause.head, g, occursCheck)) {
        s.b.undo(mark)
        continue
      }
      const barrier = s.cps.length
      if (more >= 0) s.cps.push({ kind: 'clauses', goal: g, clauses, index: more, next, trail: mark, depth, node })
      const exit: Cont = { item: { kind: 'exit', goal: g, depth }, next }
      s.cont = pushGoals(renamed.clause.body, barrier, depth, exit)
      s.inferences += 1
      s.depth = depth
      const via = `${indicator(g)} #${i + 1}`
      emit(s, { kind: redo ? 'redo' : 'call', goal: g, depth, via })
      if (record) {
        s.node = node
        addNode(s, node, via)
      }
      return true
    }
    emit(s, { kind: 'fail', goal: g, depth })
    if (record) addNode(s, node, `${indicator(g)}: no ${redo ? 'more ' : ''}clauses`, 'failure')
    return false
  }

  const backtrack = (s: Machine): void => {
    for (;;) {
      const cp = s.cps.pop()
      if (cp === undefined) {
        s.status = 'exhausted'
        return
      }
      s.b.undo(cp.trail)
      switch (cp.kind) {
        case 'clauses':
          if (tryClauses(s, cp.goal, cp.clauses, cp.index, cp.next, cp.depth, cp.node, true)) return
          break
        case 'alt':
          if (cp.disabled) break
          s.cont = cp.cont
          s.node = cp.node
          return
        case 'gen': {
          for (let r = cp.iter.next(); r.done !== true; r = cp.iter.next()) {
            if (r.value()) {
              s.cps.push(cp)
              s.cont = cp.next
              s.node = cp.node
              s.inferences += 1
              emit(s, { kind: 'redo', goal: cp.goal, depth: s.depth })
              if (record) addNode(s, cp.node, indicator(cp.goal))
              return
            }
            s.b.undo(cp.trail)
          }
          break
        }
        case 'catch':
          break
        case 'findall': {
          const alternatives = cp.finish(cp.frame.results)
          s.node = cp.node
          s.cps.push({
            kind: 'gen',
            iter: alternatives[Symbol.iterator](),
            goal: cp.goal,
            next: cp.next,
            trail: s.b.mark(),
            node: cp.node,
          })
          break
        }
      }
    }
  }

  const raise = (s: Machine, ball: Term): void => {
    const b = s.b.resolve(ball)
    emit(s, { kind: 'exception', goal: b, depth: s.depth })
    for (;;) {
      const cp = s.cps.pop()
      if (cp === undefined) {
        s.status = 'error'
        s.error = b
        if (s.tree !== undefined)
          s.tree.push({ id: s.tree.length, parent: s.node, goals: termToString(b), via: 'throw', kind: 'error' })
        return
      }
      if (cp.kind !== 'catch' || !cp.active) continue
      s.b.undo(cp.trail)
      const m = s.b.mark()
      if (s.b.unify(cp.catcher, b, occursCheck)) {
        s.cont = { item: { kind: 'goal', goal: cp.recovery, cut: s.cps.length, depth: cp.depth }, next: cp.next }
        s.node = cp.node
        emit(s, { kind: 'exception', goal: b, depth: cp.depth, via: 'caught' })
        return
      }
      s.b.undo(m)
    }
  }

  const callUser = (s: Machine, g: Term, depth: number, next: Cont | null): void => {
    const key = indicator(g)
    const clauses = s.db.preds.get(key)
    if (clauses === undefined) {
      if (unknown === 'fail' || s.db.dynamic.has(key)) {
        emit(s, { kind: 'fail', goal: g, depth })
        if (record) addNode(s, s.node, `${key}: undefined`, 'failure')
        backtrack(s)
        return
      }
      const slash = key.lastIndexOf('/')
      existenceError('procedure', compound('/', [atom(key.slice(0, slash)), num(Number(key.slice(slash + 1)))]), key)
    }
    if (depth + 1 > maxDepth) {
      s.depthLimited = true
      emit(s, { kind: 'depth', goal: g, depth })
      if (record) addNode(s, s.node, 'depth limit', 'failure')
      backtrack(s)
      return
    }
    if (!tryClauses(s, g, clauses ?? [], 0, next, depth + 1, s.node, false)) backtrack(s)
  }

  const run = (s: Machine, item: Extract<Item, { kind: 'goal' }>): void => {
    const g = s.b.deref(item.goal)
    const { cut, depth } = item
    const next = s.cont
    const goalItem = (t: Term, c: number, d = depth): Item => ({ kind: 'goal', goal: t, cut: c, depth: d })
    if (g.kind === 'var') instantiationError('call/1')
    if (g.kind === 'num') typeError('callable', g, 'call/1')
    const key = indicator(g)
    const args = g.kind === 'compound' ? g.args : []
    const [a0, a1, a2] = args as [Term, Term, Term]
    switch (key) {
      case 'true/0':
        return
      case 'fail/0':
      case 'false/0':
        emit(s, { kind: 'fail', goal: g, depth })
        backtrack(s)
        return
      case '!/0': {
        const removed = s.cps.length - cut
        if (removed > 0) s.cps.length = cut
        emit(s, { kind: 'cut', goal: null, depth, via: String(Math.max(removed, 0)) })
        return
      }
      case ',/2':
        s.cont = { item: goalItem(a0, cut), next: { item: goalItem(a1, cut), next } }
        return
      case ';/2': {
        const left = s.b.deref(a0)
        if (left.kind === 'compound' && left.args.length === 2 && (left.functor === '->' || left.functor === '*->')) {
          const [c, t] = left.args as [Term, Term]
          const b = s.cps.length
          const alt: AltPoint = {
            kind: 'alt',
            cont: { item: goalItem(a1, cut), next },
            trail: s.b.mark(),
            node: s.node,
            disabled: false,
          }
          s.cps.push(alt)
          const commit: Item = left.functor === '->' ? { kind: 'cut', to: b } : { kind: 'softcut', cp: alt }
          s.cont = { item: goalItem(c, b + 1), next: { item: commit, next: { item: goalItem(t, cut), next } } }
          return
        }
        s.cps.push({
          kind: 'alt',
          cont: { item: goalItem(a1, cut), next },
          trail: s.b.mark(),
          node: s.node,
          disabled: false,
        })
        s.cont = { item: goalItem(a0, cut), next }
        return
      }
      case '->/2': {
        const b = s.cps.length
        s.cps.push({
          kind: 'alt',
          cont: { item: goalItem(FAIL, cut), next },
          trail: s.b.mark(),
          node: s.node,
          disabled: false,
        })
        s.cont = {
          item: goalItem(a0, b + 1),
          next: { item: { kind: 'cut', to: b }, next: { item: goalItem(a1, cut), next } },
        }
        return
      }
      case '*->/2':
        s.cont = { item: goalItem(a0, s.cps.length), next: { item: goalItem(a1, cut), next } }
        return
      case '\\+/1':
      case 'not/1': {
        const b = s.cps.length
        s.cps.push({ kind: 'alt', cont: next, trail: s.b.mark(), node: s.node, disabled: false })
        s.cont = {
          item: goalItem(a0, b + 1),
          next: { item: { kind: 'cut', to: b }, next: { item: goalItem(FAIL, cut), next: null } },
        }
        return
      }
      case 'call/1':
      case 'call/2':
      case 'call/3':
      case 'call/4':
      case 'call/5':
      case 'call/6':
      case 'call/7':
      case 'call/8': {
        const target = addArgs(s.b.deref(a0), args.slice(1), key)
        s.cont = { item: goalItem(target, s.cps.length), next }
        return
      }
      case 'findall/3':
      case 'findall/4': {
        const frame: Collector = { results: [] }
        const result = a2
        const tail = key === 'findall/4' ? (args[3] as Term) : null
        s.cps.push({
          kind: 'findall',
          frame,
          goal: g,
          finish: (results) => [() => s.b.unify(result, list([...results], tail ?? atom('[]')), occursCheck)],
          next,
          trail: s.b.mark(),
          node: s.node,
        })
        s.cont = {
          item: goalItem(a1, s.cps.length),
          next: { item: { kind: 'collect', frame, template: a0 }, next: null },
        }
        return
      }
      case 'bagof/3':
      case 'setof/3': {
        let inner = s.b.deref(a1)
        const existential: string[] = []
        while (inner.kind === 'compound' && inner.functor === '^' && inner.args.length === 2) {
          for (const v of termVariables(s.b.resolve(inner.args[0] as Term))) existential.push(v.name)
          inner = s.b.deref(inner.args[1] as Term)
        }
        const templateVars = new Set(termVariables(s.b.resolve(a0)).map((v) => v.name))
        const witness: Var[] = termVariables(s.b.resolve(inner)).filter(
          (v) => !templateVars.has(v.name) && !existential.includes(v.name),
        )
        const w = list(witness)
        const frame: Collector = { results: [] }
        const isSet = key === 'setof/3'
        const finishList = (items: Term[]): Term => {
          if (!isSet) return list(items)
          const sorted = [...items].sort(compareTerms)
          return list(sorted.filter((t, i) => i === 0 || compareTerms(sorted[i - 1] as Term, t) !== 0))
        }
        s.cps.push({
          kind: 'findall',
          frame,
          goal: g,
          finish: (results) => {
            if (results.length === 0) return []
            const pairs = results.map((r) => (r as { args: readonly Term[] }).args as [Term, Term])
            if (witness.length === 0) return [() => s.b.unify(a2, finishList(pairs.map((p) => p[1])), occursCheck)]
            const sorted = pairs.map((p, i) => ({ p, i })).sort((x, y) => compareTerms(x.p[0], y.p[0]) || x.i - y.i)
            const groups: [Term, Term[]][] = []
            for (const { p } of sorted) {
              const last = groups[groups.length - 1]
              if (last !== undefined && isVariant(last[0], p[0])) last[1].push(p[1])
              else groups.push([p[0], [p[1]]])
            }
            return groups.map(
              ([key0, items]) =>
                () =>
                  s.b.unify(w, key0, occursCheck) && s.b.unify(a2, finishList(items), occursCheck),
            )
          },
          next,
          trail: s.b.mark(),
          node: s.node,
        })
        s.cont = {
          item: goalItem(inner, s.cps.length),
          next: { item: { kind: 'collect', frame, template: compound('-', [w, a0]) }, next: null },
        }
        return
      }
      case 'catch/3': {
        const cp: CatchPoint = {
          kind: 'catch',
          catcher: a1,
          recovery: a2,
          next,
          trail: s.b.mark(),
          depth,
          node: s.node,
          active: true,
        }
        s.cps.push(cp)
        s.cont = { item: goalItem(a0, s.cps.length), next: { item: { kind: 'catchexit', cp }, next } }
        return
      }
      case 'throw/1': {
        const ball = s.b.resolve(a0)
        if (ball.kind === 'var') instantiationError('throw/1')
        throw new PrologThrow(copyTerm(ball, engine(s)))
      }
    }
    const builtin = BUILTINS.get(key)
    if (builtin !== undefined) {
      const r = builtin(args, engine(s))
      s.inferences += 1
      if (r === true) {
        emit(s, { kind: 'call', goal: g, depth, via: 'built-in' })
        if (record) addNode(s, s.node, key)
        return
      }
      if (r === false) {
        emit(s, { kind: 'fail', goal: g, depth })
        if (record) addNode(s, s.node, key, 'failure')
        backtrack(s)
        return
      }
      const iter = r[Symbol.iterator]()
      const cp: GenPoint = { kind: 'gen', iter, goal: g, next, trail: s.b.mark(), node: s.node }
      for (let x = iter.next(); x.done !== true; x = iter.next()) {
        if (x.value()) {
          s.cps.push(cp)
          emit(s, { kind: 'call', goal: g, depth, via: 'built-in' })
          if (record) addNode(s, s.node, key)
          return
        }
        s.b.undo(cp.trail)
      }
      emit(s, { kind: 'fail', goal: g, depth })
      if (record) addNode(s, s.node, key, 'failure')
      backtrack(s)
      return
    }
    callUser(s, g, depth, next)
  }

  return {
    name: 'sld-resolution',
    config: { maxDepth, occursCheck, unknown, query: typeof query === 'string' ? query : termToString(query) },
    schema: SLD_SCHEMA,
    init: () => {
      const preds = new Map(db0.predicates)
      const db: MutableDb = { preds, dynamic: new Set(db0.dynamic) }
      const s: Machine = {
        status: 'running',
        answers: [],
        inferences: 0,
        choicepoints: 0,
        depth: 0,
        depthLimited: false,
        error: null,
        output: '',
        events: [],
        tree: record ? [] : undefined,
        database: db0,
        b: new Bindings(),
        cps: [],
        db,
        cont: { item: { kind: 'goal', goal, cut: 0, depth: 0 }, next: null },
        counter: 0,
        node: 0,
        backtrackNext: false,
      }
      if (s.tree !== undefined)
        s.tree.push({ id: 0, parent: null, goals: termToString(goal, { operators: ops }), via: 'query', kind: 'goal' })
      return s
    },
    step: (s) => {
      s.events = []
      try {
        if (s.backtrackNext) {
          s.backtrackNext = false
          backtrack(s)
          return s
        }
        const c = s.cont
        if (c === null) {
          const bindings: Record<string, Term> = {}
          for (const n of names) {
            const v = s.b.resolve({ kind: 'var', name: n })
            if (!(v.kind === 'var' && v.name === n)) bindings[n] = v
          }
          const answer: Answer = { bindings, text: answerText(names, bindings, ops) }
          s.answers.push(answer)
          emit(s, { kind: 'solution', goal: null, depth: 0, via: answer.text })
          if (s.tree !== undefined) {
            const node = s.tree[s.node]
            if (node !== undefined) node.kind = 'success'
          }
          if (s.answers.length >= maxSolutions) s.status = 'stopped'
          else s.backtrackNext = true
          return s
        }
        s.cont = c.next
        const it = c.item
        switch (it.kind) {
          case 'goal':
            run(s, it)
            break
          case 'cut':
            if (s.cps.length > it.to) s.cps.length = it.to
            break
          case 'softcut':
            it.cp.disabled = true
            break
          case 'exit':
            emit(s, { kind: 'exit', goal: it.goal, depth: it.depth })
            s.depth = Math.max(0, it.depth - 1)
            break
          case 'collect':
            it.frame.results.push(copyTerm(it.template, engine(s)))
            backtrack(s)
            break
          case 'catchexit': {
            const cp = it.cp
            cp.active = false
            s.b.onUndo(() => {
              cp.active = true
            })
            break
          }
        }
      } catch (e) {
        if (!(e instanceof PrologThrow)) throw e
        raise(s, e.ball)
      }
      s.choicepoints = s.cps.length
      if (s.status !== 'running') s.database = snapshot(s.db, ops, db0.directives)
      return s
    },
    done: (s) => (s.status === 'running' ? undefined : s.status === 'error' ? 'stopped' : 'converged'),
    observe: (s) => {
      const messages: { kind: string; payload: Json }[] = []
      for (const e of s.events) {
        const goalText = e.goal === null ? null : termToString(e.goal, { operators: ops })
        if (e.kind === 'solution') messages.push({ kind: 'solution', payload: { answer: e.via ?? 'true' } })
        else if (e.kind === 'cut') messages.push({ kind: 'cut', payload: { removed: Number(e.via ?? 0) } })
        else if (e.kind === 'exception')
          messages.push({ kind: 'exception', payload: { ball: goalText, caught: e.via === 'caught' } })
        else
          messages.push({
            kind: e.kind,
            payload: { goal: goalText, depth: e.depth, ...(e.via === undefined ? {} : { via: e.via }) },
          })
      }
      return {
        metrics: { inferences: s.inferences, choicepoints: s.cps.length, depth: s.depth, answers: s.answers.length },
        messages,
      }
    },
  }
}

function finalise(s: Machine, ops: Operators, directives: readonly Term[]): Machine {
  s.database = snapshot(s.db, ops, directives)
  return s
}

/**
 * SLD resolution as a traced {@link Algorithm}: Prolog's depth-first, left-to-right proof search with chronological
 * backtracking. Each step is one call (resolving the selected goal against the first matching clause or a built-in),
 * one exit, one backtrack into a choice point, or one answer; messages follow Byrd's box model (call, exit, redo,
 * fail) plus solutions, cuts and exceptions, so a figure can animate a proof. Calls beyond `maxDepth` fail and mark the
 * run depth-limited; the run's step budget bounds infinite derivations.
 *
 * Supported: cut (with call/N opaque to it), if-then-else and soft cut, negation as failure, disjunction,
 * findall/bagof/setof (with `^`), catch/throw with ISO error terms, arithmetic, term inspection, atoms, sorting, the
 * dynamic database (assert/retract, with the logical update view), output to a string, and the standard library.
 *
 * @param db - The database.
 * @param query - The query text or goal term.
 * @param options - Answers wanted, depth limit, occurs check, unknown-predicate behaviour and tree recording.
 * @returns The algorithm.
 * @concept sld-resolution
 * @example
 * ```ts
 * import { consult, sldAlgorithm } from 'whitebox-engine/logic'
 * sldAlgorithm(consult('p(1).'), 'p(X)').name // => 'sld-resolution'
 * ```
 */
export function sldAlgorithm(db: Database, query: string | Term, options: SolveOptions = {}): Algorithm<SldState> {
  return machine(db, query, options)
}

/** The outcome of {@link solve}. */
export interface SolveResult {
  /**
   * `success` (at least one answer), `failure` (none, search complete), `depth` (none, but the depth limit cut some
   * branches), `budget` (the step budget ran out first) or `error` (an uncaught exception).
   */
  readonly status: 'success' | 'failure' | 'depth' | 'budget' | 'error'
  readonly answers: readonly Answer[]
  /** Whether every branch was explored (false when stopped by `maxSolutions`, the budget or an error). */
  readonly exhausted: boolean
  readonly depthLimited: boolean
  /** The uncaught exception term, if any. */
  readonly error: Term | null
  /** Text written by write/1, format/2 and friends. */
  readonly output: string
  readonly steps: number
  readonly inferences: number
  /** The database after assert and retract. */
  readonly database: Database
  /** The SLD tree, when recorded. */
  readonly tree?: readonly SldNode[]
}

/**
 * Run a Prolog query to completion (or to `maxSolutions` answers) and collect its answers.
 *
 * @param program - A database, or program text to consult with the standard library.
 * @param query - The query text (`?- ...` optional) or a goal term.
 * @param options - Answers wanted, budgets, occurs check, unknown-predicate behaviour, tree recording and tracing.
 * @returns The status, answers, output and effort.
 * @concept sld-resolution
 * @example
 * ```ts
 * import { solve } from 'whitebox-engine/logic'
 * const family = 'parent(tom, bob). parent(tom, liz). parent(bob, ann). grandparent(X, Z) :- parent(X, Y), parent(Y, Z).'
 * solve(family, 'grandparent(tom, W)').answers.map((a) => a.text) // => ['W = ann']
 * solve('', 'append(X, Y, [1,2])').answers.map((a) => a.text) // => ['X = [], Y = [1,2]', 'X = [1], Y = [2]', 'X = [1,2], Y = []']
 * solve('', 'X is 7 / 2, Y is 7 // 2').answers[0].text // => 'X = 3.5, Y = 3'
 * solve('loop :- loop.', 'loop', { maxSteps: 1000 }).status // => 'budget'
 * solve('nat(0). nat(s(X)) :- nat(X).', 'nat(X), fail', { maxDepth: 5 }).status // => 'depth'
 * solve('', 'catch(X is foo + 1, error(E, _), true)').answers[0].text // => 'E = type_error(evaluable,foo/0)'
 * ```
 */
export function solve(program: Database | string, query: string | Term, options: SolveOptions = {}): SolveResult {
  const db = typeof program === 'string' ? consult(program) : program
  const {
    maxSteps = 1_000_000,
    maxSolutions: _m,
    maxDepth: _d,
    occursCheck: _o,
    unknown: _u,
    recordTree: _r,
    ...run
  } = options
  const algorithm = machine(db, query, options)
  const { state, steps } = runAlgorithm(algorithm, { ...run, maxSteps })
  finalise(state, db.operators, db.directives)
  const status: SolveResult['status'] =
    state.status === 'error'
      ? 'error'
      : state.status === 'running'
        ? 'budget'
        : state.answers.length > 0
          ? 'success'
          : state.depthLimited
            ? 'depth'
            : 'failure'
  return {
    status,
    answers: state.answers,
    exhausted: state.status === 'exhausted',
    depthLimited: state.depthLimited,
    error: state.error,
    output: state.output,
    steps,
    inferences: state.inferences,
    database: state.database,
    ...(state.tree === undefined ? {} : { tree: state.tree }),
  }
}

/**
 * Whether a goal has at least one proof (stopping at the first answer).
 *
 * @param program - A database or program text.
 * @param query - The goal.
 * @param options - Budgets and flags as for {@link solve}.
 * @returns True when provable.
 * @throws WhiteboxError with code `convergence` when the step budget runs out first, and code `value` for an uncaught
 *   Prolog exception.
 * @example
 * ```ts
 * import { prove } from 'whitebox-engine/logic'
 * [prove('', 'member(2, [1,2,3])'), prove('', '\\+ member(4, [1,2,3])'), prove('', 'member(4, [1,2,3])')] // => [true, true, false]
 * ```
 */
export function prove(program: Database | string, query: string | Term, options: SolveOptions = {}): boolean {
  const r = solve(program, query, { ...options, maxSolutions: 1 })
  if (r.status === 'budget') throw new WhiteboxError('convergence', 'prove: the step budget ran out')
  if (r.status === 'error') {
    throw new WhiteboxError('value', `prove: uncaught exception ${termToString(r.error ?? atom('error'))}`)
  }
  return r.answers.length > 0
}

/**
 * The SLD tree of a run recorded with `recordTree: true`, as a {@link Structure}: one node per resolvent, edges labelled
 * by the clause or built-in used, success leaves marked as outputs and failure leaves as leaves.
 *
 * @param result - A solve result with its tree.
 * @returns The structure.
 * @throws WhiteboxError with code `value` when the run did not record its tree.
 * @concept sld-resolution
 * @example
 * ```ts
 * import { sldTree, solve } from 'whitebox-engine/logic'
 * const r = solve('p(1). p(2). q(2).', 'p(X), q(X)', { recordTree: true })
 * sldTree(r).nodes.map((n) => n.label) // => ['p(X),q(X)', 'q(1)', 'fail', 'q(2)', '[]']
 * ```
 */
export function sldTree(result: SolveResult): Structure {
  const tree = result.tree
  if (tree === undefined) throw new WhiteboxError('value', 'sldTree: solve with recordTree: true')
  const nodes: StructureNode[] = tree.map((n) => ({
    id: `n${n.id}`,
    label: n.goals,
    role: n.parent === null ? 'input' : n.kind === 'success' ? 'output' : n.kind === 'goal' ? 'state' : 'leaf',
    detail: n.via,
    data: { kind: n.kind },
  }))
  const edges = tree.flatMap((n) =>
    n.parent === null ? [] : [{ from: `n${n.parent}`, to: `n${n.id}`, directed: true, label: n.via }],
  )
  const annotations = tree
    .filter((n) => n.kind === 'success' || n.kind === 'error')
    .map((n) => ({
      target: `n${n.id}`,
      text: n.kind === 'success' ? 'success' : 'exception',
      kind: 'highlight' as const,
    }))
  return structure('sld-tree', nodes, edges, { direction: 'down', annotations })
}

/**
 * Write an uncaught error or answer term as SWI-Prolog does in messages (a convenience for displaying results).
 *
 * @param t - The term.
 * @param db - The database whose operators to use (default the standard table).
 * @returns The text.
 * @example
 * ```ts
 * import { formatTerm, solve } from 'whitebox-engine/logic'
 * formatTerm(solve('', 'X is 1 / 0').error) // => 'error(evaluation_error(zero_divisor),context((is)/2,_))'
 * ```
 */
export function formatTerm(t: Term | null, db?: Database): string {
  if (t === null) return ''
  return termToString(t, { operators: db?.operators ?? standardOperators() })
}
