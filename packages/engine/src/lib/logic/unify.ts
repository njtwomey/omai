import { type Json, WhiteboxError } from '../base/index.ts'
import { type Algorithm, channels, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { type Clause, type Compound, type Term, termToString, termVariables, type Var } from './terms.ts'

/**
 * A substitution: variable names bound to terms. Bindings may be triangular (a bound term can mention other bound
 * variables); {@link applySubstitution} resolves them fully.
 *
 * @concept unification
 */
export type Substitution = ReadonlyMap<string, Term>

/** The empty substitution. */
export const EMPTY_SUBSTITUTION: Substitution = new Map()

/** A store of bindings that unification reads and extends (a substitution map, or the trailing {@link Bindings}). */
export interface BindingStore {
  get(name: string): Term | undefined
  bind(name: string, t: Term): void
}

/**
 * Follow variable bindings from a term until an unbound variable or a non-variable is reached.
 *
 * @param t - The term.
 * @param s - The substitution.
 * @returns The dereferenced term (only its top is resolved).
 * @example
 * ```ts
 * import { parseTerm, termToString, walk } from 'whitebox-engine/logic'
 * const s = new Map([['X', parseTerm('Y')], ['Y', parseTerm('f(Z)')]])
 * termToString(walk(parseTerm('X'), s)) // => 'f(Z)'
 * ```
 */
export function walk(t: Term, s: Substitution | BindingStore): Term {
  let cur = t
  while (cur.kind === 'var') {
    const next = s.get(cur.name)
    if (next === undefined) return cur
    cur = next
  }
  return cur
}

/**
 * Apply a substitution to a term, resolving bindings fully (cycles from unification without the occurs check are
 * written as the variable itself at the point of recurrence).
 *
 * @param t - The term.
 * @param s - The substitution.
 * @returns The instantiated term.
 * @example
 * ```ts
 * import { applySubstitution, parseTerm, termToString } from 'whitebox-engine/logic'
 * termToString(applySubstitution(parseTerm('p(X, Y)'), new Map([['X', parseTerm('Y')], ['Y', parseTerm('a')]]))) // => 'p(a,a)'
 * ```
 */
export function applySubstitution(t: Term, s: Substitution | BindingStore): Term {
  const active = new Set<string>()
  const go = (x: Term): Term => {
    if (x.kind === 'var') {
      const b = s.get(x.name)
      if (b === undefined || active.has(x.name)) return x
      active.add(x.name)
      const r = go(b)
      active.delete(x.name)
      return r
    }
    if (x.kind !== 'compound') return x
    let changed = false
    const args = x.args.map((a) => {
      const r = go(a)
      if (r !== a) changed = true
      return r
    })
    return changed ? { kind: 'compound', functor: x.functor, args } : x
  }
  return go(t)
}

/**
 * Whether a variable occurs in a term under a substitution (the occurs check).
 *
 * @param name - The variable's name.
 * @param t - The term.
 * @param s - The substitution (default empty).
 * @returns True when the variable occurs.
 * @example
 * ```ts
 * import { occursIn, parseTerm } from 'whitebox-engine/logic'
 * [occursIn('X', parseTerm('f(g(X))')), occursIn('X', parseTerm('f(Y)'))] // => [true, false]
 * ```
 */
export function occursIn(name: string, t: Term, s: Substitution | BindingStore = EMPTY_SUBSTITUTION): boolean {
  const stack: Term[] = [t]
  const seen = new Set<string>()
  for (let x = stack.pop(); x !== undefined; x = stack.pop()) {
    const y = walk(x, s)
    if (y.kind === 'var') {
      if (y.name === name) return true
    } else if (y.kind === 'compound') {
      for (const a of y.args) {
        if (a.kind === 'var') {
          if (seen.has(a.name)) continue
          seen.add(a.name)
        }
        stack.push(a)
      }
    }
  }
  return false
}

/** Options of unification. */
export interface UnifyOptions {
  /** Refuse to bind a variable to a term containing it (default false, as in standard Prolog). */
  readonly occursCheck?: boolean
}

// Unify two terms into a store, iteratively with an explicit stack of pairs. Returns false (with the store possibly
// partly extended) on failure. Shared by the persistent `unify` and the trailing `Bindings` of the resolution engine.
function unifyInto(store: BindingStore, a: Term, b: Term, occursCheck: boolean): boolean {
  const stack: Term[] = [a, b]
  while (stack.length > 0) {
    const y = walk(stack.pop() as Term, store)
    const x = walk(stack.pop() as Term, store)
    if (x === y) continue
    if (x.kind === 'var') {
      if (y.kind === 'var' && x.name === y.name) continue
      if (occursCheck && occursIn(x.name, y, store)) return false
      store.bind(x.name, y)
      continue
    }
    if (y.kind === 'var') {
      if (occursCheck && occursIn(y.name, x, store)) return false
      store.bind(y.name, x)
      continue
    }
    if (x.kind === 'atom') {
      if (y.kind !== 'atom' || x.name !== y.name) return false
      continue
    }
    if (x.kind === 'num') {
      if (y.kind !== 'num' || x.value !== y.value || x.float !== y.float) {
        if (!(y.kind === 'num' && Number.isNaN(x.value) && Number.isNaN(y.value) && x.float === y.float)) return false
      }
      continue
    }
    if (y.kind !== 'compound' || x.functor !== y.functor || x.args.length !== y.args.length) return false
    for (let i = x.args.length - 1; i >= 0; i--) stack.push(x.args[i] as Term, y.args[i] as Term)
  }
  return true
}

/**
 * Unify two terms, extending a substitution (Robinson unification, iterative). Variables are bound to the other side
 * as found, so the result is triangular; {@link mgu} returns the resolved most general unifier.
 *
 * @param a - A term.
 * @param b - Another term.
 * @param s - The substitution to extend (default empty; it is not modified).
 * @param options - Whether to perform the occurs check.
 * @returns The extended substitution, or null when the terms do not unify.
 * @concept unification
 * @example
 * ```ts
 * import { applySubstitution, parseTerm, termToString, unify } from 'whitebox-engine/logic'
 * const s = unify(parseTerm('f(X, g(Y))'), parseTerm('f(a, g(X))'))
 * termToString(applySubstitution(parseTerm('Y'), s)) // => 'a'
 * unify(parseTerm('f(a)'), parseTerm('f(b)')) // => null
 * unify(parseTerm('X'), parseTerm('f(X)'), undefined, { occursCheck: true }) // => null
 * ```
 */
export function unify(
  a: Term,
  b: Term,
  s: Substitution = EMPTY_SUBSTITUTION,
  options: UnifyOptions = {},
): Substitution | null {
  const m = new Map(s)
  const store: BindingStore = { get: (n) => m.get(n), bind: (n, t) => m.set(n, t) }
  return unifyInto(store, a, b, options.occursCheck ?? false) ? m : null
}

/**
 * A mutable store of variable bindings with a trail, for search procedures that bind and undo many times (the
 * resolution engine): {@link Bindings.unify} binds, {@link Bindings.mark} and {@link Bindings.undo} restore. Undo
 * entries may also be functions, run when the trail is unwound past them.
 *
 * @example
 * ```ts
 * import { Bindings, parseTerm, termToString } from 'whitebox-engine/logic'
 * const b = new Bindings()
 * const m = b.mark()
 * b.unify(parseTerm('f(X)'), parseTerm('f(a)'))
 * const bound = termToString(b.resolve(parseTerm('X')))
 * b.undo(m)
 * [bound, termToString(b.resolve(parseTerm('X')))] // => ['a', 'X']
 * ```
 */
export class Bindings implements BindingStore {
  private readonly map = new Map<string, Term>()
  private readonly trail: (string | (() => void))[] = []

  /** The binding of a variable, if any. */
  get(name: string): Term | undefined {
    return this.map.get(name)
  }

  /** Bind a variable (recorded on the trail). */
  bind(name: string, t: Term): void {
    this.map.set(name, t)
    this.trail.push(name)
  }

  /** Record an action to run when the trail is unwound past this point. */
  onUndo(f: () => void): void {
    this.trail.push(f)
  }

  /** The current trail height. */
  mark(): number {
    return this.trail.length
  }

  /** Unwind the trail to a mark, unbinding variables and running undo actions. */
  undo(mark: number): void {
    while (this.trail.length > mark) {
      const e = this.trail.pop()
      if (typeof e === 'string') this.map.delete(e)
      else if (e !== undefined) e()
    }
  }

  /** Unify two terms under the current bindings; on failure the partial bindings stay until the caller undoes. */
  unify(a: Term, b: Term, occursCheck = false): boolean {
    return unifyInto(this, a, b, occursCheck)
  }

  /** Dereference the top of a term. */
  deref(t: Term): Term {
    return walk(t, this)
  }

  /** Apply the bindings to a term fully. */
  resolve(t: Term): Term {
    return applySubstitution(t, this)
  }

  /** The number of bound variables. */
  get size(): number {
    return this.map.size
  }
}

/**
 * The most general unifier of two terms, fully resolved (idempotent: no bound variable occurs in a binding), with
 * the occurs check on by default.
 *
 * @param a - A term.
 * @param b - Another term.
 * @param options - The occurs check (default true here).
 * @returns The unifier, or null.
 * @concept unification
 * @example
 * ```ts
 * import { mgu, parseTerm, substitutionToString } from 'whitebox-engine/logic'
 * substitutionToString(mgu(parseTerm('p(X, f(Y))'), parseTerm('p(g(Z), f(a))'))) // => '{X/g(Z), Y/a}'
 * ```
 */
export function mgu(a: Term, b: Term, options: UnifyOptions = {}): Substitution | null {
  const s = unify(a, b, EMPTY_SUBSTITUTION, { occursCheck: options.occursCheck ?? true })
  return s === null ? null : resolveSubstitution(s)
}

/**
 * Resolve a triangular substitution: apply it to each of its own bindings and drop trivial `X/X` bindings.
 *
 * @param s - The substitution.
 * @returns An idempotent substitution with the same effect.
 * @example
 * ```ts
 * import { parseTerm, resolveSubstitution, substitutionToString } from 'whitebox-engine/logic'
 * substitutionToString(resolveSubstitution(new Map([['X', parseTerm('f(Y)')], ['Y', parseTerm('a')]]))) // => '{X/f(a), Y/a}'
 * ```
 */
export function resolveSubstitution(s: Substitution): Substitution {
  const out = new Map<string, Term>()
  for (const name of s.keys()) {
    const t = applySubstitution({ kind: 'var', name }, s)
    if (!(t.kind === 'var' && t.name === name)) out.set(name, t)
  }
  return out
}

/**
 * Compose substitutions: applying the result equals applying `first` and then `second` (`t(first∘second)`).
 *
 * @param first - Applied first.
 * @param second - Applied second.
 * @returns The composition.
 * @example
 * ```ts
 * import { applySubstitution, composeSubstitutions, parseTerm, termToString } from 'whitebox-engine/logic'
 * const c = composeSubstitutions(new Map([['X', parseTerm('f(Y)')]]), new Map([['Y', parseTerm('a')], ['X', parseTerm('b')]]))
 * termToString(applySubstitution(parseTerm('g(X, Y)'), c)) // => 'g(f(a),a)'
 * ```
 */
export function composeSubstitutions(first: Substitution, second: Substitution): Substitution {
  const out = new Map<string, Term>()
  for (const [k, v] of first) {
    const t = applySubstitution(v, second)
    if (!(t.kind === 'var' && t.name === k)) out.set(k, t)
  }
  for (const [k, v] of second) if (!first.has(k)) out.set(k, v)
  return out
}

/**
 * Write a substitution as `{X/t, ...}`, bindings sorted by variable name.
 *
 * @param s - The substitution (null writes as `fail`).
 * @returns The text.
 * @example
 * ```ts
 * import { parseTerm, substitutionToString } from 'whitebox-engine/logic'
 * substitutionToString(new Map([['Y', parseTerm('b')], ['X', parseTerm('a')]])) // => '{X/a, Y/b}'
 * substitutionToString(null) // => 'fail'
 * ```
 */
export function substitutionToString(s: Substitution | null): string {
  if (s === null) return 'fail'
  const parts = [...s.entries()]
    .sort((p, q) => (p[0] < q[0] ? -1 : p[0] > q[0] ? 1 : 0))
    .map(([k, v]) => `${k}/${termToString(v)}`)
  return `{${parts.join(', ')}}`
}

/**
 * Rename every variable of a term with a function of its name (renaming apart).
 *
 * @param t - The term.
 * @param rename - The new name of each variable.
 * @returns The renamed term.
 * @example
 * ```ts
 * import { parseTerm, renameTerm, termToString } from 'whitebox-engine/logic'
 * termToString(renameTerm(parseTerm('f(X, g(Y, X))'), (n) => `${n}_1`)) // => 'f(X_1,g(Y_1,X_1))'
 * ```
 */
export function renameTerm(t: Term, rename: (name: string) => string): Term {
  if (t.kind === 'var') return { kind: 'var', name: rename(t.name) }
  if (t.kind !== 'compound') return t
  return { kind: 'compound', functor: t.functor, args: t.args.map((a) => renameTerm(a, rename)) }
}

/**
 * Rename a clause's variables apart from all others: each distinct variable becomes `_<n>` for consecutive `n`
 * starting at `start` (the convention of the resolution engine's fresh variables).
 *
 * @param c - The clause.
 * @param start - The first fresh number.
 * @returns The renamed clause and the next free number.
 * @example
 * ```ts
 * import { clauseToString, parseClause, renameApart } from 'whitebox-engine/logic'
 * const r = renameApart(parseClause('p(X, Y) :- q(Y, X).'), 7)
 * [clauseToString(r.clause), r.next] // => ['p(_7,_8) :- q(_8,_7).', 9]
 * ```
 */
export function renameApart(c: Clause, start: number): { readonly clause: Clause; readonly next: number } {
  const names = new Map<string, string>()
  let next = start
  const rename = (n: string): string => {
    let r = names.get(n)
    if (r === undefined) {
      r = `_${next++}`
      names.set(n, r)
    }
    return r
  }
  const head = renameTerm(c.head, rename)
  const body = c.body.map((g) => renameTerm(g, rename))
  return { clause: { head, body }, next }
}

/**
 * One-way matching: a substitution of the pattern's variables that makes it identical to the term (the term's
 * variables are treated as constants).
 *
 * @param pattern - The general term.
 * @param t - The instance.
 * @param s - Bindings to extend (default empty).
 * @returns The matcher, or null.
 * @example
 * ```ts
 * import { matchTerm, parseTerm, substitutionToString } from 'whitebox-engine/logic'
 * substitutionToString(matchTerm(parseTerm('f(X, Y)'), parseTerm('f(a, Z)'))) // => '{X/a, Y/Z}'
 * matchTerm(parseTerm('f(a)'), parseTerm('f(X)')) // => null
 * ```
 */
export function matchTerm(pattern: Term, t: Term, s: Substitution = EMPTY_SUBSTITUTION): Substitution | null {
  const m = new Map(s)
  const stack: [Term, Term][] = [[pattern, t]]
  for (let pair = stack.pop(); pair !== undefined; pair = stack.pop()) {
    const [p, x] = pair
    if (p.kind === 'var') {
      const b = m.get(p.name)
      if (b === undefined) m.set(p.name, x)
      else if (!identical(b, x)) return null
      continue
    }
    if (p.kind === 'atom') {
      if (x.kind !== 'atom' || x.name !== p.name) return null
      continue
    }
    if (p.kind === 'num') {
      if (x.kind !== 'num' || x.value !== p.value || x.float !== p.float) return null
      continue
    }
    if (x.kind !== 'compound' || x.functor !== p.functor || x.args.length !== p.args.length) return null
    for (let i = 0; i < p.args.length; i++) stack.push([p.args[i] as Term, x.args[i] as Term])
  }
  return m
}

function identical(a: Term, b: Term): boolean {
  if (a.kind !== b.kind) return false
  if (a.kind === 'var') return a.name === (b as Var).name
  if (a.kind === 'atom') return a.name === (b as typeof a).name
  if (a.kind === 'num') return a.value === (b as typeof a).value && a.float === (b as typeof a).float
  const y = b as Compound
  return (
    a.functor === y.functor &&
    a.args.length === y.args.length &&
    a.args.every((x, i) => identical(x, y.args[i] as Term))
  )
}

/**
 * Whether two terms are variants: equal up to a consistent one-to-one renaming of variables.
 *
 * @param a - A term.
 * @param b - Another term.
 * @returns True for variants.
 * @example
 * ```ts
 * import { isVariant, parseTerm } from 'whitebox-engine/logic'
 * [isVariant(parseTerm('f(X,Y,X)'), parseTerm('f(A,B,A)')), isVariant(parseTerm('f(X,Y)'), parseTerm('f(A,A)'))] // => [true, false]
 * ```
 */
export function isVariant(a: Term, b: Term): boolean {
  const ab = matchTerm(a, b)
  if (ab === null) return false
  const images = new Set<string>()
  for (const v of ab.values()) {
    if (v.kind !== 'var' || images.has(v.name)) return false
    images.add(v.name)
  }
  return true
}

// ------------------------------------------------------------------------------------------------ Martelli-Montanari

/** An equation `left = right` of a unification problem. */
export interface Equation {
  readonly left: Term
  readonly right: Term
}

/** The Martelli-Montanari rule applied in a step. */
export type MMRule = 'delete' | 'decompose' | 'conflict' | 'swap' | 'eliminate' | 'check' | 'solved'

/** One step of the Martelli-Montanari algorithm: the rule, the equation it acted on, and the equations after it. */
export interface MMStep {
  readonly rule: MMRule
  readonly equation: Equation | null
  readonly equations: readonly Equation[]
}

/** The state of a traced Martelli-Montanari run. */
export interface MMState {
  /** Unsolved equations. */
  readonly pending: readonly Equation[]
  /** Solved equations `X = t`, each variable occurring nowhere else. */
  readonly solved: readonly Equation[]
  readonly status: 'running' | 'unified' | 'clash' | 'occurs'
  readonly last: MMStep | null
}

const MM_SCHEMA: TraceSchema = channels({
  metrics: {
    pending: { description: 'equations still to process' },
    solved: { description: 'equations in solved form' },
  },
  messages: {
    rule: {
      description: 'the rule applied to an equation',
      payload: { rule: 'delete, decompose, conflict, swap, eliminate or check', equation: 'the equation' },
    },
  },
})

function substituteEq(e: Equation, name: string, t: Term): Equation {
  const s = new Map([[name, t]])
  return { left: applySubstitution(e.left, s), right: applySubstitution(e.right, s) }
}

function mmStep(state: MMState): MMState {
  const [e, ...rest] = state.pending
  if (e === undefined) return { ...state, status: 'unified', last: { rule: 'solved', equation: null, equations: [] } }
  const { left: l, right: r } = e
  const done = (
    rule: MMRule,
    pending: Equation[],
    solved: readonly Equation[],
    status: MMState['status'],
  ): MMState => ({
    pending,
    solved,
    status,
    last: { rule, equation: e, equations: [...pending, ...solved] },
  })
  if (l.kind === 'var' && r.kind === 'var' && l.name === r.name) return done('delete', rest, state.solved, 'running')
  if (l.kind !== 'var' && r.kind === 'var')
    return done('swap', [{ left: r, right: l }, ...rest], state.solved, 'running')
  if (l.kind === 'var') {
    if (occursIn(l.name, r)) return done('check', rest, state.solved, 'occurs')
    const pending = rest.map((x) => substituteEq(x, l.name, r))
    const solved = [...state.solved.map((x) => substituteEq(x, l.name, r)), e]
    return done('eliminate', pending, solved, 'running')
  }
  if (l.kind === 'compound' && r.kind === 'compound') {
    if (l.functor !== r.functor || l.args.length !== r.args.length) return done('conflict', rest, state.solved, 'clash')
    const args = l.args.map((a, i) => ({ left: a, right: r.args[i] as Term }))
    return done('decompose', [...args, ...rest], state.solved, 'running')
  }
  const same =
    (l.kind === 'atom' && r.kind === 'atom' && l.name === r.name) ||
    (l.kind === 'num' && r.kind === 'num' && l.value === r.value && l.float === r.float)
  return same ? done('delete', rest, state.solved, 'running') : done('conflict', rest, state.solved, 'clash')
}

/**
 * The Martelli-Montanari unification algorithm as a traced {@link Algorithm}: each step applies one rule (delete,
 * decompose, conflict, swap, eliminate, occurs check) to the first pending equation, so a figure can animate how a
 * set of equations reaches solved form. The occurs check is part of the algorithm: `X = f(X)` fails with `occurs`.
 *
 * @param equations - The equations to solve.
 * @returns The algorithm.
 * @concept unification
 * @example
 * ```ts
 * import { parseTerm, unificationAlgorithm } from 'whitebox-engine/logic'
 * unificationAlgorithm([{ left: parseTerm('f(X)'), right: parseTerm('f(a)') }]).name // => 'martelli-montanari'
 * ```
 */
export function unificationAlgorithm(equations: readonly Equation[]): Algorithm<MMState> {
  return {
    name: 'martelli-montanari',
    config: { equations: equations.length },
    schema: MM_SCHEMA,
    init: () => ({ pending: [...equations], solved: [], status: 'running', last: null }),
    step: (s) => mmStep(s),
    done: (s) => (s.status === 'unified' ? 'converged' : s.status === 'running' ? undefined : 'stopped'),
    observe: (s) => ({
      metrics: { pending: s.pending.length, solved: s.solved.length },
      messages:
        s.last === null || s.last.equation === null
          ? []
          : [
              {
                kind: 'rule',
                payload: {
                  rule: s.last.rule,
                  equation: `${termToString(s.last.equation.left)} = ${termToString(s.last.equation.right)}`,
                } as Json,
              },
            ],
    }),
  }
}

/** The outcome of {@link martelliMontanari}. */
export interface MMResult {
  /** The most general unifier (solved form), or null on a clash or occurs-check failure. */
  readonly unifier: Substitution | null
  readonly status: 'unified' | 'clash' | 'occurs' | 'budget'
  /** Every rule application in order. */
  readonly steps: readonly MMStep[]
}

/**
 * Solve unification equations by the Martelli-Montanari rules, recording each step.
 *
 * @param a - A term, or a list of equations.
 * @param b - The other term (when `a` is a term).
 * @param options - A step budget (default 10000).
 * @returns The unifier, the status and the steps.
 * @throws WhiteboxError with code `value` when a term is given without its partner.
 * @concept unification
 * @example
 * ```ts
 * import { martelliMontanari, parseTerm, substitutionToString } from 'whitebox-engine/logic'
 * const r = martelliMontanari(parseTerm('f(X, g(X))'), parseTerm('f(a, Y)'))
 * substitutionToString(r.unifier) // => '{X/a, Y/g(a)}'
 * r.steps.map((s) => s.rule) // => ['decompose', 'eliminate', 'swap', 'eliminate', 'solved']
 * martelliMontanari(parseTerm('X'), parseTerm('f(X)')).status // => 'occurs'
 * ```
 */
export function martelliMontanari(
  a: Term | readonly Equation[],
  b?: Term,
  options: { readonly maxSteps?: number } = {},
): MMResult {
  let equations: readonly Equation[]
  if (Array.isArray(a)) equations = a as readonly Equation[]
  else {
    if (b === undefined) throw new WhiteboxError('value', 'martelliMontanari: two terms or a list of equations')
    equations = [{ left: a as Term, right: b }]
  }
  const steps: MMStep[] = []
  const algorithm = unificationAlgorithm(equations)
  const recording: Algorithm<MMState> = {
    ...algorithm,
    step: (s, ctx) => {
      const n = algorithm.step(s, ctx)
      if (n.last !== null) steps.push(n.last)
      return n
    },
  }
  const { state } = runAlgorithm(recording, { maxSteps: options.maxSteps ?? 10_000 })
  const status = state.status === 'running' ? 'budget' : state.status
  const unifier =
    status === 'unified' ? new Map(state.solved.map((e) => [(e.left as Var).name, e.right] as [string, Term])) : null
  return { unifier, status, steps }
}

/**
 * The variables bound by a substitution that also occur in a term (to restrict an answer to a query's variables).
 *
 * @param s - The substitution.
 * @param t - The term whose variables to keep.
 * @returns The restricted, resolved substitution.
 * @example
 * ```ts
 * import { parseTerm, restrictSubstitution, substitutionToString } from 'whitebox-engine/logic'
 * substitutionToString(restrictSubstitution(new Map([['X', parseTerm('f(Y)')], ['Y', parseTerm('b')]]), parseTerm('p(X)'))) // => '{X/f(b)}'
 * ```
 */
export function restrictSubstitution(s: Substitution, t: Term): Substitution {
  const out = new Map<string, Term>()
  for (const v of termVariables(t)) {
    const r = applySubstitution(v, s)
    if (!(r.kind === 'var' && r.name === v.name)) out.set(v.name, r)
  }
  return out
}
