/**
 * First-order logic and automated theorem proving by resolution: parsing and printing formulas with quantifiers,
 * conversion to clausal form (negation normal form, Skolemisation), clause subsumption, and an Otter-style given-clause
 * prover as a traced algorithm. Proofs can be checked and drawn as structures, and small clause sets can be tested for
 * finite models.
 */

import { type Json, WhiteboxError } from '../base/index.ts'
import { type Structure, type StructureNode, structure } from '../scene/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { atom, compound, type Term, termSize, termToString, termVariables, variable } from './terms.ts'
import { applySubstitution, matchTerm, mgu, renameTerm, type Substitution } from './unify.ts'

/**
 * A first-order formula: atoms (predicate applications over terms), connectives and quantifiers. Equality, when used,
 * is an ordinary predicate `=` (no built-in equality reasoning).
 *
 * @concept first-order-logic
 */
export type FolFormula =
  | { readonly kind: 'atom'; readonly atom: Term }
  | { readonly kind: 'const'; readonly value: boolean }
  | { readonly kind: 'not'; readonly arg: FolFormula }
  | { readonly kind: 'and'; readonly args: readonly FolFormula[] }
  | { readonly kind: 'or'; readonly args: readonly FolFormula[] }
  | { readonly kind: 'implies'; readonly left: FolFormula; readonly right: FolFormula }
  | { readonly kind: 'iff'; readonly left: FolFormula; readonly right: FolFormula }
  | { readonly kind: 'forall'; readonly variable: string; readonly body: FolFormula }
  | { readonly kind: 'exists'; readonly variable: string; readonly body: FolFormula }

/** A literal of a clause: an atom and its sign. */
export interface FolLiteral {
  readonly positive: boolean
  readonly atom: Term
}

/** A clause: a disjunction of literals whose variables are implicitly universally quantified. */
export type FolClause = readonly FolLiteral[]

// ------------------------------------------------------------------------------------------------ syntax

interface Tok {
  readonly t: string
  readonly v: string
  readonly at: number
}

const FOL_SYMBOLS: readonly [string, string][] = [
  ['<->', 'iff'],
  ['<=>', 'iff'],
  ['↔', 'iff'],
  ['->', 'implies'],
  ['=>', 'implies'],
  ['→', 'implies'],
  ['&', 'and'],
  ['∧', 'and'],
  ['|', 'or'],
  ['∨', 'or'],
  ['~', 'not'],
  ['-', 'not'],
  ['¬', 'not'],
  ['!', 'not'],
  ['∀', 'all'],
  ['∃', 'exists'],
  ['(', '('],
  [')', ')'],
  [',', ','],
  ['.', '.'],
  ['=', '='],
]

function folLex(text: string): Tok[] {
  const out: Tok[] = []
  let i = 0
  while (i < text.length) {
    const c = text[i] as string
    if (/\s/.test(c)) {
      i++
      continue
    }
    if (text.startsWith('!=', i)) {
      out.push({ t: '!=', v: '!=', at: i })
      i += 2
      continue
    }
    const sym = FOL_SYMBOLS.find(([s]) => text.startsWith(s, i))
    if (sym !== undefined) {
      out.push({ t: sym[1], v: sym[0], at: i })
      i += sym[0].length
      continue
    }
    const m = /^[A-Za-z_0-9][A-Za-z0-9_']*/.exec(text.slice(i))
    if (m === null) throw new WhiteboxError('value', `parseFol: unexpected ${JSON.stringify(c)} at ${i}`)
    const w = m[0]
    const kw: Readonly<Record<string, string>> = {
      all: 'all',
      forall: 'all',
      exists: 'exists',
      some: 'exists',
      not: 'not',
      and: 'and',
      or: 'or',
      implies: 'implies',
      iff: 'iff',
      true: 'true',
      false: 'false',
    }
    out.push({ t: kw[w] ?? 'id', v: w, at: i })
    i += w.length
  }
  return out
}

/**
 * Parse a first-order formula. Quantifiers are `all x.`, `forall x.`, `exists x.` (or `∀x.`, `∃x.`), possibly binding
 * several variables (`all x y.`), and scope as far to the right as possible; connectives are `-`/`~`/`¬`/`not`, `&`,
 * `|`, `->`, `<->`, and `=` / `!=` build equality atoms. Inside a quantifier's scope its variables are variables;
 * other identifiers that start with an upper-case letter are free variables (read as universally quantified) in term
 * position and propositions in formula position, and the rest are constants, functions and predicates.
 *
 * @param text - The formula.
 * @returns The formula.
 * @throws WhiteboxError with code `value` on a syntax error.
 * @concept first-order-logic
 * @example
 * ```ts
 * import { folToString, parseFol } from 'whitebox-engine/logic'
 * folToString(parseFol('all x.(man(x) -> mortal(x))')) // => '∀x (man(x) → mortal(x))'
 * folToString(parseFol('exists y. loves(y, f(Z)) & -hates(Z, y)')) // => '∃y (loves(y,f(Z)) ∧ ¬hates(Z,y))'
 * ```
 */
export function parseFol(text: string): FolFormula {
  const toks = folLex(text)
  let pos = 0
  const peek = (): Tok | undefined => toks[pos]
  const fail = (what: string): never => {
    throw new WhiteboxError('value', `parseFol: ${what} at ${toks[pos]?.at ?? text.length}`)
  }
  const expect = (t: string): void => {
    if (peek()?.t !== t) fail(`expected ${t}`)
    pos++
  }
  const bound: string[] = []
  const term = (): Term => {
    const tok = peek()
    if (tok?.t !== 'id') return fail('expected a term')
    pos++
    if (peek()?.t === '(') {
      pos++
      const args = [term()]
      while (peek()?.t === ',') {
        pos++
        args.push(term())
      }
      expect(')')
      return compound(tok.v, args)
    }
    if (bound.includes(tok.v) || /^[A-Z_]/.test(tok.v)) return variable(tok.v)
    return atom(tok.v)
  }
  const primary = (): FolFormula => {
    const tok = peek()
    if (tok === undefined) return fail('unexpected end')
    if (tok.t === 'not') {
      pos++
      return { kind: 'not', arg: primary() }
    }
    if (tok.t === 'all' || tok.t === 'exists') {
      pos++
      const vars: string[] = []
      while (peek()?.t === 'id') vars.push((toks[pos++] as Tok).v)
      if (vars.length === 0) fail('expected a variable')
      expect('.')
      bound.push(...vars)
      const body = iff()
      bound.splice(bound.length - vars.length, vars.length)
      let out = body
      for (let k = vars.length - 1; k >= 0; k--) {
        out = { kind: tok.t === 'all' ? 'forall' : 'exists', variable: vars[k] as string, body: out }
      }
      return out
    }
    if (tok.t === '(') {
      pos++
      const f = iff()
      expect(')')
      return f
    }
    if (tok.t === 'true' || tok.t === 'false') {
      pos++
      return { kind: 'const', value: tok.t === 'true' }
    }
    const left = term()
    const op = peek()?.t
    if (op === '=' || op === '!=') {
      pos++
      const eq: FolFormula = { kind: 'atom', atom: compound('=', [left, term()]) }
      return op === '=' ? eq : { kind: 'not', arg: eq }
    }
    if (left.kind === 'var') {
      // An upper-case name in formula position is a proposition (a 0-ary predicate) unless it is bound.
      if (bound.includes(left.name)) fail(`the variable ${left.name} is not a formula`)
      return { kind: 'atom', atom: atom(left.name) }
    }
    return { kind: 'atom', atom: left }
  }
  const nary =
    (op: 'and' | 'or', next: () => FolFormula): (() => FolFormula) =>
    () => {
      const args = [next()]
      while (peek()?.t === op) {
        pos++
        args.push(next())
      }
      return args.length === 1 ? (args[0] as FolFormula) : { kind: op, args }
    }
  const conj = nary('and', primary)
  const disj = nary('or', conj)
  const impl = (): FolFormula => {
    const left = disj()
    if (peek()?.t !== 'implies') return left
    pos++
    return { kind: 'implies', left, right: impl() }
  }
  const iff = (): FolFormula => {
    let left = impl()
    while (peek()?.t === 'iff') {
      pos++
      left = { kind: 'iff', left, right: impl() }
    }
    return left
  }
  const f = iff()
  if (pos < toks.length) fail('unexpected trailing input')
  return f
}

/**
 * Write a first-order formula in logical notation (`∀ ∃ ¬ ∧ ∨ → ↔`), bracketing compound operands.
 *
 * @param f - The formula.
 * @returns The text.
 * @example
 * ```ts
 * import { folToString, parseFol } from 'whitebox-engine/logic'
 * folToString(parseFol('-(p(a) | q(a)) <-> exists x. r(x)')) // => '¬(p(a) ∨ q(a)) ↔ ∃x r(x)'
 * ```
 */
export function folToString(f: FolFormula): string {
  const go = (x: FolFormula, top: boolean): string => {
    const wrap = (s: string): string => (top ? s : `(${s})`)
    switch (x.kind) {
      case 'atom':
        return termToString(x.atom)
      case 'const':
        return x.value ? '⊤' : '⊥'
      case 'not':
        return `¬${go(x.arg, false)}`
      case 'and':
        return wrap(x.args.map((a) => go(a, false)).join(' ∧ '))
      case 'or':
        return wrap(x.args.map((a) => go(a, false)).join(' ∨ '))
      case 'implies':
        return wrap(`${go(x.left, false)} → ${go(x.right, false)}`)
      case 'iff':
        return wrap(`${go(x.left, false)} ↔ ${go(x.right, false)}`)
      case 'forall':
      case 'exists':
        return `${x.kind === 'forall' ? '∀' : '∃'}${x.variable} ${go(x.body, false)}`
    }
  }
  return go(f, true)
}

/**
 * Write a clause as a disjunction (`□` for the empty clause).
 *
 * @param c - The clause.
 * @returns The text.
 * @example
 * ```ts
 * import { clauseText } from 'whitebox-engine/logic'
 * clauseText([{ positive: false, atom: { kind: 'atom', name: 'p' } }, { positive: true, atom: { kind: 'atom', name: 'q' } }]) // => '¬p ∨ q'
 * ```
 */
export function clauseText(c: FolClause): string {
  if (c.length === 0) return '□'
  return c.map((l) => `${l.positive ? '' : '¬'}${termToString(l.atom)}`).join(' ∨ ')
}

// ------------------------------------------------------------------------------------------------ clausal form

function freeVariables(f: FolFormula): string[] {
  const out = new Set<string>()
  const go = (x: FolFormula, bound: ReadonlySet<string>): void => {
    switch (x.kind) {
      case 'atom':
        for (const v of termVariables(x.atom)) if (!bound.has(v.name)) out.add(v.name)
        return
      case 'const':
        return
      case 'not':
        go(x.arg, bound)
        return
      case 'and':
      case 'or':
        for (const a of x.args) go(a, bound)
        return
      case 'implies':
      case 'iff':
        go(x.left, bound)
        go(x.right, bound)
        return
      case 'forall':
      case 'exists':
        go(x.body, new Set([...bound, x.variable]))
    }
  }
  go(f, new Set())
  return [...out].sort()
}

function functionSymbols(f: FolFormula): Set<string> {
  const out = new Set<string>()
  const term = (t: Term): void => {
    if (t.kind === 'atom') out.add(t.name)
    else if (t.kind === 'compound') {
      out.add(t.functor)
      for (const a of t.args) term(a)
    }
  }
  const go = (x: FolFormula): void => {
    if (x.kind === 'atom') {
      if (x.atom.kind === 'compound') for (const a of x.atom.args) term(a)
    } else if (x.kind === 'not') go(x.arg)
    else if (x.kind === 'and' || x.kind === 'or') for (const a of x.args) go(a)
    else if (x.kind === 'implies' || x.kind === 'iff') {
      go(x.left)
      go(x.right)
    } else if (x.kind === 'forall' || x.kind === 'exists') go(x.body)
  }
  go(f)
  return out
}

type Nnf =
  | { readonly kind: 'lit'; readonly lit: FolLiteral }
  | { readonly kind: 'const'; readonly value: boolean }
  | { readonly kind: 'and' | 'or'; readonly args: readonly Nnf[] }
  | { readonly kind: 'forall' | 'exists'; readonly variable: string; readonly body: Nnf }

function nnf(f: FolFormula, positive: boolean): Nnf {
  switch (f.kind) {
    case 'atom':
      return { kind: 'lit', lit: { positive, atom: f.atom } }
    case 'const':
      return { kind: 'const', value: f.value === positive }
    case 'not':
      return nnf(f.arg, !positive)
    case 'and':
    case 'or': {
      const kind = (f.kind === 'and') === positive ? 'and' : 'or'
      return { kind, args: f.args.map((a) => nnf(a, positive)) }
    }
    case 'implies':
      return nnf({ kind: 'or', args: [{ kind: 'not', arg: f.left }, f.right] }, positive)
    case 'iff':
      return nnf(
        {
          kind: 'and',
          args: [
            { kind: 'or', args: [{ kind: 'not', arg: f.left }, f.right] },
            { kind: 'or', args: [f.left, { kind: 'not', arg: f.right }] },
          ],
        },
        positive,
      )
    case 'forall':
    case 'exists': {
      const kind = (f.kind === 'forall') === positive ? 'forall' : 'exists'
      return { kind, variable: f.variable, body: nnf(f.body, positive) }
    }
  }
}

/** Options of {@link clausalForm}. */
export interface ClausalFormOptions {
  /** The prefix of Skolem function names (default `sk`; numbers are appended, skipping names already in use). */
  readonly skolemPrefix?: string
  /** The first Skolem number (default 1); successive calls can continue numbering. */
  readonly skolemStart?: number
}

/**
 * Convert a formula to clausal form: implications and equivalences eliminated, negations pushed inwards (NNF),
 * quantified variables standardised apart, existentials Skolemised (by functions of the enclosing universals),
 * universals dropped and the matrix distributed into clauses. Free variables are read as universally quantified.
 * Tautological clauses are dropped and duplicate literals merged; the clause set is equisatisfiable with the formula.
 *
 * @param f - The formula (or several, whose conjunction is converted).
 * @param options - Skolem naming.
 * @returns The clauses (each with variables renamed `X1`, `X2`, ... in order of occurrence).
 * @concept first-order-logic
 * @example
 * ```ts
 * import { clausalForm, clauseText, parseFol } from 'whitebox-engine/logic'
 * clausalForm(parseFol('all x.(person(x) -> exists y. mother(y, x))')).map(clauseText) // => ['¬person(X1) ∨ mother(sk1(X1),X1)']
 * clausalForm(parseFol('-(all x. p(x))')).map(clauseText) // => ['¬p(sk1)']
 * ```
 */
export function clausalForm(f: FolFormula | readonly FolFormula[], options: ClausalFormOptions = {}): FolClause[] {
  const formulas = Array.isArray(f) ? (f as readonly FolFormula[]) : [f as FolFormula]
  const prefix = options.skolemPrefix ?? 'sk'
  const used = new Set<string>()
  for (const g of formulas) for (const s of functionSymbols(g)) used.add(s)
  let skolem = options.skolemStart ?? 1
  const nextSkolem = (): string => {
    let name = `${prefix}${skolem++}`
    while (used.has(name)) name = `${prefix}${skolem++}`
    used.add(name)
    return name
  }
  let fresh = 0
  const clauses: FolClause[] = []
  for (const g of formulas) {
    // Free variables are universal: bind them first.
    let closed: FolFormula = g
    for (const v of freeVariables(g).reverse()) closed = { kind: 'forall', variable: v, body: closed }
    // Standardise apart and Skolemise in one pass over the NNF: universals become fresh variables, existentials terms.
    const skol = (x: Nnf, env: ReadonlyMap<string, Term>, universals: readonly Term[]): Nnf => {
      switch (x.kind) {
        case 'lit':
          return {
            kind: 'lit',
            lit: { positive: x.lit.positive, atom: substitute(x.lit.atom, env) },
          }
        case 'const':
          return x
        case 'and':
        case 'or':
          return { kind: x.kind, args: x.args.map((a) => skol(a, env, universals)) }
        case 'forall': {
          const v = variable(`V${++fresh}`)
          return skol(x.body, new Map([...env, [x.variable, v]]), [...universals, v])
        }
        case 'exists': {
          const name = nextSkolem()
          const t = universals.length === 0 ? atom(name) : compound(name, universals)
          return skol(x.body, new Map([...env, [x.variable, t]]), universals)
        }
      }
    }
    const matrix = skol(nnf(closed, true), new Map(), [])
    for (const c of distribute(matrix)) clauses.push(c)
  }
  return clauses.map(tidyClause).filter((c): c is FolClause => c !== null)
}

function substitute(t: Term, env: ReadonlyMap<string, Term>): Term {
  if (t.kind === 'var') return env.get(t.name) ?? t
  if (t.kind !== 'compound') return t
  return { kind: 'compound', functor: t.functor, args: t.args.map((a) => substitute(a, env)) }
}

function distribute(x: Nnf): FolLiteral[][] {
  switch (x.kind) {
    case 'lit':
      return [[x.lit]]
    case 'const':
      return x.value ? [] : [[]]
    case 'and':
      return x.args.flatMap(distribute)
    case 'or': {
      let acc: FolLiteral[][] = [[]]
      for (const a of x.args) {
        const part = distribute(a)
        acc = acc.flatMap((c) => part.map((d) => [...c, ...d]))
      }
      return acc
    }
    case 'forall':
    case 'exists':
      return distribute(x.body)
  }
}

const literalKey = (l: FolLiteral): string => `${l.positive ? '+' : '-'}${termToString(l.atom)}`

/** Merge duplicate literals, drop tautologies, rename variables X1, X2, ... in order of occurrence. */
function tidyClause(c: readonly FolLiteral[]): FolClause | null {
  const seen = new Map<string, FolLiteral>()
  for (const l of c) seen.set(literalKey(l), l)
  for (const l of seen.values()) if (seen.has(literalKey({ positive: !l.positive, atom: l.atom }))) return null
  const lits = [...seen.values()]
  const names = new Map<string, string>()
  for (const v of termVariables(lits.map((l) => l.atom))) names.set(v.name, `X${names.size + 1}`)
  return lits.map((l) => ({ positive: l.positive, atom: renameTerm(l.atom, (n) => names.get(n) ?? n) }))
}

// ------------------------------------------------------------------------------------------------ subsumption

/**
 * Clause subsumption (theta-subsumption on literal sets): whether some substitution maps every literal of `c` onto a
 * literal of `d` with the same sign. With `strict` (default false) `c` must also have no more literals than `d`, the
 * form a resolution prover uses for deletion.
 *
 * @param c - The general clause.
 * @param d - The specific clause.
 * @param options - Whether to require `|c| <= |d|`.
 * @returns The substitution found, or null.
 * @concept first-order-logic
 * @example
 * ```ts
 * import { clausalForm, parseFol, subsumes } from 'whitebox-engine/logic'
 * const [c] = clausalForm(parseFol('p(X) | q(X)'))
 * const [d] = clausalForm(parseFol('p(a) | q(a) | r'))
 * subsumes(c, d) === null // => false
 * subsumes(d, c) // => null
 * ```
 */
export function subsumes(c: FolClause, d: FolClause, options: { readonly strict?: boolean } = {}): Substitution | null {
  if ((options.strict ?? false) && c.length > d.length) return null
  // Most constrained literals first: those with fewest candidate partners.
  const candidates = c.map((l) => d.filter((m) => m.positive === l.positive && sameHead(l.atom, m.atom)))
  if (candidates.some((x) => x.length === 0)) return null
  const order = c.map((_, i) => i).sort((a, b) => (candidates[a]?.length ?? 0) - (candidates[b]?.length ?? 0))
  const go = (k: number, s: Substitution): Substitution | null => {
    if (k === order.length) return s
    const i = order[k] as number
    const lit = c[i] as FolLiteral
    for (const m of candidates[i] ?? []) {
      const s2 = matchTerm(lit.atom, m.atom, s)
      if (s2 === null) continue
      const r = go(k + 1, s2)
      if (r !== null) return r
    }
    return null
  }
  return go(0, new Map())
}

function sameHead(a: Term, b: Term): boolean {
  if (a.kind === 'atom') return b.kind === 'atom' && b.name === a.name
  if (a.kind === 'compound') return b.kind === 'compound' && b.functor === a.functor && b.args.length === a.args.length
  return false
}

// ------------------------------------------------------------------------------------------------ resolution

/** How a clause of a refutation was obtained. */
export interface ProofStep {
  readonly id: number
  readonly clause: FolClause
  readonly rule: 'axiom' | 'negated-conjecture' | 'resolution' | 'factoring'
  readonly parents: readonly number[]
}

/** Options of {@link proveFol}. */
export interface ProverOptions extends Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> {
  /** Given clauses to process before giving up with `unknown` (default 2000). */
  readonly maxSteps?: number
  /** Kept clauses before giving up with `unknown` (default 20000). */
  readonly maxClauses?: number
  /** Set of support (default when a conjecture is given): only clauses descended from the negated conjecture are given. */
  readonly setOfSupport?: boolean
  /** Weight of a clause for selection (default symbol count); the lightest unprocessed clause is given next. */
  readonly weight?: (c: FolClause) => number
}

/** The observable state of the given-clause loop. */
export interface ProverState {
  readonly status: 'running' | 'proved' | 'saturated'
  readonly given: ProofStep | null
  /** Clauses kept from the last step. */
  readonly kept: readonly ProofStep[]
  readonly processed: number
  readonly unprocessed: number
  readonly generated: number
  readonly subsumed: number
  readonly empty: ProofStep | null
}

interface ProverMachine extends ProverState {
  status: 'running' | 'proved' | 'saturated'
  given: ProofStep | null
  kept: ProofStep[]
  processed: number
  unprocessed: number
  generated: number
  subsumed: number
  empty: ProofStep | null
  readonly steps: ProofStep[]
  readonly usable: ProofStep[]
  sos: ProofStep[]
}

const PROVER_SCHEMA: TraceSchema = channels({
  metrics: {
    processed: { description: 'clauses given so far (the usable set)' },
    unprocessed: { description: 'clauses waiting to be given (the set of support)' },
    generated: { description: 'resolvents and factors generated' },
    subsumed: { description: 'clauses deleted as tautologies or by subsumption' },
  },
  messages: {
    given: { description: 'the clause selected for inference', payload: { clause: 'text', id: 'id' } },
    kept: { description: 'new clauses kept', payload: { clauses: 'texts with parents' } },
    proof: { description: 'the empty clause was derived', payload: { id: 'id', length: 'clauses in the proof' } },
  },
})

const defaultWeight = (c: FolClause): number => c.reduce((s, l) => s + termSize(l.atom), 0)

function renameClause(c: FolClause, suffix: string): FolClause {
  return c.map((l) => ({ positive: l.positive, atom: renameTerm(l.atom, (n) => `${n}${suffix}`) }))
}

function applyClause(c: FolClause, s: Substitution): FolLiteral[] {
  return c.map((l) => ({ positive: l.positive, atom: applySubstitution(l.atom, s) }))
}

/** Binary resolvents of two clauses (the second renamed apart). */
function resolvents(a: FolClause, b: FolClause): FolLiteral[][] {
  const b2 = renameClause(b, "'")
  const out: FolLiteral[][] = []
  for (let i = 0; i < a.length; i++) {
    const la = a[i] as FolLiteral
    for (let j = 0; j < b2.length; j++) {
      const lb = b2[j] as FolLiteral
      if (la.positive === lb.positive || !sameHead(la.atom, lb.atom)) continue
      const s = mgu(la.atom, lb.atom)
      if (s === null) continue
      out.push([
        ...applyClause(
          a.filter((_, k) => k !== i),
          s,
        ),
        ...applyClause(
          b2.filter((_, k) => k !== j),
          s,
        ),
      ])
    }
  }
  return out
}

/** Factors of a clause: unify two literals of the same sign. */
function factors(c: FolClause): FolLiteral[][] {
  const out: FolLiteral[][] = []
  for (let i = 0; i < c.length; i++) {
    for (let j = i + 1; j < c.length; j++) {
      const li = c[i] as FolLiteral
      const lj = c[j] as FolLiteral
      if (li.positive !== lj.positive || !sameHead(li.atom, lj.atom)) continue
      const s = mgu(li.atom, lj.atom)
      if (s !== null)
        out.push(
          applyClause(
            c.filter((_, k) => k !== j),
            s,
          ),
        )
    }
  }
  return out
}

function proverMachine(
  axioms: readonly FolClause[],
  goals: readonly FolClause[],
  options: ProverOptions,
): Algorithm<ProverMachine> {
  const maxClauses = options.maxClauses ?? 20_000
  const weight = options.weight ?? defaultWeight
  const sosMode = options.setOfSupport ?? goals.length > 0
  return {
    name: 'given-clause-resolution',
    config: { setOfSupport: sosMode, maxClauses },
    schema: PROVER_SCHEMA,
    init: () => {
      const steps: ProofStep[] = []
      const add = (c: FolClause, rule: ProofStep['rule']): ProofStep => {
        const step = { id: steps.length, clause: c, rule, parents: [] }
        steps.push(step)
        return step
      }
      const ax = axioms.map((c) => add(c, 'axiom'))
      const gl = goals.map((c) => add(c, 'negated-conjecture'))
      const s: ProverMachine = {
        status: 'running',
        given: null,
        kept: [],
        processed: 0,
        unprocessed: 0,
        generated: 0,
        subsumed: 0,
        empty: null,
        steps,
        usable: sosMode ? ax : [],
        sos: sosMode ? gl : [...ax, ...gl],
      }
      const empty = steps.find((x) => x.clause.length === 0)
      if (empty !== undefined) {
        s.status = 'proved'
        s.empty = empty
      }
      s.processed = s.usable.length
      s.unprocessed = s.sos.length
      return s
    },
    step: (s) => {
      s.kept = []
      if (s.sos.length === 0) {
        s.status = 'saturated'
        s.given = null
        return s
      }
      // The lightest clause, oldest first among equals.
      let best = 0
      for (let i = 1; i < s.sos.length; i++) {
        const w = weight((s.sos[i] as ProofStep).clause)
        const wb = weight((s.sos[best] as ProofStep).clause)
        if (w < wb || (w === wb && (s.sos[i] as ProofStep).id < (s.sos[best] as ProofStep).id)) best = i
      }
      const given = s.sos[best] as ProofStep
      s.sos.splice(best, 1)
      s.given = given
      s.usable.push(given)
      const fresh: { clause: FolLiteral[]; rule: ProofStep['rule']; parents: number[] }[] = []
      for (const f of factors(given.clause)) fresh.push({ clause: f, rule: 'factoring', parents: [given.id] })
      for (const u of s.usable) {
        for (const r of resolvents(given.clause, u.clause)) {
          fresh.push({ clause: r, rule: 'resolution', parents: u.id === given.id ? [given.id] : [given.id, u.id] })
        }
      }
      s.generated += fresh.length
      for (const x of fresh) {
        const c = tidyClause(x.clause)
        if (c === null) {
          s.subsumed++
          continue
        }
        const all = [...s.usable, ...s.sos]
        if (all.some((o) => subsumes(o.clause, c, { strict: true }) !== null)) {
          s.subsumed++
          continue
        }
        const step: ProofStep = { id: s.steps.length, clause: c, rule: x.rule, parents: x.parents }
        s.steps.push(step)
        s.kept.push(step)
        if (c.length === 0) {
          s.status = 'proved'
          s.empty = step
          break
        }
        // Back subsumption: drop kept clauses the new one subsumes.
        const before = s.usable.length + s.sos.length
        const keep = (o: ProofStep): boolean => o === given || subsumes(c, o.clause, { strict: true }) === null
        const usable = s.usable.filter(keep)
        s.usable.length = 0
        s.usable.push(...usable)
        s.sos = s.sos.filter(keep)
        s.subsumed += before - (s.usable.length + s.sos.length)
        s.sos.push(step)
      }
      s.processed = s.usable.length
      s.unprocessed = s.sos.length
      return s
    },
    done: (s) =>
      s.status !== 'running' ? 'converged' : s.usable.length + s.sos.length >= maxClauses ? 'stopped' : undefined,
    observe: (s) => {
      const messages: { kind: string; payload: Json }[] = []
      if (s.given !== null)
        messages.push({ kind: 'given', payload: { clause: clauseText(s.given.clause), id: s.given.id } })
      if (s.kept.length > 0) {
        messages.push({
          kind: 'kept',
          payload: {
            clauses: s.kept.map((k) => ({ id: k.id, clause: clauseText(k.clause), parents: [...k.parents] })),
          },
        })
      }
      if (s.empty !== null && s.status === 'proved') {
        messages.push({ kind: 'proof', payload: { id: s.empty.id, length: proofOf(s.steps, s.empty).length } })
      }
      return {
        metrics: { processed: s.processed, unprocessed: s.unprocessed, generated: s.generated, subsumed: s.subsumed },
        messages,
      }
    },
  }
}

function proofOf(steps: readonly ProofStep[], empty: ProofStep): ProofStep[] {
  const need = new Set<number>()
  const stack = [empty.id]
  for (let id = stack.pop(); id !== undefined; id = stack.pop()) {
    if (need.has(id)) continue
    need.add(id)
    for (const p of steps[id]?.parents ?? []) stack.push(p)
  }
  return steps.filter((s) => need.has(s.id))
}

/**
 * The given-clause resolution loop (Otter's) as a traced {@link Algorithm}: each step selects the lightest unprocessed
 * clause, moves it to the processed set, and adds its factors and its binary resolvents with every processed clause,
 * discarding tautologies and clauses subsumed by existing ones (and deleting existing clauses the new one subsumes).
 * The run proves the input unsatisfiable when the empty clause appears, and saturates when nothing new can be added.
 *
 * @param axioms - Axiom clauses.
 * @param goals - Clauses of the negated conjecture (the set of support).
 * @param options - Set of support, clause selection weight and the clause budget.
 * @returns The algorithm.
 * @concept resolution-proving
 * @example
 * ```ts
 * import { clausalForm, parseFol, resolutionAlgorithm } from 'whitebox-engine/logic'
 * resolutionAlgorithm(clausalForm(parseFol('p')), clausalForm(parseFol('-p'))).name // => 'given-clause-resolution'
 * ```
 */
export function resolutionAlgorithm(
  axioms: readonly FolClause[],
  goals: readonly FolClause[] = [],
  options: ProverOptions = {},
): Algorithm<ProverState> {
  return proverMachine(axioms, goals, options)
}

/** The outcome of {@link proveFol}. */
export interface ProverResult {
  /**
   * `theorem` (the conjecture follows: a refutation was found), `countersatisfiable` (saturation without the empty
   * clause: the conjecture does not follow), `unknown` (a budget ran out).
   */
  readonly status: 'theorem' | 'countersatisfiable' | 'unknown'
  /** The refutation, in derivation order, when one was found. */
  readonly proof: readonly ProofStep[]
  readonly steps: number
  readonly generated: number
  readonly kept: number
}

/**
 * Prove a conjecture from axioms by refutation: the axioms and the negated conjecture are put in clausal form and
 * resolved until the empty clause appears (a theorem), nothing new follows (countersatisfiable), or a budget runs out.
 * Without a conjecture the axioms themselves are refuted (`theorem` then means they are unsatisfiable).
 *
 * @param axioms - Axiom formulas (text or parsed).
 * @param conjecture - The conjecture (text or parsed), or null.
 * @param options - Budgets, set of support, selection weight and tracing.
 * @returns The status, the proof, and the effort.
 * @throws WhiteboxError with code `value` on a syntax error.
 * @concept resolution-proving
 * @example
 * ```ts
 * import { proveFol } from 'whitebox-engine/logic'
 * proveFol(['all x.(man(x) -> mortal(x))', 'man(socrates)'], 'mortal(socrates)').status // => 'theorem'
 * proveFol(['all x.(p(x) -> q(x))', 'q(a)'], 'p(a)').status // => 'countersatisfiable'
 * // The barber paradox needs factoring.
 * proveFol([], '-exists x. all y.(shaves(x, y) <-> -shaves(y, y))').status // => 'theorem'
 * ```
 */
export function proveFol(
  axioms: readonly (FolFormula | string)[],
  conjecture: FolFormula | string | null,
  options: ProverOptions = {},
): ProverResult {
  const read = (f: FolFormula | string): FolFormula => (typeof f === 'string' ? parseFol(f) : f)
  const ax = clausalForm(axioms.map(read))
  const goal =
    conjecture === null
      ? []
      : clausalForm({ kind: 'not', arg: read(conjecture) }, { skolemPrefix: 'sk', skolemStart: 1000 })
  const { maxSteps = 2000, maxClauses: _c, setOfSupport: _s, weight: _w, ...run } = options
  const { state, steps } = runAlgorithm(proverMachine(ax, goal, options), { ...run, maxSteps })
  const status = state.status === 'proved' ? 'theorem' : state.status === 'saturated' ? 'countersatisfiable' : 'unknown'
  return {
    status,
    proof: state.empty === null ? [] : proofOf(state.steps, state.empty),
    steps,
    generated: state.generated,
    kept: state.steps.length,
  }
}

/**
 * Check a refutation: every resolution or factoring step's clause is a resolvent or factor of its parents (up to
 * variable renaming, via mutual subsumption), and the last clause is empty.
 *
 * @param proof - The proof steps.
 * @returns True for a valid refutation.
 * @example
 * ```ts
 * import { checkProof, proveFol } from 'whitebox-engine/logic'
 * checkProof(proveFol(['p(a)', 'all x.(p(x) -> q(x))'], 'q(a)').proof) // => true
 * ```
 */
export function checkProof(proof: readonly ProofStep[]): boolean {
  const byId = new Map(proof.map((s) => [s.id, s]))
  const last = proof[proof.length - 1]
  if (last === undefined || last.clause.length !== 0) return false
  const same = (a: FolClause, b: FolClause): boolean => subsumes(a, b) !== null && subsumes(b, a) !== null
  return proof.every((s) => {
    if (s.rule === 'axiom' || s.rule === 'negated-conjecture') return true
    const ps = s.parents.map((p) => byId.get(p)?.clause)
    if (ps.some((p) => p === undefined)) return false
    const [a, b] = ps as FolClause[]
    if (a === undefined) return false
    const candidates = s.rule === 'factoring' ? factors(a) : resolvents(a, b ?? a)
    return candidates.some((c) => {
      const t = tidyClause(c)
      return t !== null && same(t, s.clause)
    })
  })
}

/**
 * A refutation as a {@link Structure}: one node per clause, edges from parents to the clauses inferred from them,
 * inputs marked, the empty clause as the output.
 *
 * @param proof - The proof steps.
 * @returns The structure.
 * @concept resolution-proving
 * @example
 * ```ts
 * import { proofStructure, proveFol } from 'whitebox-engine/logic'
 * proofStructure(proveFol(['p(a)', 'all x.(p(x) -> q(x))'], 'q(a)').proof).nodes.map((n) => n.label) // => ['p(a)', '¬p(X1) ∨ q(X1)', '¬q(a)', '¬p(a)', '□']
 * ```
 */
export function proofStructure(proof: readonly ProofStep[]): Structure {
  const nodes: StructureNode[] = proof.map((s) => ({
    id: `c${s.id}`,
    label: clauseText(s.clause),
    role: s.clause.length === 0 ? 'output' : s.parents.length === 0 ? 'input' : 'internal',
    detail: s.rule,
  }))
  const edges = proof.flatMap((s) => s.parents.map((p) => ({ from: `c${p}`, to: `c${s.id}`, directed: true })))
  return structure('refutation', nodes, edges, { direction: 'down' })
}

/**
 * Herbrand-style check of a clause set over a finite domain of constants: whether some interpretation of the
 * predicates over `domain` satisfies every ground instance (exhaustive; for small signatures, as a sanity check that a
 * saturated set is satisfiable).
 *
 * @param clauses - Function-free clauses.
 * @param domain - The constants to instantiate variables with.
 * @returns True when a model over the domain exists.
 * @throws WhiteboxError with code `value` for function symbols or more than 20 ground atoms.
 * @example
 * ```ts
 * import { clausalForm, hasFiniteModel, parseFol } from 'whitebox-engine/logic'
 * hasFiniteModel(clausalForm([parseFol('all x.(p(x) -> q(x))'), parseFol('q(a)'), parseFol('-p(a)')]), ['a']) // => true
 * ```
 */
export function hasFiniteModel(clauses: readonly FolClause[], domain: readonly string[]): boolean {
  const ground: FolLiteral[][] = []
  const atoms = new Map<string, number>()
  const consts = domain.map((d) => atom(d))
  for (const c of clauses) {
    const vars = termVariables(c.map((l) => l.atom)).map((v) => v.name)
    const total = consts.length ** vars.length
    for (let m = 0; m < total; m++) {
      const s = new Map<string, Term>()
      let k = m
      for (const v of vars) {
        s.set(v, consts[k % consts.length] as Term)
        k = Math.floor(k / consts.length)
      }
      const g = c.map((l) => ({ positive: l.positive, atom: applySubstitution(l.atom, s) }))
      for (const l of g) {
        if (l.atom.kind === 'compound' && l.atom.args.some((a) => a.kind === 'compound')) {
          throw new WhiteboxError('value', 'hasFiniteModel: function symbols are not supported')
        }
        const key = termToString(l.atom)
        if (!atoms.has(key)) atoms.set(key, atoms.size)
      }
      ground.push(g)
    }
  }
  if (atoms.size > 20) throw new WhiteboxError('value', 'hasFiniteModel: more than 20 ground atoms')
  for (let m = 0; m < 2 ** atoms.size; m++) {
    const value = (l: FolLiteral): boolean => (((m >> (atoms.get(termToString(l.atom)) ?? 0)) & 1) === 1) === l.positive
    if (ground.every((c) => c.some(value))) return true
  }
  return false
}
