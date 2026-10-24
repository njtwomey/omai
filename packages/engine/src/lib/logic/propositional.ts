import { WhiteboxError } from '../base/index.ts'
import { type Cnf, cdcl } from './sat.ts'

/**
 * A propositional formula: variables, constants, negation, n-ary conjunction and disjunction, implication,
 * equivalence and exclusive or.
 *
 * @concept propositional-logic
 */
export type Formula =
  | { readonly kind: 'var'; readonly name: string }
  | { readonly kind: 'const'; readonly value: boolean }
  | { readonly kind: 'not'; readonly arg: Formula }
  | { readonly kind: 'and'; readonly args: readonly Formula[] }
  | { readonly kind: 'or'; readonly args: readonly Formula[] }
  | { readonly kind: 'implies'; readonly left: Formula; readonly right: Formula }
  | { readonly kind: 'iff'; readonly left: Formula; readonly right: Formula }
  | { readonly kind: 'xor'; readonly left: Formula; readonly right: Formula }

/** An assignment of truth values to variable names. */
export type Valuation = Readonly<Record<string, boolean>>

const T: Formula = { kind: 'const', value: true }
const F: Formula = { kind: 'const', value: false }
const v = (name: string): Formula => ({ kind: 'var', name })
const not = (arg: Formula): Formula => ({ kind: 'not', arg })
const and = (args: readonly Formula[]): Formula =>
  args.length === 0 ? T : args.length === 1 ? (args[0] as Formula) : { kind: 'and', args }
const or = (args: readonly Formula[]): Formula =>
  args.length === 0 ? F : args.length === 1 ? (args[0] as Formula) : { kind: 'or', args }

// ------------------------------------------------------------------------------------------------ syntax

interface Tok {
  readonly t: string
  readonly at: number
}

const SYMBOLS: readonly [string, string][] = [
  ['<->', 'iff'],
  ['<=>', 'iff'],
  ['↔', 'iff'],
  ['->', 'implies'],
  ['=>', 'implies'],
  ['→', 'implies'],
  ['&&', 'and'],
  ['||', 'or'],
  ['&', 'and'],
  ['∧', 'and'],
  ['|', 'or'],
  ['∨', 'or'],
  ['~', 'not'],
  ['!', 'not'],
  ['¬', 'not'],
  ['^', 'xor'],
  ['⊕', 'xor'],
  ['⊤', 'true'],
  ['⊥', 'false'],
  ['(', '('],
  [')', ')'],
]
const WORDS: Readonly<Record<string, string>> = {
  and: 'and',
  or: 'or',
  not: 'not',
  xor: 'xor',
  implies: 'implies',
  iff: 'iff',
  true: 'true',
  false: 'false',
  True: 'true',
  False: 'false',
}

function lex(text: string): Tok[] {
  const out: Tok[] = []
  let i = 0
  while (i < text.length) {
    const c = text[i] as string
    if (/\s/.test(c)) {
      i++
      continue
    }
    const sym = SYMBOLS.find(([s]) => text.startsWith(s, i))
    if (sym !== undefined) {
      out.push({ t: sym[1], at: i })
      i += sym[0].length
      continue
    }
    if (c === '1' || c === '0') {
      out.push({ t: c === '1' ? 'true' : 'false', at: i })
      i++
      continue
    }
    const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(text.slice(i))
    if (m === null) throw new WhiteboxError('value', `parseFormula: unexpected ${JSON.stringify(c)} at ${i}`)
    const word = m[0]
    out.push({ t: WORDS[word] ?? `v:${word}`, at: i })
    i += word.length
  }
  return out
}

/**
 * Parse a propositional formula. Operators from tightest to loosest: `~`/`!`/`¬`/`not`, `&`/`∧`/`and`, `^`/`⊕`/`xor`,
 * `|`/`∨`/`or`, `->`/`=>`/`→` (right associative), `<->`/`<=>`/`↔`; `true`/`false` (or `1`/`0`, `⊤`/`⊥`) are constants and any
 * other identifier is a variable.
 *
 * @param text - The formula.
 * @returns The formula.
 * @throws WhiteboxError with code `value` on a syntax error.
 * @concept propositional-logic
 * @example
 * ```ts
 * import { formulaToString, parseFormula } from 'whitebox-engine/logic'
 * formulaToString(parseFormula('p & q -> ~r | s')) // => '(p & q) -> (~r | s)'
 * ```
 */
export function parseFormula(text: string): Formula {
  const toks = lex(text)
  let pos = 0
  const peek = (): string | undefined => toks[pos]?.t
  const fail = (what: string): never => {
    throw new WhiteboxError('value', `parseFormula: ${what} at ${toks[pos]?.at ?? text.length}`)
  }
  const binary =
    (op: string, next: () => Formula, build: (args: Formula[]) => Formula): (() => Formula) =>
    () => {
      const args = [next()]
      while (peek() === op) {
        pos++
        args.push(next())
      }
      return args.length === 1 ? (args[0] as Formula) : build(args)
    }
  const atom = (): Formula => {
    const t = peek()
    if (t === undefined) return fail('unexpected end')
    pos++
    if (t === 'not') return not(atom())
    if (t === '(') {
      const f = iff()
      if (peek() !== ')') fail('expected )')
      pos++
      return f
    }
    if (t === 'true') return T
    if (t === 'false') return F
    if (t.startsWith('v:')) return v(t.slice(2))
    return fail(`unexpected ${t}`)
  }
  const conj = binary('and', atom, (args) => ({ kind: 'and', args }))
  const exclusive = binary('xor', conj, (args) =>
    args.slice(1).reduce((l, r) => ({ kind: 'xor', left: l, right: r }), args[0] as Formula),
  )
  const disj = binary('or', exclusive, (args) => ({ kind: 'or', args }))
  const impl = (): Formula => {
    const left = disj()
    if (peek() !== 'implies') return left
    pos++
    return { kind: 'implies', left, right: impl() }
  }
  const iff = (): Formula => {
    let left = impl()
    while (peek() === 'iff') {
      pos++
      left = { kind: 'iff', left, right: impl() }
    }
    return left
  }
  const f = iff()
  if (pos < toks.length) fail('unexpected trailing input')
  return f
}

/** Options of {@link formulaToString}. */
export interface FormulaStyle {
  /** `ascii` (`~ & | -> <-> ^`, default), `unicode` (`¬ ∧ ∨ → ↔ ⊕`) or `sympy` (SymPy's `~ & | Implies Equivalent Xor`). */
  readonly style?: 'ascii' | 'unicode' | 'sympy'
}

/**
 * Write a formula with full brackets around compound operands.
 *
 * @param f - The formula.
 * @param options - The notation.
 * @returns The text.
 * @example
 * ```ts
 * import { formulaToString, parseFormula } from 'whitebox-engine/logic'
 * const f = parseFormula('(p -> q) <-> ~(p & ~q)')
 * formulaToString(f, { style: 'unicode' }) // => '(p → q) ↔ ¬(p ∧ ¬q)'
 * formulaToString(f, { style: 'sympy' }) // => 'Equivalent(Implies(p, q), ~(p & ~q))'
 * ```
 */
export function formulaToString(f: Formula, options: FormulaStyle = {}): string {
  const style = options.style ?? 'ascii'
  const sym =
    style === 'unicode'
      ? { not: '¬', and: ' ∧ ', or: ' ∨ ', implies: ' → ', iff: ' ↔ ', xor: ' ⊕ ', t: '⊤', f: '⊥' }
      : style === 'sympy'
        ? { not: '~', and: ' & ', or: ' | ', implies: '', iff: '', xor: '', t: 'True', f: 'False' }
        : { not: '~', and: ' & ', or: ' | ', implies: ' -> ', iff: ' <-> ', xor: ' ^ ', t: 'true', f: 'false' }
  const go = (x: Formula, top: boolean): string => {
    const wrap = (s: string): string => (top ? s : `(${s})`)
    switch (x.kind) {
      case 'var':
        return x.name
      case 'const':
        return x.value ? sym.t : sym.f
      case 'not':
        return `${sym.not}${go(x.arg, false)}`
      case 'and':
        return wrap(x.args.map((a) => go(a, false)).join(sym.and))
      case 'or':
        return wrap(x.args.map((a) => go(a, false)).join(sym.or))
      case 'implies':
      case 'iff':
      case 'xor':
        if (style === 'sympy') {
          const name = x.kind === 'implies' ? 'Implies' : x.kind === 'iff' ? 'Equivalent' : 'Xor'
          return `${name}(${go(x.left, true)}, ${go(x.right, true)})`
        }
        return wrap(`${go(x.left, false)}${sym[x.kind]}${go(x.right, false)}`)
    }
  }
  return go(f, true)
}

// ------------------------------------------------------------------------------------------------ semantics

/**
 * The variables of a formula, sorted.
 *
 * @param f - The formula.
 * @returns The variable names.
 * @example
 * ```ts
 * import { formulaVariables, parseFormula } from 'whitebox-engine/logic'
 * formulaVariables(parseFormula('q | (p & ~q) -> r')) // => ['p', 'q', 'r']
 * ```
 */
export function formulaVariables(f: Formula): string[] {
  const out = new Set<string>()
  const go = (x: Formula): void => {
    if (x.kind === 'var') out.add(x.name)
    else if (x.kind === 'not') go(x.arg)
    else if (x.kind === 'and' || x.kind === 'or') for (const a of x.args) go(a)
    else if (x.kind !== 'const') {
      go(x.left)
      go(x.right)
    }
  }
  go(f)
  return [...out].sort()
}

/**
 * The truth value of a formula under a valuation.
 *
 * @param f - The formula.
 * @param valuation - The truth value of each variable.
 * @returns The truth value.
 * @throws WhiteboxError with code `value` for a variable without a value.
 * @example
 * ```ts
 * import { evaluateFormula, parseFormula } from 'whitebox-engine/logic'
 * evaluateFormula(parseFormula('p -> q'), { p: true, q: false }) // => false
 * ```
 */
export function evaluateFormula(f: Formula, valuation: Valuation): boolean {
  switch (f.kind) {
    case 'var': {
      const x = valuation[f.name]
      if (x === undefined) throw new WhiteboxError('value', `evaluateFormula: no value for ${f.name}`)
      return x
    }
    case 'const':
      return f.value
    case 'not':
      return !evaluateFormula(f.arg, valuation)
    case 'and':
      return f.args.every((a) => evaluateFormula(a, valuation))
    case 'or':
      return f.args.some((a) => evaluateFormula(a, valuation))
    case 'implies':
      return !evaluateFormula(f.left, valuation) || evaluateFormula(f.right, valuation)
    case 'iff':
      return evaluateFormula(f.left, valuation) === evaluateFormula(f.right, valuation)
    case 'xor':
      return evaluateFormula(f.left, valuation) !== evaluateFormula(f.right, valuation)
  }
}

/** A truth table: the variables, and one row per valuation in binary counting order (first variable most significant). */
export interface TruthTable {
  readonly variables: readonly string[]
  readonly rows: readonly { readonly valuation: Valuation; readonly value: boolean }[]
}

/**
 * The truth table of a formula.
 *
 * @param f - The formula.
 * @param variables - The variables to tabulate (default the formula's own, sorted); at most 20.
 * @returns The table.
 * @throws WhiteboxError with code `value` for more than 20 variables.
 * @concept propositional-logic
 * @example
 * ```ts
 * import { parseFormula, truthTable } from 'whitebox-engine/logic'
 * truthTable(parseFormula('p ^ q')).rows.map((r) => r.value) // => [false, true, true, false]
 * ```
 */
export function truthTable(f: Formula, variables: readonly string[] = formulaVariables(f)): TruthTable {
  if (variables.length > 20) throw new WhiteboxError('value', 'truthTable: more than 20 variables')
  const k = variables.length
  const rows: { valuation: Valuation; value: boolean }[] = []
  for (let m = 0; m < 2 ** k; m++) {
    const valuation: Record<string, boolean> = {}
    variables.forEach((name, i) => {
      valuation[name] = ((m >> (k - 1 - i)) & 1) === 1
    })
    rows.push({ valuation, value: evaluateFormula(f, valuation) })
  }
  return { variables, rows }
}

/**
 * Every model of a formula (valuations making it true), in truth-table order.
 *
 * @param f - The formula.
 * @param variables - The variables (default the formula's own).
 * @returns The models.
 * @example
 * ```ts
 * import { models, parseFormula } from 'whitebox-engine/logic'
 * models(parseFormula('p & ~q')) // => [{ p: true, q: false }]
 * ```
 */
export function models(f: Formula, variables?: readonly string[]): Valuation[] {
  return truthTable(f, variables)
    .rows.filter((r) => r.value)
    .map((r) => r.valuation)
}

/**
 * Whether a formula has a model, decided by CDCL on its Tseitin encoding (so formulas of any size can be checked).
 *
 * @param f - The formula.
 * @returns True when satisfiable.
 * @concept propositional-logic
 * @example
 * ```ts
 * import { isSatisfiable, parseFormula } from 'whitebox-engine/logic'
 * [isSatisfiable(parseFormula('p & ~p')), isSatisfiable(parseFormula('p | q'))] // => [false, true]
 * ```
 */
export function isSatisfiable(f: Formula): boolean {
  const t = tseitin(f)
  return cdcl(t.cnf).status === 'sat'
}

/**
 * A satisfying valuation of a formula, or null (by CDCL on the Tseitin encoding).
 *
 * @param f - The formula.
 * @returns A model over the formula's variables, or null.
 * @example
 * ```ts
 * import { findModel, parseFormula } from 'whitebox-engine/logic'
 * findModel(parseFormula('(p | q) & ~p')) // => { p: false, q: true }
 * ```
 */
export function findModel(f: Formula): Valuation | null {
  const t = tseitin(f)
  const r = cdcl(t.cnf)
  if (r.model === null) return null
  const out: Record<string, boolean> = {}
  for (const [name, index] of t.variables) out[name] = (r.model[index - 1] ?? -1) > 0
  return out
}

/**
 * Whether a formula is valid (a tautology): its negation is unsatisfiable.
 *
 * @param f - The formula.
 * @returns True when valid.
 * @example
 * ```ts
 * import { isValid, parseFormula } from 'whitebox-engine/logic'
 * [isValid(parseFormula('p | ~p')), isValid(parseFormula('p -> q'))] // => [true, false]
 * ```
 */
export function isValid(f: Formula): boolean {
  return !isSatisfiable(not(f))
}

/**
 * Whether premises entail a conclusion: every model of the premises is a model of the conclusion.
 *
 * @param premises - The premises.
 * @param conclusion - The conclusion.
 * @returns True when entailed.
 * @concept propositional-logic
 * @example
 * ```ts
 * import { entails, parseFormula } from 'whitebox-engine/logic'
 * entails([parseFormula('p -> q'), parseFormula('p')], parseFormula('q')) // => true
 * entails([parseFormula('p -> q'), parseFormula('q')], parseFormula('p')) // => false
 * ```
 */
export function entails(premises: readonly Formula[], conclusion: Formula): boolean {
  return !isSatisfiable(and([...premises, not(conclusion)]))
}

/**
 * Whether two formulas are logically equivalent.
 *
 * @param a - A formula.
 * @param b - Another.
 * @returns True when they have the same models.
 * @example
 * ```ts
 * import { equivalent, parseFormula } from 'whitebox-engine/logic'
 * equivalent(parseFormula('~(p & q)'), parseFormula('~p | ~q')) // => true
 * ```
 */
export function equivalent(a: Formula, b: Formula): boolean {
  return isValid({ kind: 'iff', left: a, right: b })
}

// ------------------------------------------------------------------------------------------------ normal forms

/**
 * Negation normal form: implications, equivalences and exclusive ors expanded, negations pushed onto variables, and
 * conjunctions and disjunctions flattened.
 *
 * @param f - The formula.
 * @returns An equivalent formula in NNF.
 * @concept normal-forms
 * @example
 * ```ts
 * import { formulaToString, parseFormula, toNnf } from 'whitebox-engine/logic'
 * formulaToString(toNnf(parseFormula('~(p -> (q & ~r))'))) // => 'p & (~q | r)'
 * ```
 */
export function toNnf(f: Formula): Formula {
  const go = (x: Formula, positive: boolean): Formula => {
    switch (x.kind) {
      case 'var':
        return positive ? x : not(x)
      case 'const':
        return { kind: 'const', value: x.value === positive }
      case 'not':
        return go(x.arg, !positive)
      case 'and':
      case 'or': {
        const isAnd = (x.kind === 'and') === positive
        const args = x.args.map((a) => go(a, positive))
        return flatten(isAnd ? 'and' : 'or', args)
      }
      case 'implies':
        return go({ kind: 'or', args: [not(x.left), x.right] }, positive)
      case 'iff':
        return go(
          {
            kind: 'and',
            args: [
              { kind: 'or', args: [not(x.left), x.right] },
              { kind: 'or', args: [x.left, not(x.right)] },
            ],
          },
          positive,
        )
      case 'xor':
        return go(
          {
            kind: 'and',
            args: [
              { kind: 'or', args: [x.left, x.right] },
              { kind: 'or', args: [not(x.left), not(x.right)] },
            ],
          },
          positive,
        )
    }
  }
  return go(f, true)
}

function flatten(kind: 'and' | 'or', args: readonly Formula[]): Formula {
  const out: Formula[] = []
  for (const a of args) {
    if (a.kind === kind) out.push(...a.args)
    else out.push(a)
  }
  return kind === 'and' ? and(out) : or(out)
}

/** A clause of literals over named variables: `{ name, positive }` pairs. */
export type NamedClause = readonly { readonly name: string; readonly positive: boolean }[]

function literalKey(l: { name: string; positive: boolean }): string {
  return `${l.positive ? '' : '~'}${l.name}`
}

/** Distribute to clauses (each a set of literals), dropping tautologies and duplicates; constants are folded. */
function clausesOf(f: Formula, outer: 'and' | 'or'): { name: string; positive: boolean }[][] {
  const nnf = toNnf(f)
  const inner = outer === 'and' ? 'or' : 'and'
  const go = (x: Formula): { name: string; positive: boolean }[][] | 'true' | 'false' => {
    // Returns the outer-level list of inner groups; 'true'/'false' are the identities.
    if (x.kind === 'const') return x.value === (outer === 'and') ? 'true' : 'false'
    if (x.kind === 'var') return [[{ name: x.name, positive: true }]]
    if (x.kind === 'not' && x.arg.kind === 'var') return [[{ name: x.arg.name, positive: false }]]
    if (x.kind === outer) {
      const parts: { name: string; positive: boolean }[][] = []
      for (const a of (x as { args: readonly Formula[] }).args) {
        const r = go(a)
        if (r === 'false') return 'false'
        if (r !== 'true') parts.push(...r)
      }
      return parts.length === 0 ? 'true' : parts
    }
    if (x.kind === inner) {
      let acc: { name: string; positive: boolean }[][] | 'true' | 'false' = 'false'
      for (const a of (x as { args: readonly Formula[] }).args) {
        const r = go(a)
        if (r === 'true') return 'true'
        if (r === 'false') continue
        acc = acc === 'false' ? r : acc.flatMap((g) => r.map((h) => [...g, ...h]))
      }
      return acc
    }
    throw new WhiteboxError('value', 'normal form: unexpected connective after NNF')
  }
  const r = go(nnf)
  if (r === 'true') return []
  if (r === 'false') return [[]]
  const seen = new Set<string>()
  const out: { name: string; positive: boolean }[][] = []
  for (const group of r) {
    const keys = new Map<string, { name: string; positive: boolean }>()
    for (const l of group) keys.set(literalKey(l), l)
    if ([...keys.values()].some((l) => keys.has(literalKey({ name: l.name, positive: !l.positive })))) continue
    const sorted = [...keys.values()].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : a.positive ? -1 : 1))
    const key = sorted.map(literalKey).join(',')
    if (seen.has(key)) continue
    seen.add(key)
    out.push(sorted)
  }
  // Drop groups subsumed by smaller ones (absorption).
  return out.filter(
    (g, i) =>
      !out.some(
        (h, j) =>
          j !== i &&
          h.length < g.length &&
          h.every((l) => g.some((x) => x.name === l.name && x.positive === l.positive)),
      ),
  )
}

function fromGroups(groups: readonly NamedClause[], outer: 'and' | 'or'): Formula {
  const lit = (l: { name: string; positive: boolean }): Formula => (l.positive ? v(l.name) : not(v(l.name)))
  const inner = (g: NamedClause): Formula => (outer === 'and' ? or(g.map(lit)) : and(g.map(lit)))
  return outer === 'and' ? and(groups.map(inner)) : or(groups.map(inner))
}

/**
 * Conjunctive normal form by distribution (exponential in the worst case; see {@link tseitin} for a linear,
 * equisatisfiable encoding), with tautological and subsumed clauses removed.
 *
 * @param f - The formula.
 * @returns An equivalent formula in CNF.
 * @concept normal-forms
 * @example
 * ```ts
 * import { formulaToString, parseFormula, toCnf } from 'whitebox-engine/logic'
 * formulaToString(toCnf(parseFormula('(p & q) | r'))) // => '(p | r) & (q | r)'
 * ```
 */
export function toCnf(f: Formula): Formula {
  return fromGroups(clausesOf(f, 'and'), 'and')
}

/**
 * Disjunctive normal form by distribution, with contradictory and subsumed terms removed.
 *
 * @param f - The formula.
 * @returns An equivalent formula in DNF.
 * @concept normal-forms
 * @example
 * ```ts
 * import { formulaToString, parseFormula, toDnf } from 'whitebox-engine/logic'
 * formulaToString(toDnf(parseFormula('(p | q) & ~p'))) // => '~p & q'
 * ```
 */
export function toDnf(f: Formula): Formula {
  return fromGroups(clausesOf(f, 'or'), 'or')
}

/**
 * The clauses of a formula's CNF over named literals (empty for a valid formula, one empty clause for an
 * unsatisfiable constant).
 *
 * @param f - The formula.
 * @returns The clauses.
 * @example
 * ```ts
 * import { cnfClauses, parseFormula } from 'whitebox-engine/logic'
 * cnfClauses(parseFormula('p -> q')) // => [[{ name: 'p', positive: false }, { name: 'q', positive: true }]]
 * ```
 */
export function cnfClauses(f: Formula): NamedClause[] {
  return clausesOf(f, 'and')
}

/** The result of {@link tseitin}: an equisatisfiable CNF and the DIMACS variable of each original variable. */
export interface TseitinEncoding {
  readonly cnf: Cnf
  /** Original variable name to DIMACS variable (1-based); auxiliary variables follow them. */
  readonly variables: ReadonlyMap<string, number>
  /** The DIMACS variable standing for the whole formula. */
  readonly root: number
}

/**
 * The Tseitin transformation: one fresh variable per connective with clauses defining it, plus a unit clause
 * asserting the root. Linear in the formula's size and equisatisfiable with it (the models agree on the original
 * variables).
 *
 * @param f - The formula.
 * @returns The CNF, the variable map and the root variable.
 * @concept normal-forms
 * @example
 * ```ts
 * import { parseFormula, tseitin } from 'whitebox-engine/logic'
 * const t = tseitin(parseFormula('(p & q) | ~p'))
 * [t.cnf.numVars, t.cnf.clauses.length] // => [4, 7]
 * ```
 */
export function tseitin(f: Formula): TseitinEncoding {
  const variables = new Map<string, number>()
  for (const name of formulaVariables(f)) variables.set(name, variables.size + 1)
  let next = variables.size
  const clauses: number[][] = []
  const fresh = (): number => ++next
  const go = (x: Formula): number => {
    switch (x.kind) {
      case 'var':
        return variables.get(x.name) as number
      case 'const': {
        const t = fresh()
        clauses.push([x.value ? t : -t])
        return t
      }
      case 'not':
        return -go(x.arg)
      case 'and':
      case 'or': {
        const args = x.args.map(go)
        const t = fresh()
        if (x.kind === 'and') {
          for (const a of args) clauses.push([-t, a])
          clauses.push([t, ...args.map((a) => -a)])
        } else {
          for (const a of args) clauses.push([t, -a])
          clauses.push([-t, ...args])
        }
        return t
      }
      case 'implies':
      case 'iff':
      case 'xor': {
        const a = go(x.left)
        const b = go(x.right)
        const t = fresh()
        if (x.kind === 'implies') clauses.push([-t, -a, b], [t, a], [t, -b])
        else if (x.kind === 'iff') clauses.push([-t, -a, b], [-t, a, -b], [t, a, b], [t, -a, -b])
        else clauses.push([-t, a, b], [-t, -a, -b], [t, -a, b], [t, a, -b])
        return t
      }
    }
  }
  const root = go(f)
  clauses.push([root])
  return { cnf: { numVars: next, clauses }, variables, root: Math.abs(root) }
}

/**
 * Simplify a formula algebraically: constants folded, double negations removed, nested conjunctions and disjunctions
 * flattened, duplicate operands dropped, complementary operands collapsed (`p & ~p` is false, `p | ~p` true), and
 * absorption (`p & (p | q)` is `p`). The result is equivalent to the input.
 *
 * @param f - The formula.
 * @returns The simplified formula.
 * @concept normal-forms
 * @example
 * ```ts
 * import { formulaToString, parseFormula, simplifyFormula } from 'whitebox-engine/logic'
 * formulaToString(simplifyFormula(parseFormula('~~p & (p | q) & true'))) // => 'p'
 * formulaToString(simplifyFormula(parseFormula('(q -> false) | (r & ~r)'))) // => '~q'
 * ```
 */
export function simplifyFormula(f: Formula): Formula {
  const key = (x: Formula): string => formulaToString(x)
  const go = (x: Formula): Formula => {
    switch (x.kind) {
      case 'var':
      case 'const':
        return x
      case 'not': {
        const a = go(x.arg)
        if (a.kind === 'const') return { kind: 'const', value: !a.value }
        if (a.kind === 'not') return a.arg
        return not(a)
      }
      case 'implies':
        return go({ kind: 'or', args: [not(x.left), x.right] })
      case 'iff':
      case 'xor': {
        const a = go(x.left)
        const b = go(x.right)
        const flip = x.kind === 'xor'
        if (a.kind === 'const') return go(a.value !== flip ? b : not(b))
        if (b.kind === 'const') return go(b.value !== flip ? a : not(a))
        if (key(a) === key(b)) return { kind: 'const', value: !flip }
        return { kind: x.kind, left: a, right: b }
      }
      case 'and':
      case 'or': {
        const isAnd = x.kind === 'and'
        const unit = isAnd // the identity: true for and, false for or
        const flat: Formula[] = []
        for (const a0 of x.args) {
          const a = go(a0)
          if (a.kind === x.kind) flat.push(...a.args)
          else flat.push(a)
        }
        const kept: Formula[] = []
        const seen = new Set<string>()
        for (const a of flat) {
          if (a.kind === 'const') {
            if (a.value === unit) continue
            return { kind: 'const', value: !unit }
          }
          const k = key(a)
          if (seen.has(k)) continue
          seen.add(k)
          kept.push(a)
        }
        for (const a of kept) {
          if (seen.has(key(a.kind === 'not' ? a.arg : not(a)))) return { kind: 'const', value: !unit }
        }
        // Absorption: drop an operand of the dual kind that contains another operand.
        const dual = isAnd ? 'or' : 'and'
        const absorbed = kept.filter(
          (a) => !(a.kind === dual && a.args.some((b) => kept.some((c) => c !== a && key(c) === key(b)))),
        )
        return isAnd ? and(absorbed) : or(absorbed)
      }
    }
  }
  // Iterate to a fixpoint (absorption can expose further simplifications).
  let cur = go(f)
  for (let i = 0; i < 8; i++) {
    const nxt = go(cur)
    if (key(nxt) === key(cur)) break
    cur = nxt
  }
  return cur
}

/**
 * The prime implicants of a formula (Quine-McCluskey): the conjunctions of literals that imply it and from which no
 * literal can be removed. Each is a map from variable to its required value.
 *
 * @param f - The formula.
 * @param variables - The variables (default the formula's own; at most 16).
 * @returns The prime implicants, sorted by size then text.
 * @throws WhiteboxError with code `value` for more than 16 variables.
 * @concept normal-forms
 * @example
 * ```ts
 * import { parseFormula, primeImplicants } from 'whitebox-engine/logic'
 * primeImplicants(parseFormula('(p & q) | (p & ~q) | (~p & q)')) // => [{ p: true }, { q: true }]
 * ```
 */
export function primeImplicants(f: Formula, variables: readonly string[] = formulaVariables(f)): Valuation[] {
  if (variables.length > 16) throw new WhiteboxError('value', 'primeImplicants: more than 16 variables')
  const k = variables.length
  // A cube: per variable 0 (false), 1 (true) or 2 (absent); as a string of digits.
  let cubes = new Set(
    truthTable(f, variables)
      .rows.map((r, m) => ({ r, m }))
      .filter((x) => x.r.value)
      .map((x) => Array.from({ length: k }, (_, i) => String((x.m >> (k - 1 - i)) & 1)).join('')),
  )
  const primes = new Set<string>()
  while (cubes.size > 0) {
    const used = new Set<string>()
    const next = new Set<string>()
    const list = [...cubes]
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        const a = list[i] as string
        const b = list[j] as string
        let diff = -1
        let count = 0
        for (let t = 0; t < k; t++) {
          if (a[t] !== b[t]) {
            count++
            diff = t
          }
        }
        if (count === 1 && a[diff] !== '2' && b[diff] !== '2') {
          next.add(`${a.slice(0, diff)}2${a.slice(diff + 1)}`)
          used.add(a)
          used.add(b)
        }
      }
    }
    for (const c of list) if (!used.has(c)) primes.add(c)
    cubes = next
  }
  const toValuation = (c: string): Valuation => {
    const out: Record<string, boolean> = {}
    for (let t = 0; t < k; t++) if (c[t] !== '2') out[variables[t] as string] = c[t] === '1'
    return out
  }
  // Fewest literals first; then positive literals and earlier variables first.
  const rank = (c: string): string => c.replace(/[012]/g, (d) => (d === '1' ? 'a' : d === '0' ? 'b' : 'c'))
  return [...primes]
    .sort((a, b) => countLits(a) - countLits(b) || (rank(a) < rank(b) ? -1 : rank(a) > rank(b) ? 1 : 0))
    .map(toValuation)
}

const countLits = (c: string): number => [...c].filter((x) => x !== '2').length

/**
 * A minimum disjunctive normal form: a smallest set of prime implicants covering every model (exact cover by
 * branch and bound over the prime implicant chart), ties broken by fewest literals.
 *
 * @param f - The formula.
 * @param variables - The variables (default the formula's own; at most 16).
 * @returns An equivalent DNF with the fewest terms, then literals.
 * @throws WhiteboxError with code `value` for more than 16 variables.
 * @concept normal-forms
 * @example
 * ```ts
 * import { formulaToString, minimalDnf, parseFormula } from 'whitebox-engine/logic'
 * formulaToString(minimalDnf(parseFormula('(p & q) | (p & ~q) | (~p & q)'))) // => 'p | q'
 * formulaToString(minimalDnf(parseFormula('p & ~p'))) // => 'false'
 * ```
 */
export function minimalDnf(f: Formula, variables: readonly string[] = formulaVariables(f)): Formula {
  const k = variables.length
  const primes = primeImplicants(f, variables)
  const minterms = truthTable(f, variables)
    .rows.map((r, m) => ({ r, m }))
    .filter((x) => x.r.value)
    .map((x) => x.m)
  if (minterms.length === 2 ** k) return T
  if (minterms.length === 0) return F
  const covers = primes.map((p) =>
    minterms.filter((m) =>
      variables.every((name, i) => p[name] === undefined || p[name] === (((m >> (k - 1 - i)) & 1) === 1)),
    ),
  )
  const size = (p: Valuation): number => Object.keys(p).length
  const best: { chosen: number[]; cost: [number, number] } = {
    chosen: [],
    cost: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
  }
  const search = (chosen: number[], covered: Set<number>): void => {
    const cost: [number, number] = [chosen.length, chosen.reduce((s, i) => s + size(primes[i] as Valuation), 0)]
    if (cost[0] > best.cost[0] || (cost[0] === best.cost[0] && cost[1] >= best.cost[1])) return
    const missing = minterms.find((m) => !covered.has(m))
    if (missing === undefined) {
      best.chosen = [...chosen]
      best.cost = cost
      return
    }
    // Branch on the implicants covering the first uncovered minterm (none of them is chosen yet).
    for (let i = 0; i < primes.length; i++) {
      if (!(covers[i] as number[]).includes(missing)) continue
      const added = (covers[i] as number[]).filter((m) => !covered.has(m))
      for (const m of added) covered.add(m)
      chosen.push(i)
      search(chosen, covered)
      chosen.pop()
      for (const m of added) covered.delete(m)
    }
  }
  search([], new Set())
  const chosen = [...best.chosen].sort((a, b) => a - b)
  const lit = (name: string, value: boolean): Formula => (value ? v(name) : not(v(name)))
  return or(
    chosen.map((i) => {
      const p = primes[i] as Valuation
      return and(variables.filter((n) => p[n] !== undefined).map((n) => lit(n, p[n] as boolean)))
    }),
  )
}

/**
 * A minimum conjunctive normal form (the dual of {@link minimalDnf}: a minimum DNF of the negation, negated).
 *
 * @param f - The formula.
 * @param variables - The variables (default the formula's own; at most 16).
 * @returns An equivalent CNF with the fewest clauses, then literals.
 * @throws WhiteboxError with code `value` for more than 16 variables.
 * @concept normal-forms
 * @example
 * ```ts
 * import { formulaToString, minimalCnf, parseFormula } from 'whitebox-engine/logic'
 * formulaToString(minimalCnf(parseFormula('(p | q) & (p | ~q) & r'))) // => 'p & r'
 * ```
 */
export function minimalCnf(f: Formula, variables: readonly string[] = formulaVariables(f)): Formula {
  return toNnf(not(minimalDnf(not(f), variables)))
}
