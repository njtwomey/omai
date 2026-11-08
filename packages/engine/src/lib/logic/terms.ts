/**
 * First-order terms, the data of logic programming: atoms, variables, numbers, compounds and lists, with clauses built
 * from them. Constructors, groundness and variable collection, structural equality and Prolog's standard order of
 * terms, conjunction helpers, the operator table, and printing terms and clauses as Prolog text.
 *
 * @index structure First-order terms and clauses, the data of logic programming.
 */

import { OmaiError } from '../base/index.ts'

/** A constant symbol (Prolog atom): `a`, `'hello world'`, `[]`. */
export interface Atom {
  readonly kind: 'atom'
  readonly name: string
}

/** A logic variable, identified by its name. */
export interface Var {
  readonly kind: 'var'
  readonly name: string
}

/** A number; `float` distinguishes `2.0` from `2`, as Prolog does. */
export interface Num {
  readonly kind: 'num'
  readonly value: number
  readonly float: boolean
}

/** A compound term `f(t1, ..., tn)` with at least one argument. */
export interface Compound {
  readonly kind: 'compound'
  readonly functor: string
  readonly args: readonly Term[]
}

/**
 * A first-order term: an atom, a variable, a number or a compound. Lists are the compounds `'.'(Head, Tail)` ending in
 * the atom `[]`, as in ISO Prolog.
 *
 * @concept logic-terms
 */
export type Term = Atom | Var | Num | Compound

/**
 * A definite or normal clause `head :- body1, ..., bodyn` (a fact when the body is empty). Body goals may themselves be
 * control constructs (`;`, `->`, `\+`) in Prolog programs.
 */
export interface Clause {
  readonly head: Term
  readonly body: readonly Term[]
}

/**
 * The empty list `[]`.
 *
 * @example
 * ```ts
 * import { NIL, termToString } from 'omai-engine/logic'
 * termToString(NIL) // => '[]'
 * ```
 */
export const NIL: Atom = { kind: 'atom', name: '[]' }

/**
 * The list constructor functor (ISO's `'.'`).
 *
 * @example
 * ```ts
 * import { LIST_FUNCTOR, parseTerm } from 'omai-engine/logic'
 * parseTerm('[a]').functor === LIST_FUNCTOR // => true
 * ```
 */
export const LIST_FUNCTOR = '.'

/**
 * An atom.
 *
 * @param name - The symbol.
 * @returns The atom.
 * @example
 * ```ts
 * import { atom } from 'omai-engine/logic'
 * atom('socrates') // => { kind: 'atom', name: 'socrates' }
 * ```
 */
export function atom(name: string): Atom {
  return { kind: 'atom', name }
}

/**
 * A variable.
 *
 * @param name - The variable's name (by convention capitalised or starting with `_`).
 * @returns The variable.
 * @example
 * ```ts
 * import { variable } from 'omai-engine/logic'
 * variable('X').kind // => 'var'
 * ```
 */
export function variable(name: string): Var {
  return { kind: 'var', name }
}

/**
 * A number term.
 *
 * @param value - The value.
 * @param float - Whether it is a float (default: when the value is not an integer).
 * @returns The number.
 * @throws OmaiError with code `value` for an integer term that is not an integer.
 * @example
 * ```ts
 * import { num } from 'omai-engine/logic'
 * [num(3).float, num(2.5).float, num(2, true).float] // => [false, true, true]
 * ```
 */
export function num(value: number, float?: boolean): Num {
  const isFloat = float ?? !Number.isInteger(value)
  if (!isFloat && !Number.isInteger(value)) throw new OmaiError('value', `num: ${value} is not an integer`)
  return { kind: 'num', value, float: isFloat }
}

/**
 * A compound term; with no arguments it is the atom of that name.
 *
 * @param functor - The function symbol.
 * @param args - The arguments.
 * @returns The term.
 * @example
 * ```ts
 * import { atom, compound, termToString, variable } from 'omai-engine/logic'
 * termToString(compound('parent', [atom('tom'), variable('X')])) // => 'parent(tom,X)'
 * compound('nil', []).kind // => 'atom'
 * ```
 */
export function compound(functor: string, args: readonly Term[]): Compound | Atom {
  return args.length === 0 ? atom(functor) : { kind: 'compound', functor, args }
}

/**
 * A list term `[t1, ..., tn | tail]`.
 *
 * @param items - The elements.
 * @param tail - The tail (default `[]`).
 * @returns The list term.
 * @example
 * ```ts
 * import { atom, list, termToString, variable } from 'omai-engine/logic'
 * termToString(list([atom('a'), atom('b')])) // => '[a,b]'
 * termToString(list([atom('a')], variable('T'))) // => '[a|T]'
 * ```
 */
export function list(items: readonly Term[], tail: Term = NIL): Term {
  let out = tail
  for (let i = items.length - 1; i >= 0; i--) {
    out = { kind: 'compound', functor: LIST_FUNCTOR, args: [items[i] as Term, out] }
  }
  return out
}

/**
 * The elements of a proper list, or null when the term is not one (a partial list or not a list).
 *
 * @param t - The term.
 * @returns The elements, or null.
 * @example
 * ```ts
 * import { listItems, parseTerm } from 'omai-engine/logic'
 * listItems(parseTerm('[1,2,3]')).length // => 3
 * listItems(parseTerm('[1|T]')) // => null
 * ```
 */
export function listItems(t: Term): Term[] | null {
  const out: Term[] = []
  let cur = t
  while (cur.kind === 'compound' && cur.functor === LIST_FUNCTOR && cur.args.length === 2) {
    out.push(cur.args[0] as Term)
    cur = cur.args[1] as Term
  }
  return cur.kind === 'atom' && cur.name === '[]' ? out : null
}

/**
 * Whether a term contains no variables.
 *
 * @param t - The term.
 * @returns True when ground.
 * @example
 * ```ts
 * import { isGround, parseTerm } from 'omai-engine/logic'
 * [isGround(parseTerm('f(a,[b])')), isGround(parseTerm('f(a,X)'))] // => [true, false]
 * ```
 */
export function isGround(t: Term): boolean {
  const stack: Term[] = [t]
  for (let x = stack.pop(); x !== undefined; x = stack.pop()) {
    if (x.kind === 'var') return false
    if (x.kind === 'compound') for (const a of x.args) stack.push(a)
  }
  return true
}

/**
 * The distinct variables of terms, in depth-first left-to-right order of first occurrence (Prolog `term_variables/2`).
 *
 * @param terms - A term or several.
 * @returns The variables.
 * @example
 * ```ts
 * import { parseTerm, termVariables } from 'omai-engine/logic'
 * termVariables(parseTerm('f(Y,g(X,Y),Z)')).map((v) => v.name) // => ['Y', 'X', 'Z']
 * ```
 */
export function termVariables(terms: Term | readonly Term[]): Var[] {
  const seen = new Set<string>()
  const out: Var[] = []
  const stack: Term[] = Array.isArray(terms) ? [...(terms as readonly Term[])].reverse() : [terms as Term]
  for (let x = stack.pop(); x !== undefined; x = stack.pop()) {
    if (x.kind === 'var') {
      if (!seen.has(x.name)) {
        seen.add(x.name)
        out.push(x)
      }
    } else if (x.kind === 'compound') for (let i = x.args.length - 1; i >= 0; i--) stack.push(x.args[i] as Term)
  }
  return out
}

/**
 * Structural identity (Prolog `==`): the same shape with the same atoms, numbers and variable names. `1` and `1.0` are
 * not identical.
 *
 * @param a - A term.
 * @param b - Another term.
 * @returns True when identical.
 * @example
 * ```ts
 * import { parseTerm, termEquals } from 'omai-engine/logic'
 * [termEquals(parseTerm('f(X,a)'), parseTerm('f(X,a)')), termEquals(parseTerm('f(X)'), parseTerm('f(Y)'))] // => [true, false]
 * ```
 */
export function termEquals(a: Term, b: Term): boolean {
  return compareTerms(a, b) === 0
}

const KIND_ORDER: Readonly<Record<Term['kind'], number>> = { var: 0, num: 1, atom: 2, compound: 3 }

function compareStrings(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0
}

/**
 * The standard order of terms (Prolog `compare/3`): variables before numbers before atoms before compounds; numbers by
 * value (a float before an equal integer), atoms alphabetically by character code, compounds by arity, then name, then
 * arguments left to right. Variables are ordered by name.
 *
 * @param a - A term.
 * @param b - Another term.
 * @returns A negative number, zero or a positive number.
 * @example
 * ```ts
 * import { compareTerms, parseTerm } from 'omai-engine/logic'
 * const ts = ['f(a)', 'b', '1', 'X', 'g(a,b)', '2.0', 'a'].map((s) => parseTerm(s))
 * import { termToString } from 'omai-engine/logic'
 * ts.sort(compareTerms).map((t) => termToString(t)) // => ['X', '1', '2.0', 'a', 'b', 'f(a)', 'g(a,b)']
 * ```
 */
export function compareTerms(a: Term, b: Term): number {
  if (a.kind !== b.kind) return KIND_ORDER[a.kind] - KIND_ORDER[b.kind]
  switch (a.kind) {
    case 'var':
      return compareStrings(a.name, (b as Var).name)
    case 'atom':
      return compareStrings(a.name, (b as Atom).name)
    case 'num': {
      const y = b as Num
      if (a.value !== y.value) return a.value < y.value ? -1 : 1
      return a.float === y.float ? 0 : a.float ? -1 : 1
    }
    case 'compound': {
      const y = b as Compound
      if (a.args.length !== y.args.length) return a.args.length - y.args.length
      const c = compareStrings(a.functor, y.functor)
      if (c !== 0) return c
      for (let i = 0; i < a.args.length; i++) {
        const d = compareTerms(a.args[i] as Term, y.args[i] as Term)
        if (d !== 0) return d
      }
      return 0
    }
  }
}

/**
 * The predicate indicator `name/arity` of a callable term.
 *
 * @param t - An atom or compound.
 * @returns The indicator.
 * @throws OmaiError with code `value` for a variable or number.
 * @example
 * ```ts
 * import { indicator, parseTerm } from 'omai-engine/logic'
 * [indicator(parseTerm('parent(a,b)')), indicator(parseTerm('halt'))] // => ['parent/2', 'halt/0']
 * ```
 */
export function indicator(t: Term): string {
  if (t.kind === 'atom') return `${t.name}/0`
  if (t.kind === 'compound') return `${t.functor}/${t.args.length}`
  throw new OmaiError('value', `indicator: ${termToString(t)} is not callable`)
}

/**
 * The goals of a conjunction `(a, b, c)` as a list (a non-conjunction is a one-goal list; `true` is empty).
 *
 * @param t - The term.
 * @returns The conjuncts.
 * @example
 * ```ts
 * import { conjuncts, parseTerm } from 'omai-engine/logic'
 * conjuncts(parseTerm('(a, (b, c))')).length // => 3
 * ```
 */
export function conjuncts(t: Term): Term[] {
  const out: Term[] = []
  const stack: Term[] = [t]
  for (let x = stack.pop(); x !== undefined; x = stack.pop()) {
    if (x.kind === 'compound' && x.functor === ',' && x.args.length === 2) {
      stack.push(x.args[1] as Term, x.args[0] as Term)
    } else if (!(x.kind === 'atom' && x.name === 'true')) out.push(x)
  }
  return out
}

/**
 * The conjunction of goals (`true` when empty).
 *
 * @param goals - The goals.
 * @returns The term `(g1, (g2, ...))`.
 * @example
 * ```ts
 * import { conjunction, parseTerm, termToString } from 'omai-engine/logic'
 * termToString(conjunction([parseTerm('a'), parseTerm('b')])) // => 'a,b'
 * ```
 */
export function conjunction(goals: readonly Term[]): Term {
  if (goals.length === 0) return atom('true')
  let out = goals[goals.length - 1] as Term
  for (let i = goals.length - 2; i >= 0; i--) out = { kind: 'compound', functor: ',', args: [goals[i] as Term, out] }
  return out
}

/**
 * A clause from a term: `H :- B` gives head `H` and the conjuncts of `B`; any other term is a fact.
 *
 * @param t - The term.
 * @returns The clause.
 * @example
 * ```ts
 * import { parseTerm, termToClause } from 'omai-engine/logic'
 * termToClause(parseTerm('p(X) :- q(X), r(X)')).body.length // => 2
 * ```
 */
export function termToClause(t: Term): Clause {
  if (t.kind === 'compound' && t.functor === ':-' && t.args.length === 2) {
    return { head: t.args[0] as Term, body: conjuncts(t.args[1] as Term) }
  }
  return { head: t, body: [] }
}

/**
 * A clause as a term: `H :- B1, ..., Bn`, or `H` for a fact.
 *
 * @param c - The clause.
 * @returns The term.
 * @example
 * ```ts
 * import { clauseToTerm, parseClause, termToString } from 'omai-engine/logic'
 * termToString(clauseToTerm(parseClause('p(X) :- q(X).'))) // => 'p(X):-q(X)'
 * ```
 */
export function clauseToTerm(c: Clause): Term {
  return c.body.length === 0 ? c.head : { kind: 'compound', functor: ':-', args: [c.head, conjunction(c.body)] }
}

// ------------------------------------------------------------------------------------------------ printing

/** The operator table used for printing (and parsing): `type` is one of xfx, xfy, yfx, fy, fx, xf, yf. */
export interface OperatorDef {
  readonly priority: number
  readonly type: 'xfx' | 'xfy' | 'yfx' | 'fy' | 'fx' | 'xf' | 'yf'
}

/** Operators by name: prefix, infix and postfix definitions. */
export interface Operators {
  readonly prefix: ReadonlyMap<string, OperatorDef>
  readonly infix: ReadonlyMap<string, OperatorDef>
  readonly postfix: ReadonlyMap<string, OperatorDef>
}

const INFIX: readonly [number, OperatorDef['type'], string][] = [
  [1200, 'xfx', ':- -->'],
  [1100, 'xfy', '; |'],
  [1105, 'xfy', '|'],
  [1050, 'xfy', '-> *->'],
  [1000, 'xfy', ','],
  [990, 'xfx', ':='],
  [700, 'xfx', '= \\= == \\== @< @> @=< @>= =.. is =:= =\\= < > =< >= as >:< :<'],
  [600, 'xfy', ':'],
  [500, 'yfx', '+ - /\\ \\/ xor'],
  [400, 'yfx', '* / // rdiv << >> mod rem div divmod'],
  [200, 'xfx', '**'],
  [200, 'xfy', '^'],
]
const PREFIX: readonly [number, OperatorDef['type'], string][] = [
  [1200, 'fx', ':- ?-'],
  [1150, 'fx', 'dynamic discontiguous initialization table'],
  [900, 'fy', '\\+'],
  [200, 'fy', '- + \\'],
  [100, 'fx', '@'],
  [1, 'fx', '$'],
]

function table(rows: readonly [number, OperatorDef['type'], string][]): Map<string, OperatorDef> {
  const m = new Map<string, OperatorDef>()
  for (const [priority, type, names] of rows) for (const n of names.split(' ')) m.set(n, { priority, type })
  return m
}

/**
 * The standard operator table (ISO plus the common SWI-Prolog additions).
 *
 * @returns A fresh table.
 * @example
 * ```ts
 * import { standardOperators } from 'omai-engine/logic'
 * standardOperators().infix.get(':-') // => { priority: 1200, type: 'xfx' }
 * ```
 */
export function standardOperators(): Operators {
  return { prefix: table(PREFIX), infix: table(INFIX), postfix: new Map() }
}

const DEFAULT_OPS = standardOperators()

const SYMBOL_CHARS = '+-*/\\^<>=~:.?@#&$'
const isSymbolChar = (c: string): boolean => SYMBOL_CHARS.includes(c)
const isAlnum = (c: string): boolean => /[A-Za-z0-9_]/.test(c)

/**
 * Whether an atom can be written without quotes: a lower-case identifier, a run of symbol characters, or one of the
 * solo atoms `[]`, `!`, `;`, `{}`.
 *
 * @param name - The atom's name.
 * @returns True when no quotes are needed.
 * @example
 * ```ts
 * import { isPlainAtom } from 'omai-engine/logic'
 * ['foo', 'Foo', 'a b', '+', '[]', ''].map(isPlainAtom) // => [true, false, false, true, true, false]
 * ```
 */
export function isPlainAtom(name: string): boolean {
  if (name === '[]' || name === '!' || name === ';' || name === '{}') return true
  if (name.length === 0) return false
  const c = name[0] as string
  if (c >= 'a' && c <= 'z') return [...name].every(isAlnum)
  return [...name].every(isSymbolChar)
}

function quoteAtom(name: string): string {
  if (isPlainAtom(name)) return name
  let out = "'"
  for (const c of name) {
    if (c === "'") out += "\\'"
    else if (c === '\\') out += '\\\\'
    else if (c === '\n') out += '\\n'
    else if (c === '\t') out += '\\t'
    else out += c
  }
  return `${out}'`
}

function numberToString(n: Num): string {
  if (!n.float) return String(n.value)
  if (Number.isNaN(n.value)) return 'nan'
  if (!Number.isFinite(n.value)) return n.value > 0 ? 'inf' : '-inf'
  const s = String(n.value)
  if (/[.e]/.test(s)) return s.includes('.') || !s.includes('e') ? s : s.replace('e', '.0e')
  return `${s}.0`
}

/** Options of {@link termToString}. */
export interface WriteOptions {
  /** Quote atoms that need it (default true, as `writeq`; false gives `write`). */
  readonly quoted?: boolean
  /** Write operators in canonical functional notation (default false). */
  readonly ignoreOps?: boolean
  /** The operator table (default the standard one). */
  readonly operators?: Operators
  /** Write lists in functional notation too (only with `ignoreOps`; default false). */
  readonly listNotation?: boolean
}

/**
 * Write a term as Prolog text (`writeq/1` by default): operators in infix or prefix form with brackets only where the
 * priorities require them, lists in bracket notation, atoms quoted where needed. With `ignoreOps` the term is written in
 * canonical functional notation (`+(1,2)`), as `write_canonical/1` does.
 *
 * @param t - The term.
 * @param options - Quoting, operator handling and the operator table.
 * @returns The text.
 * @example
 * ```ts
 * import { parseTerm, termToString } from 'omai-engine/logic'
 * termToString(parseTerm('(1+2)*3-f(-1)')) // => '(1+2)*3-f(-1)'
 * termToString(parseTerm('a:-b,c;d')) // => 'a:-b,c;d'
 * termToString(parseTerm("'hello world'(X)"), { quoted: false }) // => 'hello world(X)'
 * termToString(parseTerm('1+2'), { ignoreOps: true }) // => '+(1,2)'
 * ```
 */
export function termToString(t: Term, options: WriteOptions = {}): string {
  const quoted = options.quoted ?? true
  const ops = options.operators ?? DEFAULT_OPS
  const ignoreOps = options.ignoreOps ?? false
  const name = (s: string): string => (quoted ? quoteAtom(s) : s)
  const write = (x: Term, max: number): string => {
    switch (x.kind) {
      case 'var':
        return x.name
      case 'num':
        return numberToString(x)
      case 'atom':
        return name(x.name)
      case 'compound':
        return writeCompound(x, max)
    }
  }
  const writeArg = (x: Term): string => write(x, 999)
  // An atom that is itself an operator is bracketed when it is an operand, so that it reads back as an atom.
  const isOpAtom = (x: Term): boolean =>
    x.kind === 'atom' && (ops.prefix.has(x.name) || ops.infix.has(x.name) || ops.postfix.has(x.name))
  const operand = (x: Term, max: number): string => (isOpAtom(x) ? `(${name((x as Atom).name)})` : write(x, max))
  const writeCompound = (x: Compound, max: number): string => {
    if (x.functor === LIST_FUNCTOR && x.args.length === 2 && !(ignoreOps && options.listNotation === true)) {
      const parts: string[] = []
      let cur: Term = x
      while (cur.kind === 'compound' && cur.functor === LIST_FUNCTOR && cur.args.length === 2) {
        parts.push(writeArg(cur.args[0] as Term))
        cur = cur.args[1] as Term
      }
      const tail = cur.kind === 'atom' && cur.name === '[]' ? '' : `|${writeArg(cur)}`
      return `[${parts.join(',')}${tail}]`
    }
    if (!ignoreOps && x.functor === '{}' && x.args.length === 1) return `{${write(x.args[0] as Term, 1200)}}`
    if (!ignoreOps && x.args.length === 2) {
      const op = ops.infix.get(x.functor)
      if (op !== undefined) {
        const lmax = op.type === 'yfx' ? op.priority : op.priority - 1
        const rmax = op.type === 'xfy' ? op.priority : op.priority - 1
        const left = operand(x.args[0] as Term, lmax)
        const right = operand(x.args[1] as Term, rmax)
        const alpha = /^[a-z]/.test(x.functor)
        const sym = x.functor === ',' ? ',' : name(x.functor)
        // Separate the operator from symbolic neighbours that would otherwise run together into one token.
        const glueL = alpha || (isSymbolChar(left.slice(-1)) && isSymbolChar(sym[0] ?? '')) ? ' ' : ''
        const glueR = alpha || (isSymbolChar(right[0] ?? '') && isSymbolChar(sym.slice(-1))) ? ' ' : ''
        const s = `${left}${glueL}${sym}${glueR}${right}`
        return op.priority > max ? `(${s})` : s
      }
    }
    if (!ignoreOps && x.args.length === 1) {
      const op = ops.prefix.get(x.functor)
      const arg = x.args[0] as Term
      if (op !== undefined && x.functor !== '-' && x.functor !== '+') {
        const amax = op.type === 'fy' ? op.priority : op.priority - 1
        const inner = operand(arg, amax)
        const alpha = /^[a-z]/.test(x.functor)
        const sep = alpha || isSymbolChar(inner[0] ?? '') || inner[0] === '(' ? ' ' : ''
        const s = `${name(x.functor)}${sep}${inner}`
        return op.priority > max ? `(${s})` : s
      }
      if (op !== undefined) {
        // -(1) is written `- 1` so that it does not read back as the number -1; -(a) is `-a`.
        const amax = op.type === 'fy' ? op.priority : op.priority - 1
        const inner = operand(arg, amax)
        const sep = /^[0-9(]/.test(inner) || isSymbolChar(inner[0] ?? '') ? ' ' : ''
        const s = `${x.functor}${sep}${inner}`
        return op.priority > max ? `(${s})` : s
      }
    }
    return `${name(x.functor)}(${x.args.map(writeArg).join(',')})`
  }
  return write(t, 1200)
}

/**
 * Write a clause as Prolog source text, ending with a full stop.
 *
 * @param c - The clause.
 * @param options - Writing options as for {@link termToString}.
 * @returns The text.
 * @example
 * ```ts
 * import { clauseToString, parseClause } from 'omai-engine/logic'
 * clauseToString(parseClause('grandparent(X,Z) :- parent(X,Y), parent(Y,Z).')) // => 'grandparent(X,Z) :- parent(X,Y), parent(Y,Z).'
 * ```
 */
export function clauseToString(c: Clause, options: WriteOptions = {}): string {
  const head = termToString(c.head, options)
  if (c.body.length === 0) return `${head}.`
  return `${head} :- ${c.body.map((g) => termToString(g, { ...options })).join(', ')}.`
}

/**
 * The number of symbol occurrences in a term (atoms, numbers, variables and functors each count one).
 *
 * @param t - The term.
 * @returns The size.
 * @example
 * ```ts
 * import { parseTerm, termSize } from 'omai-engine/logic'
 * termSize(parseTerm('f(a,g(X))')) // => 4
 * ```
 */
export function termSize(t: Term): number {
  let n = 0
  const stack: Term[] = [t]
  for (let x = stack.pop(); x !== undefined; x = stack.pop()) {
    n += 1
    if (x.kind === 'compound') for (const a of x.args) stack.push(a)
  }
  return n
}
