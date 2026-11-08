/**
 * A Prolog reader: a tokeniser for Prolog text, and parsers for terms, clauses, whole programs with directives and
 * `op/3` operator declarations, and queries that keep their variable names. Also the translation of definite clause
 * grammar rules into ordinary clauses with difference lists.
 */

import { OmaiError } from '../base/index.ts'
import {
  atom,
  type Clause,
  compound,
  conjuncts,
  list,
  listItems,
  NIL,
  num,
  type OperatorDef,
  type Operators,
  standardOperators,
  type Term,
  termToClause,
  variable,
} from './terms.ts'

/** A token of Prolog text. `layoutBefore` records whether white space or a comment preceded it. */
export interface Token {
  readonly type: 'atom' | 'var' | 'int' | 'float' | 'string' | 'punct' | 'end' | 'backquote'
  readonly text: string
  readonly value?: number
  readonly layoutBefore: boolean
  readonly offset: number
}

const SYMBOL = '+-*/\\^<>=~:.?@#&$'
const isDigit = (c: string | undefined): boolean => c !== undefined && c >= '0' && c <= '9'
const isAlnum = (c: string | undefined): boolean => c !== undefined && /[A-Za-z0-9_]/.test(c)
const isLayout = (c: string | undefined): boolean => c !== undefined && /\s/.test(c)

function syntaxError(message: string, offset: number): OmaiError {
  return new OmaiError('value', `syntax error at ${offset}: ${message}`, { offset })
}

function readEscape(text: string, i: number, quote: string): [string, number] {
  const c = text[i]
  const simple: Readonly<Record<string, string>> = {
    n: '\n',
    t: '\t',
    r: '\r',
    a: '\x07',
    b: '\b',
    f: '\f',
    v: '\v',
    '0': '\0',
    '\\': '\\',
    "'": "'",
    '"': '"',
    '`': '`',
    e: '\x1b',
    s: ' ',
  }
  if (c === undefined) throw syntaxError('unterminated escape', i)
  if (c === '\n') return ['', i + 1]
  if (c === 'x') {
    let j = i + 1
    let hex = ''
    while (j < text.length && /[0-9a-fA-F]/.test(text[j] ?? '')) hex += text[j++]
    if (text[j] === '\\') j++
    return [String.fromCodePoint(Number.parseInt(hex, 16)), j]
  }
  if (isDigit(c) && c !== '0') {
    let j = i
    let oct = ''
    while (isDigit(text[j])) oct += text[j++]
    if (text[j] === '\\') j++
    return [String.fromCodePoint(Number.parseInt(oct, 8)), j]
  }
  const s = simple[c]
  if (s === undefined) throw syntaxError(`unknown escape \\${c} in ${quote}-quoted text`, i)
  return [s, i + 1]
}

/**
 * Split Prolog text into tokens: atoms (identifiers, symbol runs, quoted, solo), variables, integers (decimal, `0x`,
 * `0o`, `0b`, `0'c` character codes), floats, double-quoted strings, punctuation and clause ends. `%` line comments and
 * `/* *\/` block comments are skipped.
 *
 * @param text - The source.
 * @returns The tokens.
 * @throws OmaiError with code `value` for an unterminated quote or comment.
 * @example
 * ```ts
 * import { tokenize } from 'omai-engine/logic'
 * tokenize("p(X) :- q('a b', 0'c). % done").map((t) => t.text) // => ['p', '(', 'X', ')', ':-', 'q', '(', 'a b', ',', '99', ')', '.']
 * ```
 */
export function tokenize(text: string): Token[] {
  const out: Token[] = []
  let i = 0
  let layout = true
  const push = (type: Token['type'], tokText: string, offset: number, value?: number): void => {
    out.push({ type, text: tokText, layoutBefore: layout, offset, ...(value === undefined ? {} : { value }) })
    layout = false
  }
  while (i < text.length) {
    const c = text[i] as string
    if (isLayout(c)) {
      layout = true
      i++
      continue
    }
    if (c === '%') {
      while (i < text.length && text[i] !== '\n') i++
      layout = true
      continue
    }
    if (c === '/' && text[i + 1] === '*') {
      const end = text.indexOf('*/', i + 2)
      if (end < 0) throw syntaxError('unterminated block comment', i)
      i = end + 2
      layout = true
      continue
    }
    const start = i
    if (isDigit(c)) {
      if (c === '0' && text[i + 1] === "'") {
        const q = text[i + 2]
        if (q === '\\') {
          const [ch, j] = readEscape(text, i + 3, "'")
          push('int', String(ch.codePointAt(0) ?? 0), start, ch.codePointAt(0) ?? 0)
          i = j
          continue
        }
        if (q === "'" && text[i + 3] === "'") {
          push('int', '39', start, 39)
          i += 4
          continue
        }
        if (q !== undefined) {
          const cp = q.codePointAt(0) ?? 0
          push('int', String(cp), start, cp)
          i += 3
          continue
        }
      }
      const radix =
        c === '0' ? ({ x: 16, o: 8, b: 2 } as Readonly<Record<string, number>>)[text[i + 1] ?? ''] : undefined
      if (radix !== undefined) {
        let j = i + 2
        let digits = ''
        while (j < text.length && /[0-9a-fA-F]/.test(text[j] ?? '') && Number.parseInt(text[j] ?? '', 16) < radix)
          digits += text[j++]
        if (digits.length > 0) {
          const v = Number.parseInt(digits, radix)
          push('int', String(v), start, v)
          i = j
          continue
        }
      }
      let j = i
      while (isDigit(text[j])) j++
      let isFloat = false
      if (text[j] === '.' && isDigit(text[j + 1])) {
        isFloat = true
        j++
        while (isDigit(text[j])) j++
      }
      if (
        (text[j] === 'e' || text[j] === 'E') &&
        (isDigit(text[j + 1]) || (/[+-]/.test(text[j + 1] ?? '') && isDigit(text[j + 2])))
      ) {
        isFloat = true
        j += 2
        while (isDigit(text[j])) j++
      }
      const s = text.slice(i, j)
      push(isFloat ? 'float' : 'int', s, start, Number(s))
      i = j
      continue
    }
    if (c === '_' || (c >= 'A' && c <= 'Z')) {
      let j = i + 1
      while (isAlnum(text[j])) j++
      push('var', text.slice(i, j), start)
      i = j
      continue
    }
    if (c >= 'a' && c <= 'z') {
      let j = i + 1
      while (isAlnum(text[j])) j++
      push('atom', text.slice(i, j), start)
      i = j
      continue
    }
    if (c === "'" || c === '"' || c === '`') {
      let j = i + 1
      let s = ''
      for (;;) {
        const d = text[j]
        if (d === undefined) throw syntaxError(`unterminated ${c}-quoted text`, start)
        if (d === c) {
          if (text[j + 1] === c) {
            s += c
            j += 2
            continue
          }
          j++
          break
        }
        if (d === '\\') {
          const [e, k] = readEscape(text, j + 1, c)
          s += e
          j = k
          continue
        }
        s += d
        j++
      }
      push(c === "'" ? 'atom' : c === '"' ? 'string' : 'backquote', s, start)
      i = j
      continue
    }
    if (c === '(' || c === ')' || c === '[' || c === ']' || c === '{' || c === '}' || c === ',' || c === '|') {
      if (c === '|' && text[i + 1] === '|') {
        push('atom', '||', start)
        i += 2
        continue
      }
      push('punct', c, start)
      i++
      continue
    }
    if (c === '!' || c === ';') {
      push('atom', c, start)
      i++
      continue
    }
    if (SYMBOL.includes(c)) {
      if (c === '.' && (i + 1 >= text.length || isLayout(text[i + 1]) || text[i + 1] === '%')) {
        push('end', '.', start)
        i++
        continue
      }
      let j = i
      while (j < text.length && SYMBOL.includes(text[j] ?? '')) j++
      push('atom', text.slice(i, j), start)
      i = j
      continue
    }
    throw syntaxError(`unexpected character ${JSON.stringify(c)}`, i)
  }
  return out
}

/** Options of the parsers. */
export interface ParseOptions {
  /** The operator table (default the standard one); `op/3` directives in a program extend a copy of it. */
  readonly operators?: Operators
  /** How double-quoted text reads: a list of character codes (default, ISO), of one-character atoms, or an atom. */
  readonly doubleQuotes?: 'codes' | 'chars' | 'atom'
}

class Parser {
  private pos = 0
  private anonymous = 0
  /** Source names of the variables of the current term, in order of first occurrence. */
  names: string[] = []

  constructor(
    private readonly tokens: readonly Token[],
    private readonly ops: Operators,
    private readonly doubleQuotes: 'codes' | 'chars' | 'atom',
  ) {}

  atEnd(): boolean {
    return this.pos >= this.tokens.length
  }

  private peek(k = 0): Token | undefined {
    return this.tokens[this.pos + k]
  }

  private next(): Token {
    const t = this.tokens[this.pos]
    if (t === undefined) throw syntaxError('unexpected end of text', this.tokens[this.tokens.length - 1]?.offset ?? 0)
    this.pos++
    return t
  }

  private expect(type: Token['type'], text: string): void {
    const t = this.next()
    if (t.type !== type || t.text !== text) throw syntaxError(`expected ${text}, found ${t.text}`, t.offset)
  }

  /** Read one clause-level term up to its end token. */
  readClauseTerm(): Term {
    this.names = []
    this.anonymous = 0
    const t = this.parse(1200)
    const end = this.next()
    if (end.type !== 'end') throw syntaxError(`operator expected, found ${end.text}`, end.offset)
    return t
  }

  /** Read one term to the end of the tokens (an optional final full stop is allowed). */
  readWhole(): Term {
    this.names = []
    const t = this.parse(1200)
    if (this.peek()?.type === 'end') this.pos++
    const rest = this.peek()
    if (rest !== undefined) throw syntaxError(`operator expected, found ${rest.text}`, rest.offset)
    return t
  }

  /** Whether the token `k` places ahead can start a term. */
  private isTermStart(k: number): boolean {
    const t = this.peek(k)
    if (t === undefined || t.type === 'end') return false
    if (t.type === 'punct') return t.text === '(' || t.text === '[' || t.text === '{'
    if (t.type === 'atom' && this.ops.infix.has(t.text) && !this.ops.prefix.has(t.text)) {
      // An infix operator starts a term only in functional notation, e.g. `\+ =(a, b)`.
      const n = this.peek(k + 1)
      return n !== undefined && n.type === 'punct' && n.text === '(' && !n.layoutBefore
    }
    return true
  }

  private parse(max: number): Term {
    let [left, leftPri] = this.primary(max)
    for (;;) {
      const t = this.peek()
      if (t === undefined) break
      let name: string | undefined
      if (t.type === 'atom') name = t.text
      else if (t.type === 'punct' && (t.text === ',' || t.text === '|')) name = t.text
      if (name === undefined) break
      const infix = this.ops.infix.get(name)
      if (infix !== undefined) {
        const p = infix.priority
        const lmax = infix.type === 'yfx' ? p : p - 1
        const rmax = infix.type === 'xfy' ? p : p - 1
        if (p <= max && leftPri <= lmax) {
          this.pos++
          const right = this.parse(rmax)
          const functor = name === '|' ? ';' : name
          left = compound(functor, [left, right])
          leftPri = p
          continue
        }
      }
      const postfix = this.ops.postfix.get(name)
      if (postfix !== undefined) {
        const p = postfix.priority
        const amax = postfix.type === 'yf' ? p : p - 1
        if (p <= max && leftPri <= amax) {
          this.pos++
          left = compound(name, [left])
          leftPri = p
          continue
        }
      }
      break
    }
    return left
  }

  private variableNamed(name: string): Term {
    if (name === '_') {
      this.anonymous += 1
      return variable(`_G${this.anonymous}`)
    }
    if (!this.names.includes(name)) this.names.push(name)
    return variable(name)
  }

  private args(): Term[] {
    const out: Term[] = [this.parse(999)]
    while (this.peek()?.type === 'punct' && this.peek()?.text === ',') {
      this.pos++
      out.push(this.parse(999))
    }
    return out
  }

  private stringTerm(s: string): Term {
    const chars = [...s]
    if (this.doubleQuotes === 'atom') return atom(s)
    if (this.doubleQuotes === 'chars') return list(chars.map((c) => atom(c)))
    return list(chars.map((c) => num(c.codePointAt(0) ?? 0)))
  }

  private primary(max: number): [Term, number] {
    const t = this.next()
    switch (t.type) {
      case 'int':
        return [num(t.value ?? 0, false), 0]
      case 'float':
        return [num(t.value ?? 0, true), 0]
      case 'var':
        return [this.variableNamed(t.text), 0]
      case 'string':
        return [this.stringTerm(t.text), 0]
      case 'backquote':
        return [list([...t.text].map((c) => num(c.codePointAt(0) ?? 0))), 0]
      case 'end':
        throw syntaxError('unexpected end of clause', t.offset)
      case 'punct':
        return this.punct(t)
      case 'atom':
        return this.atomOrOperator(t, max)
    }
  }

  private punct(t: Token): [Term, number] {
    if (t.text === '(') {
      const inner = this.parse(1200)
      this.expect('punct', ')')
      return [inner, 0]
    }
    if (t.text === '[') {
      const n = this.peek()
      if (n?.type === 'punct' && n.text === ']') {
        this.pos++
        return this.maybeCall('[]', 0)
      }
      const items = this.args()
      let tail: Term = NIL
      if (this.peek()?.type === 'punct' && this.peek()?.text === '|') {
        this.pos++
        tail = this.parse(999)
      }
      this.expect('punct', ']')
      return [list(items, tail), 0]
    }
    if (t.text === '{') {
      const n = this.peek()
      if (n?.type === 'punct' && n.text === '}') {
        this.pos++
        return this.maybeCall('{}', 0)
      }
      const inner = this.parse(1200)
      this.expect('punct', '}')
      return [compound('{}', [inner]), 0]
    }
    if (t.text === ',') throw syntaxError('unexpected comma', t.offset)
    if (t.text === '|') return [atom('|'), 0]
    throw syntaxError(`unexpected ${t.text}`, t.offset)
  }

  private maybeCall(name: string, pri: number): [Term, number] {
    const n = this.peek()
    if (n?.type === 'punct' && n.text === '(' && !n.layoutBefore) {
      this.pos++
      const args = this.args()
      this.expect('punct', ')')
      return [compound(name, args), 0]
    }
    return [atom(name), pri]
  }

  private atomOrOperator(t: Token, max: number): [Term, number] {
    const name = t.text
    const n = this.peek()
    if (n?.type === 'punct' && n.text === '(' && !n.layoutBefore) return this.maybeCall(name, 0)
    // A minus sign directly before a number is part of the number.
    if (name === '-' && (n?.type === 'int' || n?.type === 'float') && !n.layoutBefore) {
      this.pos++
      return [num(-(n.value ?? 0), n.type === 'float'), 0]
    }
    const prefix: OperatorDef | undefined = this.ops.prefix.get(name)
    if (prefix !== undefined && this.isTermStart(0)) {
      // An operator followed by an infix operator is an atom operand (e.g. `- = x`), unless that could be a prefix.
      const isInfixNext =
        n?.type === 'atom' && this.ops.infix.has(n.text) && !this.ops.prefix.has(n.text) && !this.isTermStart(0)
      const isClose = n?.type === 'punct' && (n.text === ')' || n.text === ']' || n.text === '}' || n.text === ',')
      if (!isInfixNext && !isClose) {
        let p = prefix.priority
        let amax = prefix.type === 'fy' ? p : p - 1
        if (p > max) {
          p = 999
          amax = 999
        }
        const save = this.pos
        try {
          const arg = this.parse(amax)
          return [compound(name, [arg]), p]
        } catch (e) {
          // `- (1)` style operands failed: fall back to reading the operator as an atom.
          this.pos = save
          if (!(e instanceof OmaiError)) throw e
        }
      }
    }
    const isOp = prefix !== undefined || this.ops.infix.has(name)
    return [atom(name), isOp ? Math.min(prefix?.priority ?? this.ops.infix.get(name)?.priority ?? 0, max) : 0]
  }

  /** Apply an `op/3` directive. */
  applyOp(priority: number, type: OperatorDef['type'], names: readonly string[]): void {
    const target =
      type === 'xfx' || type === 'xfy' || type === 'yfx'
        ? (this.ops.infix as Map<string, OperatorDef>)
        : type === 'fy' || type === 'fx'
          ? (this.ops.prefix as Map<string, OperatorDef>)
          : (this.ops.postfix as Map<string, OperatorDef>)
    for (const n of names) {
      if (priority === 0) target.delete(n)
      else target.set(n, { priority, type })
    }
  }
}

function copyOperators(ops: Operators | undefined): Operators {
  const base = ops ?? standardOperators()
  return { prefix: new Map(base.prefix), infix: new Map(base.infix), postfix: new Map(base.postfix) }
}

/**
 * Parse one Prolog term (a final full stop is optional). Variables are named as in the source; each `_` becomes a
 * distinct variable `_G1`, `_G2`, ....
 *
 * @param text - The text.
 * @param options - Operators and the reading of double-quoted text.
 * @returns The term.
 * @throws OmaiError with code `value` on a syntax error.
 * @example
 * ```ts
 * import { parseTerm, termToString } from 'omai-engine/logic'
 * termToString(parseTerm('X is 2 + 3 * 4')) // => 'X is 2+3*4'
 * termToString(parseTerm('[H|T]')) // => '[H|T]'
 * parseTerm('- 1').kind // => 'compound'
 * parseTerm('-1').kind // => 'num'
 * ```
 */
export function parseTerm(text: string, options: ParseOptions = {}): Term {
  return new Parser(tokenize(text), copyOperators(options.operators), options.doubleQuotes ?? 'codes').readWhole()
}

/**
 * Parse one clause (`head :- body.` or a fact).
 *
 * @param text - The clause text.
 * @param options - Operators and double-quote reading.
 * @returns The clause.
 * @throws OmaiError with code `value` on a syntax error.
 * @example
 * ```ts
 * import { parseClause, termToString } from 'omai-engine/logic'
 * const c = parseClause('ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).')
 * [termToString(c.head), c.body.length] // => ['ancestor(X,Y)', 2]
 * ```
 */
export function parseClause(text: string, options: ParseOptions = {}): Clause {
  return termToClause(parseTerm(text, options))
}

/** A parsed program: its clauses in order, its directives (`:- goal.`) and the operator table after `op/3` directives. */
export interface ParsedProgram {
  readonly clauses: readonly Clause[]
  readonly directives: readonly Term[]
  readonly operators: Operators
}

function opNames(t: Term): string[] {
  if (t.kind === 'atom') return t.name === '[]' ? [] : [t.name]
  const out: string[] = []
  let cur: Term = t
  while (cur.kind === 'compound' && cur.functor === '.' && cur.args.length === 2) {
    const h = cur.args[0] as Term
    if (h.kind === 'atom') out.push(h.name)
    cur = cur.args[1] as Term
  }
  return out
}

/**
 * Parse Prolog source text: clauses and directives, with `%` and block comments. `:- op(P, T, Names).` directives take
 * effect immediately for the rest of the text.
 *
 * @param text - The program.
 * @param options - Initial operators and double-quote reading.
 * @returns The clauses, directives and final operator table.
 * @throws OmaiError with code `value` on a syntax error.
 * @example
 * ```ts
 * import { parseProgram } from 'omai-engine/logic'
 * const p = parseProgram(`
 *   % family
 *   parent(tom, bob).  parent(bob, ann).
 *   :- dynamic(seen/1).
 *   grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
 * `)
 * [p.clauses.length, p.directives.length] // => [3, 1]
 * ```
 */
export function parseProgram(text: string, options: ParseOptions = {}): ParsedProgram {
  const ops = copyOperators(options.operators)
  const tokens = tokenize(text)
  const clauses: Clause[] = []
  const directives: Term[] = []
  // Directives that change the operator table must apply before the rest is tokenised into terms, so parse clause by
  // clause from the shared token stream.
  const parser = new Parser(tokens, ops, options.doubleQuotes ?? 'codes')
  while (!parser.atEnd()) {
    const t = parser.readClauseTerm()
    if (t.kind === 'compound' && t.functor === ':-' && t.args.length === 1) {
      const d = t.args[0] as Term
      directives.push(d)
      if (d.kind === 'compound' && d.functor === 'op' && d.args.length === 3) {
        const [p, ty, names] = d.args as [Term, Term, Term]
        if (p.kind === 'num' && ty.kind === 'atom') {
          parser.applyOp(p.value, ty.name as OperatorDef['type'], opNames(names))
        }
      }
      continue
    }
    if (t.kind === 'compound' && t.functor === '?-' && t.args.length === 1) {
      directives.push(t.args[0] as Term)
      continue
    }
    if (t.kind === 'compound' && t.functor === '-->' && t.args.length === 2) {
      clauses.push(translateDcg(t.args[0] as Term, t.args[1] as Term))
      continue
    }
    clauses.push(termToClause(t))
  }
  return { clauses, directives, operators: ops }
}

/**
 * Translate a definite clause grammar rule `Head --> Body` into a clause with two extra difference-list arguments:
 * terminal lists become `S0 = [t1, ..., tn | S]`, `{Goal}` runs `Goal`, `!` and `\+` keep their meaning, `call//N`
 * gets the two arguments appended, and a non-terminal `p(X)` becomes `p(X, S0, S)`.
 *
 * @param head - The rule's head (a non-terminal).
 * @param body - The rule's body.
 * @returns The clause.
 * @throws OmaiError with code `value` for a head that is not callable.
 * @example
 * ```ts
 * import { clauseToString, parseTerm, translateDcg } from 'omai-engine/logic'
 * clauseToString(translateDcg(parseTerm('greeting'), parseTerm('[hello], name'))) // => 'greeting(_DCG0,_DCG1) :- _DCG0=[hello|_DCG2], name(_DCG2,_DCG1).'
 * ```
 */
export function translateDcg(head: Term, body: Term): Clause {
  let n = 0
  const fresh = (): Term => variable(`_DCG${n++}`)
  const extend = (t: Term, s0: Term, s: Term): Term => {
    if (t.kind === 'atom') return compound(t.name, [s0, s])
    if (t.kind === 'compound') return compound(t.functor, [...t.args, s0, s])
    throw new OmaiError('value', `translateDcg: ${t.kind} is not a non-terminal`)
  }
  const eq = (a: Term, b: Term): Term => compound('=', [a, b])
  const trans = (t: Term, s0: Term, s: Term): Term => {
    if (t.kind === 'var') return compound('phrase', [t, s0, s])
    if (t.kind === 'atom' && t.name === '[]') return eq(s0, s)
    if (t.kind === 'atom' && t.name === '!') return compound(',', [atom('!'), eq(s0, s)])
    if (t.kind === 'compound') {
      const [a, b] = t.args as [Term, Term]
      if (t.functor === ',' && t.args.length === 2) {
        const mid = fresh()
        return compound(',', [trans(a, s0, mid), trans(b, mid, s)])
      }
      if ((t.functor === ';' || t.functor === '|') && t.args.length === 2) {
        return compound(';', [trans(a, s0, s), trans(b, s0, s)])
      }
      if (t.functor === '->' && t.args.length === 2) {
        const mid = fresh()
        return compound('->', [trans(a, s0, mid), trans(b, mid, s)])
      }
      if (t.functor === '\\+' && t.args.length === 1) {
        return compound(',', [compound('\\+', [trans(a, s0, fresh())]), eq(s0, s)])
      }
      if (t.functor === '{}' && t.args.length === 1) return compound(',', [a, eq(s0, s)])
      if (t.functor === 'call') return compound('call', [...t.args, s0, s])
      if (t.functor === '.' && t.args.length === 2) {
        const items = listItems(t)
        if (items !== null) return eq(s0, list(items, s))
      }
    }
    return extend(t, s0, s)
  }
  const s0 = fresh()
  const s = fresh()
  return { head: extend(head, s0, s), body: conjuncts(trans(body, s0, s)) }
}

/** A parsed query: the goal and the names of its variables in order of first occurrence (anonymous `_` excluded). */
export interface ParsedQuery {
  readonly goal: Term
  readonly variables: readonly string[]
}

/**
 * Parse a query (`?- goal.` or just `goal`), keeping the order of its named variables for printing answers.
 *
 * @param text - The query.
 * @param options - Operators and double-quote reading.
 * @returns The goal and its variable names.
 * @throws OmaiError with code `value` on a syntax error.
 * @example
 * ```ts
 * import { parseQuery } from 'omai-engine/logic'
 * parseQuery('?- append(X, Y, [1,2]), _ = Y.').variables // => ['X', 'Y']
 * ```
 */
export function parseQuery(text: string, options: ParseOptions = {}): ParsedQuery {
  const tokens = tokenize(text)
  const first = tokens[0]
  const body = first?.type === 'atom' && first.text === '?-' ? tokens.slice(1) : tokens
  const parser = new Parser(body, copyOperators(options.operators), options.doubleQuotes ?? 'codes')
  const goal = parser.readWhole()
  return { goal, variables: [...parser.names] }
}
