/**
 * Parsing text into symbolic expressions: `parse` reads infix notation (`x^2 + sin(x)`) and `parseLatex` reads a useful
 * subset of LaTeX (fractions, roots, powers and subscripts, implicit multiplication, absolute values).
 */

import { MdmlError } from '../base/index.ts'
import {
  E,
  FALSE,
  I,
  MINUS_ONE,
  makeAdd,
  makeDerivative,
  makeIntegral,
  makeLogic,
  makeMul,
  makeOrder,
  makePiecewise,
  makePow,
  makeRel,
  NAN,
  numFromRational,
  ONE,
  OO,
  PI,
  TRUE,
  ZERO,
  ZOO,
} from './_canon.ts'
import { makeFn } from './_functions.ts'
import { parseLatexSource } from './_latex-parse.ts'
import { rawFloat } from './_node.ts'
import { rat, rfromDecimal } from './_rational.ts'
import { type Expr, type ExprSym, type RelOp, sym } from './expr.ts'

/** Options for {@link parse} and {@link parseLatex}. */
export interface ParseOptions {
  /** Symbols to use by name (for example ones carrying assumptions); other names become plain symbols. */
  readonly symbols?: Readonly<Record<string, ExprSym>>
  /** Read decimals as exact rationals (`0.1` is 1/10) instead of floats, as SymPy's `rational=True`. */
  readonly exact?: boolean
}

type Token = { readonly t: 'num' | 'id' | 'op' | 'end'; readonly v: string; readonly pos: number }

const OPS = ['**', '==', '!=', '<=', '>=', '(', ')', ',', '+', '-', '*', '/', '^', '<', '>', '=', '!', '&', '|', '~']

function tokenize(src: string): Token[] {
  const out: Token[] = []
  let i = 0
  while (i < src.length) {
    const ch = src[i] as string
    if (/\s/.test(ch)) {
      i++
      continue
    }
    const num = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(src.slice(i))
    if (num !== null) {
      out.push({ t: 'num', v: num[0], pos: i })
      i += num[0].length
      continue
    }
    const id = /^[A-Za-z_Ͱ-Ͽ][A-Za-z0-9_'Ͱ-Ͽ]*/.exec(src.slice(i))
    if (id !== null) {
      out.push({ t: 'id', v: id[0], pos: i })
      i += id[0].length
      continue
    }
    const op = OPS.find((o) => src.startsWith(o, i))
    if (op === undefined) throw new MdmlError('value', `parse: unexpected '${ch}' at ${i}`)
    out.push({ t: 'op', v: op, pos: i })
    i += op.length
  }
  out.push({ t: 'end', v: '', pos: src.length })
  return out
}

/** Function names accepted by the text parser, with SymPy-compatible aliases. */
const FUNCTION_ALIASES: Readonly<Record<string, string>> = {
  abs: 'Abs',
  arcsin: 'asin',
  arccos: 'acos',
  arctan: 'atan',
  arccot: 'acot',
  arcsinh: 'asinh',
  arccosh: 'acosh',
  arctanh: 'atanh',
  ln: 'log',
  ceil: 'ceiling',
}

function constant(name: string): Expr | undefined {
  switch (name) {
    case 'pi':
      return PI
    case 'E':
      return E
    case 'I':
      return I
    case 'oo':
      return OO
    case 'zoo':
      return ZOO
    case 'nan':
      return NAN
    case 'True':
      return TRUE
    case 'False':
      return FALSE
    default:
      return undefined
  }
}

class TextParser {
  private i = 0
  private readonly toks: readonly Token[]
  private readonly options: ParseOptions
  constructor(toks: readonly Token[], options: ParseOptions) {
    this.toks = toks
    this.options = options
  }

  private peek(): Token {
    return this.toks[this.i] as Token
  }
  private next(): Token {
    return this.toks[this.i++] as Token
  }
  private isOp(v: string): boolean {
    const t = this.peek()
    return t.t === 'op' && t.v === v
  }
  private expect(v: string): void {
    const t = this.next()
    if (t.t !== 'op' || t.v !== v) throw new MdmlError('value', `parse: expected '${v}' at ${t.pos}`)
  }

  parseAll(): Expr {
    const e = this.logic()
    if (this.peek().t !== 'end')
      throw new MdmlError('value', `parse: unexpected '${this.peek().v}' at ${this.peek().pos}`)
    return e
  }

  private logic(): Expr {
    let left = this.conj()
    while (this.isOp('|')) {
      this.next()
      left = makeLogic('or', [left, this.conj()])
    }
    return left
  }
  private conj(): Expr {
    let left = this.negation()
    while (this.isOp('&')) {
      this.next()
      left = makeLogic('and', [left, this.negation()])
    }
    return left
  }
  private negation(): Expr {
    if (this.isOp('~')) {
      this.next()
      return makeLogic('not', [this.negation()])
    }
    return this.relation()
  }
  private relation(): Expr {
    const left = this.sum()
    const t = this.peek()
    if (t.t === 'op' && ['==', '=', '!=', '<', '<=', '>', '>='].includes(t.v)) {
      this.next()
      const op = (t.v === '=' ? '==' : t.v) as RelOp
      return makeRel(op, left, this.sum())
    }
    return left
  }
  private sum(): Expr {
    let left = this.term()
    for (;;) {
      if (this.isOp('+')) {
        this.next()
        left = makeAdd([left, this.term()])
      } else if (this.isOp('-')) {
        this.next()
        left = makeAdd([left, makeMul([MINUS_ONE, this.term()])])
      } else return left
    }
  }
  private startsPrimary(): boolean {
    const t = this.peek()
    return t.t === 'num' || t.t === 'id' || (t.t === 'op' && t.v === '(')
  }
  // A product chain is gathered and built once, so `2*(x + y)*z` stays a three-factor product (pairwise building would
  // distribute the 2 over the sum first, as a two-factor number-times-sum product does).
  private term(): Expr {
    const factors = [this.unary()]
    for (;;) {
      if (this.isOp('*')) {
        this.next()
        factors.push(this.unary())
      } else if (this.isOp('/')) {
        this.next()
        factors.push(makePow(this.unary(), MINUS_ONE))
      } else if (this.startsPrimary()) factors.push(this.power())
      else return factors.length === 1 ? (factors[0] as Expr) : makeMul(factors)
    }
  }
  private unary(): Expr {
    if (this.isOp('-')) {
      this.next()
      return makeMul([MINUS_ONE, this.unary()])
    }
    if (this.isOp('+')) {
      this.next()
      return this.unary()
    }
    return this.power()
  }
  private power(): Expr {
    const base = this.postfix()
    if (this.isOp('^') || this.isOp('**')) {
      this.next()
      return makePow(base, this.unary())
    }
    return base
  }
  private postfix(): Expr {
    let e = this.primary()
    while (this.isOp('!')) {
      this.next()
      e = makeFn('factorial', [e])
    }
    return e
  }
  private args(): Expr[] {
    this.expect('(')
    const out: Expr[] = []
    if (this.isOp(')')) {
      this.next()
      return out
    }
    for (;;) {
      out.push(this.logic())
      if (this.isOp(',')) {
        this.next()
        continue
      }
      this.expect(')')
      return out
    }
  }
  /** A parenthesised tuple `(a, b, ...)` as a list. */
  private tuple(): Expr[] {
    if (!this.isOp('(')) return [this.logic()]
    return this.args()
  }
  private special(name: string): Expr | undefined {
    const symOf = (e: Expr | undefined, what: string): ExprSym => {
      if (e?.kind !== 'sym') throw new MdmlError('value', `parse: ${what} needs a symbol`)
      return e
    }
    switch (name) {
      case 'Piecewise': {
        this.expect('(')
        const pieces: [Expr, Expr][] = []
        for (;;) {
          const [v, c] = this.tuple()
          pieces.push([v as Expr, c ?? TRUE])
          if (this.isOp(',')) {
            this.next()
            continue
          }
          this.expect(')')
          return makePiecewise(pieces)
        }
      }
      case 'Integral':
      case 'Derivative':
      case 'O': {
        this.expect('(')
        const body = this.logic()
        const specs: Expr[][] = []
        while (this.isOp(',')) {
          this.next()
          specs.push(this.tuple())
        }
        this.expect(')')
        if (name === 'Integral') {
          const [x, a, b] = specs[0] ?? []
          return makeIntegral(body, symOf(x, 'Integral'), a !== undefined && b !== undefined ? [a, b] : undefined)
        }
        if (name === 'Derivative') {
          return makeDerivative(
            body,
            specs.map(([x, k]) => [symOf(x, 'Derivative'), k?.kind === 'num' ? Number(k.n) : 1] as const),
          )
        }
        const [x, point] = specs[0] ?? []
        const v = x ?? body
        return makeOrder(body, symOf(v.kind === 'sym' ? v : firstSymbol(body), 'O'), point ?? ZERO)
      }
      default:
        return undefined
    }
  }
  private call(name: string): Expr {
    const special = this.special(name)
    if (special !== undefined) return special
    const args = this.args()
    const a = args[0]
    switch (name) {
      case 'Eq':
      case 'Ne':
      case 'Lt':
      case 'Le':
      case 'Gt':
      case 'Ge': {
        const ops: Record<string, RelOp> = { Eq: '==', Ne: '!=', Lt: '<', Le: '<=', Gt: '>', Ge: '>=' }
        return makeRel(ops[name] as RelOp, a ?? ZERO, args[1] ?? ZERO)
      }
      case 'And':
        return makeLogic('and', args)
      case 'Or':
        return makeLogic('or', args)
      case 'Not':
        return makeLogic('not', args)
      case 'sqrt':
        return makePow(a ?? ZERO, numFromRational(rat(1n, 2n)))
      case 'cbrt':
        return makePow(a ?? ZERO, numFromRational(rat(1n, 3n)))
      case 'root':
        return makePow(a ?? ZERO, makePow(args[1] ?? ONE, MINUS_ONE))
      case 'Rational':
        return makeMul([a ?? ZERO, makePow(args[1] ?? ONE, MINUS_ONE)])
      case 'Integer':
      case 'S':
      case 'Float':
        return a ?? ZERO
      case 'log':
        if (args.length === 2) {
          return makeMul([makeFn('log', [a ?? ZERO]), makePow(makeFn('log', [args[1] as Expr]), MINUS_ONE)])
        }
        return makeFn('log', args)
      default:
        return makeFn(FUNCTION_ALIASES[name] ?? name, args)
    }
  }
  private primary(): Expr {
    const t = this.next()
    if (t.t === 'num') {
      if (/[.eE]/.test(t.v)) {
        const r = rfromDecimal(t.v)
        return this.options.exact ? numFromRational(r) : rawFloat(Number(t.v))
      }
      return numFromRational(rat(BigInt(t.v)))
    }
    if (t.t === 'id') {
      if (this.isOp('(')) return this.call(t.v)
      const c = constant(t.v)
      if (c !== undefined) return c
      return this.options.symbols?.[t.v] ?? sym(t.v)
    }
    if (t.t === 'op' && t.v === '(') {
      const e = this.logic()
      this.expect(')')
      return e
    }
    throw new MdmlError('value', `parse: unexpected '${t.v || 'end of input'}' at ${t.pos}`)
  }
}

function firstSymbol(e: Expr): ExprSym | undefined {
  if (e.kind === 'sym') return e
  if (e.kind === 'add' || e.kind === 'mul' || e.kind === 'fn') {
    for (const a of e.args) {
      const s = firstSymbol(a)
      if (s !== undefined) return s
    }
  }
  if (e.kind === 'pow') return firstSymbol(e.base) ?? firstSymbol(e.exp)
  return undefined
}

/**
 * Parse infix text into an expression. The syntax is SymPy's (`**` or `^` for powers, `sqrt`, `exp`, `log(x, b)`,
 * `Abs`, `pi`, `E`, `I`, `oo`, `Piecewise((a, cond), ...)`, `Eq(a, b)`, `O(x**3)`, `Integral(f, (x, a, b))`) plus
 * implicit multiplication between a number and what follows (`2x`, `3(x + 1)`, `(x + 1)(x - 1)`). A name followed by
 * an opening parenthesis is a function application, so `x(y + 1)` is the undefined function `x`, as in SymPy. In
 * text, `|` is the logical or of SymPy's output; write `Abs(x)` for an absolute value.
 *
 * @param src - The text.
 * @param options - Symbols to bind by name and whether decimals are exact.
 * @returns The expression in canonical form.
 * @throws MdmlError with code `value` on a syntax error.
 * @index function Parse text into an expression.
 * @example
 * ```ts
 * import { parse, toText } from 'mdml-engine/symbolic'
 * toText(parse('2x^2 + 3x - x')) // => '2*x**2 + 2*x'
 * toText(parse('sqrt(8) * sin(pi/6)')) // => 'sqrt(2)'
 * ```
 */
export function parse(src: string, options: ParseOptions = {}): Expr {
  return new TextParser(tokenize(src), options).parseAll()
}

/**
 * Parse a useful subset of LaTeX: `\frac{a}{b}`, `\sqrt{x}`, `\sqrt[n]{x}`, powers and subscripts with or without
 * braces, `\cdot` and `\times`, implicit multiplication, `\left( \right)`, `|x|` and `\left|{x}\right|`, Greek letters
 * (`\alpha`), `\pi`, `e` (Euler's number), `\infty`, `\sin`, `\cos`, `\tan`, `\sec`, `\csc`, `\cot`, their inverses
 * (`\arcsin` and SymPy's `\operatorname{asin}`) and hyperbolic forms, `\exp`, `\log`, `\ln`, `\log_{b}`,
 * `\operatorname{name}`, powers of functions (`\sin^{2}{x}`), `x!`, `\lfloor x \rfloor`, `\lceil x \rceil`,
 * `\Gamma\left(x\right)` and relations (`=`, `\neq`, `<`, `\leq`, `>`, `\geq`). A function written without
 * parentheses takes the implicit product that follows it (`\sin 2x` is sin(2x)). Single letters are separate symbols
 * (`xy` is x*y); a subscript joins its letter (`x_{1}` is the symbol `x_1`).
 *
 * @param src - The LaTeX source.
 * @param options - Symbols to bind by name and whether decimals are exact.
 * @returns The expression.
 * @throws MdmlError with code `value` on unsupported or malformed input.
 * @example
 * ```ts
 * import { parseLatex, toText } from 'mdml-engine/symbolic'
 * toText(parseLatex('\\frac{x^{2}}{2} + \\sin{\\left(x \\right)}')) // => 'x**2/2 + sin(x)'
 * toText(parseLatex('2xy')) // => '2*x*y'
 * ```
 */
export function parseLatex(src: string, options: ParseOptions = {}): Expr {
  return parseLatexSource(src, options)
}
