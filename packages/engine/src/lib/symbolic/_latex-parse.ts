// The LaTeX subset parser behind `parseLatex` (see parse.ts for the accepted syntax).
import { WhiteboxError } from '../base/index.ts'
import { E, HALF, MINUS_ONE, makeAdd, makeMul, makePow, makeRel, numFromRational, OO, PI } from './_canon.ts'
import { makeFn } from './_functions.ts'
import { rawFloat } from './_node.ts'
import { rat, rfromDecimal } from './_rational.ts'
import { type Expr, type ExprSym, type RelOp, sym } from './expr.ts'

interface Options {
  readonly symbols?: Readonly<Record<string, ExprSym>>
  readonly exact?: boolean
}

type Tok = { readonly t: 'cmd' | 'char' | 'digit' | 'letter' | 'end'; readonly v: string; readonly pos: number }

function tokenize(src: string): Tok[] {
  const out: Tok[] = []
  let i = 0
  while (i < src.length) {
    const ch = src[i] as string
    if (/\s/.test(ch)) {
      i++
      continue
    }
    if (ch === '\\') {
      const m = /^\\([A-Za-z]+)/.exec(src.slice(i))
      if (m !== null) {
        out.push({ t: 'cmd', v: m[1] as string, pos: i })
        i += m[0].length
        continue
      }
      const next = src[i + 1] ?? ''
      // Spacing commands are ignored; \{ and \} and \| are delimiters.
      if (next === '{' || next === '}') out.push({ t: 'char', v: next === '{' ? '(' : ')', pos: i })
      else if (next === '|') out.push({ t: 'char', v: '|', pos: i })
      i += 2
      continue
    }
    if (/\d/.test(ch) || (ch === '.' && /\d/.test(src[i + 1] ?? ''))) out.push({ t: 'digit', v: ch, pos: i })
    else if (/[A-Za-z]/.test(ch)) out.push({ t: 'letter', v: ch, pos: i })
    else out.push({ t: 'char', v: ch, pos: i })
    i++
  }
  out.push({ t: 'end', v: '', pos: src.length })
  return out
}

const FUNCTIONS: Readonly<Record<string, string>> = {
  sin: 'sin',
  cos: 'cos',
  tan: 'tan',
  cot: 'cot',
  sec: 'sec',
  csc: 'csc',
  arcsin: 'asin',
  arccos: 'acos',
  arctan: 'atan',
  arccot: 'acot',
  sinh: 'sinh',
  cosh: 'cosh',
  tanh: 'tanh',
  coth: 'coth',
  arsinh: 'asinh',
  arcsinh: 'asinh',
  arcosh: 'acosh',
  arccosh: 'acosh',
  artanh: 'atanh',
  arctanh: 'atanh',
  exp: 'exp',
  log: 'log',
  ln: 'log',
  arg: 'arg',
  Re: 're',
  Im: 'im',
}
const INVERSE: Readonly<Record<string, string>> = {
  sin: 'asin',
  cos: 'acos',
  tan: 'atan',
  cot: 'acot',
  sinh: 'asinh',
  cosh: 'acosh',
  tanh: 'atanh',
}
const GREEK = new Set([
  'alpha',
  'beta',
  'gamma',
  'delta',
  'epsilon',
  'varepsilon',
  'zeta',
  'eta',
  'theta',
  'vartheta',
  'iota',
  'kappa',
  'lambda',
  'mu',
  'nu',
  'xi',
  'rho',
  'varrho',
  'sigma',
  'varsigma',
  'tau',
  'upsilon',
  'phi',
  'varphi',
  'chi',
  'psi',
  'omega',
  'Gamma',
  'Delta',
  'Theta',
  'Lambda',
  'Xi',
  'Pi',
  'Sigma',
  'Upsilon',
  'Phi',
  'Psi',
  'Omega',
])
const RELATIONS: Readonly<Record<string, RelOp>> = {
  '=': '==',
  '<': '<',
  '>': '>',
  neq: '!=',
  ne: '!=',
  leq: '<=',
  le: '<=',
  geq: '>=',
  ge: '>=',
  lt: '<',
  gt: '>',
}
const IGNORED = new Set(['displaystyle', 'textstyle', 'mathrm', 'limits', 'nolimits'])

class LatexParser {
  private i = 0
  private absDepth = 0
  private readonly toks: readonly Tok[]
  private readonly options: Options
  constructor(toks: readonly Tok[], options: Options) {
    this.toks = toks
    this.options = options
  }

  private peek(k = 0): Tok {
    return this.toks[Math.min(this.i + k, this.toks.length - 1)] as Tok
  }
  private next(): Tok {
    return this.toks[this.i++] as Tok
  }
  private isChar(v: string, k = 0): boolean {
    const t = this.peek(k)
    return t.t === 'char' && t.v === v
  }
  private isCmd(v: string, k = 0): boolean {
    const t = this.peek(k)
    return t.t === 'cmd' && t.v === v
  }
  private fail(msg: string): never {
    throw new WhiteboxError('value', `parseLatex: ${msg} at ${this.peek().pos}`)
  }
  private expectChar(v: string): void {
    if (!this.isChar(v)) this.fail(`expected '${v}'`)
    this.next()
  }
  private skipIgnored(): void {
    while (this.peek().t === 'cmd' && IGNORED.has(this.peek().v)) this.next()
  }

  parseAll(): Expr {
    const e = this.relation()
    if (this.peek().t !== 'end') this.fail(`unexpected '${this.peek().v}'`)
    return e
  }

  private relation(): Expr {
    const left = this.sum()
    const t = this.peek()
    const op = t.t === 'char' || t.t === 'cmd' ? RELATIONS[t.v] : undefined
    if (op !== undefined && (t.t === 'cmd' || ['=', '<', '>'].includes(t.v))) {
      this.next()
      return makeRel(op, left, this.sum())
    }
    return left
  }
  private sum(): Expr {
    let left = this.term()
    for (;;) {
      if (this.isChar('+')) {
        this.next()
        left = makeAdd([left, this.term()])
      } else if (this.isChar('-')) {
        this.next()
        left = makeAdd([left, makeMul([MINUS_ONE, this.term()])])
      } else return left
    }
  }
  private startsFactor(): boolean {
    this.skipIgnored()
    const t = this.peek()
    if (t.t === 'digit' || t.t === 'letter') return true
    if (t.t === 'char') return t.v === '(' || t.v === '[' || t.v === '{' || (t.v === '|' && this.absDepth === 0)
    if (t.t === 'cmd') {
      if (t.v === 'left') {
        const n = this.peek(1)
        return !(n.t === 'char' && n.v === '|' && this.absDepth > 0)
      }
      return (
        !['right', 'cdot', 'times', 'div', 'rfloor', 'rceil', 'pm', 'mp'].includes(t.v) && RELATIONS[t.v] === undefined
      )
    }
    return false
  }
  private isFunctionStart(): boolean {
    const t = this.peek()
    return t.t === 'cmd' && (FUNCTIONS[t.v] !== undefined || t.v === 'operatorname')
  }
  // Product chains are built once, as in the text parser.
  private term(): Expr {
    const factors = [this.unary()]
    for (;;) {
      if (this.isCmd('cdot') || this.isCmd('times') || this.isChar('*')) {
        this.next()
        factors.push(this.unary())
      } else if (this.isChar('/') || this.isCmd('div')) {
        this.next()
        factors.push(makePow(this.unary(), MINUS_ONE))
      } else if (this.startsFactor()) factors.push(this.power())
      else return factors.length === 1 ? (factors[0] as Expr) : makeMul(factors)
    }
  }
  private unary(): Expr {
    if (this.isChar('-')) {
      this.next()
      return makeMul([MINUS_ONE, this.unary()])
    }
    if (this.isChar('+')) {
      this.next()
      return this.unary()
    }
    return this.power()
  }
  /** The argument of ^ or _: a braced group or a single token. */
  private script(): Expr {
    if (this.isChar('{')) return this.group()
    const t = this.peek()
    if (t.t === 'digit') {
      this.next()
      return numFromRational(rat(BigInt(t.v)))
    }
    if (t.t === 'char' && t.v === '-') {
      this.next()
      return makeMul([MINUS_ONE, this.script()])
    }
    return this.primary()
  }
  private power(): Expr {
    let base = this.postfix()
    while (this.isChar('^')) {
      this.next()
      base = makePow(base, this.script())
    }
    return base
  }
  private postfix(): Expr {
    let e = this.primary()
    while (this.isChar('!')) {
      this.next()
      e = makeFn('factorial', [e])
    }
    return e
  }
  private group(): Expr {
    this.expectChar('{')
    const e = this.relation()
    this.expectChar('}')
    return e
  }
  /** Raw text of a braced group or one token (for subscripts and operator names). */
  private rawScript(): string {
    if (!this.isChar('{')) {
      const t = this.next()
      return t.v
    }
    this.next()
    let depth = 1
    let s = ''
    for (;;) {
      const t = this.next()
      if (t.t === 'end') this.fail('unclosed group')
      if (t.t === 'char' && t.v === '{') depth++
      if (t.t === 'char' && t.v === '}' && --depth === 0) return s
      s += t.t === 'cmd' ? t.v : t.v
    }
  }
  private symbolNamed(name: string): Expr {
    let full = name
    if (this.isChar('_')) {
      this.next()
      full = `${name}_${this.rawScript()}`
    }
    return this.options.symbols?.[full] ?? sym(full)
  }
  private delimited(close: string): Expr {
    const e = this.relation()
    if (this.isCmd('right')) {
      this.next()
      this.next()
    } else this.expectChar(close)
    return e
  }
  /** A comma-separated list up to the closing delimiter (after an opening `(` or `\left(` was consumed). */
  private list(close: string): Expr[] {
    const out = [this.relation()]
    while (this.isChar(',')) {
      this.next()
      out.push(this.relation())
    }
    if (this.isCmd('right')) {
      this.next()
      this.next()
    } else this.expectChar(close)
    return out
  }
  private functionArguments(): Expr[] {
    if (this.isChar('{')) {
      if (this.isCmd('left', 1) && this.isChar('(', 2)) {
        this.next()
        this.next()
        this.next()
        const args = this.list(')')
        this.expectChar('}')
        return args
      }
      return [this.group()]
    }
    if (this.isChar('(')) {
      this.next()
      return this.list(')')
    }
    if (this.isCmd('left') && this.isChar('(', 1)) {
      this.next()
      this.next()
      return this.list(')')
    }
    // Without brackets: the implicit product that follows, up to an operator or the next function.
    let e = this.power()
    while (this.startsFactor() && !this.isFunctionStart()) e = makeMul([e, this.power()])
    return [e]
  }
  private applyFunction(name: string): Expr {
    let exponent: Expr | undefined
    let base: Expr | undefined
    for (;;) {
      if (this.isChar('^')) {
        this.next()
        exponent = this.script()
      } else if (this.isChar('_') && name === 'log') {
        this.next()
        base = this.script()
      } else break
    }
    let f = name
    if (exponent !== undefined && exponent.kind === 'num' && exponent.n === -1n && exponent.d === 1n && INVERSE[name]) {
      f = INVERSE[name] as string
      exponent = undefined
    }
    let v = makeFn(f, this.functionArguments())
    if (base !== undefined) v = makeMul([v, makePow(makeFn('log', [base]), MINUS_ONE)])
    return exponent === undefined ? v : makePow(v, exponent)
  }
  private number(first: Tok): Expr {
    let s = first.v
    while (this.peek().t === 'digit') s += this.next().v
    if (s.includes('.')) {
      const r = rfromDecimal(s)
      return this.options.exact ? numFromRational(r) : rawFloat(Number(s))
    }
    return numFromRational(rat(BigInt(s)))
  }
  private command(name: string): Expr {
    switch (name) {
      case 'frac':
      case 'dfrac':
      case 'tfrac': {
        const a = this.script()
        const b = this.script()
        return makeMul([a, makePow(b, MINUS_ONE)])
      }
      case 'sqrt': {
        if (this.isChar('[')) {
          this.next()
          const n = this.relation()
          this.expectChar(']')
          return makePow(this.script(), makePow(n, MINUS_ONE))
        }
        return makePow(this.script(), HALF)
      }
      case 'left': {
        const t = this.next()
        if (t.t === 'char' && t.v === '|') {
          this.absDepth++
          const e = this.delimited('|')
          this.absDepth--
          return makeFn('Abs', [e])
        }
        if (t.t === 'cmd' && (t.v === 'lfloor' || t.v === 'lceil')) {
          const e = this.delimited(t.v === 'lfloor' ? 'rfloor' : 'rceil')
          return makeFn(t.v === 'lfloor' ? 'floor' : 'ceiling', [e])
        }
        if (t.t === 'cmd' && (t.v === 'lvert' || t.v === 'vert')) {
          this.absDepth++
          const e = this.delimited('|')
          this.absDepth--
          return makeFn('Abs', [e])
        }
        return this.delimited(t.v === '[' ? ']' : ')')
      }
      case 'lfloor':
      case 'lceil': {
        const e = this.relation()
        if (!this.isCmd(name === 'lfloor' ? 'rfloor' : 'rceil')) this.fail('unclosed floor or ceiling')
        this.next()
        return makeFn(name === 'lfloor' ? 'floor' : 'ceiling', [e])
      }
      case 'pi':
        return PI
      case 'infty':
        return OO
      case 'operatorname': {
        let fname = this.rawScript().replace(/\s/g, '')
        if (this.isChar('_')) {
          this.next()
          fname += this.rawScript()
        }
        return this.applyFunction(fname)
      }
      case 'overline':
        return makeFn('conjugate', [this.script()])
      case 'exp':
        return this.applyFunction('exp')
      default:
        break
    }
    const f = FUNCTIONS[name]
    if (f !== undefined) return this.applyFunction(f)
    if (name === 'Gamma' && (this.isChar('(') || this.isCmd('left'))) {
      return makeFn('gamma', this.functionArguments())
    }
    if (name === 'Gamma' && this.isChar('^')) {
      // \Gamma^{k}\left(x\right) is gamma(x)^k as SymPy prints it; otherwise the symbol Gamma to a power.
      const save = this.i
      this.next()
      const k = this.script()
      if (this.isChar('(') || this.isCmd('left')) return makePow(makeFn('gamma', this.functionArguments()), k)
      this.i = save
    }
    if (GREEK.has(name)) return this.symbolNamed(name)
    return this.fail(`unsupported command \\${name}`)
  }
  private primary(): Expr {
    this.skipIgnored()
    const t = this.next()
    switch (t.t) {
      case 'digit':
        return this.number(t)
      case 'letter':
        if (t.v === 'e' && this.options.symbols?.['e'] === undefined && !this.isChar('_')) return E
        // A letter applied to a braced, bracketed argument list is an undefined function, as SymPy prints f(x).
        if (this.isChar('{') && this.isCmd('left', 1) && this.isChar('(', 2)) {
          return makeFn(t.v, this.functionArguments())
        }
        return this.symbolNamed(t.v)
      case 'cmd':
        return this.command(t.v)
      case 'char':
        if (t.v === '(') return this.delimited(')')
        if (t.v === '[') return this.delimited(']')
        if (t.v === '{') {
          const e = this.relation()
          this.expectChar('}')
          return e
        }
        if (t.v === '|') {
          this.absDepth++
          const e = this.relation()
          this.absDepth--
          this.expectChar('|')
          return makeFn('Abs', [e])
        }
        return this.fail(`unexpected '${t.v}'`)
      default:
        return this.fail('unexpected end of input')
    }
  }
}

/** Parse LaTeX source (the implementation of `parseLatex`). @internal */
export function parseLatexSource(src: string, options: Options = {}): Expr {
  return new LatexParser(tokenize(src), options).parseAll()
}
