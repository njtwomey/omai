import { asCoeffMul, E, isNumeric, MINUS_ONE, makeMul, makePow, numFromRational, ONE } from './_canon.ts'
import { couldExtractMinus, orderedFactors, orderedTerms, sortBySympyKey } from './_order.ts'
import { rat } from './_rational.ts'
import type { Derivation, Expr, ExprFn, ExprMul, ExprPow, Step } from './expr.ts'
import type { Matrix } from './matrices.ts'

const PREC = { Lambda: 1, Or: 20, And: 30, Relational: 35, Add: 40, Mul: 50, Pow: 60, Func: 70, Not: 100, Atom: 1000 }

function precedence(e: Expr, traditional: boolean): number {
  switch (e.kind) {
    case 'add':
      return PREC.Add
    case 'mul':
      return couldExtractMinus(e) ? PREC.Add : PREC.Mul
    case 'pow':
      return PREC.Pow
    case 'fn':
    case 'piecewise':
      return PREC.Func
    case 'num':
      return e.n < 0n ? PREC.Add : e.d !== 1n ? PREC.Mul : PREC.Atom
    case 'float':
      return e.value < 0 ? PREC.Add : PREC.Atom
    case 'const':
      return e.name === '-oo' ? PREC.Add : PREC.Atom
    case 'rel':
      return PREC.Relational
    case 'logic':
      return e.op === 'and' ? PREC.And : e.op === 'or' ? PREC.Or : PREC.Not
    case 'integral':
    case 'derivative':
      return traditional ? PREC.Mul : PREC.Atom
    default:
      return PREC.Atom
  }
}

function floatText(x: number): string {
  return Number.isInteger(x) && Math.abs(x) < 1e16 ? `${x}.0` : `${x}`
}

// ---------------------------------------------------------------------------------------------------------- text

function strParen(e: Expr, level: number, strict = false): string {
  const p = precedence(e, false)
  return p < level || (!strict && p <= level) ? `(${str(e)})` : str(e)
}

/** Whether a power's exponent has a negative numeric coefficient (it prints in a denominator). */
function negativeExponent(p: ExprPow): boolean {
  const [c] = asCoeffMul(p.exp)
  return c.kind === 'num' ? c.n < 0n : c.value < 0
}

function strMul(e: ExprMul): string {
  const prec = precedence(e, false)
  const [c, rest] = asCoeffMul(e)
  let sign = ''
  let body: Expr = e
  if ((c.kind === 'num' && c.n < 0n) || (c.kind === 'float' && c.value < 0)) {
    sign = '-'
    body = makeMul([makeMul([MINUS_ONE, c]), rest])
  }
  const a: Expr[] = []
  const b: Expr[] = []
  const parenBases: string[] = []
  for (const item of orderedFactors(body)) {
    if (item.kind === 'pow' && negativeExponent(item)) {
      if (!(item.exp.kind === 'num' && item.exp.n === -1n && item.exp.d === 1n)) {
        b.push(makePow(item.base, makeMul([MINUS_ONE, item.exp])))
      } else {
        if (item.base.kind === 'mul' || item.base.kind === 'pow') parenBases.push(item.base.key)
        b.push(item.base)
      }
    } else if (item.kind === 'num') {
      if (item.n !== 1n) a.push(numFromRational(rat(item.n)))
      if (item.d !== 1n) b.push(numFromRational(rat(item.d)))
    } else a.push(item)
  }
  const aStr = (a.length === 0 ? [ONE as Expr] : a).map((x) => strParen(x, prec))
  const bStr = b.map((x) => (parenBases.includes(x.key) ? `(${strParen(x, prec)})` : strParen(x, prec)))
  if (b.length === 0) return sign + aStr.join('*')
  if (b.length === 1) return `${sign}${aStr.join('*')}/${bStr[0]}`
  return `${sign}${aStr.join('*')}/(${bStr.join('*')})`
}

function strPow(e: ExprPow): string {
  const ex = e.exp
  if (ex.kind === 'num' && ex.n === 1n && ex.d === 2n) return `sqrt(${str(e.base)})`
  if (ex.kind === 'num' && ex.n === -1n && ex.d === 2n) return `1/sqrt(${str(e.base)})`
  if (ex.kind === 'num' && ex.n === -1n && ex.d === 1n) return `1/${strParen(e.base, PREC.Pow)}`
  return `${strParen(e.base, PREC.Pow)}**${strParen(ex, PREC.Pow)}`
}

function str(e: Expr): string {
  switch (e.kind) {
    case 'num':
      return e.d === 1n ? `${e.n}` : `${e.n}/${e.d}`
    case 'float':
      return floatText(e.value)
    case 'sym':
      return e.name
    case 'const':
      return e.name === 'true' ? 'True' : e.name === 'false' ? 'False' : e.name
    case 'add': {
      const parts: string[] = []
      for (const term of orderedTerms(e)) {
        let t = str(term)
        let sign = '+'
        if (t.startsWith('-') && term.kind !== 'add') {
          sign = '-'
          t = t.slice(1)
        }
        parts.push(sign, precedence(term, false) < PREC.Add ? `(${t})` : t)
      }
      const first = parts.shift()
      return (first === '-' ? '-' : '') + parts.join(' ')
    }
    case 'mul':
      return strMul(e)
    case 'pow':
      return strPow(e)
    case 'fn':
      return `${e.name}(${e.args.map(str).join(', ')})`
    case 'piecewise':
      return `Piecewise(${e.pieces.map(([v, c]) => `(${str(v)}, ${str(c)})`).join(', ')})`
    case 'rel':
      if (e.op === '==') return `Eq(${str(e.lhs)}, ${str(e.rhs)})`
      if (e.op === '!=') return `Ne(${str(e.lhs)}, ${str(e.rhs)})`
      return `${strParen(e.lhs, PREC.Relational)} ${e.op} ${strParen(e.rhs, PREC.Relational)}`
    case 'logic': {
      if (e.op === 'not') return `~${strParen(e.args[0] as Expr, PREC.Not)}`
      const args = sortBySympyKey(e.args).map((a) => (a.kind === 'rel' || a.kind === 'logic' ? `(${str(a)})` : str(a)))
      return args.join(e.op === 'and' ? ' & ' : ' | ')
    }
    case 'order':
      return e.point.kind === 'num' && e.point.n === 0n
        ? `O(${str(e.expr)})`
        : `O(${str(e.expr)}, (${e.variable.name}, ${str(e.point)}))`
    case 'derivative': {
      const vs = e.variables.map(([v, k]) => (k === 1 ? v.name : `(${v.name}, ${k})`))
      return `Derivative(${[str(e.expr), ...vs].join(', ')})`
    }
    case 'integral':
      return e.bounds === undefined
        ? `Integral(${str(e.expr)}, ${e.variable.name})`
        : `Integral(${str(e.expr)}, (${e.variable.name}, ${str(e.bounds[0])}, ${str(e.bounds[1])}))`
  }
}

/**
 * Plain-text form, matching SymPy's `str()` (Python syntax: `x**2 + 2*x + 1`, `sqrt(x)`, `Piecewise(...)`).
 *
 * Terms and factors are printed in SymPy's order (lex order of monomials over sorted generators), so the text equals
 * SymPy's for the same canonical expression. Floats print in the shortest round-trip form rather than with fifteen
 * digits.
 *
 * @param e - The expression.
 * @returns The text.
 * @example
 * ```ts
 * import { parse, toText } from 'whitebox-engine/symbolic'
 * toText(parse('(x + 1)^2 + sin(x)/2')) // => '(x + 1)**2 + sin(x)/2'
 * ```
 */
export function toText(e: Expr): string {
  return str(e)
}

// --------------------------------------------------------------------------------------------------------- latex

const GREEK = new Set([
  'alpha',
  'beta',
  'chi',
  'delta',
  'epsilon',
  'eta',
  'gamma',
  'iota',
  'kappa',
  'lambda',
  'mu',
  'nu',
  'omega',
  'omicron',
  'phi',
  'pi',
  'psi',
  'rho',
  'sigma',
  'tau',
  'theta',
  'upsilon',
  'xi',
  'zeta',
])
const GREEK_TEX: Readonly<Record<string, string>> = {
  Alpha: '\\mathrm{A}',
  Beta: '\\mathrm{B}',
  Gamma: '\\Gamma',
  Delta: '\\Delta',
  Epsilon: '\\mathrm{E}',
  Zeta: '\\mathrm{Z}',
  Eta: '\\mathrm{H}',
  Theta: '\\Theta',
  Iota: '\\mathrm{I}',
  Kappa: '\\mathrm{K}',
  Lambda: '\\Lambda',
  Mu: '\\mathrm{M}',
  Nu: '\\mathrm{N}',
  Xi: '\\Xi',
  omicron: 'o',
  Omicron: '\\mathrm{O}',
  Pi: '\\Pi',
  Rho: '\\mathrm{P}',
  Sigma: '\\Sigma',
  Tau: '\\mathrm{T}',
  Upsilon: '\\Upsilon',
  Phi: '\\Phi',
  Chi: '\\mathrm{X}',
  Psi: '\\Psi',
  Omega: '\\Omega',
  lamda: '\\lambda',
  Lamda: '\\Lambda',
  khi: '\\chi',
  Khi: '\\mathrm{X}',
  varepsilon: '\\varepsilon',
  varkappa: '\\varkappa',
  varphi: '\\varphi',
  varpi: '\\varpi',
  varrho: '\\varrho',
  varsigma: '\\varsigma',
  vartheta: '\\vartheta',
}
const OTHER_SYMBOLS = new Set(['aleph', 'beth', 'daleth', 'gimel', 'ell', 'eth', 'hbar', 'hslash', 'mho', 'wp'])
const MODIFIERS: readonly [string, (s: string) => string][] = (
  [
    ['mathring', (s) => `\\mathring{${s}}`],
    ['ddddot', (s) => `\\ddddot{${s}}`],
    ['dddot', (s) => `\\dddot{${s}}`],
    ['ddot', (s) => `\\ddot{${s}}`],
    ['dot', (s) => `\\dot{${s}}`],
    ['check', (s) => `\\check{${s}}`],
    ['breve', (s) => `\\breve{${s}}`],
    ['acute', (s) => `\\acute{${s}}`],
    ['grave', (s) => `\\grave{${s}}`],
    ['tilde', (s) => `\\tilde{${s}}`],
    ['hat', (s) => `\\hat{${s}}`],
    ['bar', (s) => `\\bar{${s}}`],
    ['vec', (s) => `\\vec{${s}}`],
    ['prime', (s) => `{${s}}'`],
    ['prm', (s) => `{${s}}'`],
    ['bold', (s) => `\\boldsymbol{${s}}`],
    ['bm', (s) => `\\boldsymbol{${s}}`],
    ['cal', (s) => `\\mathcal{${s}}`],
    ['scr', (s) => `\\mathscr{${s}}`],
    ['frak', (s) => `\\mathfrak{${s}}`],
    ['norm', (s) => `\\left\\|{${s}}\\right\\|`],
    ['avg', (s) => `\\left\\langle{${s}}\\right\\rangle`],
    ['abs', (s) => `\\left|{${s}}\\right|`],
    ['mag', (s) => `\\left|{${s}}\\right|`],
  ] as [string, (s: string) => string][]
).sort((a, b) => b[0].length - a[0].length)

function translate(s: string): string {
  const tex = GREEK_TEX[s]
  if (tex !== undefined) return tex
  if (GREEK.has(s.toLowerCase())) return `\\${s.toLowerCase()}`
  if (OTHER_SYMBOLS.has(s)) return `\\${s}`
  for (const [k, f] of MODIFIERS) {
    if (s.toLowerCase().endsWith(k) && s.length > k.length) return f(translate(s.slice(0, -k.length)))
  }
  return s
}

/** SymPy's `split_super_sub`: name, superscripts and subscripts (trailing digits become a subscript). */
function splitSuperSub(text: string): [string, string[], string[]] {
  let pos = 0
  let name: string | undefined
  const supers: string[] = []
  const subs: string[] = []
  while (pos < text.length) {
    let start = pos + 1
    if (text.slice(pos, pos + 2) === '__') start += 1
    let hat = text.indexOf('^', start)
    if (hat < 0) hat = text.length
    let usc = text.indexOf('_', start)
    if (usc < 0) usc = text.length
    const next = Math.min(hat, usc)
    const part = text.slice(pos, next)
    pos = next
    if (name === undefined) name = part
    else if (part.startsWith('^')) supers.push(part.slice(1))
    else if (part.startsWith('__')) supers.push(part.slice(2))
    else if (part.startsWith('_')) subs.push(part.slice(1))
  }
  const base = name ?? ''
  const m = /^([A-Za-zͰ-Ͽ]+)(\d+)$/.exec(base)
  if (m !== null) return [m[1] as string, supers, [m[2] as string, ...subs]]
  return [base, supers, subs]
}

function superSub(text: string): string {
  const [name, supers, subs] = splitSuperSub(text)
  let out = translate(name)
  if (supers.length > 0) out += `^{${supers.map(translate).join(' ')}}`
  if (subs.length > 0) out += `_{${subs.map(translate).join(' ')}}`
  return out
}

const LATEX_FUNCTIONS = new Set([
  'arcsin',
  'arccos',
  'arctan',
  'sin',
  'cos',
  'tan',
  'sinh',
  'cosh',
  'tanh',
  'sqrt',
  'ln',
  'log',
  'sec',
  'csc',
  'cot',
  'coth',
  're',
  'im',
  'frac',
  'root',
  'arg',
])

function functionName(func: string): string {
  const f = superSub(func)
  const sup = f.indexOf('^')
  const subi = f.indexOf('_')
  if (LATEX_FUNCTIONS.has(f)) return `\\${f}`
  if (f.length === 1 || f.startsWith('\\') || subi === 1 || sup === 1) return f
  const cut = sup > 0 && subi > 0 ? Math.min(sup, subi) : sup > 0 ? sup : subi
  if (cut > 0) return `\\operatorname{${f.slice(0, cut)}}${f.slice(cut)}`
  return `\\operatorname{${f}}`
}

function parenSuper(s: string): string {
  return s.includes('^') ? `\\left(${s}\\right)` : s
}

function texParen(e: Expr, level: number, strict = false): string {
  const p = precedence(e, true)
  return p < level || (!strict && p <= level) ? `\\left(${tex(e)}\\right)` : tex(e)
}

function needsMulBrackets(e: Expr, first: boolean, last: boolean): boolean {
  if (e.kind === 'mul') {
    if (!first && couldExtractMinus(e)) return true
  } else if (precedence(e, true) < PREC.Mul) return true
  else if (e.kind === 'rel') return true
  if (e.kind === 'piecewise') return true
  if (!last && containsIntegral(e)) return true
  return false
}

function containsIntegral(e: Expr): boolean {
  if (e.kind === 'integral') return true
  if (e.kind === 'add' || e.kind === 'mul' || e.kind === 'fn') return e.args.some(containsIntegral)
  if (e.kind === 'pow') return containsIntegral(e.base) || containsIntegral(e.exp)
  return false
}

const NUMBER_END = /[0-9][} ]*$/
const NUMBER_START = /^(\d|\\frac\{\d+\}\{\d+\})/

function convertArgs(args: readonly Expr[]): string {
  let out = ''
  let last = ''
  args.forEach((term, i) => {
    let t = tex(term)
    if (needsMulBrackets(term, i === 0, i === args.length - 1)) t = `\\left(${t}\\right)`
    if (NUMBER_END.test(last) && NUMBER_START.test(t)) out += ' \\cdot '
    else if (out.length > 0) out += ' '
    out += t
    last = t
  })
  return out
}

/** SymPy's `fraction(expr, exact=True)`: numerator and denominator factors. */
function fraction(e: Expr): [Expr[], Expr[]] {
  const numer: Expr[] = []
  const denom: Expr[] = []
  for (const term of e.kind === 'mul' ? e.args : [e]) {
    if (term.kind === 'pow' || (term.kind === 'fn' && term.name === 'exp')) {
      const [b, ex] = term.kind === 'pow' ? [term.base, term.exp] : [E, term.args[0] as Expr]
      if (isNumeric(ex) && (ex.kind === 'num' ? ex.n < 0n : ex.value < 0)) {
        if (ex.kind === 'num' && ex.n === -1n && ex.d === 1n) denom.push(b)
        else denom.push(makePow(b, makeMul([MINUS_ONE, ex])))
      } else numer.push(term)
    } else if (term.kind === 'num' && term.d !== 1n) {
      if (term.n !== 1n) numer.push(numFromRational(rat(term.n)))
      denom.push(numFromRational(rat(term.d)))
    } else numer.push(term)
  }
  return [numer, denom]
}

function texMul(e: Expr): string {
  let body = e
  let out = ''
  if (couldExtractMinus(e)) {
    body = makeMul([MINUS_ONE, e])
    out = '- '
  }
  const [numer, denom] = fraction(body)
  const convert = (xs: readonly Expr[]): string =>
    xs.length === 0 ? '1' : xs.length === 1 ? tex(xs[0] as Expr) : convertArgs(sortBySympyKey(xs))
  if (denom.length === 0) return out + (body.kind === 'mul' ? convertArgs(orderedFactors(body)) : tex(body))
  return `${out}\\frac{${convert(numer)}}{${convert(denom)}}`
}

function texFunction(e: ExprFn, exp?: string): string {
  const arg = (i = 0): string => tex(e.args[i] as Expr)
  const pow = (t: string): string => (exp === undefined ? t : `${t}^{${exp}}`)
  switch (e.name) {
    case 'exp': {
      const t = `e^{${arg()}}`
      return exp === undefined ? t : `\\left(${t}\\right)^{${exp}}`
    }
    case 'log':
      return pow(`\\log{\\left(${arg()} \\right)}`)
    case 'Abs':
      return pow(`\\left|{${arg()}}\\right|`)
    case 'floor':
      return pow(`\\left\\lfloor{${arg()}}\\right\\rfloor`)
    case 'ceiling':
      return pow(`\\left\\lceil{${arg()}}\\right\\rceil`)
    case 'gamma':
      return exp === undefined ? `\\Gamma\\left(${arg()}\\right)` : `\\Gamma^{${exp}}\\left(${arg()}\\right)`
    case 'erf':
      return exp === undefined
        ? `\\operatorname{erf}{\\left(${arg()} \\right)}`
        : `\\operatorname{erf}^{${exp}}{\\left(${arg()} \\right)}`
    case 'factorial':
      return pow(`${texParen(e.args[0] as Expr, PREC.Func)}!`)
    case 'conjugate':
      return pow(`\\overline{${arg()}}`)
    case 're':
    case 'im': {
      const t = `\\operatorname{${e.name}}{${texParen(e.args[0] as Expr, PREC.Atom)}}`
      return exp === undefined ? t : `\\left(${t}\\right)^{${exp}}`
    }
    default: {
      const name = exp === undefined ? functionName(e.name) : `${parenSuper(functionName(e.name))}^{${exp}}`
      return `${name}{\\left(${e.args.map(tex).join(',')} \\right)}`
    }
  }
}

function texPow(e: ExprPow): string {
  const ex = e.exp
  if (ex.kind === 'num') {
    const p = ex.n < 0n ? -ex.n : ex.n
    if (p === 1n && ex.d !== 1n) {
      const base = tex(e.base)
      const root = ex.d === 2n ? `\\sqrt{${base}}` : `\\sqrt[${ex.d}]{${base}}`
      return ex.n < 0n ? `\\frac{1}{${root}}` : root
    }
    if (ex.n < 0n) return texMul(e)
  }
  if (e.base.kind === 'fn') return texFunction(e.base, tex(ex))
  let base = texParen(e.base, PREC.Pow)
  if (e.base.kind === 'sym') base = parenSuper(base)
  else if (e.base.kind === 'float') base = `{${base}}`
  return `${base}^{${tex(ex)}}`
}

function texFloat(x: number): string {
  const s = `${x}`
  const m = /^(-?[\d.]+)e([+-]\d+)$/.exec(s)
  if (m !== null) return `${m[1]} \\cdot 10^{${(m[2] as string).replace('+', '')}}`
  return Number.isInteger(x) ? `${x}.0` : s
}

const REL_TEX: Readonly<Record<string, string>> = {
  '==': '=',
  '!=': '\\neq',
  '<': '<',
  '<=': '\\leq',
  '>': '>',
  '>=': '\\geq',
}

function tex(e: Expr): string {
  switch (e.kind) {
    case 'num':
      if (e.d === 1n) return `${e.n}`
      return `${e.n < 0n ? '- ' : ''}\\frac{${e.n < 0n ? -e.n : e.n}}{${e.d}}`
    case 'float':
      return texFloat(e.value)
    case 'sym':
      return superSub(e.name)
    case 'const':
      return {
        pi: '\\pi',
        E: 'e',
        I: 'i',
        oo: '\\infty',
        '-oo': '-\\infty',
        zoo: '\\tilde{\\infty}',
        nan: '\\text{NaN}',
        true: '\\text{True}',
        false: '\\text{False}',
      }[e.name]
    case 'add': {
      let out = ''
      orderedTerms(e).forEach((term, i) => {
        let t = term
        if (i > 0) {
          if (couldExtractMinus(term)) {
            out += ' - '
            t = makeMul([MINUS_ONE, term])
          } else out += ' + '
        }
        const s = tex(t)
        out += t.kind === 'add' || t.kind === 'rel' ? `\\left(${s}\\right)` : s
      })
      return out
    }
    case 'mul':
      return texMul(e)
    case 'pow':
      return texPow(e)
    case 'fn':
      return texFunction(e)
    case 'piecewise': {
      const rows = e.pieces.map(([v, c], i) =>
        i === e.pieces.length - 1 && c.kind === 'const' && c.name === 'true'
          ? `${tex(v)} & \\text{otherwise}`
          : `${tex(v)} & \\text{for}\\: ${tex(c)}`,
      )
      return `\\begin{cases} ${rows.join(' \\\\')} \\end{cases}`
    }
    case 'rel':
      return `${tex(e.lhs)} ${REL_TEX[e.op]} ${tex(e.rhs)}`
    case 'logic': {
      if (e.op === 'not') {
        const a = e.args[0] as Expr
        return a.kind === 'logic' ? `\\neg \\left(${tex(a)}\\right)` : `\\neg ${tex(a)}`
      }
      const args = sortBySympyKey(e.args)
      const char = e.op === 'and' ? '\\wedge' : '\\vee'
      return args
        .map((a) => (a.kind === 'logic' && a.op !== 'not' ? `\\left(${tex(a)}\\right)` : tex(a)))
        .join(` ${char} `)
    }
    case 'order': {
      let s = tex(e.expr)
      if (!(e.point.kind === 'num' && e.point.n === 0n)) s += `; ${tex(e.variable)}\\rightarrow ${tex(e.point)}`
      return `O\\left(${s}\\right)`
    }
    case 'derivative': {
      const partial = new Set(collectSymbols(e.expr)).size > 1
      const d = partial ? '\\partial' : 'd'
      let dim = 0
      let den = ''
      for (const [v, k] of [...e.variables].reverse()) {
        dim += k
        den += k === 1 ? `${d} ${tex(v)}` : `${d} ${parenSuper(tex(v))}^{${k}}`
      }
      const head = dim === 1 ? `\\frac{${d}}{${den}}` : `\\frac{${d}^{${dim}}}{${den}}`
      return `${head} ${texParen(e.expr, PREC.Mul, true)}`
    }
    case 'integral': {
      const body = texParen(e.expr, PREC.Mul, true)
      if (e.bounds === undefined) return `\\int ${body}\\, d${tex(e.variable)}`
      return `\\int\\limits_{${tex(e.bounds[0])}}^{${tex(e.bounds[1])}} ${body}\\, d${tex(e.variable)}`
    }
  }
}

function collectSymbols(e: Expr): string[] {
  if (e.kind === 'sym') return [e.key]
  if (e.kind === 'add' || e.kind === 'mul' || e.kind === 'fn') return e.args.flatMap(collectSymbols)
  if (e.kind === 'pow') return [...collectSymbols(e.base), ...collectSymbols(e.exp)]
  return []
}

function isMatrix(x: Expr | Matrix): x is Matrix {
  return Array.isArray(x)
}

/**
 * LaTeX for an expression or a matrix, matching SymPy's `latex()` with default settings for the common cases
 * (`\frac{}{}` for quotients, `\sqrt{}` for roots, `\sin{\left(x \right)}`, `\operatorname{asin}`, `e^{x}`,
 * `\left[\begin{matrix}...\end{matrix}\right]`).
 *
 * @param e - The expression or matrix (rows of expressions).
 * @returns The LaTeX source.
 * @example
 * ```ts
 * import { parse, toLatex } from 'whitebox-engine/symbolic'
 * toLatex(parse('x^2/2 + sin(x)')) // => '\\frac{x^{2}}{2} + \\sin{\\left(x \\right)}'
 * ```
 */
export function toLatex(e: Expr | Matrix): string {
  if (isMatrix(e)) {
    const rows = e.map((row) => row.map(tex).join(' & ')).join('\\\\')
    const cols = e[0]?.length ?? 0
    const env = cols <= 10 ? 'matrix' : 'array'
    const spec = env === 'array' ? `{${'c'.repeat(cols)}}` : ''
    return `\\left[\\begin{${env}}${spec}${rows}\\end{${env}}\\right]`
  }
  return tex(e)
}

function ruleText(rule: string): string {
  return `\\text{${rule.replace(/([\\{}$&#^_%~])/g, '\\$1')}}`
}

/**
 * A derivation as a LaTeX `aligned` block: one line per step with the rule on the right. Consecutive steps that
 * continue from the previous result are chained with `&= ...`.
 *
 * @param d - A derivation (or a list of steps).
 * @returns The LaTeX source.
 * @example
 * ```ts
 * import { diffSteps, parse, stepsToLatex } from 'whitebox-engine/symbolic'
 * stepsToLatex(diffSteps(parse('x^2'), 'x')).startsWith('\\begin{aligned}') // => true
 * ```
 */
export function stepsToLatex(d: Derivation | readonly Step[]): string {
  const steps = Array.isArray(d) ? (d as readonly Step[]) : (d as Derivation).steps
  const lines: string[] = []
  let previous: Expr | undefined
  for (const s of steps) {
    const lhs = previous !== undefined && previous.key === s.before.key ? '' : tex(s.before)
    lines.push(`${lhs} &= ${tex(s.after)} && ${ruleText(s.rule)}`)
    previous = s.after
  }
  return `\\begin{aligned}${lines.join(' \\\\ ')}\\end{aligned}`
}
