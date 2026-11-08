/**
 * Numeric evaluation of symbolic expressions, private to symbolic: real (double) and complex values of an expression
 * under an environment of numbers, with caller-supplied implementations for functions symbolic cannot evaluate, and an
 * adaptive Gauss-Kronrod rule for the numeric fallback of definite integrals (kept here so symbolic does not depend on
 * numerics).
 */

import { type Complex, cadd, cdiv, cexp, clog, cmul, OmaiError } from '../base/index.ts'
import { children } from './_node.ts'
import { KNOWN_FUNCTIONS } from './_order.ts'
import type { Expr } from './expr.ts'

/** Numeric implementations supplied by the caller for functions symbolic cannot evaluate itself (erf, gamma, f). */
export type NumericFunctions = Readonly<Record<string, (...args: number[]) => number>>

function csqrt(a: Complex): Complex {
  const r = Math.hypot(a[0], a[1])
  const re = Math.sqrt((r + a[0]) / 2)
  const im = Math.sqrt(Math.max(0, (r - a[0]) / 2))
  // A negative zero imaginary part selects the lower side of the branch cut, as numpy's csqrt.
  return [re, a[1] < 0 || Object.is(a[1], -0) ? -im : im]
}
function cpow(a: Complex, b: Complex): Complex {
  if (b[1] === 0 && Number.isInteger(b[0]) && Math.abs(b[0]) <= 64) {
    let k = Math.abs(b[0])
    let base = a
    let out: Complex = [1, 0]
    while (k > 0) {
      if (k & 1) out = cmul(out, base)
      base = cmul(base, base)
      k >>= 1
    }
    return b[0] < 0 ? cdiv([1, 0], out) : out
  }
  if (a[0] === 0 && a[1] === 0) return b[0] > 0 ? [0, 0] : [Number.NaN, Number.NaN]
  if (b[1] === 0 && b[0] === 0.5) return csqrt(a)
  return cexp(cmul(b, clog(a)))
}
const csin = (a: Complex): Complex => [Math.sin(a[0]) * Math.cosh(a[1]), Math.cos(a[0]) * Math.sinh(a[1])]
const ccos = (a: Complex): Complex => [Math.cos(a[0]) * Math.cosh(a[1]), -Math.sin(a[0]) * Math.sinh(a[1])]
const csinh = (a: Complex): Complex => [Math.sinh(a[0]) * Math.cos(a[1]), Math.cosh(a[0]) * Math.sin(a[1])]
const ccosh = (a: Complex): Complex => [Math.cosh(a[0]) * Math.cos(a[1]), Math.sinh(a[0]) * Math.sin(a[1])]
const I1: Complex = [0, 1]
const ONE_C: Complex = [1, 0]
const neg = (a: Complex): Complex => [-a[0], -a[1]]

function complexFn(name: string, z: Complex): Complex | undefined {
  switch (name) {
    case 'exp':
      return cexp(z)
    case 'log':
      return clog(z)
    case 'sin':
      return csin(z)
    case 'cos':
      return ccos(z)
    case 'tan':
      return cdiv(csin(z), ccos(z))
    case 'cot':
      return cdiv(ccos(z), csin(z))
    case 'sec':
      return cdiv(ONE_C, ccos(z))
    case 'csc':
      return cdiv(ONE_C, csin(z))
    case 'sinh':
      return csinh(z)
    case 'cosh':
      return ccosh(z)
    case 'tanh':
      return cdiv(csinh(z), ccosh(z))
    case 'coth':
      return cdiv(ccosh(z), csinh(z))
    case 'asin':
      return cmul(neg(I1), clog(cadd(cmul(I1, z), csqrt(cadd(ONE_C, neg(cmul(z, z)))))))
    case 'acos': {
      const s = complexFn('asin', z) as Complex
      return [Math.PI / 2 - s[0], -s[1]]
    }
    case 'atan':
      return cmul([0, 0.5], cadd(clog(cadd(ONE_C, neg(cmul(I1, z)))), neg(clog(cadd(ONE_C, cmul(I1, z))))))
    case 'acot':
      return complexFn('atan', cdiv(ONE_C, z))
    case 'asinh':
      return clog(cadd(z, csqrt(cadd(cmul(z, z), ONE_C))))
    case 'acosh':
      return clog(cadd(z, cmul(csqrt(cadd(z, ONE_C)), csqrt(cadd(z, neg(ONE_C))))))
    case 'atanh':
      return cmul([0.5, 0], cadd(clog(cadd(ONE_C, z)), neg(clog(cadd(ONE_C, neg(z))))))
    case 'Abs':
      return [Math.hypot(z[0], z[1]), 0]
    case 'sign': {
      const r = Math.hypot(z[0], z[1])
      return r === 0 ? [0, 0] : [z[0] / r, z[1] / r]
    }
    case 're':
      return [z[0], 0]
    case 'im':
      return [z[1], 0]
    case 'conjugate':
      return [z[0], -z[1]]
    case 'arg':
      return [Math.atan2(z[1], z[0]), 0]
    case 'floor':
      return z[1] === 0 ? [Math.floor(z[0]), 0] : undefined
    case 'ceiling':
      return z[1] === 0 ? [Math.ceil(z[0]), 0] : undefined
    default:
      return undefined
  }
}

/**
 * Evaluate an expression as a complex number, with symbols bound by `env`; undefined when something cannot be
 * evaluated (an unbound symbol, an undefined function, a relation).
 * @internal
 */
export function evalComplex(
  e: Expr,
  env: ReadonlyMap<string, Complex>,
  fns: NumericFunctions = {},
): Complex | undefined {
  switch (e.kind) {
    case 'num':
      return [Number(e.n) / Number(e.d), 0]
    case 'float':
      return [e.value, 0]
    case 'sym':
      return env.get(e.name)
    case 'const':
      if (e.name === 'pi') return [Math.PI, 0]
      if (e.name === 'E') return [Math.E, 0]
      if (e.name === 'I') return I1
      if (e.name === 'oo') return [Number.POSITIVE_INFINITY, 0]
      if (e.name === '-oo') return [Number.NEGATIVE_INFINITY, 0]
      return undefined
    case 'add': {
      let s: Complex = [0, 0]
      for (const a of e.args) {
        const v = evalComplex(a, env, fns)
        if (v === undefined) return undefined
        s = cadd(s, v)
      }
      return s
    }
    case 'mul': {
      // A real coefficient scales both parts (so -x of a real x has a negative zero imaginary part, as in numpy).
      let s: Complex = ONE_C
      let scale = 1
      for (const a of e.args) {
        if (a.kind === 'num' || a.kind === 'float') {
          scale *= a.kind === 'num' ? Number(a.n) / Number(a.d) : a.value
          continue
        }
        const v = evalComplex(a, env, fns)
        if (v === undefined) return undefined
        s = cmul(s, v)
      }
      return [scale * s[0], scale * s[1]]
    }
    case 'pow': {
      const b = evalComplex(e.base, env, fns)
      const x = evalComplex(e.exp, env, fns)
      return b === undefined || x === undefined ? undefined : cpow(b, x)
    }
    case 'fn': {
      const args: Complex[] = []
      for (const a of e.args) {
        const v = evalComplex(a, env, fns)
        if (v === undefined) return undefined
        args.push(v)
      }
      const user = fns[e.name]
      if (user !== undefined) {
        if (args.some((z) => z[1] !== 0)) return undefined
        return [user(...args.map((z) => z[0])), 0]
      }
      if (e.name === 'atan2' && args.length === 2) {
        const [y, x] = args as [Complex, Complex]
        return y[1] === 0 && x[1] === 0 ? [Math.atan2(y[0], x[0]), 0] : undefined
      }
      const z = args[0]
      return z === undefined || args.length !== 1 ? undefined : complexFn(e.name, z)
    }
    case 'piecewise': {
      for (const [v, c] of e.pieces) {
        const t = evalCondition(c, env, fns)
        if (t === undefined) return undefined
        if (t) return evalComplex(v, env, fns)
      }
      return undefined
    }
    default:
      return undefined
  }
}

function evalCondition(c: Expr, env: ReadonlyMap<string, Complex>, fns: NumericFunctions): boolean | undefined {
  if (c.kind === 'const') return c.name === 'true' ? true : c.name === 'false' ? false : undefined
  if (c.kind === 'rel') {
    const a = evalComplex(c.lhs, env, fns)
    const b = evalComplex(c.rhs, env, fns)
    if (a === undefined || b === undefined) return undefined
    if (c.op === '==') return a[0] === b[0] && a[1] === b[1]
    if (c.op === '!=') return a[0] !== b[0] || a[1] !== b[1]
    if (Math.abs(a[1]) > 1e-12 || Math.abs(b[1]) > 1e-12) return undefined
    return { '<': a[0] < b[0], '<=': a[0] <= b[0], '>': a[0] > b[0], '>=': a[0] >= b[0] }[c.op]
  }
  if (c.kind === 'logic') {
    const vs = c.args.map((a) => evalCondition(a, env, fns))
    if (vs.some((v) => v === undefined)) return undefined
    if (c.op === 'not') return !vs[0]
    return c.op === 'and' ? vs.every(Boolean) : vs.some(Boolean)
  }
  return undefined
}

/** Whether an expression has no free symbols and no undefined functions. */
function isConstant(e: Expr): boolean {
  if (e.kind === 'sym' || e.kind === 'order' || e.kind === 'integral' || e.kind === 'derivative') return false
  if (e.kind === 'rel' || e.kind === 'logic' || e.kind === 'piecewise') return false
  if (e.kind === 'fn' && !KNOWN_FUNCTIONS.has(e.name)) return false
  return children(e).every(isConstant)
}

/** The complex value of a constant expression (SymPy's `is_number` and `complex()`), or undefined. @internal */
export function evalComplexConstant(e: Expr): Complex | undefined {
  if (!isConstant(e)) return undefined
  if (e.kind === 'const' && (e.name === 'zoo' || e.name === 'nan')) return undefined
  const v = evalComplex(e, new Map())
  return v === undefined || Number.isNaN(v[0]) || Number.isNaN(v[1]) ? undefined : v
}

/**
 * Evaluate an expression to a real double; complex results give NaN.
 * @internal
 */
export function evalReal(e: Expr, env: Readonly<Record<string, number>>, fns: NumericFunctions = {}): number {
  const m = new Map<string, Complex>()
  for (const [k, v] of Object.entries(env)) m.set(k, [v, 0])
  const missing = freeNames(e).filter((n) => !m.has(n))
  if (missing.length > 0) throw new OmaiError('value', `evaluate: no value for ${missing.join(', ')}`)
  const v = evalReal1(e, m, fns)
  return v
}

function evalReal1(e: Expr, env: ReadonlyMap<string, Complex>, fns: NumericFunctions): number {
  // Real arithmetic where possible (exact branch cuts as numpy), complex only for intermediate values.
  switch (e.kind) {
    case 'pow': {
      const b = evalReal1(e.base, env, fns)
      const x = evalReal1(e.exp, env, fns)
      return b ** x
    }
    case 'add': {
      let s = 0
      for (const a of e.args) s += evalReal1(a, env, fns)
      return s
    }
    case 'mul': {
      let s = 1
      for (const a of e.args) s *= evalReal1(a, env, fns)
      return s
    }
    case 'fn': {
      const user = fns[e.name]
      const xs = e.args.map((a) => evalReal1(a, env, fns))
      if (user !== undefined) return user(...xs)
      const x = xs[0] ?? Number.NaN
      const real = REAL[e.name]
      if (real !== undefined) return real(x, xs[1] ?? Number.NaN)
      throw new OmaiError('not-implemented', `evaluate: no numeric implementation of ${e.name}; pass functions`)
    }
    case 'const':
      if (e.name === 'I' || e.name === 'zoo' || e.name === 'nan') return Number.NaN
      break
    case 'piecewise':
      for (const [v, c] of e.pieces) {
        const t = evalCondition(c, env, fns)
        if (t === undefined) return Number.NaN
        if (t) return evalReal1(v, env, fns)
      }
      return Number.NaN
    default:
      break
  }
  const z = evalComplex(e, env, fns)
  if (z === undefined) throw new OmaiError('value', `evaluate: cannot evaluate ${e.kind}`)
  return Math.abs(z[1]) > 0 ? Number.NaN : z[0]
}

const REAL: Readonly<Record<string, (x: number, y: number) => number>> = {
  exp: Math.exp,
  log: Math.log,
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  cot: (x) => 1 / Math.tan(x),
  sec: (x) => 1 / Math.cos(x),
  csc: (x) => 1 / Math.sin(x),
  asin: Math.asin,
  acos: Math.acos,
  atan: Math.atan,
  acot: (x) => Math.atan(1 / x),
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  coth: (x) => 1 / Math.tanh(x),
  asinh: Math.asinh,
  acosh: Math.acosh,
  atanh: Math.atanh,
  Abs: Math.abs,
  sign: Math.sign,
  floor: Math.floor,
  ceiling: Math.ceil,
  re: (x) => x,
  im: () => 0,
  conjugate: (x) => x,
  arg: (x) => (x < 0 ? Math.PI : 0),
  atan2: (y, x) => Math.atan2(y, x),
}

/** Names of the free symbols (by name). @internal */
export function freeNames(e: Expr): string[] {
  const out = new Set<string>()
  const walk = (x: Expr): void => {
    if (x.kind === 'sym') out.add(x.name)
    else for (const c of children(x)) walk(c)
  }
  walk(e)
  return [...out]
}

// Gauss-Kronrod 7-15 nodes and weights on [-1, 1].
const XGK = [
  0.9914553711208126, 0.9491079123427585, 0.8648644233597691, 0.7415311855993945, 0.5860872354676911,
  0.4058451513773972, 0.20778495500789848, 0,
]
const WGK = [
  0.022935322010529224, 0.06309209262997856, 0.10479001032225019, 0.14065325971552592, 0.1690047266392679,
  0.19035057806478542, 0.20443294007529889, 0.20948214108472782,
]
const WG = [0.1294849661688697, 0.27970539148927664, 0.3818300505051189, 0.4179591836734694]

function kronrod(f: (x: number) => number, a: number, b: number): [number, number] {
  const c = (a + b) / 2
  const h = (b - a) / 2
  let k = 0
  let g = 0
  for (let i = 0; i < 8; i++) {
    const x = XGK[i] as number
    const w = WGK[i] as number
    const fs = x === 0 ? f(c) : f(c - h * x) + f(c + h * x)
    k += w * fs
    if (i % 2 === 1) g += (WG[(i - 1) / 2] as number) * fs
  }
  return [k * h, Math.abs((k - g) * h)]
}

/**
 * Adaptive Gauss-Kronrod quadrature of a scalar function, with infinite limits mapped to a finite interval.
 * @internal
 */
export function quadrature(f: (x: number) => number, a: number, b: number, tol = 1e-10): number {
  if (a === b) return 0
  if (a > b) return -quadrature(f, b, a, tol)
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    // x = t / (1 - t^2) maps (-1, 1) onto the real line; half-lines map from (0, 1) or (-1, 0).
    const g = (t: number): number => {
      const d = 1 - t * t
      return (f(t / d) * (1 + t * t)) / (d * d)
    }
    const lo = Number.isFinite(a) ? a : Number.NEGATIVE_INFINITY
    const hi = Number.isFinite(b) ? b : Number.POSITIVE_INFINITY
    if (lo === Number.NEGATIVE_INFINITY && hi === Number.POSITIVE_INFINITY) return quadrature(g, -1, 1, tol)
    if (hi === Number.POSITIVE_INFINITY) return quadrature((t) => f(lo + t / (1 - t)) / ((1 - t) * (1 - t)), 0, 1, tol)
    return quadrature((t) => f(hi - t / (1 - t)) / ((1 - t) * (1 - t)), 0, 1, tol)
  }
  const stack: [number, number, number][] = []
  let total = 0
  const first = kronrod(f, a, b)
  stack.push([a, b, 0])
  let budget = 2000
  while (stack.length > 0 && budget-- > 0) {
    const [lo, hi, depth] = stack.pop() as [number, number, number]
    const [v, err] = lo === a && hi === b && depth === 0 ? first : kronrod(f, lo, hi)
    if (err <= Math.max(tol * Math.abs(v), 1e-15 * (hi - lo)) || depth > 40) total += v
    else {
      const mid = (lo + hi) / 2
      stack.push([lo, mid, depth + 1], [mid, hi, depth + 1])
    }
  }
  for (const [lo, hi] of stack) total += kronrod(f, lo, hi)[0]
  return total
}
