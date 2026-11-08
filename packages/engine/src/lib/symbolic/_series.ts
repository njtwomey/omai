// Truncated Laurent series in a variable t at 0, as exact coefficient lists with a known absolute order, composed
// node by node (sums, products, inverses, powers, exp, log, sin, cos and, for any other known function, Taylor
// composition from its derivatives). A logarithm of t is carried as a placeholder symbol in the coefficients.
import { OmaiError } from '../base/index.ts'
import { isNegative, isPositive } from './_assume.ts'
import { WorkExhausted } from './_budget.ts'
import { isInfinite, isRat, MINUS_ONE, makeAdd, makeMul, makePow, NAN, numFromRational, ONE, ZERO } from './_canon.ts'
import { expandExpr } from './_expand.ts'
import { makeFn } from './_functions.ts'
import { hasSymbol } from './_node.ts'
import { evalComplexConstant, freeNames } from './_numeric.ts'
import { rat } from './_rational.ts'
import { cancel } from './_rational-functions.ts'
import { diff } from './differentiate.ts'
import { type Expr, type ExprSym, type Step, substitute, sym } from './expr.ts'

/** A truncated series: sum of c[i] t^(v + i) for i < n - v, plus O(t^n). @internal */
export interface S {
  readonly v: number
  readonly c: readonly Expr[]
  readonly n: number
}

/** The placeholder for log(t) inside coefficients. @internal */
export const LOG_T: ExprSym = sym('log_t__')

function norm(e: Expr): Expr {
  const x = expandExpr(e)
  return x.kind === 'add' || x.kind === 'mul' || x.kind === 'pow' ? cancel(x) : x
}

function isZero(e: Expr): boolean {
  if (e.kind === 'num') return e.n === 0n
  if (freeNames(e).length === 0) {
    const z = evalComplexConstant(e)
    if (z !== undefined && Math.hypot(z[0], z[1]) < 1e-13)
      return norm(e).key === ZERO.key || Math.hypot(z[0], z[1]) < 1e-15
  }
  return false
}

function make(v: number, c: readonly Expr[], n: number): S {
  const cs = c.map(norm)
  let k = 0
  while (k < cs.length && isZero(cs[k] as Expr)) k++
  if (k === cs.length) return { v: n, c: [], n }
  return { v: v + k, c: cs.slice(k, Math.max(k, n - v)), n }
}

function coeff(s: S, k: number): Expr {
  return k < s.v ? ZERO : (s.c[k - s.v] ?? ZERO)
}

/** A constant. @internal */
export function sConst(c: Expr, n: number): S {
  return n <= 0 ? { v: n, c: [], n } : make(0, [c], n)
}

/** The variable t. @internal */
export function sVar(n: number): S {
  return make(0, [ZERO, ONE], n)
}

/** Sum. @internal */
export function sAdd(a: S, b: S): S {
  const n = Math.min(a.n, b.n)
  const v = Math.min(a.v, b.v)
  const out: Expr[] = []
  for (let k = v; k < n; k++) out.push(makeAdd([coeff(a, k), coeff(b, k)]))
  return make(v, out, n)
}

/** Scale by a constant. @internal */
export function sScale(a: S, c: Expr): S {
  return make(
    a.v,
    a.c.map((x) => makeMul([x, c])),
    a.n,
  )
}

/** Product. @internal */
export function sMul(a: S, b: S): S {
  if (a.c.length === 0 && b.c.length === 0) return { v: a.n + b.n, c: [], n: a.n + b.n }
  const n = Math.min(a.n + b.v, b.n + a.v)
  const v = a.v + b.v
  const out: Expr[] = []
  for (let k = v; k < n; k++) {
    const terms: Expr[] = []
    for (let i = a.v; i <= k - b.v; i++) terms.push(makeMul([coeff(a, i), coeff(b, k - i)]))
    out.push(makeAdd(terms))
  }
  return make(v, out, n)
}

/** Split a = c0 t^v (1 + u): returns [c0, v, u] with u of valuation at least 1. */
function splitLead(a: S): [Expr, number, S] {
  const c0 = a.c[0]
  if (c0 === undefined) throw new OmaiError('convergence', 'series: not enough terms (cancellation)')
  const inv0 = makePow(c0, MINUS_ONE)
  const m = a.n - a.v
  const u = make(
    1,
    a.c.slice(1).map((x) => makeMul([x, inv0])),
    m,
  )
  return [c0, a.v, u]
}

/** Σ coeffs[k] u^k for u of valuation >= 1, to relative order m. */
function compose(coeffs: (k: number) => Expr, u: S, m: number): S {
  let out = sConst(coeffs(0), m)
  let pw = sConst(ONE, m)
  for (let k = 1; k < m; k++) {
    pw = sMul(pw, u)
    if (pw.v >= m) break
    out = sAdd(out, sScale(pw, coeffs(k)))
  }
  return out
}

function shift(a: S, k: number): S {
  return { v: a.v + k, c: a.c, n: a.n + k }
}

/** Reciprocal. @internal */
export function sInv(a: S): S {
  const [c0, v, u] = splitLead(a)
  const m = a.n - a.v
  const geo = compose((k) => (k % 2 === 0 ? ONE : MINUS_ONE), u, m)
  return shift(sScale(geo, makePow(c0, MINUS_ONE)), -v)
}

/** Power with an exponent free of t. @internal */
export function sPow(a: S, r: Expr): S {
  if (r.kind === 'num' && r.d === 1n) {
    const k = Number(r.n)
    if (k === 0) return sConst(ONE, a.n - a.v)
    let base = k < 0 ? sInv(a) : a
    let out: S | undefined
    let e = Math.abs(k)
    while (e > 0) {
      if (e & 1) out = out === undefined ? base : sMul(out, base)
      e >>= 1
      if (e > 0) base = sMul(base, base)
    }
    return out as S
  }
  const [c0, v, u] = splitLead(a)
  const vr = makeMul([numFromRational(rat(BigInt(v))), r])
  if (vr.kind !== 'num' || vr.d !== 1n) {
    throw new OmaiError('not-implemented', 'series: fractional powers of t (Puiseux series) are not supported')
  }
  const m = a.n - a.v
  // (1 + u)^r = Σ binom(r, k) u^k
  const binom: Expr[] = [ONE]
  const b = (k: number): Expr => {
    while (binom.length <= k) {
      const j = binom.length
      binom.push(
        makeMul([
          binom[j - 1] as Expr,
          makeAdd([r, numFromRational(rat(BigInt(-(j - 1))))]),
          numFromRational(rat(1n, BigInt(j))),
        ]),
      )
    }
    return binom[k] as Expr
  }
  const body = compose(b, u, m)
  return shift(sScale(body, makePow(c0, r)), Number(vr.n))
}

function factorial(k: number): bigint {
  let f = 1n
  for (let i = 2; i <= k; i++) f *= BigInt(i)
  return f
}

/** Split a = a0 + u with u of valuation >= 1 (a must have no negative powers). */
function splitConstant(a: S, what: string): [Expr, S] {
  if (a.v < 0) throw new OmaiError('not-implemented', `series: ${what} has an essential singularity here`)
  const a0 = a.v === 0 ? (a.c[0] as Expr) : ZERO
  const u = a.v === 0 ? make(1, a.c.slice(1), a.n) : a
  return [a0, u]
}

/** exp of a series. @internal */
export function sExp(a: S): S {
  const [a0, u] = splitConstant(a, 'exp')
  const m = a.n
  const body = compose((k) => numFromRational(rat(1n, factorial(k))), u, m)
  return sScale(body, makeFn('exp', [a0]))
}

/** log of a series (log t enters as the placeholder symbol). @internal */
export function sLog(a: S): S {
  const [c0, v, u] = splitLead(a)
  const m = a.n - a.v
  const body = compose((k) => (k === 0 ? ZERO : numFromRational(rat(BigInt(k % 2 === 1 ? 1 : -1), BigInt(k)))), u, m)
  const constant = makeAdd([makeFn('log', [c0]), makeMul([numFromRational(rat(BigInt(v))), LOG_T])])
  return sAdd(sConst(constant, m), body)
}

/** sin or cos of a series by sin(a0 + u) = sin a0 cos u + cos a0 sin u. @internal */
export function sTrig(a: S, name: 'sin' | 'cos' | 'sinh' | 'cosh'): S {
  const [a0, u] = splitConstant(a, name)
  const m = a.n
  const hyp = name.endsWith('h')
  const sinU = compose(
    (k) => (k % 2 === 1 ? numFromRational(rat(hyp || k % 4 === 1 ? 1n : -1n, factorial(k))) : ZERO),
    u,
    m,
  )
  const cosU = compose(
    (k) => (k % 2 === 0 ? numFromRational(rat(hyp || k % 4 === 0 ? 1n : -1n, factorial(k))) : ZERO),
    u,
    m,
  )
  const s0 = makeFn(hyp ? 'sinh' : 'sin', [a0])
  const c0 = makeFn(hyp ? 'cosh' : 'cos', [a0])
  if (name === 'sin' || name === 'sinh') return sAdd(sScale(cosU, s0), sScale(sinU, c0))
  // cos(a0 + u) = cos a0 cos u - sin a0 sin u; cosh(a0 + u) = cosh a0 cosh u + sinh a0 sinh u.
  return sAdd(sScale(cosU, c0), sScale(sinU, hyp ? s0 : makeMul([MINUS_ONE, s0])))
}

/** f(a) for any function analytic at a0, by Taylor composition from symbolic derivatives. @internal */
export function sGeneric(name: string, a: S): S {
  const [a0, u] = splitConstant(a, name)
  const z = sym('z_series__')
  const m = a.n
  const ds: Expr[] = []
  let cur: Expr = makeFn(name, [z])
  for (let k = 0; k < m; k++) {
    const val = substitute(cur, [[z, a0]])
    if (isInfinite(val) || val.key === NAN.key) {
      throw new OmaiError('not-implemented', `series: ${name} is not analytic at ${a0.key}`)
    }
    ds.push(makeMul([val, numFromRational(rat(1n, factorial(k)))]))
    cur = diff(cur, z)
  }
  return compose((k) => ds[k] ?? ZERO, u, m)
}

/** Series of an expression in t at 0+, to absolute order n (the result may know fewer terms). @internal */
export function toSeries(e: Expr, t: ExprSym, n: number, steps?: Step[]): S {
  const out = toSeries1(e, t, n, steps)
  if (steps !== undefined && hasSymbol(e, t) && e.kind !== 'sym') {
    steps.push({ rule: `series of ${e.kind === 'fn' ? e.name : e.kind}`, before: e, after: seriesToExpr(out, t) })
  }
  return out
}

function toSeries1(e: Expr, t: ExprSym, n: number, steps?: Step[]): S {
  if (!hasSymbol(e, t)) return sConst(e, n)
  const go = (x: Expr): S => toSeries(x, t, n, steps)
  switch (e.kind) {
    case 'sym':
      return sVar(n)
    case 'add':
      return e.args.map(go).reduce(sAdd)
    case 'mul':
      return e.args.map(go).reduce(sMul)
    case 'pow':
      if (!hasSymbol(e.exp, t)) return sPow(go(e.base), e.exp)
      return sExp(sMul(go(e.exp), sLog(go(e.base))))
    case 'fn': {
      const a = e.args[0] as Expr
      switch (e.name) {
        case 'exp':
          return sExp(go(a))
        case 'log':
          return sLog(go(a))
        case 'sin':
        case 'cos':
        case 'sinh':
        case 'cosh':
          return sTrig(go(a), e.name)
        case 'tan':
          return sMul(sTrig(go(a), 'sin'), sInv(sTrig(go(a), 'cos')))
        case 'cot':
          return sMul(sTrig(go(a), 'cos'), sInv(sTrig(go(a), 'sin')))
        case 'sec':
          return sInv(sTrig(go(a), 'cos'))
        case 'csc':
          return sInv(sTrig(go(a), 'sin'))
        case 'tanh':
          return sMul(sTrig(go(a), 'sinh'), sInv(sTrig(go(a), 'cosh')))
        case 'coth':
          return sMul(sTrig(go(a), 'cosh'), sInv(sTrig(go(a), 'sinh')))
        case 'Abs': {
          const s = go(a)
          const lead = s.c[0]
          if (lead !== undefined && isPositive(lead) === true) return s
          if (lead !== undefined && isNegative(lead) === true) return sScale(s, MINUS_ONE)
          const z = lead === undefined ? undefined : evalComplexConstant(lead)
          if (z !== undefined && z[1] === 0) return z[0] >= 0 ? s : sScale(s, MINUS_ONE)
          throw new OmaiError('not-implemented', 'series: Abs of a series with unknown sign')
        }
        default:
          if (e.args.length !== 1) throw new OmaiError('not-implemented', `series: ${e.name} of several arguments`)
          return sGeneric(e.name, go(a))
      }
    }
    default:
      throw new OmaiError('not-implemented', `series: cannot expand a ${e.kind}`)
  }
}

/** The series as an expression in t, without the order term. @internal */
export function seriesToExpr(s: S, t: Expr): Expr {
  const terms: Expr[] = []
  s.c.forEach((c, i) => {
    const k = s.v + i
    if (k < s.n) terms.push(makeMul([c, makePow(t, numFromRational(rat(BigInt(k))))]))
  })
  return makeAdd(terms)
}

/** Series to absolute order n, retrying with more terms when cancellation loses precision. @internal */
export function seriesAt0(e: Expr, t: ExprSym, n: number, steps?: Step[]): S {
  for (const extra of [0, 2, 4, 8, 16]) {
    let s: S
    try {
      s = toSeries(e, t, n + extra, extra === 0 ? steps : undefined)
    } catch (err) {
      // Cancellation is retried with more terms; running out of work is not.
      if (err instanceof OmaiError && err.code === 'convergence' && !(err instanceof WorkExhausted)) continue
      throw err
    }
    if (s.n >= n) return { v: s.v, c: s.c.slice(0, Math.max(0, n - s.v)), n }
  }
  throw new OmaiError('convergence', 'series: too much cancellation to reach the requested order')
}

/** Whether a series coefficient is identically 1 (helper for callers). @internal */
export function isOne(e: Expr): boolean {
  return isRat(e, 1n)
}
