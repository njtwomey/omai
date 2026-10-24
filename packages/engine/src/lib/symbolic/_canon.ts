// Canonicalising constructors, following SymPy's automatic evaluation: sums and products are flattened, numeric
// coefficients combined, like terms collected, equal bases merged, numeric powers evaluated exactly (perfect powers are
// pulled out of radicals and denominators rationalised), and infinities and big-O terms absorbed.
import { isNonnegative, isReal } from './_assume.ts'
import { makeFn } from './_functions.ts'
import {
  byKey,
  hasSymbol,
  rawAdd,
  rawConst,
  rawDerivative,
  rawFloat,
  rawIntegral,
  rawLogic,
  rawMul,
  rawNum,
  rawOrder,
  rawPiecewise,
  rawPow,
  rawRel,
} from './_node.ts'
import {
  bgcd,
  factorInteger,
  R0,
  R1,
  type Rational,
  rabs,
  radd,
  rat,
  rcmp,
  rfloor,
  risInt,
  rmul,
  rneg,
  rnum,
  rpow,
  rsub,
} from './_rational.ts'
import type { Expr, ExprFloat, ExprNum, ExprOrder, ExprSym, RelOp } from './expr.ts'

/** Shared constants. @internal */
export const ZERO: ExprNum = rawNum(R0)
/** @internal */
export const ONE: ExprNum = rawNum(R1)
/** @internal */
export const MINUS_ONE: ExprNum = rawNum({ n: -1n, d: 1n })
/** @internal */
export const TWO: ExprNum = rawNum({ n: 2n, d: 1n })
/** @internal */
export const HALF: ExprNum = rawNum({ n: 1n, d: 2n })
/** @internal */
export const PI: Expr = rawConst('pi')
/** @internal */
export const E: Expr = rawConst('E')
/** @internal */
export const I: Expr = rawConst('I')
/** @internal */
export const OO: Expr = rawConst('oo')
/** @internal */
export const NEG_INF: Expr = rawConst('-oo')
/** @internal */
export const ZOO: Expr = rawConst('zoo')
/** @internal */
export const NAN: Expr = rawConst('nan')
/** @internal */
export const TRUE: Expr = rawConst('true')
/** @internal */
export const FALSE: Expr = rawConst('false')

/** A number node from a rational. @internal */
export function numFromRational(r: Rational): ExprNum {
  return r.n === 0n ? ZERO : r.n === 1n && r.d === 1n ? ONE : rawNum(r)
}

/** A number node from a double: NaN and infinities map to their constants. @internal */
export function numFromFloat(x: number): Expr {
  if (Number.isNaN(x)) return NAN
  if (x === Number.POSITIVE_INFINITY) return OO
  if (x === Number.NEGATIVE_INFINITY) return NEG_INF
  return rawFloat(x)
}

/** Whether an expression is a rational or float number. @internal */
export function isNumeric(e: Expr): e is ExprNum | ExprFloat {
  return e.kind === 'num' || e.kind === 'float'
}

/** The rational of a number node. @internal */
export function ratOf(e: ExprNum): Rational {
  return { n: e.n, d: e.d }
}

/** Whether `e` is the rational `n / d`. @internal */
export function isRat(e: Expr, n: bigint, d = 1n): boolean {
  return e.kind === 'num' && e.n === n && e.d === d
}

/** Whether `e` is one of the infinities. @internal */
export function isInfinite(e: Expr): boolean {
  return e.kind === 'const' && (e.name === 'oo' || e.name === '-oo' || e.name === 'zoo')
}

function numValue(e: ExprNum | ExprFloat): number {
  return e.kind === 'float' ? e.value : rnum(ratOf(e))
}

function numAdd(a: ExprNum | ExprFloat, b: ExprNum | ExprFloat): ExprNum | ExprFloat {
  if (a.kind === 'num' && b.kind === 'num') return numFromRational(radd(ratOf(a), ratOf(b)))
  return rawFloat(numValue(a) + numValue(b))
}

function isZeroNum(e: ExprNum | ExprFloat): boolean {
  return e.kind === 'num' ? e.n === 0n : e.value === 0
}

/** Split off the numeric coefficient: 3*x*y gives [3, x*y]. @internal */
export function asCoeffMul(e: Expr): [ExprNum | ExprFloat, Expr] {
  if (isNumeric(e)) return [e, ONE]
  if (e.kind === 'mul') {
    const c = e.args[0]
    if (c !== undefined && isNumeric(c)) {
      const rest = e.args.length === 2 ? (e.args[1] as Expr) : rawMul(e.args.slice(1))
      return [c, rest]
    }
  }
  return [ONE, e]
}

/**
 * Base and exponent for collecting equal bases: x^2 gives [x, 2], anything else [e, 1]. As in SymPy, exp(x)*exp(y)
 * stays a product (exp is a function, not a power of E), while exp(x)*exp(x) merges into exp(2*x).
 * @internal
 */
export function asBaseExp(e: Expr): [Expr, Expr] {
  if (e.kind === 'pow') return [e.base, e.exp]
  return [e, ONE]
}

function withCoeff(c: ExprNum | ExprFloat, rest: Expr): Expr {
  if (c.kind === 'num' && c.n === 1n && c.d === 1n) return rest
  if (rest.kind === 'num' && rest.n === 1n && rest.d === 1n) return c
  return rest.kind === 'mul' ? rawMul([c, ...rest.args]) : rawMul([c, rest])
}

/** The x-degree of a term c * x^k * (factors free of x), or undefined for other shapes. */
function monomialDegree(t: Expr, x: ExprSym): Rational | undefined {
  if (!hasSymbol(t, x)) return R0
  if (t.key === x.key) return R1
  if (t.kind === 'pow' && t.base.key === x.key && t.exp.kind === 'num') return ratOf(t.exp)
  if (t.kind === 'mul') {
    let deg: Rational | undefined = R0
    for (const f of t.args) {
      const d = monomialDegree(f, x)
      if (d === undefined || deg === undefined) return undefined
      deg = radd(deg, d)
    }
    return deg
  }
  return undefined
}

/** Whether O(order) absorbs the term t. */
function absorbs(o: ExprOrder, t: Expr): boolean {
  const n = monomialDegree(o.expr, o.variable)
  const k = monomialDegree(t, o.variable)
  if (n === undefined || k === undefined) return false
  return o.point.key === OO.key ? rcmp(k, n) <= 0 : rcmp(k, n) >= 0
}

/**
 * The canonical sum. @internal
 */
export function makeAdd(input: readonly Expr[]): Expr {
  const flat: Expr[] = []
  for (const a of input) {
    if (a.kind === 'add') flat.push(...a.args)
    else flat.push(a)
  }
  if (flat.length === 1) return flat[0] as Expr
  let constant: ExprNum | ExprFloat = ZERO
  let posInf = false
  let negInf = false
  let complexInf = false
  const orders: ExprOrder[] = []
  const groups = new Map<string, { rest: Expr; coeff: ExprNum | ExprFloat }>()
  for (const t of flat) {
    if (isNumeric(t)) {
      constant = numAdd(constant, t)
      continue
    }
    if (t.kind === 'const') {
      if (t.name === 'nan') return NAN
      if (t.name === 'oo') {
        posInf = true
        continue
      }
      if (t.name === '-oo') {
        negInf = true
        continue
      }
      if (t.name === 'zoo') {
        complexInf = true
        continue
      }
    }
    if (t.kind === 'order') {
      orders.push(t)
      continue
    }
    const [c, rest] = asCoeffMul(t)
    const g = groups.get(rest.key)
    if (g === undefined) groups.set(rest.key, { rest, coeff: c })
    else g.coeff = numAdd(g.coeff, c)
  }
  if ((posInf && negInf) || (complexInf && (posInf || negInf))) return NAN
  let terms: Expr[] = []
  for (const g of groups.values()) if (!isZeroNum(g.coeff)) terms.push(withCoeff(g.coeff, g.rest))
  if (complexInf) terms.push(ZOO)
  else if (posInf) terms.push(OO)
  else if (negInf) terms.push(NEG_INF)
  else if (!isZeroNum(constant)) terms.push(constant)
  if (orders.length > 0) {
    const merged = mergeOrders(orders)
    terms = terms.filter((t) => !merged.some((o) => absorbs(o, t)))
    terms.push(...merged)
  }
  if (terms.length === 0) return constant.kind === 'float' && groups.size === 0 ? constant : ZERO
  if (terms.length === 1) return terms[0] as Expr
  terms.sort(byKey)
  return rawAdd(terms)
}

function mergeOrders(orders: readonly ExprOrder[]): ExprOrder[] {
  const out: ExprOrder[] = []
  for (const o of orders) {
    const same = out.findIndex((p) => p.variable.key === o.variable.key && p.point.key === o.point.key)
    if (same < 0) out.push(o)
    else if (absorbs(o, (out[same] as ExprOrder).expr)) out[same] = o
  }
  return out
}

/**
 * Normalise a product of positive rational bases raised to rational powers: each prime's total exponent is split into
 * an integer part (into the coefficient) and a fractional part in (0, 1); primes with equal fractional exponents share
 * one radical. So sqrt(2)*sqrt(3) = sqrt(6), sqrt(8) = 2*sqrt(2) and 1/sqrt(2) = sqrt(2)/2.
 */
/**
 * b^(p/q) for an integer b > 0 and a positive non-integer exponent, as SymPy's `Integer._eval_power`: each prime's
 * multiple of q leaves the radical; a remainder sharing a factor with q becomes its own lower root; the other
 * remainders share one q-th root (12^(2/3) = 2*18^(1/3), 8^(1/2) = 2*sqrt(2), 4^(1/3) = 2^(2/3)). Returns the
 * rational coefficient and the radicals.
 */
function integerRoot(b: bigint, e: Rational): [Rational, Expr[]] {
  const { n: p, d: q } = e
  let outInt = 1n
  const out: Expr[] = []
  const sqr = new Map<bigint, bigint>()
  for (const [prime, m] of factorInteger(b)) {
    const exponent = BigInt(m) * p
    const divE = exponent / q
    const divM = exponent % q
    if (divE > 0n) outInt *= prime ** divE
    if (divM > 0n) {
      const g = bgcd(divM, q)
      if (g !== 1n) out.push(rawPow(numFromRational(rat(prime)), numFromRational(rat(divM / g, q / g))))
      else sqr.set(prime, divM)
    }
  }
  let sqrGcd = 0n
  for (const v of sqr.values()) sqrGcd = bgcd(sqrGcd, v)
  let sqrInt = 1n
  for (const [k, v] of sqr) sqrInt *= k ** (v / sqrGcd)
  if (sqr.size > 0) out.push(rawPow(numFromRational(rat(sqrInt)), numFromRational(rat(sqrGcd, q))))
  return [rat(outInt), out]
}

/** b^e for a rational b > 0 and a non-integer rational e, as a coefficient and radicals (SymPy's rules). */
function positiveRoot(b: Rational, e: Rational): [Rational, Expr[]] {
  if (e.n < 0n) {
    // b^-e = (1/b)^e; for an integer b SymPy writes (1/b)^(p/q) = b^(p(q-1)/q) / b^p.
    if (b.d === 1n) {
      const ne = rneg(e)
      const [c, pows] = integerRoot(b.n, rat(ne.n * (ne.d - 1n), ne.d))
      return [rmul(c, rat(1n, b.n ** ne.n)), pows]
    }
    return positiveRoot({ n: b.d, d: b.n }, rneg(e))
  }
  if (b.d === 1n) return integerRoot(b.n, e)
  if (b.n === 1n) return positiveRoot({ n: b.d, d: 1n }, rneg(e))
  const [c1, p1] = positiveRoot({ n: b.n, d: 1n }, e)
  const [c2, p2] = positiveRoot({ n: b.d, d: 1n }, rneg(e))
  return [rmul(c1, c2), [...p1, ...p2]]
}

/**
 * Products of numeric radicals, as SymPy's Mul: each base^exponent is evaluated, then radicals with exactly the same
 * exponent share their base (sqrt(2)*sqrt(3) = sqrt(6), sqrt(2)*sqrt(6) = 2*sqrt(3)).
 */
function radicals(coeff: Rational, items: readonly (readonly [Rational, Rational])[]): [Rational, Expr[]] {
  let c = coeff
  const byExp = new Map<string, { exp: Rational; bases: Rational[] }>()
  const add = (pows: readonly Expr[]): void => {
    for (const pw of pows) {
      if (pw.kind !== 'pow' || pw.base.kind !== 'num' || pw.exp.kind !== 'num') continue
      const k = pw.exp.key
      const g = byExp.get(k)
      if (g === undefined) byExp.set(k, { exp: ratOf(pw.exp), bases: [ratOf(pw.base)] })
      else g.bases.push(ratOf(pw.base))
    }
  }
  for (const [b, e] of items) {
    if (risInt(e)) {
      c = rmul(c, rpow(b, Number(e.n)))
      continue
    }
    const [k, pows] = positiveRoot(b, e)
    c = rmul(c, k)
    add(pows)
  }
  const out: Expr[] = []
  for (const g of byExp.values()) {
    if (g.bases.length === 1) {
      out.push(rawPow(numFromRational(g.bases[0] as Rational), numFromRational(g.exp)))
      continue
    }
    let prod: Rational = R1
    for (const b of g.bases) prod = rmul(prod, b)
    const [k, pows] = positiveRoot(prod, g.exp)
    c = rmul(c, k)
    out.push(...pows)
  }
  return [c, out]
}

function buildMul(coeff: Rational, factors: Expr[]): Expr {
  factors.sort(byKey)
  if (coeff.n === 0n) return ZERO
  const isOne = coeff.n === 1n && coeff.d === 1n
  if (factors.length === 0) return numFromRational(coeff)
  if (factors.length === 1 && isOne) return factors[0] as Expr
  return rawMul(isOne ? factors : [numFromRational(coeff), ...factors])
}

/** (-1)^e for rational e: (-1)^floor(e) * (-1)^frac(e), with (-1)^(1/2) = I. */
function powMinusOne(e: Rational): Expr {
  const fl = rfloor(e)
  const frac = rsub(e, rat(fl))
  const sign = fl % 2n === 0n ? ONE : MINUS_ONE
  if (frac.n === 0n) return sign
  if (frac.n === 1n && frac.d === 2n) return sign === ONE ? I : rawMul([MINUS_ONE, I])
  const p = rawPow(MINUS_ONE, numFromRational(frac))
  return sign === ONE ? p : rawMul([MINUS_ONE, p])
}

const MAX_POW_BITS = 1_000_000

function ratPow(b: Rational, e: Rational): Expr {
  if (b.n === 0n) return e.n > 0n ? ZERO : ZOO
  if (risInt(e)) {
    const k = Number(e.n)
    const bits = Math.max(b.n.toString(2).length, b.d.toString(2).length)
    if (Math.abs(k) * bits > MAX_POW_BITS && rabs(b).n !== rabs(b).d) {
      return rawPow(numFromRational(b), numFromRational(e))
    }
    return numFromRational(rpow(b, k))
  }
  if (b.n < 0n) {
    const s = powMinusOne(e)
    if (b.n === -1n && b.d === 1n) return s
    return makeMul([ratPow(rneg(b), e), s])
  }
  const [c, pows] = radicals(R1, [[b, e]])
  return buildMul(c, pows)
}

/**
 * The canonical product. @internal
 */
export function makeMul(input: readonly Expr[], depth = 0): Expr {
  const flat: Expr[] = []
  for (const a of input) {
    if (a.kind === 'mul') flat.push(...a.args)
    else flat.push(a)
  }
  if (flat.length === 0) return ONE
  if (flat.length === 1) return flat[0] as Expr
  let coeff: Rational = R1
  let fcoeff: number | undefined
  let infinities = 0
  let complexInf = false
  const orders: ExprOrder[] = []
  const groups = new Map<string, { base: Expr; exps: Expr[] }>()
  for (const f of flat) {
    if (f.kind === 'num') {
      coeff = rmul(coeff, ratOf(f))
      continue
    }
    if (f.kind === 'float') {
      fcoeff = (fcoeff ?? 1) * f.value
      continue
    }
    if (f.kind === 'const') {
      if (f.name === 'nan') return NAN
      if (f.name === 'oo' || f.name === '-oo') {
        infinities++
        if (f.name === '-oo') coeff = rneg(coeff)
        continue
      }
      if (f.name === 'zoo') {
        complexInf = true
        continue
      }
    }
    if (f.kind === 'order') {
      orders.push(f)
      continue
    }
    // As SymPy's Mul.flatten: equal bases merge when their exponents share the same non-numeric part
    // (x*x^2 = x^3 and x^y*x^(2y) = x^(3y), but x^y*x^z stays a product).
    const [b, e] = asBaseExp(f)
    const [, term] = asCoeffMul(e)
    const k = `${b.key}|${term.key}`
    const g = groups.get(k)
    if (g === undefined) groups.set(k, { base: b, exps: [e] })
    else g.exps.push(e)
  }
  if (coeff.n === 0n || fcoeff === 0) return infinities > 0 || complexInf ? NAN : ZERO
  const radicalItems: [Rational, Rational][] = []
  const factors: Expr[] = []
  let again = false
  for (const g of groups.values()) {
    const e = g.exps.length === 1 ? (g.exps[0] as Expr) : makeAdd(g.exps)
    if (g.base.kind === 'num' && e.kind === 'num' && g.base.n > 0n) {
      radicalItems.push([ratOf(g.base), ratOf(e)])
      continue
    }
    const p = isRat(e, 1n) ? g.base : makePow(g.base, e)
    for (const q of p.kind === 'mul' ? p.args : [p]) {
      if (q.kind === 'num') coeff = rmul(coeff, ratOf(q))
      else if (q.kind === 'float') fcoeff = (fcoeff ?? 1) * q.value
      else {
        if (p.kind === 'mul' || isInfinite(q) || q.key === NAN.key || q.kind === 'order') again = true
        factors.push(q)
      }
    }
  }
  if (radicalItems.length > 0) {
    const [c, pows] = radicals(coeff, radicalItems)
    coeff = c
    factors.push(...pows)
  }
  if (again && depth < 3) {
    // A power expanded into a product (or produced a constant); merge its factors with the others.
    const parts: Expr[] = [numFromRational(coeff), ...factors]
    if (fcoeff !== undefined) parts.push(rawFloat(fcoeff))
    for (let i = 0; i < infinities; i++) parts.push(OO)
    if (complexInf) parts.push(ZOO)
    parts.push(...orders)
    return makeMul(parts, depth + 1)
  }
  if (coeff.n === 0n) return ZERO
  if (orders.length > 0) return mulOrders(orders, factors)
  if (infinities > 0 || complexInf) {
    const sign = rnum(coeff) * (fcoeff ?? 1)
    const inf = complexInf ? ZOO : sign > 0 ? OO : NEG_INF
    if (factors.length === 0) return inf
    factors.push(inf)
    return buildMul(R1, factors)
  }
  if (fcoeff !== undefined) {
    const c = rawFloat(fcoeff * rnum(coeff))
    if (factors.length === 1 && (factors[0] as Expr).kind === 'add') return distribute(c, factors[0] as Expr)
    factors.sort(byKey)
    if (factors.length === 0) return c
    return rawMul([c, ...factors])
  }
  if (factors.length === 1 && (factors[0] as Expr).kind === 'add' && !(coeff.n === 1n && coeff.d === 1n)) {
    return distribute(numFromRational(coeff), factors[0] as Expr)
  }
  return buildMul(coeff, factors)
}

/**
 * c * e without distributing a numeric c over a sum (SymPy's `_keep_coeff`), for factored and combined forms such
 * as 2*(x + 1) or (3*x + 2)/6.
 * @internal
 */
export function keepCoeff(c: Expr, e: Expr): Expr {
  if (isRat(c, 1n)) return e
  if (isRat(c, 0n)) return ZERO
  // -1 distributes (SymPy returns -factors), any other number stays in front of the sum.
  if (e.kind === 'add' && isNumeric(c) && !isRat(c, -1n)) return rawMul([c, e])
  return makeMul([c, e])
}

function distribute(c: Expr, sum: Expr): Expr {
  if (sum.kind !== 'add') return makeMul([c, sum])
  return makeAdd(sum.args.map((t) => makeMul([c, t])))
}

function mulOrders(orders: readonly ExprOrder[], factors: readonly Expr[]): Expr {
  let o = orders[0] as ExprOrder
  for (const p of orders.slice(1)) {
    const m = makeOrder(makeMul([o.expr, p.expr]), o.variable, o.point)
    if (m.kind === 'order') o = m
  }
  const dependent = factors.filter((f) => hasSymbol(f, o.variable))
  return makeOrder(makeMul([o.expr, ...dependent]), o.variable, o.point)
}

/** Big-O in canonical form: constants dropped, a sum reduced to its dominant monomial. @internal */
export function makeOrder(expr: Expr, variable: ExprSym, point: Expr): Expr {
  if (expr.kind === 'num' && expr.n === 0n) return ZERO
  if (!hasSymbol(expr, variable)) return rawOrder(ONE, variable, point)
  const atInf = point.key === OO.key
  const lead = (t: Expr): Expr => {
    if (t.kind === 'mul') {
      const dep = t.args.filter((f) => hasSymbol(f, variable))
      return dep.length === 1 ? (dep[0] as Expr) : rawMul(dep)
    }
    return t
  }
  if (expr.kind === 'add') {
    let best: Rational | undefined
    for (const t of expr.args) {
      const d = monomialDegree(t, variable)
      if (d === undefined) return rawOrder(expr, variable, point)
      if (best === undefined || (atInf ? rcmp(d, best) > 0 : rcmp(d, best) < 0)) best = d
    }
    return rawOrder(makePow(variable, numFromRational(best ?? R0)), variable, point)
  }
  const l = lead(expr)
  const d = monomialDegree(l, variable)
  if (d !== undefined) return rawOrder(d.n === 0n ? ONE : makePow(variable, numFromRational(d)), variable, point)
  return rawOrder(l, variable, point)
}

function floatPow(b: number, e: number): Expr {
  if (b < 0 && !Number.isInteger(e)) return rawPow(rawFloat(b), rawFloat(e))
  return numFromFloat(b ** e)
}

/**
 * The canonical power. @internal
 */
export function makePow(b: Expr, e: Expr): Expr {
  if (e.kind === 'num' && e.n === 0n) return ONE
  if (e.kind === 'num' && e.n === 1n && e.d === 1n) return b
  if (b.key === NAN.key || e.key === NAN.key) return NAN
  if (b.kind === 'num' && b.n === 1n && b.d === 1n) return isInfinite(e) ? NAN : ONE
  if (b.key === E.key) return makeFn('exp', [e])
  if (b.kind === 'num' && e.kind === 'num') return ratPow(ratOf(b), ratOf(e))
  if (isNumeric(b) && isNumeric(e)) return floatPow(numValue(b), numValue(e))
  if (b.kind === 'num' && b.n === 0n) {
    if (isNumeric(e)) return numValue(e) > 0 ? ZERO : ZOO
    if (e.key === OO.key) return ZERO
    if (e.key === NEG_INF.key) return ZOO
    return rawPow(b, e)
  }
  if (b.key === I.key && e.kind === 'num' && e.d === 1n) {
    const k = Number(((e.n % 4n) + 4n) % 4n)
    return [ONE, I, MINUS_ONE, rawMul([MINUS_ONE, I])][k] as Expr
  }
  if (isInfinite(b) && isNumeric(e)) {
    const v = numValue(e)
    if (v < 0) return ZERO
    if (b.key === ZOO.key) return ZOO
    if (b.key === NEG_INF.key && e.kind === 'num' && e.d === 1n) return e.n % 2n === 0n ? OO : NEG_INF
    if (b.key === OO.key) return OO
    return rawPow(b, e)
  }
  if (isInfinite(e) && isNumeric(b)) {
    const v = Math.abs(numValue(b))
    if (e.key === ZOO.key) return NAN
    if (v === 1) return NAN
    const big = e.key === OO.key ? v > 1 : v < 1
    return big ? (numValue(b) > 0 ? OO : ZOO) : ZERO
  }
  if (b.kind === 'pow') {
    const a = b.base
    const k = b.exp
    const integerOuter = e.kind === 'num' && e.d === 1n
    if (integerOuter) return makePow(a, makeMul([k, e]))
    if (isNumeric(k) && isNumeric(e)) {
      const kv = numValue(k)
      if (Math.abs(kv) < 1 || isNonnegative(a) === true) return makePow(a, makeMul([k, e]))
      if (k.kind === 'num' && k.d === 1n && k.n % 2n === 0n && isReal(a) === true) {
        return makePow(makeFn('Abs', [a]), makeMul([k, e]))
      }
    }
    return rawPow(b, e)
  }
  if (b.kind === 'fn' && b.name === 'exp' && e.kind === 'num' && e.d === 1n) {
    return makeFn('exp', [makeMul([b.args[0] as Expr, e])])
  }
  if (b.kind === 'mul') {
    if (e.kind === 'num' && e.d === 1n) return makeMul(b.args.map((f) => makePow(f, e)))
    if (isNumeric(e) || isReal(e) === true) {
      const out: Expr[] = []
      const keep: Expr[] = []
      for (const f of b.args) {
        if (f.kind === 'num' && rcmp(ratOf(f), R0) < 0 && !(f.n === -1n && f.d === 1n)) {
          out.push(makePow(numFromRational(rabs(ratOf(f))), e))
          keep.push(MINUS_ONE)
        } else if ((isNumeric(f) && numValue(f) > 0) || (!isNumeric(f) && isNonnegative(f) === true)) {
          out.push(makePow(f, e))
        } else keep.push(f)
      }
      if (out.length > 0) {
        const inner = makeMul(keep)
        out.push(isRat(inner, 1n) ? ONE : makePow(inner, e))
        return makeMul(out)
      }
    }
    return rawPow(b, e)
  }
  return rawPow(b, e)
}

/** A piecewise expression in canonical form: false pieces dropped, pieces after a true condition dropped. @internal */
export function makePiecewise(pieces: readonly (readonly [Expr, Expr])[]): Expr {
  const kept: [Expr, Expr][] = []
  for (const [v, c] of pieces) {
    if (c.key === FALSE.key) continue
    kept.push([v, c])
    if (c.key === TRUE.key) break
  }
  if (kept.length === 0) return NAN
  const first = kept[0] as [Expr, Expr]
  if (first[1].key === TRUE.key) return first[0]
  return rawPiecewise(kept)
}

function realValue(e: Expr): number | undefined {
  if (isNumeric(e)) return numValue(e)
  if (e.key === PI.key) return Math.PI
  if (e.key === E.key) return Math.E
  if (e.key === OO.key) return Number.POSITIVE_INFINITY
  if (e.key === NEG_INF.key) return Number.NEGATIVE_INFINITY
  return undefined
}

/** A relation, evaluated to true or false when both sides are real numbers. @internal */
export function makeRel(op: RelOp, lhs: Expr, rhs: Expr): Expr {
  if (op === '==' && lhs.key === rhs.key) return TRUE
  if (op === '!=' && lhs.key === rhs.key) return FALSE
  let cmp: number | undefined
  if (lhs.kind === 'num' && rhs.kind === 'num') cmp = rcmp(ratOf(lhs), ratOf(rhs))
  else {
    const a = realValue(lhs)
    const b = realValue(rhs)
    if (a !== undefined && b !== undefined) cmp = a < b ? -1 : a > b ? 1 : 0
  }
  if (cmp === undefined) return rawRel(op, lhs, rhs)
  const truth = { '==': cmp === 0, '!=': cmp !== 0, '<': cmp < 0, '<=': cmp <= 0, '>': cmp > 0, '>=': cmp >= 0 }[op]
  return truth ? TRUE : FALSE
}

const NEGATED: Readonly<Record<RelOp, RelOp>> = { '==': '!=', '!=': '==', '<': '>=', '<=': '>', '>': '<=', '>=': '<' }

/** A logical connective in canonical form. @internal */
export function makeLogic(op: 'and' | 'or' | 'not', args: readonly Expr[]): Expr {
  if (op === 'not') {
    const a = args[0] as Expr
    if (a.key === TRUE.key) return FALSE
    if (a.key === FALSE.key) return TRUE
    if (a.kind === 'logic' && a.op === 'not') return a.args[0] as Expr
    if (a.kind === 'rel') return makeRel(NEGATED[a.op], a.lhs, a.rhs)
    return rawLogic('not', [a])
  }
  const absorbing = op === 'and' ? FALSE : TRUE
  const identity = op === 'and' ? TRUE : FALSE
  const seen = new Map<string, Expr>()
  for (const a of args) {
    for (const b of a.kind === 'logic' && a.op === op ? a.args : [a]) {
      if (b.key === absorbing.key) return absorbing
      if (b.key !== identity.key) seen.set(b.key, b)
    }
  }
  const kept = [...seen.values()].sort(byKey)
  if (kept.length === 0) return identity
  if (kept.length === 1) return kept[0] as Expr
  return rawLogic(op, kept)
}

/** An unevaluated derivative (variables merged by symbol). @internal */
export function makeDerivative(expr: Expr, variables: readonly (readonly [ExprSym, number])[]): Expr {
  const merged: [ExprSym, number][] = []
  for (const [v, k] of variables) {
    const last = merged[merged.length - 1]
    if (last !== undefined && last[0].key === v.key) last[1] += k
    else if (k > 0) merged.push([v, k])
  }
  if (merged.length === 0) return expr
  if (isNumeric(expr)) return ZERO
  return rawDerivative(expr, merged)
}

/** An unevaluated integral. @internal */
export function makeIntegral(expr: Expr, variable: ExprSym, bounds?: readonly [Expr, Expr]): Expr {
  return rawIntegral(expr, variable, bounds)
}

/** Rebuild a node from transformed children, re-canonicalising. @internal */
export function mapChildren(e: Expr, f: (x: Expr) => Expr): Expr {
  switch (e.kind) {
    case 'add':
      return makeAdd(e.args.map(f))
    case 'mul':
      return makeMul(e.args.map(f))
    case 'pow':
      return makePow(f(e.base), f(e.exp))
    case 'fn':
      return makeFn(e.name, e.args.map(f))
    case 'piecewise':
      return makePiecewise(e.pieces.map(([v, c]) => [f(v), f(c)] as const))
    case 'rel':
      return makeRel(e.op, f(e.lhs), f(e.rhs))
    case 'logic':
      return makeLogic(e.op, e.args.map(f))
    case 'order': {
      const v = f(e.variable)
      return makeOrder(f(e.expr), v.kind === 'sym' ? v : e.variable, f(e.point))
    }
    case 'derivative':
      return makeDerivative(f(e.expr), e.variables)
    case 'integral':
      return makeIntegral(f(e.expr), e.variable, e.bounds === undefined ? undefined : [f(e.bounds[0]), f(e.bounds[1])])
    default:
      return e
  }
}

/** Numeric helpers shared by other modules. @internal */
export { numValue, rsub as ratSub }
