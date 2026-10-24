// Sparse multivariate polynomials over the rationals (lex order, highest term first), dense univariate helpers, and
// the conversion between expressions and polynomials over automatically chosen generators.
import { WhiteboxError } from '../base/index.ts'
import { HALF, isNumeric, makeAdd, makeMul, makePow, numFromRational, ONE, ratOf } from './_canon.ts'
import { byKey, hasSymbol } from './_node.ts'
import {
  bgcd,
  blcm,
  R0,
  R1,
  type Rational,
  radd,
  rat,
  rcmp,
  rdiv,
  rfromNumber,
  rmul,
  rneg,
  rpow,
  rsub,
} from './_rational.ts'
import type { Expr, ExprSym } from './expr.ts'

/** One term: exponent vector and non-zero coefficient. @internal */
export interface Term {
  readonly e: readonly number[]
  readonly c: Rational
}

/** A polynomial in `n` variables: terms sorted in decreasing lex order. @internal */
export interface P {
  readonly n: number
  readonly terms: readonly Term[]
}

/** Lex comparison of exponent vectors. @internal */
export function cmpMono(a: readonly number[], b: readonly number[]): number {
  for (let i = 0; i < a.length; i++) {
    const d = (a[i] as number) - (b[i] as number)
    if (d !== 0) return d
  }
  return 0
}

function monoKey(e: readonly number[]): string {
  return e.join(',')
}

/** Build from unsorted terms (combining equal monomials). @internal */
export function fromTerms(n: number, terms: Iterable<Term>): P {
  const m = new Map<string, Term>()
  for (const t of terms) {
    const k = monoKey(t.e)
    const old = m.get(k)
    m.set(k, old === undefined ? t : { e: t.e, c: radd(old.c, t.c) })
  }
  const out = [...m.values()].filter((t) => t.c.n !== 0n)
  out.sort((a, b) => cmpMono(b.e, a.e))
  return { n, terms: out }
}

/** The zero polynomial. @internal */
export function pzero(n: number): P {
  return { n, terms: [] }
}

/** A constant. @internal */
export function pconst(n: number, c: Rational): P {
  return c.n === 0n ? pzero(n) : { n, terms: [{ e: new Array<number>(n).fill(0), c }] }
}

/** The variable x_i. @internal */
export function pvar(n: number, i: number): P {
  const e = new Array<number>(n).fill(0)
  e[i] = 1
  return { n, terms: [{ e, c: R1 }] }
}

/** Whether zero. @internal */
export function pIsZero(a: P): boolean {
  return a.terms.length === 0
}

/** Whether constant (including zero). @internal */
export function pIsConst(a: P): boolean {
  return a.terms.length === 0 || (a.terms.length === 1 && (a.terms[0] as Term).e.every((k) => k === 0))
}

/** Sum. @internal */
export function padd(a: P, b: P): P {
  return fromTerms(a.n, [...a.terms, ...b.terms])
}

/** Negation. @internal */
export function pneg(a: P): P {
  return { n: a.n, terms: a.terms.map((t) => ({ e: t.e, c: rneg(t.c) })) }
}

/** Difference. @internal */
export function psub(a: P, b: P): P {
  return padd(a, pneg(b))
}

/** Scale by a rational. @internal */
export function pscale(a: P, c: Rational): P {
  if (c.n === 0n) return pzero(a.n)
  return { n: a.n, terms: a.terms.map((t) => ({ e: t.e, c: rmul(t.c, c) })) }
}

/** Multiply by a monomial c * x^e. @internal */
export function pmulTerm(a: P, t: Term): P {
  if (t.c.n === 0n) return pzero(a.n)
  return { n: a.n, terms: a.terms.map((s) => ({ e: s.e.map((k, i) => k + (t.e[i] as number)), c: rmul(s.c, t.c) })) }
}

/** Product. @internal */
export function pmul(a: P, b: P): P {
  const out: Term[] = []
  for (const s of a.terms)
    for (const t of b.terms) out.push({ e: s.e.map((k, i) => k + (t.e[i] as number)), c: rmul(s.c, t.c) })
  return fromTerms(a.n, out)
}

/** Non-negative integer power. @internal */
export function ppow(a: P, k: number): P {
  let out = pconst(a.n, R1)
  let base = a
  let m = k
  while (m > 0) {
    if (m & 1) out = pmul(out, base)
    m >>= 1
    if (m > 0) base = pmul(base, base)
  }
  return out
}

/** Degree in variable i (-1 for zero). @internal */
export function pdeg(a: P, i: number): number {
  let d = -1
  for (const t of a.terms) d = Math.max(d, t.e[i] as number)
  return d
}

/** Total degree (-1 for zero). @internal */
export function ptotalDeg(a: P): number {
  let d = -1
  for (const t of a.terms)
    d = Math.max(
      d,
      t.e.reduce((s, k) => s + k, 0),
    )
  return d
}

/** Leading term in lex order. @internal */
export function plt(a: P): Term {
  const t = a.terms[0]
  if (t === undefined) throw new WhiteboxError('value', 'poly: leading term of zero')
  return t
}

/** Partial derivative in variable i. @internal */
export function pdiff(a: P, i: number): P {
  const out: Term[] = []
  for (const t of a.terms) {
    const k = t.e[i] as number
    if (k === 0) continue
    const e = [...t.e]
    e[i] = k - 1
    out.push({ e, c: rmul(t.c, rat(BigInt(k))) })
  }
  return fromTerms(a.n, out)
}

/** Whether monomial a divides monomial b. */
function monoDivides(a: readonly number[], b: readonly number[]): boolean {
  return a.every((k, i) => k <= (b[i] as number))
}

/** Multivariate division by lex leading terms: a = q*b + r with no term of r divisible by lt(b). @internal */
export function pdivmod(a: P, b: P): [P, P] {
  const lb = plt(b)
  let r = a
  let q = pzero(a.n)
  let rem = pzero(a.n)
  while (!pIsZero(r)) {
    const lr = plt(r)
    if (monoDivides(lb.e, lr.e)) {
      const t: Term = { e: lr.e.map((k, i) => k - (lb.e[i] as number)), c: rdiv(lr.c, lb.c) }
      q = padd(q, { n: a.n, terms: [t] })
      r = psub(r, pmulTerm(b, t))
    } else {
      rem = padd(rem, { n: a.n, terms: [lr] })
      r = { n: r.n, terms: r.terms.slice(1) }
    }
  }
  return [q, rem]
}

/** Exact quotient a / b, or undefined when b does not divide a. @internal */
export function pdivExact(a: P, b: P): P | undefined {
  const [q, r] = pdivmod(a, b)
  return pIsZero(r) ? q : undefined
}

/** Evaluate variable i at a rational value. @internal */
export function pevalVar(a: P, i: number, v: Rational): P {
  return fromTerms(
    a.n,
    a.terms.map((t) => {
      const e = [...t.e]
      const k = e[i] as number
      e[i] = 0
      return { e, c: rmul(t.c, rpow(v, k)) }
    }),
  )
}

/** Coefficients in variable i: map from degree to a polynomial free of x_i. @internal */
export function pcoeffsIn(a: P, i: number): Map<number, P> {
  const groups = new Map<number, Term[]>()
  for (const t of a.terms) {
    const k = t.e[i] as number
    const e = [...t.e]
    e[i] = 0
    const g = groups.get(k)
    if (g === undefined) groups.set(k, [{ e, c: t.c }])
    else g.push({ e, c: t.c })
  }
  const out = new Map<number, P>()
  for (const [k, ts] of groups) out.set(k, fromTerms(a.n, ts))
  return out
}

/** Leading coefficient in variable i (a polynomial free of x_i). @internal */
export function plcIn(a: P, i: number): P {
  return pcoeffsIn(a, i).get(pdeg(a, i)) ?? pzero(a.n)
}

/** x_i^k. */
function pvarPow(n: number, i: number, k: number): Term {
  const e = new Array<number>(n).fill(0)
  e[i] = k
  return { e, c: R1 }
}

/** Pseudo-remainder of a by b in variable i. @internal */
export function pprem(a: P, b: P, i: number): P {
  const db = pdeg(b, i)
  const lb = plcIn(b, i)
  let r = a
  let dr = pdeg(r, i)
  while (!pIsZero(r) && dr >= db) {
    const lr = plcIn(r, i)
    r = psub(pmul(lb, r), pmul(pmulTerm(lr, pvarPow(a.n, i, dr - db)), b))
    dr = pdeg(r, i)
  }
  return r
}

/** The rational content (positive) and the primitive integer polynomial with positive leading coefficient. @internal */
export function pnumericContent(a: P): [Rational, P] {
  if (pIsZero(a)) return [R1, a]
  let den = 1n
  for (const t of a.terms) den = blcm(den, t.c.d)
  let num = 0n
  for (const t of a.terms) num = bgcd(num, (t.c.n * den) / t.c.d)
  let c = rat(num, den)
  if (plt(a).c.n < 0n) c = rneg(c)
  return [c, pscale(a, rdiv(R1, c))]
}

/** Variables that occur. @internal */
export function pvars(a: P): number[] {
  const out: number[] = []
  for (let i = 0; i < a.n; i++) if (a.terms.some((t) => (t.e[i] as number) > 0)) out.push(i)
  return out
}

/** Content with respect to variable i: the gcd of the coefficients in x_i. @internal */
export function pcontentIn(a: P, i: number): P {
  let g = pzero(a.n)
  for (const c of pcoeffsIn(a, i).values()) g = pgcd(g, c)
  return g
}

/**
 * Greatest common divisor over the rationals, normalised to a primitive integer polynomial with positive leading
 * coefficient (as SymPy's `gcd`). Recursive on the variables with primitive pseudo-remainder sequences.
 * @internal
 */
export function pgcd(a: P, b: P): P {
  if (pIsZero(a)) return pIsZero(b) ? b : pnumericContent(b)[1]
  if (pIsZero(b)) return pnumericContent(a)[1]
  if (pIsConst(a) || pIsConst(b)) return pconst(a.n, R1)
  // A monomial's gcd with anything is the monomial of the smallest exponents.
  if (a.terms.length === 1 || b.terms.length === 1) {
    const e = [...((a.terms.length === 1 ? a : b).terms[0] as Term).e]
    for (const t of (a.terms.length === 1 ? b : a).terms)
      t.e.forEach((k, i) => {
        if (k < (e[i] as number)) e[i] = k
      })
    return { n: a.n, terms: [{ e, c: R1 }] }
  }
  const vs = [...new Set([...pvars(a), ...pvars(b)])].sort((x, y) => x - y)
  const v = vs[0] as number
  if (vs.length === 1) {
    // One variable: Euclid on dense coefficients over the rationals, then the primitive integer form.
    return pnumericContent(fromDense(ugcd(toDense(a, v), toDense(b, v)), a.n, v))[1]
  }
  if (pdeg(a, v) <= 0) return pgcd(a, pcontentIn(b, v))
  if (pdeg(b, v) <= 0) return pgcd(pcontentIn(a, v), b)
  const ca = pcontentIn(a, v)
  const cb = pcontentIn(b, v)
  let pa = pdivExact(a, ca) ?? a
  let pb = pdivExact(b, cb) ?? b
  const g = pgcd(ca, cb)
  if (pdeg(pa, v) < pdeg(pb, v)) [pa, pb] = [pb, pa]
  while (!pIsZero(pb) && pdeg(pb, v) > 0) {
    const r = pprem(pa, pb, v)
    pa = pb
    if (pIsZero(r)) {
      pb = r
      break
    }
    // Primitive part (polynomial and numeric content removed) keeps the coefficients from growing.
    const cr = pcontentIn(r, v)
    pb = pnumericContent(pdivExact(r, cr) ?? r)[1]
  }
  const prim = pIsZero(pb) ? pa : pconst(a.n, R1)
  const cp = pcontentIn(prim, v)
  const out = pmul(g, pdivExact(prim, cp) ?? prim)
  return pnumericContent(out)[1]
}

// ----------------------------------------------------------------------------------------- dense univariate

/** Dense univariate coefficients, constant term first. @internal */
export type U = Rational[]

/** Trim trailing zeros. @internal */
export function utrim(a: U): U {
  let n = a.length
  while (n > 0 && (a[n - 1] as Rational).n === 0n) n--
  return a.slice(0, n)
}

/** Degree (-1 for zero). @internal */
export function udeg(a: U): number {
  return utrim(a).length - 1
}

/** Univariate polynomial from P in variable i (others must be absent). @internal */
export function toDense(a: P, i = 0): U {
  const out: U = new Array<Rational>(Math.max(0, pdeg(a, i) + 1)).fill(R0)
  for (const t of a.terms) out[t.e[i] as number] = radd(out[t.e[i] as number] as Rational, t.c)
  return out
}

/** P in n variables from dense coefficients in variable i. @internal */
export function fromDense(a: U, n = 1, i = 0): P {
  return fromTerms(
    n,
    a.map((c, k) => {
      const e = new Array<number>(n).fill(0)
      e[i] = k
      return { e, c }
    }),
  )
}

/** Sum. @internal */
export function uadd(a: U, b: U): U {
  const n = Math.max(a.length, b.length)
  const out: U = []
  for (let k = 0; k < n; k++) out.push(radd(a[k] ?? R0, b[k] ?? R0))
  return utrim(out)
}

/** Difference. @internal */
export function usub(a: U, b: U): U {
  return uadd(
    a,
    b.map((c) => rneg(c)),
  )
}

/** Product. @internal */
export function umul(a: U, b: U): U {
  if (a.length === 0 || b.length === 0) return []
  const out: U = new Array<Rational>(a.length + b.length - 1).fill(R0)
  a.forEach((x, i) => {
    if (x.n === 0n) return
    b.forEach((y, j) => {
      out[i + j] = radd(out[i + j] as Rational, rmul(x, y))
    })
  })
  return utrim(out)
}

/** Scale. @internal */
export function uscale(a: U, c: Rational): U {
  return utrim(a.map((x) => rmul(x, c)))
}

/** Quotient and remainder. @internal */
export function udivmod(a: U, b: U): [U, U] {
  const bt = utrim(b)
  const db = bt.length - 1
  if (db < 0) throw new WhiteboxError('domain', 'poly: division by zero polynomial')
  let r = utrim(a)
  const lb = bt[db] as Rational
  const q: U = new Array<Rational>(Math.max(0, r.length - db)).fill(R0)
  while (r.length - 1 >= db && r.length > 0) {
    const k = r.length - 1 - db
    const c = rdiv(r[r.length - 1] as Rational, lb)
    q[k] = c
    const sub: U = new Array<Rational>(k).fill(R0).concat(bt.map((x) => rmul(x, c)))
    r = usub(r, sub)
  }
  return [utrim(q), r]
}

/** Derivative. @internal */
export function uderiv(a: U): U {
  return utrim(a.slice(1).map((c, k) => rmul(c, rat(BigInt(k + 1)))))
}

/** Monic version. @internal */
export function umonic(a: U): U {
  const t = utrim(a)
  const lc = t[t.length - 1]
  return lc === undefined ? t : uscale(t, rdiv(R1, lc))
}

/** Monic gcd. @internal */
export function ugcd(a: U, b: U): U {
  // Monic remainders keep the rational coefficients small.
  let x = umonic(a)
  let y = umonic(b)
  while (y.length > 0) {
    const r = udivmod(x, y)[1]
    x = y
    y = umonic(r)
  }
  return umonic(x)
}

/** Evaluate at a rational. @internal */
export function ueval(a: U, x: Rational): Rational {
  let s = R0
  for (let k = a.length - 1; k >= 0; k--) s = radd(rmul(s, x), a[k] as Rational)
  return s
}

/** Square-free decomposition (Yun): factors f_i with f = lc * prod f_i^i, each monic. @internal */
export function usqf(f: U): [U, number][] {
  const a = umonic(f)
  if (udeg(a) <= 0) return []
  const out: [U, number][] = []
  const d = uderiv(a)
  const a0 = ugcd(a, d)
  let b = udivmod(a, a0)[0]
  let c = udivmod(d, a0)[0]
  let dd = usub(c, uderiv(b))
  let i = 1
  while (udeg(b) > 0) {
    const ai = ugcd(b, dd)
    b = udivmod(b, ai)[0]
    c = udivmod(dd, ai)[0]
    dd = usub(c, uderiv(b))
    if (udeg(ai) > 0) out.push([ai, i])
    i++
  }
  return out
}

// ---------------------------------------------------------------------------------- expressions and polys

/** Generators of an expression viewed as a polynomial (symbols, functions, radicals), sorted by key. @internal */
export function findGens(e: Expr): Expr[] {
  const gens = new Map<string, Expr>()
  const walk = (x: Expr): void => {
    if (isNumeric(x)) return
    if (x.kind === 'add' || x.kind === 'mul') {
      for (const a of x.args) walk(a)
      return
    }
    if (x.kind === 'pow') {
      if (x.exp.kind === 'num' && x.exp.d === 1n && x.exp.n > 0n) {
        walk(x.base)
        return
      }
      if (x.exp.kind === 'num' && x.exp.n > 0n) {
        const g = makePow(x.base, numFromRational(rat(1n, x.exp.d)))
        gens.set(g.key, g)
        return
      }
      gens.set(x.key, x)
      return
    }
    gens.set(x.key, x)
  }
  walk(e)
  return [...gens.values()].sort(byKey)
}

/** Convert an expression to a polynomial over the given generators; undefined if it is not one. @internal */
export function exprToP(e: Expr, gens: readonly Expr[]): P | undefined {
  const n = gens.length
  const index = new Map(gens.map((g, i) => [g.key, i]))
  const go = (x: Expr): P | undefined => {
    const gi = index.get(x.key)
    if (gi !== undefined) return pvar(n, gi)
    if (x.kind === 'num') return pconst(n, ratOf(x))
    if (x.kind === 'float') return pconst(n, rfromNumber(x.value))
    if (x.kind === 'add') {
      let s = pzero(n)
      for (const a of x.args) {
        const p = go(a)
        if (p === undefined) return undefined
        s = padd(s, p)
      }
      return s
    }
    if (x.kind === 'mul') {
      let s = pconst(n, R1)
      for (const a of x.args) {
        const p = go(a)
        if (p === undefined) return undefined
        s = pmul(s, p)
      }
      return s
    }
    if (x.kind === 'pow' && x.exp.kind === 'num' && x.exp.n > 0n) {
      if (x.exp.d === 1n) {
        const b = go(x.base)
        return b === undefined ? undefined : ppow(b, Number(x.exp.n))
      }
      const g = makePow(x.base, numFromRational(rat(1n, x.exp.d)))
      const j = index.get(g.key)
      if (j !== undefined) return ppow(pvar(n, j), Number(x.exp.n))
    }
    // Anything free of every generator would need a coefficient field beyond the rationals.
    return undefined
  }
  return go(e)
}

/** Expression from a polynomial over generators. @internal */
export function pToExpr(a: P, gens: readonly Expr[]): Expr {
  const terms = a.terms.map((t) =>
    makeMul([
      numFromRational(t.c),
      ...t.e.map((k, i) => (k === 0 ? ONE : makePow(gens[i] as Expr, numFromRational(rat(BigInt(k)))))),
    ]),
  )
  return makeAdd(terms)
}

/** Dense univariate expression builder (for roots and series). @internal */
export function uToExpr(a: U, x: Expr): Expr {
  return makeAdd(a.map((c, k) => makeMul([numFromRational(c), makePow(x, numFromRational(rat(BigInt(k))))])))
}

/** Whether every generator is free of the symbol. @internal */
export function gensFree(gens: readonly Expr[], s: ExprSym): boolean {
  return gens.every((g) => !hasSymbol(g, s))
}

/** Rational comparisons re-exported for neighbours. @internal */
export { HALF, rcmp, rsub }
