/**
 * Polynomials over the rationals in one or more generators: conversion to and from expressions, degrees, coefficients,
 * division, greatest common divisors and least common multiples, exact roots in radicals, real-root counting and
 * isolation by Sturm sequences, resultants and discriminants, following SymPy's `Poly`.
 */

import { WhiteboxError } from '../base/index.ts'
import { bounded } from './_budget.ts'
import { HALF, I, MINUS_ONE, makeAdd, makeMul, makePow, numFromRational, ONE, PI, ratOf, ZERO } from './_canon.ts'
import { factorUnivariate } from './_factor.ts'
import { makeFn } from './_functions.ts'
import { contains, hasSymbol } from './_node.ts'
import { evalComplexConstant } from './_numeric.ts'
import {
  exprToP,
  findGens,
  fromTerms,
  type P,
  pdivExact,
  pdivmod,
  pgcd,
  pmul,
  pscale,
  pToExpr,
  type U,
  udeg,
  uderiv,
  udivmod,
  ueval,
  ugcd,
  uscale,
  utrim,
} from './_poly.ts'
import { bgcd, blcm, R1, type Rational, rabs, radd, rat, rcmp, rdiv, rmul, rneg, rsub } from './_rational.ts'
import type { Expr, ExprNum } from './expr.ts'
import { det } from './matrices.ts'

/** One term of a polynomial: exponents of the generators and an exact rational coefficient. */
export interface PolyTerm {
  readonly monom: readonly number[]
  readonly coeff: ExprNum
}

/**
 * A polynomial over the rationals in one or more generators (symbols, or other expressions such as `sin(x)` or
 * `sqrt(2)` treated as variables), with terms in decreasing lexicographic order. Parameters that appear in the
 * coefficients become further generators after the requested ones.
 */
export interface Poly {
  readonly kind: 'poly'
  readonly gens: readonly Expr[]
  readonly terms: readonly PolyTerm[]
}

/** A root with its multiplicity. */
export interface Root {
  readonly value: Expr
  readonly multiplicity: number
}

function fromP(p: P, gens: readonly Expr[]): Poly {
  return { kind: 'poly', gens, terms: p.terms.map((t) => ({ monom: t.e, coeff: numFromRational(t.c) })) }
}

function toP(p: Poly): P {
  return fromTerms(
    p.gens.length,
    p.terms.map((t) => ({ e: t.monom, c: ratOf(t.coeff) })),
  )
}

/**
 * Convert an expression to a polynomial. With no generators given, every symbol, function application and radical
 * is a generator; with generators given, anything else must be free of them and becomes an extra generator.
 *
 * @param e - The expression (sums, products and non-negative integer powers of generators and rationals).
 * @param gens - The generators, in order.
 * @returns The polynomial.
 * @throws WhiteboxError with code `value` when `e` is not a polynomial in the generators.
 * @example
 * ```ts
 * import { parse, poly } from 'whitebox-engine/symbolic'
 * poly(parse('(x + 1)^2'), parse('x')).terms.map((t) => t.monom[0]) // => [2, 1, 0]
 * ```
 */
export function poly(e: Expr, ...gens: readonly Expr[]): Poly {
  return bounded('poly', () => {
    const all = [...gens]
    for (const g of findGens(e)) {
      if (all.some((h) => h.key === g.key)) continue
      if (gens.some((h) => contains(g, h) || (h.kind === 'sym' && hasSymbol(g, h)))) {
        throw new WhiteboxError('value', `poly: not a polynomial in the given generators (contains ${g.key})`)
      }
      all.push(g)
    }
    const p = exprToP(e, all)
    if (p === undefined) throw new WhiteboxError('value', 'poly: not a polynomial (negative or fractional powers)')
    return fromP(p, all)
  })
}

/**
 * The expression of a polynomial.
 *
 * @param p - The polynomial.
 * @returns Its expanded expression.
 * @example
 * ```ts
 * import { parse, poly, polyToExpr, toText } from 'whitebox-engine/symbolic'
 * toText(polyToExpr(poly(parse('(x + 1)*(x - 1)')))) // => 'x**2 - 1'
 * ```
 */
export function polyToExpr(p: Poly): Expr {
  return pToExpr(toP(p), p.gens)
}

function asPoly(f: Expr | Poly, gen?: Expr): Poly {
  if ((f as Poly).kind === 'poly') return f as Poly
  return gen === undefined ? poly(f as Expr) : poly(f as Expr, gen)
}

function genIndex(p: Poly, gen?: Expr): number {
  if (gen === undefined) return 0
  const i = p.gens.findIndex((g) => g.key === gen.key)
  return i
}

/**
 * Degree in a generator (the first by default); -Infinity for the zero polynomial, as SymPy's -oo.
 *
 * @param f - A polynomial or expression.
 * @param gen - The generator.
 * @returns The degree.
 * @example
 * ```ts
 * import { degree, parse } from 'whitebox-engine/symbolic'
 * degree(parse('x^3*y + y^5'), parse('x')) // => 3
 * ```
 */
export function degree(f: Expr | Poly, gen?: Expr): number {
  const p = asPoly(f, gen)
  if (p.terms.length === 0) return Number.NEGATIVE_INFINITY
  const i = genIndex(p, gen)
  if (i < 0) return 0
  let d = 0
  for (const t of p.terms) d = Math.max(d, t.monom[i] as number)
  return d
}

/**
 * Coefficients in one generator, highest degree first (SymPy's `all_coeffs`), as expressions in the other
 * generators.
 *
 * @param f - A polynomial or expression.
 * @param gen - The generator (the first by default).
 * @returns Coefficients from degree n down to 0.
 * @example
 * ```ts
 * import { coefficients, parse, toText } from 'whitebox-engine/symbolic'
 * coefficients(parse('a*x^2 + 3*x - 1'), parse('x')).map(toText) // => ['a', '3', '-1']
 * ```
 */
export function coefficients(f: Expr | Poly, gen?: Expr): Expr[] {
  const p = asPoly(f, gen)
  const i = Math.max(0, genIndex(p, gen))
  const n = degree(p, p.gens[i])
  if (!Number.isFinite(n)) return [ZERO]
  const parts: Expr[][] = Array.from({ length: n + 1 }, () => [])
  for (const t of p.terms) {
    const k = t.monom[i] as number
    const rest = makeMul([
      t.coeff,
      ...t.monom.map((m, j) =>
        j === i || m === 0 ? ONE : makePow(p.gens[j] as Expr, numFromRational(rat(BigInt(m)))),
      ),
    ])
    ;(parts[k] as Expr[]).push(rest)
  }
  return parts.map((ts) => makeAdd(ts)).reverse()
}

/**
 * Leading coefficient in a generator.
 *
 * @param f - A polynomial or expression.
 * @param gen - The generator (the first by default).
 * @returns The coefficient of the highest power.
 * @example
 * ```ts
 * import { leadingCoefficient, parse, toText } from 'whitebox-engine/symbolic'
 * toText(leadingCoefficient(parse('3*x^2 + x'), parse('x'))) // => '3'
 * ```
 */
export function leadingCoefficient(f: Expr | Poly, gen?: Expr): Expr {
  return coefficients(f, gen)[0] ?? ZERO
}

function common(f: Expr | Poly, g: Expr | Poly, gens: readonly Expr[]): [P, P, readonly Expr[]] {
  const fe = (f as Poly).kind === 'poly' ? polyToExpr(f as Poly) : (f as Expr)
  const ge = (g as Poly).kind === 'poly' ? polyToExpr(g as Poly) : (g as Expr)
  // The generators of both (not of their sum, which may cancel): f's, then any g adds.
  const all = poly(ge, ...poly(fe, ...gens).gens).gens
  const pf = poly(fe, ...all)
  const pg = poly(ge, ...all)
  return [toP(pf), toP(pg), all]
}

/**
 * Polynomial division f = q*g + r (long division for one generator, lexicographic division for several, as SymPy's
 * `div` over the rationals).
 *
 * @param f - Dividend.
 * @param g - Divisor (non-zero).
 * @param gens - Generators (inferred when omitted).
 * @returns The quotient and remainder as expressions.
 * @example
 * ```ts
 * import { parse, polyDiv, toText } from 'whitebox-engine/symbolic'
 * const { quotient, remainder } = polyDiv(parse('x^3 + 2x + 1'), parse('x - 1'))
 * [toText(quotient), toText(remainder)] // => ['x**2 + x + 3', '4']
 * ```
 */
export function polyDiv(f: Expr | Poly, g: Expr | Poly, ...gens: readonly Expr[]): { quotient: Expr; remainder: Expr } {
  return bounded('polyDiv', () => {
    const [pf, pg, all] = common(f, g, gens)
    if (pg.terms.length === 0) throw new WhiteboxError('domain', 'polyDiv: division by zero')
    const [q, r] = pdivmod(pf, pg)
    return { quotient: pToExpr(q, all), remainder: pToExpr(r, all) }
  })
}

/**
 * Greatest common divisor over the rationals (primitive with integer coefficients and positive leading coefficient,
 * as SymPy's `gcd`).
 *
 * @param f - First polynomial.
 * @param g - Second polynomial.
 * @param gens - Generators (inferred when omitted).
 * @returns The gcd.
 * @example
 * ```ts
 * import { parse, polyGcd, toText } from 'whitebox-engine/symbolic'
 * toText(polyGcd(parse('x^2 - 1'), parse('x^2 - 3x + 2'))) // => 'x - 1'
 * ```
 */
export function polyGcd(f: Expr | Poly, g: Expr | Poly, ...gens: readonly Expr[]): Expr {
  return bounded('polyGcd', () => {
    const [pf, pg, all] = common(f, g, gens)
    const prim = pgcd(pf, pg)
    // SymPy: over the integers the contents' gcd is kept; over the rationals the gcd is monic.
    const scale = integral(pf) && integral(pg) ? rat(bgcd(content(pf), content(pg))) : rdiv(R1, leadCoeff(prim))
    return pToExpr(pscale(prim, scale), all)
  })
}

function integral(p: P): boolean {
  return p.terms.every((t) => t.c.d === 1n)
}

function content(p: P): bigint {
  let g = 0n
  for (const t of p.terms) g = bgcd(g, t.c.n)
  return g
}

function leadCoeff(p: P): Rational {
  return p.terms[0]?.c ?? R1
}

/**
 * Least common multiple over the rationals (monic in SymPy's normalisation: primitive, positive leading coefficient).
 *
 * @param f - First polynomial.
 * @param g - Second polynomial.
 * @param gens - Generators (inferred when omitted).
 * @returns The lcm.
 * @example
 * ```ts
 * import { parse, polyLcm, toText } from 'whitebox-engine/symbolic'
 * toText(polyLcm(parse('x^2 - 1'), parse('x^2 - 3x + 2'))) // => 'x**3 - 2*x**2 - x + 2'
 * ```
 */
export function polyLcm(f: Expr | Poly, g: Expr | Poly, ...gens: readonly Expr[]): Expr {
  return bounded('polyLcm', () => {
    const [pf, pg, all] = common(f, g, gens)
    if (pf.terms.length === 0 || pg.terms.length === 0) return ZERO
    const prod = pmul(pf, pg)
    const q = pdivExact(prod, pgcd(pf, pg)) ?? prod
    const prim = pgcd(q, q)
    // As polyGcd: the lcm of the contents over the integers, monic over the rationals.
    const scale = integral(pf) && integral(pg) ? rat(blcm(content(pf), content(pg))) : rdiv(R1, leadCoeff(prim))
    return pToExpr(pscale(prim, scale), all)
  })
}

// ------------------------------------------------------------------------------------------------- roots

function q(n: number, d = 1): Expr {
  return numFromRational(rat(BigInt(n), BigInt(d)))
}
const add = (...xs: Expr[]): Expr => makeAdd(xs)
const mul = (...xs: Expr[]): Expr => makeMul(xs)
const div = (a: Expr, b: Expr): Expr => makeMul([a, makePow(b, MINUS_ONE)])
const sqrt = (a: Expr): Expr => makePow(a, HALF)
const cbrt = (a: Expr): Expr => makePow(a, q(1, 3))
const neg = (a: Expr): Expr => makeMul([MINUS_ONE, a])

function numeric(e: Expr): [number, number] | undefined {
  const z = evalComplexConstant(e)
  return z === undefined ? undefined : [z[0], z[1]]
}

/** Real cube root of a real constant (the principal power would be complex for negatives). */
function realCbrt(a: Expr): Expr {
  const v = numeric(a)
  if (v !== undefined && Math.abs(v[1]) < 1e-14 && v[0] < 0) return neg(cbrt(neg(a)))
  return cbrt(a)
}

function quadraticRoots(a: Expr, b: Expr, c: Expr): Expr[] {
  const d = sqrt(add(mul(b, b), mul(q(-4), a, c)))
  const twoA = mul(q(2), a)
  return [div(add(neg(b), neg(d)), twoA), div(add(neg(b), d), twoA)]
}

function cubicRoots(a: Expr, b: Expr, c: Expr, d: Expr): Expr[] {
  // Normalise to x^3 + A x^2 + B x + C and depress with x = t - A/3.
  const A = div(b, a)
  const B = div(c, a)
  const C = div(d, a)
  const p = add(B, mul(q(-1, 3), A, A))
  const qq = add(mul(q(2, 27), makePow(A, q(3))), mul(q(-1, 3), A, B), C)
  const shift = mul(q(-1, 3), A)
  const pv = numeric(p)
  const qv = numeric(qq)
  if (pv !== undefined && qv !== undefined && pv[1] === 0 && qv[1] === 0) {
    const disc = -(4 * pv[0] ** 3 + 27 * qv[0] ** 2)
    if (pv[0] === 0) return [0, 1, 2].map((k) => add(shift, mul(realCbrt(neg(qq)), unity(k, 3))))
    if (disc > 0) {
      // Three real roots: the trigonometric form keeps every root real.
      const r = mul(q(2), sqrt(mul(q(-1, 3), p)))
      const theta = mul(q(1, 3), makeFn('acos', [mul(div(mul(q(3), qq), mul(q(2), p)), sqrt(div(q(-3), p)))]))
      return [0, 1, 2].map((k) => add(shift, mul(r, makeFn('cos', [add(theta, mul(q(-2 * k, 3), PI))]))))
    }
    const s = sqrt(add(mul(q(1, 4), qq, qq), mul(q(1, 27), makePow(p, q(3)))))
    const u = realCbrt(add(mul(q(-1, 2), qq), s))
    const v = div(neg(p), mul(q(3), u))
    const t0 = add(u, v)
    const im = mul(div(sqrt(q(3)), q(2)), add(u, neg(v)), I)
    const re = mul(q(-1, 2), t0)
    return [add(shift, t0), add(shift, re, neg(im)), add(shift, re, im)]
  }
  // Symbolic coefficients: Cardano with principal roots, t_k = w^k u - p / (3 w^k u).
  const s = sqrt(add(mul(q(1, 4), qq, qq), mul(q(1, 27), makePow(p, q(3)))))
  const u = cbrt(add(mul(q(-1, 2), qq), s))
  return [0, 1, 2].map((k) => {
    const wu = mul(unity(k, 3), u)
    return add(shift, wu, div(neg(p), mul(q(3), wu)))
  })
}

/** exp(2 pi i k / n) with exact trigonometric values where they exist. */
function unity(k: number, n: number): Expr {
  const angle = mul(q(2 * k, n), PI)
  return add(makeFn('cos', [angle]), mul(I, makeFn('sin', [angle])))
}

function quarticRoots(a: Expr, b: Expr, c: Expr, d: Expr, e: Expr): Expr[] {
  const A = div(b, a)
  const B = div(c, a)
  const C = div(d, a)
  const D = div(e, a)
  const p = add(B, mul(q(-3, 8), A, A))
  const qq = add(C, mul(q(-1, 2), A, B), mul(q(1, 8), makePow(A, q(3))))
  const r = add(D, mul(q(-1, 4), A, C), mul(q(1, 16), A, A, B), mul(q(-3, 256), makePow(A, q(4))))
  const shift = mul(q(-1, 4), A)
  const qv = numeric(qq)
  if (qq.kind === 'num' && qq.n === 0n) {
    const zs = quadraticRoots(ONE, p, r)
    return zs.flatMap((z) => [add(shift, sqrt(z)), add(shift, neg(sqrt(z)))])
  }
  // Resolvent cubic 8m^3 + 8p m^2 + (2p^2 - 8r) m - q^2 = 0; take a root with m != 0 (a positive real one if numeric).
  const ms = rootsOfDense([mul(q(-1), qq, qq), add(mul(q(2), p, p), mul(q(-8), r)), mul(q(8), p), q(8)])
  let m = ms[0] ?? ZERO
  if (qv !== undefined) {
    let best = Number.NEGATIVE_INFINITY
    for (const cand of ms) {
      const v = numeric(cand)
      if (v !== undefined && Math.abs(v[1]) < 1e-9 && v[0] > best) {
        best = v[0]
        m = cand
      }
    }
  }
  const S = sqrt(mul(q(2), m))
  const out: Expr[] = []
  for (const s1 of [1, -1]) {
    const inner = sqrt(neg(add(mul(q(2), p), mul(q(2), m), mul(q(2 * s1), div(qq, S)))))
    for (const s2 of [1, -1]) out.push(add(shift, mul(q(1, 2), add(mul(q(s1), S), mul(q(s2), inner)))))
  }
  return out
}

/** Roots of a dense polynomial with expression coefficients (constant first), degree at most four. */
function rootsOfDense(cs: readonly Expr[]): Expr[] {
  const n = cs.length - 1
  const c = (k: number): Expr => cs[k] ?? ZERO
  if (n === 1) return [div(neg(c(0)), c(1))]
  if (n === 2) return quadraticRoots(c(2), c(1), c(0))
  if (n === 3) return cubicRoots(c(3), c(2), c(1), c(0))
  if (n === 4) return quarticRoots(c(4), c(3), c(2), c(1), c(0))
  return []
}

function binomialRoots(lead: Rational, constant: Rational, n: number): Expr[] {
  // a x^n + b = 0: x = |c|^(1/n) * exp(i (theta + 2 pi k) / n) with c = -b/a.
  const c = rneg(rdiv(constant, lead))
  const mag = makePow(numFromRational(rabs(c)), q(1, n))
  const offset = c.n < 0n ? 1 : 0
  const out: Expr[] = []
  for (let k = 0; k < n; k++) {
    const angle = mul(q(2 * k + offset, n), PI)
    out.push(mul(mag, add(makeFn('cos', [angle]), mul(I, makeFn('sin', [angle])))))
  }
  return out
}

function rootsOfRational(f: U): Expr[] {
  const t = utrim(f)
  const n = t.length - 1
  const nonzero = t.map((c, k) => [c, k] as const).filter(([c]) => c.n !== 0n)
  if (n >= 2 && nonzero.length === 2 && (nonzero[0] as readonly [Rational, number])[1] === 0) {
    return binomialRoots(t[n] as Rational, t[0] as Rational, n)
  }
  return rootsOfDense(t.map((c) => numFromRational(c)))
}

function sortRoots(rs: Root[]): Root[] {
  const keyed = rs.map((r) => ({ r, v: numeric(r.value) ?? [0, 0] }))
  keyed.sort((a, b) => {
    const ra = Math.abs(a.v[1]) < 1e-12 ? 0 : 1
    const rb = Math.abs(b.v[1]) < 1e-12 ? 0 : 1
    if (ra !== rb) return ra - rb
    // Conjugates computed from different radicals may differ in the last bits of the real part.
    if (Math.abs(a.v[0] - b.v[0]) > 1e-9 * Math.max(1, Math.abs(a.v[0]))) return a.v[0] - b.v[0]
    return a.v[1] - b.v[1]
  })
  return keyed.map((k) => k.r)
}

/**
 * Roots in radicals of a univariate polynomial. Rational polynomials are first factored over the rationals, so any
 * degree works when the irreducible factors have degree at most four or are binomials `a x^n + b`; roots of other
 * irreducible factors are omitted (as SymPy's `roots`), so the multiplicities may sum to less than the degree.
 * Symbolic coefficients use the quadratic, Cardano and Ferrari formulas directly. Real roots come first, in
 * increasing order. Three real roots of an irreducible cubic are given in trigonometric form (real expressions);
 * SymPy uses Cardano's form with complex radicals, which has the same values.
 *
 * @param f - A polynomial or expression.
 * @param x - The variable (the only generator by default).
 * @returns The roots with multiplicities.
 * @example
 * ```ts
 * import { exactRoots, parse, toText } from 'whitebox-engine/symbolic'
 * exactRoots(parse('x^3 - x^2 - 2x + 2'), parse('x')).map((r) => toText(r.value)) // => ['-sqrt(2)', '1', 'sqrt(2)']
 * ```
 */
export function exactRoots(f: Expr | Poly, x?: Expr): Root[] {
  return bounded('exactRoots', () => {
    const p = asPoly(f, x)
    if (p.gens.length === 0) return []
    const i = Math.max(0, genIndex(p, x))
    const cs = coefficients(p, p.gens[i]).reverse()
    const out: Root[] = []
    if (cs.every((c) => c.kind === 'num')) {
      const dense = cs.map((c) => ratOf(c as ExprNum))
      const [, factors] = factorUnivariate(dense)
      for (const [g, k] of factors) for (const r of rootsOfRational(g)) out.push({ value: r, multiplicity: k })
      return sortRoots(out)
    }
    // Symbolic coefficients: strip zero roots, then use the formulas.
    let lo = 0
    while (lo < cs.length - 1 && (cs[lo] as Expr).kind === 'num' && (cs[lo] as ExprNum).n === 0n) lo++
    if (lo > 0) out.push({ value: ZERO, multiplicity: lo })
    for (const r of rootsOfDense(cs.slice(lo))) out.push({ value: r, multiplicity: 1 })
    return out
  })
}

// ------------------------------------------------------------------------------------------------- Sturm

function univariate(f: Expr | Poly, x?: Expr): U {
  const p = asPoly(f, x)
  const cs = coefficients(p, p.gens[Math.max(0, genIndex(p, x))]).reverse()
  if (!cs.every((c) => c.kind === 'num')) throw new WhiteboxError('value', 'sturm: needs rational coefficients')
  return utrim(cs.map((c) => ratOf(c as ExprNum)))
}

function sturmDense(f: U): U[] {
  const seq: U[] = [f, uderiv(f)]
  for (;;) {
    const a = seq[seq.length - 2] as U
    const b = seq[seq.length - 1] as U
    if (udeg(b) <= 0) break
    const r = udivmod(a, b)[1]
    if (r.length === 0) break
    seq.push(uscale(r, rat(-1n)))
  }
  return seq.filter((s) => s.length > 0)
}

/**
 * The Sturm sequence p0 = f, p1 = f', p_(k+1) = -rem(p_(k-1), p_k), as SymPy's `sturm`.
 *
 * @param f - A univariate polynomial with rational coefficients.
 * @param x - The variable.
 * @returns The sequence as expressions.
 * @example
 * ```ts
 * import { parse, sturm, toText } from 'whitebox-engine/symbolic'
 * sturm(parse('x^3 - 2x + 1'), parse('x')).map(toText) // => ['x**3 - 2*x + 1', '3*x**2 - 2', '4*x/3 - 1', '5/16']
 * ```
 */
export function sturm(f: Expr | Poly, x?: Expr): Expr[] {
  return bounded('sturm', () => {
    const p = asPoly(f, x)
    const v = p.gens[Math.max(0, genIndex(p, x))] ?? ZERO
    return sturmDense(univariate(p, x)).map((s) =>
      makeAdd(s.map((c, k) => makeMul([numFromRational(c), makePow(v, q(k))]))),
    )
  })
}

function signAt(f: U, x: Rational | 'inf' | '-inf'): number {
  if (x === 'inf' || x === '-inf') {
    const n = f.length - 1
    const lc = f[n] as Rational
    const s = lc.n > 0n ? 1 : -1
    return x === 'inf' || n % 2 === 0 ? s : -s
  }
  const v = ueval(f, x)
  return v.n > 0n ? 1 : v.n < 0n ? -1 : 0
}

function variations(seq: readonly U[], x: Rational | 'inf' | '-inf'): number {
  let count = 0
  let last = 0
  for (const s of seq) {
    const sg = signAt(s, x)
    if (sg === 0) continue
    if (last !== 0 && sg !== last) count++
    last = sg
  }
  return count
}

function toBound(e: Expr | number | undefined, fallback: 'inf' | '-inf'): Rational | 'inf' | '-inf' {
  if (e === undefined) return fallback
  if (typeof e === 'number') {
    if (e === Number.POSITIVE_INFINITY) return 'inf'
    if (e === Number.NEGATIVE_INFINITY) return '-inf'
    return rat(BigInt(Math.round(e * 1e9)), 1000000000n)
  }
  if (e.kind === 'num') return ratOf(e)
  if (e.kind === 'const' && e.name === 'oo') return 'inf'
  if (e.kind === 'const' && e.name === '-oo') return '-inf'
  throw new WhiteboxError('value', 'countRealRoots: bounds must be rational numbers or infinities')
}

/**
 * The number of distinct real roots in the closed interval [a, b], by Sturm's theorem (SymPy's `count_roots`).
 *
 * @param f - A univariate polynomial with rational coefficients.
 * @param x - The variable.
 * @param a - Lower bound (rational expression, number or -oo; default -oo).
 * @param b - Upper bound (default oo).
 * @returns The count.
 * @example
 * ```ts
 * import { countRealRoots, parse } from 'whitebox-engine/symbolic'
 * countRealRoots(parse('x^3 - 2x + 1'), parse('x')) // => 3
 * countRealRoots(parse('x^2 - 2'), parse('x'), 0, 1) // => 0
 * ```
 */
export function countRealRoots(f: Expr | Poly, x?: Expr, a?: Expr | number, b?: Expr | number): number {
  return bounded('countRealRoots', () => {
    const u = univariate(f, x)
    const sqfree = udivmod(u, ugcd(u, uderiv(u)))[0]
    const lo = toBound(a, '-inf')
    const hi = toBound(b, 'inf')
    const seq = sturmDense(sqfree)
    const atLo = lo !== 'inf' && lo !== '-inf' && ueval(sqfree, lo).n === 0n ? 1 : 0
    return variations(seq, lo) - variations(seq, hi) + atLo
  })
}

/**
 * Disjoint isolating intervals with rational endpoints for the distinct real roots, in increasing order (bisection
 * driven by Sturm counts; an exact rational root gives a degenerate interval).
 *
 * @param f - A univariate polynomial with rational coefficients.
 * @param x - The variable.
 * @returns Pairs [lo, hi] containing exactly one root each.
 * @example
 * ```ts
 * import { parse, realRootIntervals, toText } from 'whitebox-engine/symbolic'
 * realRootIntervals(parse('x^2 - 2'), parse('x')).length // => 2
 * ```
 */
export function realRootIntervals(f: Expr | Poly, x?: Expr): [Expr, Expr][] {
  return bounded('realRootIntervals', () => {
    const u = univariate(f, x)
    if (udeg(u) <= 0) return []
    const sq = udivmod(u, ugcd(u, uderiv(u)))[0]
    const seq = sturmDense(sq)
    // Cauchy bound: every root has |x| < 1 + max |a_i / a_n|.
    const lc = sq[sq.length - 1] as Rational
    let bound = R1
    for (const c of sq.slice(0, -1)) {
      const v = rabs(rdiv(c, lc))
      if (rcmp(v, bound) > 0) bound = v
    }
    bound = radd(bound, R1)
    // Sturm counts roots in (lo, hi] when neither end is a root; split points that are roots are moved aside.
    const count = (lo: Rational, hi: Rational): number => variations(seq, lo) - variations(seq, hi)
    const out: [Rational, Rational][] = []
    const stack: [Rational, Rational][] = [[rneg(bound), bound]]
    while (stack.length > 0) {
      const [lo, hi] = stack.pop() as [Rational, Rational]
      const k = count(lo, hi)
      if (k === 0) continue
      if (k === 1) {
        out.push([lo, hi])
        continue
      }
      let mid = rmul(radd(lo, hi), rat(1n, 2n))
      for (let j = 3n; ueval(sq, mid).n === 0n; j++) mid = radd(lo, rmul(rsub(hi, lo), rat(1n, j)))
      stack.push([mid, hi], [lo, mid])
    }
    out.sort((a, b) => rcmp(a[0], b[0]))
    return out.map(([lo, hi]) => [numFromRational(lo), numFromRational(hi)])
  })
}

// --------------------------------------------------------------------------------------------- resultants

function sylvester(a: readonly Expr[], b: readonly Expr[]): Expr[][] {
  // a, b: coefficients highest first.
  const m = a.length - 1
  const n = b.length - 1
  const size = m + n
  const rows: Expr[][] = []
  for (let i = 0; i < n; i++)
    rows.push(Array.from({ length: size }, (_, j) => (j >= i && j - i <= m ? (a[j - i] as Expr) : ZERO)))
  for (let i = 0; i < m; i++)
    rows.push(Array.from({ length: size }, (_, j) => (j >= i && j - i <= n ? (b[j - i] as Expr) : ZERO)))
  return rows
}

/**
 * The resultant of two polynomials in x: the determinant of their Sylvester matrix, a polynomial in the remaining
 * symbols that vanishes exactly when they share a root (SymPy's `resultant`).
 *
 * @param f - First polynomial.
 * @param g - Second polynomial.
 * @param x - The variable eliminated.
 * @returns The resultant, expanded.
 * @example
 * ```ts
 * import { parse, resultant, toText } from 'whitebox-engine/symbolic'
 * toText(resultant(parse('x^2 + y'), parse('x - y'), parse('x'))) // => 'y**2 + y'
 * ```
 */
export function resultant(f: Expr | Poly, g: Expr | Poly, x: Expr): Expr {
  return bounded('resultant', () => {
    const a = coefficients(asPoly(f, x), x)
    const b = coefficients(asPoly(g, x), x)
    if (a.length === 1 && b.length === 1) return ONE
    if (a.length === 1) return makePow(a[0] as Expr, q(b.length - 1))
    if (b.length === 1) return makePow(b[0] as Expr, q(a.length - 1))
    return det(sylvester(a, b))
  })
}

/**
 * The discriminant (-1)^(n(n-1)/2) res(f, f') / lc(f).
 *
 * @param f - A polynomial.
 * @param x - The variable.
 * @returns The discriminant, expanded.
 * @example
 * ```ts
 * import { discriminant, parse, toText } from 'whitebox-engine/symbolic'
 * toText(discriminant(parse('a*x^2 + b*x + c'), parse('x'))) // => '-4*a*c + b**2'
 * ```
 */
export function discriminant(f: Expr | Poly, x: Expr): Expr {
  return bounded('discriminant', () => {
    const p = asPoly(f, x)
    const cs = coefficients(p, x)
    const n = cs.length - 1
    if (n < 1) return ZERO
    const fe = polyToExpr(p)
    const res = resultant(fe, derivativeIn(cs, x), x)
    const signed = ((n * (n - 1)) / 2) % 2 === 0 ? res : makeMul([MINUS_ONE, res])
    // res(f, f') is divisible by lc(f) as a polynomial in the other symbols.
    const gens = poly(makeAdd([signed, cs[0] as Expr])).gens
    const quotient = pdivExact(toP(poly(signed, ...gens)), toP(poly(cs[0] as Expr, ...gens)))
    if (quotient === undefined) throw new WhiteboxError('value', 'discriminant: resultant not divisible by lc')
    return pToExpr(quotient, gens)
  })
}

function derivativeIn(cs: readonly Expr[], x: Expr): Expr {
  const n = cs.length - 1
  return makeAdd(cs.slice(0, -1).map((c, i) => makeMul([q(n - i), c, makePow(x, q(n - i - 1))])))
}
