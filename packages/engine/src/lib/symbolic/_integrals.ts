// Integration rules that need no search: the table of elementary antiderivatives (with linear arguments a*x + b),
// Gaussian integrals, quadratic radicals, rational functions by partial fractions (logarithms, arctangents and the
// reduction formula for powers of irreducible quadratics), and powers and products of sines and cosines.
import { isNegative, isPositive } from './_assume.ts'
import {
  asCoeffMul,
  HALF,
  isRat,
  MINUS_ONE,
  makeAdd,
  makeMul,
  makePow,
  numFromRational,
  ONE,
  PI,
  TWO,
  ZERO,
} from './_canon.ts'
import { expandExpr } from './_expand.ts'
import { makeFn } from './_functions.ts'
import { hasSymbol } from './_node.ts'
import { evalComplexConstant } from './_numeric.ts'
import { exprToP, findGens, pdeg } from './_poly.ts'
import { rat } from './_rational.ts'
import { apart, cancel, numerDenom } from './_rational-functions.ts'
import { powerReduce, productToSum } from './_trigsimp.ts'
import { diff } from './differentiate.ts'
import { type Expr, type ExprSym, substitute, sym } from './expr.ts'
import { coefficients } from './polynomials.ts'

const neg = (a: Expr): Expr => makeMul([MINUS_ONE, a])
const inv = (a: Expr): Expr => makePow(a, MINUS_ONE)
const q = (n: number, d = 1): Expr => numFromRational(rat(BigInt(n), BigInt(d)))
const fn = (name: string, a: Expr): Expr => makeFn(name, [a])

/** a and b with u = a*x + b (a free of x and non-zero), or undefined. @internal */
export function linear(u: Expr, x: ExprSym): [Expr, Expr] | undefined {
  const a = cancel(diff(u, x))
  if (hasSymbol(a, x) || isRat(a, 0n)) return undefined
  const b = cancel(makeAdd([u, neg(makeMul([a, x]))]))
  return hasSymbol(b, x) ? undefined : [a, b]
}

/** Sign of a constant when it can be decided. */
function signOf(e: Expr): number | undefined {
  if (isPositive(e) === true) return 1
  if (isNegative(e) === true) return -1
  const z = evalComplexConstant(e)
  if (z !== undefined && z[1] === 0 && z[0] !== 0) return Math.sign(z[0])
  return undefined
}

function fnTable(name: string, u: Expr): Expr | undefined {
  const s = (v: Expr): Expr => makePow(v, HALF)
  switch (name) {
    case 'exp':
      return fn('exp', u)
    case 'sin':
      return neg(fn('cos', u))
    case 'cos':
      return fn('sin', u)
    case 'tan':
      return neg(fn('log', fn('cos', u)))
    case 'cot':
      return fn('log', fn('sin', u))
    case 'sec':
      return fn('log', makeAdd([fn('tan', u), fn('sec', u)]))
    case 'csc':
      return neg(fn('log', makeAdd([fn('cot', u), fn('csc', u)])))
    case 'sinh':
      return fn('cosh', u)
    case 'cosh':
      return fn('sinh', u)
    case 'tanh':
      return fn('log', fn('cosh', u))
    case 'coth':
      return fn('log', fn('sinh', u))
    case 'log':
      return makeAdd([makeMul([u, fn('log', u)]), neg(u)])
    case 'asin':
      return makeAdd([makeMul([u, fn('asin', u)]), s(makeAdd([ONE, neg(makePow(u, TWO))]))])
    case 'acos':
      return makeAdd([makeMul([u, fn('acos', u)]), neg(s(makeAdd([ONE, neg(makePow(u, TWO))])))])
    case 'atan':
      return makeAdd([makeMul([u, fn('atan', u)]), makeMul([q(-1, 2), fn('log', makeAdd([makePow(u, TWO), ONE]))])])
    case 'acot':
      return makeAdd([makeMul([u, fn('acot', u)]), makeMul([q(1, 2), fn('log', makeAdd([makePow(u, TWO), ONE]))])])
    case 'asinh':
      return makeAdd([makeMul([u, fn('asinh', u)]), neg(s(makeAdd([makePow(u, TWO), ONE])))])
    case 'acosh':
      return makeAdd([makeMul([u, fn('acosh', u)]), neg(makeMul([s(makeAdd([u, MINUS_ONE])), s(makeAdd([u, ONE]))]))])
    case 'atanh':
      return makeAdd([
        makeMul([u, fn('atanh', u)]),
        makeMul([q(1, 2), fn('log', makeAdd([ONE, neg(makePow(u, TWO))]))]),
      ])
    case 'erf':
      return makeAdd([makeMul([u, fn('erf', u)]), makeMul([makePow(PI, q(-1, 2)), fn('exp', neg(makePow(u, TWO)))])])
    default:
      return undefined
  }
}

/** Quadratic radicand p x^2 + r (no linear term) as [p, r]. */
function quadratic(u: Expr, x: ExprSym): [Expr, Expr] | undefined {
  const gens = findGens(u)
  if (gens.some((g) => g.key !== x.key && hasSymbol(g, x))) return undefined
  try {
    const cs = coefficients(u, x)
    if (cs.length !== 3 || !isRat(cs[1] as Expr, 0n)) return undefined
    return [cs[0] as Expr, cs[2] as Expr]
  } catch {
    return undefined
  }
}

function radicalTable(base: Expr, n: Expr, x: ExprSym): Expr | undefined {
  const qr = quadratic(base, x)
  if (qr === undefined) return undefined
  const [p, r] = qr
  const sp = signOf(p)
  const sr = signOf(r)
  if (sp === undefined || sr === undefined) return undefined
  const root = (v: Expr): Expr => makePow(v, HALF)
  const k = sp > 0 ? root(cancel(makeMul([p, inv(makeMul([q(sr), r]))]))) : root(cancel(neg(makeMul([p, inv(r)]))))
  const arg = makeMul([k, x])
  if (isRat(n, -1n, 2n)) {
    if (sp < 0 && sr > 0) return makeMul([inv(root(neg(p))), fn('asin', arg)])
    if (sp > 0 && sr > 0) return makeMul([inv(root(p)), fn('asinh', arg)])
    if (sp > 0 && sr < 0) return makeMul([inv(root(p)), fn('acosh', arg)])
    return undefined
  }
  if (isRat(n, 1n, 2n)) {
    const lead = makeMul([q(1, 2), x, makePow(base, HALF)])
    if (sp < 0 && sr > 0) return makeAdd([lead, makeMul([q(1, 2), r, inv(root(neg(p))), fn('asin', arg)])])
    if (sp > 0 && sr > 0) return makeAdd([lead, makeMul([q(1, 2), r, inv(root(p)), fn('asinh', arg)])])
    if (sp > 0 && sr < 0) return makeAdd([lead, makeMul([q(1, 2), r, inv(root(p)), fn('acosh', arg)])])
  }
  return undefined
}

function gaussian(arg: Expr, x: ExprSym): Expr | undefined {
  // exp(p x^2 + b x + c) with p < 0: complete the square.
  try {
    const cs = coefficients(arg, x)
    if (cs.length !== 3) return undefined
    const [p, b, c] = cs as [Expr, Expr, Expr]
    if (signOf(p) !== -1) return undefined
    const m = neg(p)
    const shift = makeMul([b, inv(makeMul([TWO, p]))])
    const constant = makeAdd([c, neg(makeMul([b, b, inv(makeMul([q(4), p]))]))])
    return makeMul([
      fn('exp', constant),
      makePow(PI, HALF),
      inv(makeMul([TWO, makePow(m, HALF)])),
      fn('erf', makeMul([makePow(m, HALF), makeAdd([x, shift])])),
    ])
  } catch {
    return undefined
  }
}

/** Table lookup for one integrand; undefined when no entry applies. @internal */
export function table(e: Expr, x: ExprSym): Expr | undefined {
  if (e.key === x.key) return makeMul([HALF, makePow(x, TWO)])
  if (e.kind === 'pow') {
    const { base, exp } = e
    if (!hasSymbol(exp, x)) {
      const l = linear(base, x)
      if (l !== undefined) {
        const [a] = l
        if (isRat(exp, -1n)) return makeMul([inv(a), fn('log', base)])
        const n1 = makeAdd([exp, ONE])
        return makeMul([makePow(base, n1), inv(makeMul([a, n1]))])
      }
      if (base.kind === 'fn' && isRat(exp, 2n)) {
        const la = linear(base.args[0] as Expr, x)
        if (la !== undefined) {
          const u = base.args[0] as Expr
          const a = la[0]
          if (base.name === 'sec') return makeMul([inv(a), fn('tan', u)])
          if (base.name === 'csc') return makeMul([neg(inv(a)), fn('cot', u)])
        }
      }
      if (base.kind === 'fn' && isRat(exp, -2n)) {
        const la = linear(base.args[0] as Expr, x)
        if (la !== undefined) {
          const u = base.args[0] as Expr
          const a = la[0]
          if (base.name === 'cos') return makeMul([inv(a), fn('tan', u)])
          if (base.name === 'sin') return makeMul([neg(inv(a)), fn('cot', u)])
          if (base.name === 'cosh') return makeMul([inv(a), fn('tanh', u)])
          if (base.name === 'sinh') return makeMul([neg(inv(a)), fn('coth', u)])
        }
      }
      return radicalTable(base, exp, x)
    }
    if (!hasSymbol(base, x)) {
      const l = linear(exp, x)
      if (l !== undefined) return makeMul([e, inv(makeMul([l[0], fn('log', base)]))])
    }
    return undefined
  }
  if (e.kind === 'fn' && e.args.length === 1) {
    const u = e.args[0] as Expr
    if (e.name === 'exp') {
      const g = gaussian(u, x)
      if (g !== undefined) return g
    }
    const l = linear(u, x)
    if (l === undefined) return undefined
    const F = fnTable(e.name, u)
    return F === undefined ? undefined : makeMul([inv(l[0]), F])
  }
  if (e.kind === 'mul' && e.args.length === 2) {
    const [f, g] = e.args as [Expr, Expr]
    const pair = (a: Expr, b: Expr): Expr | undefined => {
      if (a.kind !== 'fn' || b.kind !== 'fn' || (a.args[0] as Expr).key !== (b.args[0] as Expr).key) return undefined
      const u = a.args[0] as Expr
      const l = linear(u, x)
      if (l === undefined) return undefined
      if (a.name === 'sec' && b.name === 'tan') return makeMul([inv(l[0]), fn('sec', u)])
      if (a.name === 'cot' && b.name === 'csc') return makeMul([neg(inv(l[0])), fn('csc', u)])
      return undefined
    }
    const p = pair(f, g) ?? pair(g, f)
    if (p !== undefined) return p
    return expTrig(f, g, x) ?? expTrig(g, f, x)
  }
  return undefined
}

/** exp(a x + b) sin(c x + d) and exp(a x + b) cos(c x + d) in closed form. */
function expTrig(f: Expr, g: Expr, x: ExprSym): Expr | undefined {
  if (f.kind !== 'fn' || f.name !== 'exp' || g.kind !== 'fn' || (g.name !== 'sin' && g.name !== 'cos')) return undefined
  const la = linear(f.args[0] as Expr, x)
  const lc = linear(g.args[0] as Expr, x)
  if (la === undefined || lc === undefined) return undefined
  const [a] = la
  const [c] = lc
  const u = g.args[0] as Expr
  const den = inv(makeAdd([makePow(a, TWO), makePow(c, TWO)]))
  const s = fn('sin', u)
  const co = fn('cos', u)
  const body =
    g.name === 'sin' ? makeAdd([makeMul([a, s]), neg(makeMul([c, co]))]) : makeAdd([makeMul([a, co]), makeMul([c, s])])
  return makeMul([f, body, den])
}

// ------------------------------------------------------------------------------------------- rational

/** ∫ dx / (a x^2 + b x + c)^k by arctangent or logarithm and the reduction formula. */
function quadraticPower(a: Expr, b: Expr, c: Expr, k: number, x: ExprSym): Expr {
  const f = makeAdd([makeMul([a, makePow(x, TWO)]), makeMul([b, x]), c])
  const delta = cancel(makeAdd([makeMul([q(4), a, c]), neg(makePow(b, TWO))]))
  const lin = makeAdd([makeMul([TWO, a, x]), b])
  if (k === 1) {
    if (signOf(delta) === -1) {
      const r = makePow(neg(delta), HALF)
      return makeMul([inv(r), fn('log', makeMul([makeAdd([lin, neg(r)]), inv(makeAdd([lin, r]))]))])
    }
    const r = makePow(delta, HALF)
    return makeMul([TWO, inv(r), fn('atan', makeMul([lin, inv(r)]))])
  }
  const first = makeMul([lin, inv(makeMul([q(k - 1), delta, makePow(f, q(k - 1))]))])
  const coeff = makeMul([q(2 * (2 * k - 3)), a, inv(makeMul([q(k - 1), delta]))])
  return makeAdd([first, makeMul([coeff, quadraticPower(a, b, c, k - 1, x)])])
}

/** One partial-fraction term N(x) f(x)^(-j), deg N < deg f <= 2. */
function fractionTerm(t: Expr, x: ExprSym): Expr | undefined {
  const [c, body] = asCoeffMul(t)
  const factors = body.kind === 'mul' ? body.args : [body]
  const den = factors.filter((f) => f.kind === 'pow' && f.exp.kind === 'num' && f.exp.n < 0n && hasSymbol(f.base, x))
  if (den.length !== 1) return undefined
  const d = den[0] as Expr & { kind: 'pow' }
  const numer = makeMul([c, ...factors.filter((f) => f !== d)])
  const j = Number(-(d.exp as Expr & { kind: 'num' }).n)
  const cs = coefficients(d.base, x)
  const nc = coefficients(numer, x)
  if (cs.length === 2) {
    // A linear factor: the numerator is a constant and the power table applies.
    const T = table(makePow(d.base, q(-j)), x)
    return T === undefined || hasSymbol(numer, x) ? undefined : makeMul([numer, T])
  }
  if (cs.length !== 3 || nc.length > 2) return undefined
  const [a, b, cc] = cs as [Expr, Expr, Expr]
  const B = nc.length === 2 ? (nc[0] as Expr) : ZERO
  const C = nc[nc.length - 1] as Expr
  const f = d.base
  // N = (B / 2a) f' + (C - B b / 2a)
  const k1 = makeMul([B, inv(makeMul([TWO, a]))])
  const k2 = makeAdd([C, neg(makeMul([B, b, inv(makeMul([TWO, a]))]))])
  const logPart = j === 1 ? makeMul([k1, fn('log', f)]) : makeMul([k1, makePow(f, q(1 - j)), inv(q(1 - j))])
  return makeAdd([logPart, makeMul([k2, quadraticPower(a, b, cc, j, x)])])
}

/** Whether e is a rational function of x (other symbols allowed in the coefficients). @internal */
export function isRationalIn(e: Expr, x: ExprSym): boolean {
  const [n, d] = numerDenom(e)
  for (const part of [n, d]) {
    const gens = findGens(part)
    if (gens.some((g) => g.key !== x.key && hasSymbol(g, x))) return false
    if (exprToP(part, gens) === undefined) return false
  }
  return true
}

/**
 * Rational functions by partial fractions: logarithms, arctangents and the reduction formula. Irreducible factors of
 * degree above two would need algebraic extensions (Rothstein-Trager); those terms make it fail.
 * @internal
 */
export function rationalIntegral(e: Expr, x: ExprSym): Expr | undefined {
  if (!isRationalIn(e, x)) return undefined
  const [, d] = numerDenom(e)
  if (!hasSymbol(d, x)) return undefined
  const parts = apart(e, x)
  const terms = parts.kind === 'add' ? parts.args : [parts]
  const out: Expr[] = []
  for (const t of terms) {
    if (!hasSymbol(t, x)) {
      out.push(makeMul([t, x]))
      continue
    }
    const [, td] = numerDenom(t)
    if (!hasSymbol(td, x)) {
      const poly = expandExpr(t)
      for (const m of poly.kind === 'add' ? poly.args : [poly]) {
        const [c, rest] = asCoeffMul(m)
        const r = !hasSymbol(m, x)
          ? makeMul([m, x])
          : (() => {
              const v = table(rest, x)
              return v === undefined ? undefined : makeMul([c, v])
            })()
        if (r === undefined) return undefined
        out.push(r)
      }
      continue
    }
    const deg = pdegIn(td, x)
    if (deg > 2) return undefined
    const r = fractionTerm(t, x)
    if (r === undefined) return undefined
    out.push(r)
  }
  return makeAdd(out)
}

function pdegIn(e: Expr, x: ExprSym): number {
  const gens = [x as Expr, ...findGens(e).filter((g) => g.key !== x.key)]
  const p = exprToP(e, gens)
  if (p === undefined) return Number.POSITIVE_INFINITY
  // The degree of the irreducible base: a power f^j has degree j * deg f; divide out the exponent.
  if (e.kind === 'pow' && e.exp.kind === 'num' && e.exp.d === 1n) return pdeg(p, 0) / Number(e.exp.n)
  return pdeg(p, 0)
}

// ---------------------------------------------------------------------------------------------- trig

/** c * sin(u)^m * cos(u)^n with u linear in x: [c, u, m, n]. */
function sinCosPowers(e: Expr, x: ExprSym): [Expr, Expr, number, number] | undefined {
  const [c, body] = asCoeffMul(e)
  let u: Expr | undefined
  let m = 0
  let n = 0
  const consts: Expr[] = [c]
  for (const f of body.kind === 'mul' ? body.args : [body]) {
    const [b, k] = f.kind === 'pow' ? [f.base, f.exp] : [f, ONE]
    if (!hasSymbol(f, x)) {
      consts.push(f)
      continue
    }
    if (b.kind !== 'fn' || (b.name !== 'sin' && b.name !== 'cos') || k.kind !== 'num' || k.d !== 1n) return undefined
    const arg = b.args[0] as Expr
    if (u !== undefined && u.key !== arg.key) return undefined
    u = arg
    if (b.name === 'sin') m += Number(k.n)
    else n += Number(k.n)
  }
  if (u === undefined || linear(u, x) === undefined) return undefined
  return [makeMul(consts), u, m, n]
}

/**
 * Powers and products of sines and cosines: odd powers by the substitution u = cos or u = sin, even powers by power
 * reduction, products of different arguments by product-to-sum; `integrateU` integrates the polynomial or rational
 * function of the new variable.
 * @internal
 */
export function trigIntegral(
  e: Expr,
  x: ExprSym,
  integrateU: (f: Expr, u: ExprSym) => Expr | undefined,
): Expr | undefined {
  const sc = sinCosPowers(e, x)
  if (sc !== undefined) {
    const [c, u, m, n] = sc
    const a = (linear(u, x) as [Expr, Expr])[0]
    const U = sym('u_trig__')
    if (m > 0 && m % 2 === 1) {
      // ∫ sin^m cos^n = -(1/a) ∫ (1 - U^2)^((m-1)/2) U^n dU, U = cos u.
      const integrand = makeMul([makePow(makeAdd([ONE, neg(makePow(U, TWO))]), q((m - 1) / 2)), makePow(U, q(n))])
      const F = integrateU(expandExpr(integrand), U)
      if (F !== undefined) return makeMul([c, neg(inv(a)), substitute(F, [[U, fn('cos', u)]])])
    }
    if (n > 0 && n % 2 === 1) {
      const integrand = makeMul([makePow(U, q(m)), makePow(makeAdd([ONE, neg(makePow(U, TWO))]), q((n - 1) / 2))])
      const F = integrateU(expandExpr(integrand), U)
      if (F !== undefined) return makeMul([c, inv(a), substitute(F, [[U, fn('sin', u)]])])
    }
    if (m >= 0 && n >= 0 && m % 2 === 0 && n % 2 === 0 && m + n > 0) {
      let r = e
      for (let i = 0; i < 6; i++) {
        const next = productToSum(powerReduce(r))
        if (next.key === r.key) break
        r = next
      }
      return r.key === e.key ? undefined : integrateU(r, x)
    }
  }
  // Products of sines and cosines of different arguments.
  if (e.kind === 'mul' && e.args.filter((f) => f.kind === 'fn' && (f.name === 'sin' || f.name === 'cos')).length >= 2) {
    const r = productToSum(e)
    if (r.key !== e.key) return integrateU(r, x)
  }
  return undefined
}

/** Reduction formulas for tan^n, cot^n, sec^n and csc^n of a linear argument (n >= 2). @internal */
export function reductionIntegral(e: Expr, x: ExprSym, again: (f: Expr) => Expr | undefined): Expr | undefined {
  if (e.kind !== 'pow' || e.base.kind !== 'fn' || e.exp.kind !== 'num' || e.exp.d !== 1n || e.exp.n < 2n)
    return undefined
  const name = e.base.name
  const u = e.base.args[0] as Expr
  const l = linear(u, x)
  if (l === undefined || !['tan', 'cot', 'sec', 'csc'].includes(name)) return undefined
  const n = Number(e.exp.n)
  const a = l[0]
  const f = (k: number): Expr => makePow(fn(name, u), q(k))
  if (name === 'tan' || name === 'cot') {
    const sign = name === 'tan' ? ONE : MINUS_ONE
    const rest = n === 2 ? makeMul([MINUS_ONE, x]) : again(f(n - 2))
    if (rest === undefined) return undefined
    return makeAdd([makeMul([sign, f(n - 1), inv(makeMul([q(n - 1), a]))]), n === 2 ? rest : neg(rest)])
  }
  if (n === 2) return undefined
  const other = name === 'sec' ? fn('tan', u) : neg(fn('cot', u))
  const rest = again(f(n - 2))
  if (rest === undefined) return undefined
  return makeAdd([makeMul([f(n - 2), other, inv(makeMul([q(n - 1), a]))]), makeMul([q(n - 2, n - 1), rest])])
}
