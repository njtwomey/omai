import { WhiteboxError } from '../base/index.ts'
import { bounded } from './_budget.ts'
import {
  asCoeffMul,
  isInfinite,
  MINUS_ONE,
  makeAdd,
  makeMul,
  makePow,
  mapChildren,
  NAN,
  numFromRational,
  ONE,
  PI,
  ZERO,
} from './_canon.ts'
import { makeFn } from './_functions.ts'
import { children, hasSymbol } from './_node.ts'
import { evalComplexConstant, freeNames } from './_numeric.ts'
import { exprToP, findGens } from './_poly.ts'
import { bgcd, blcm, rat } from './_rational.ts'
import { cancel, numerDenom, together } from './_rational-functions.ts'
import { type Derivation, type Expr, type ExprSym, type Step, substitute, sym } from './expr.ts'
import { exactRoots } from './polynomials.ts'
import { factor } from './simplify.ts'

const neg = (a: Expr): Expr => makeMul([MINUS_ONE, a])

function asZero(eq: Expr): Expr {
  if (eq.kind === 'rel') {
    if (eq.op !== '==') throw new WhiteboxError('value', 'solve: inequalities are not supported')
    return makeAdd([eq.lhs, neg(eq.rhs)])
  }
  if (eq.kind === 'const' && (eq.name === 'true' || eq.name === 'false')) {
    throw new WhiteboxError('value', 'solve: the equation is identically true or false')
  }
  return eq
}

function isZero(e: Expr): boolean {
  const c = cancel(e)
  if (c.kind === 'num') return c.n === 0n
  if (freeNames(c).length === 0) {
    const z = evalComplexConstant(c)
    return z !== undefined && Math.hypot(z[0], z[1]) < 1e-12
  }
  return false
}

function collectExps(e: Expr, x: ExprSym, out: Map<string, Expr>): void {
  if (e.kind === 'fn' && e.name === 'exp' && hasSymbol(e, x)) out.set(e.key, e.args[0] as Expr)
  for (const c of children(e)) collectExps(c, x, out)
}

/**
 * Write exp(k1 u), exp(k2 u), ... as powers U^(k/g) of one generator U = exp(g u) (g the gcd of the rational k),
 * so that exp(2x) - 3 exp(x) + 2 becomes U^2 - 3U + 2. Returns the generator and the rewritten expression.
 */
function expGenerator(e: Expr, x: ExprSym, U: ExprSym): [Expr, Expr] | undefined {
  const args = new Map<string, Expr>()
  collectExps(e, x, args)
  if (args.size === 0) return undefined
  let rest: Expr | undefined
  const cs: { n: bigint; d: bigint }[] = []
  for (const a of args.values()) {
    const [c, r] = asCoeffMul(a)
    if (c.kind !== 'num' || (rest !== undefined && rest.key !== r.key)) return undefined
    rest = r
    cs.push({ n: c.n, d: c.d })
  }
  if (rest === undefined) return undefined
  let gn = 0n
  let gd = 1n
  for (const c of cs) {
    gn = bgcd(gn, c.n)
    gd = blcm(gd, c.d)
  }
  const g = numFromRational(rat(gn, gd))
  const gen = makeFn('exp', [makeMul([g, rest])])
  const go = (y: Expr): Expr => {
    if (y.kind === 'fn' && y.name === 'exp' && hasSymbol(y, x)) {
      const [c] = asCoeffMul(y.args[0] as Expr)
      return makePow(U, makeMul([c, makePow(g, MINUS_ONE)]))
    }
    return mapChildren(y, go)
  }
  return [gen, go(e)]
}

/** The inverse images of value under a known function (principal solutions, as SymPy's solve). */
function invert(name: string, value: Expr): Expr[] | undefined {
  switch (name) {
    case 'exp':
      return [makeFn('log', [value])]
    case 'log':
      return [makeFn('exp', [value])]
    case 'sin': {
      const a = makeFn('asin', [value])
      return [a, makeAdd([PI, neg(a)])]
    }
    case 'cos': {
      const a = makeFn('acos', [value])
      return [a, makeAdd([makeMul([numFromRational(rat(2n)), PI]), neg(a)])]
    }
    case 'tan':
      return [makeFn('atan', [value])]
    case 'cot':
      return [makeFn('acot', [value])]
    case 'sinh':
      return [makeFn('asinh', [value])]
    case 'cosh': {
      const a = makeFn('acosh', [value])
      return [a, neg(a)]
    }
    case 'tanh':
      return [makeFn('atanh', [value])]
    case 'asin':
      return [makeFn('sin', [value])]
    case 'acos':
      return [makeFn('cos', [value])]
    case 'atan':
      return [makeFn('tan', [value])]
    case 'asinh':
      return [makeFn('sinh', [value])]
    case 'acosh':
      return [makeFn('cosh', [value])]
    case 'atanh':
      return [makeFn('tanh', [value])]
    default:
      return undefined
  }
}

/** k with b^k = v for rational constants b, v and a rational k of small denominator, or undefined. */
function exactLog(b: Expr, v: Expr): Expr | undefined {
  if (b.kind !== 'num' || v.kind !== 'num') return undefined
  const bv = Number(b.n) / Number(b.d)
  const vv = Number(v.n) / Number(v.d)
  if (!(bv > 0 && vv > 0) || bv === 1) return undefined
  const k = Math.log(vv) / Math.log(bv)
  for (let den = 1; den <= 12; den++) {
    const num = Math.round(k * den)
    if (Math.abs(k * den - num) > 1e-9) continue
    const cand = numFromRational(rat(BigInt(num), BigInt(den)))
    if (makePow(b, cand).key === v.key) return cand
  }
  return undefined
}

function containsInfinity(e: Expr): boolean {
  return isInfinite(e) || e.key === NAN.key || children(e).some(containsInfinity)
}

function dedupe(xs: readonly Expr[]): Expr[] {
  const seen = new Map<string, Expr>()
  for (const x of xs) if (!seen.has(x.key)) seen.set(x.key, x)
  return [...seen.values()]
}

function sortSolutions(xs: readonly Expr[]): Expr[] {
  const keyed = xs.map((e) => ({ e, z: evalComplexConstant(e) }))
  keyed.sort((a, b) => {
    if (a.z === undefined || b.z === undefined) return a.z === undefined ? (b.z === undefined ? 0 : 1) : -1
    const ra = Math.abs(a.z[1]) < 1e-12 ? 0 : 1
    const rb = Math.abs(b.z[1]) < 1e-12 ? 0 : 1
    if (ra !== rb) return ra - rb
    return a.z[0] - b.z[0] || a.z[1] - b.z[1]
  })
  return keyed.map((k) => k.e)
}

function solveZero(f: Expr, x: ExprSym, steps: Step[] | undefined, depth: number): Expr[] {
  if (depth > 6) throw new WhiteboxError('not-implemented', 'solve: no method applies')
  const t = together(f)
  const [n, d] = numerDenom(t)
  if (!hasSymbol(n, x)) {
    if (isZero(n)) throw new WhiteboxError('value', 'solve: the equation holds for every x')
    return []
  }
  const keep = (sols: Expr[]): Expr[] =>
    sols.filter((s) => {
      if (isInfinite(s) || s.key === NAN.key || containsInfinity(s)) return false
      const dv = substitute(d, [[x, s]])
      const ok = !isZero(dv) && !isInfinite(cancel(dv))
      const nv = substitute(n, [[x, s]])
      return ok && nv.key !== NAN.key
    })
  // Polynomial in x.
  const gens = findGens(n)
  if (gens.every((g) => g.key === x.key || !hasSymbol(g, x)) && exprToP(n, gens) !== undefined) {
    const roots = exactRoots(n, x).map((r) => r.value)
    if (steps !== undefined) steps.push({ rule: 'roots of a polynomial', before: n, after: makeFn('List', roots) })
    return keep(roots)
  }
  // A product: each factor in turn.
  const fac = factor(n)
  if (fac.kind === 'mul' && fac.args.filter((a) => hasSymbol(a, x)).length > 1) {
    if (steps !== undefined) steps.push({ rule: 'zero product', before: n, after: fac })
    const sols: Expr[] = []
    for (const a of fac.args) if (hasSymbol(a, x)) sols.push(...solveZero(a, x, steps, depth + 1))
    return keep(sols)
  }
  if (fac.kind === 'pow' && fac.exp.kind === 'num' && fac.exp.n > 0n)
    return keep(solveZero(fac.base, x, steps, depth + 1))
  // A polynomial in a single generator g(x): solve for g, then invert g.
  const U = sym(`u_solve${depth}__`)
  const viaExp = expGenerator(n, x, U)
  const sgens = findGens(n).filter((g) => hasSymbol(g, x))
  if (viaExp !== undefined || sgens.length === 1) {
    const g = viaExp === undefined ? (sgens[0] as Expr) : viaExp[0]
    const inU = viaExp === undefined ? substitute(n, [[g, U]]) : viaExp[1]
    const ugens = findGens(inU)
    const pu = exprToP(inU, ugens)
    const iu = ugens.findIndex((h) => h.key === U.key)
    let degU = 0
    if (pu !== undefined && iu >= 0) for (const t of pu.terms) degU = Math.max(degU, t.e[iu] as number)
    if (!hasSymbol(inU, x) && degU >= 2) {
      const values = solveZero(inU, U, steps, depth + 1)
      if (steps !== undefined) steps.push({ rule: 'substitute u = g(x)', before: n, after: inU })
      const sols: Expr[] = []
      for (const v of values) sols.push(...solveZero(makeAdd([g, neg(v)]), x, steps, depth + 1))
      return keep(sols)
    }
  }
  // a*h(u) + b = 0 with h invertible: u = h^-1(-b/a).
  const terms = n.kind === 'add' ? n.args : [n]
  const dep = terms.filter((t) => hasSymbol(t, x))
  if (dep.length === 1) {
    const rest = makeAdd(terms.filter((t) => !hasSymbol(t, x)))
    const term = dep[0] as Expr
    const consts = term.kind === 'mul' ? term.args.filter((a) => !hasSymbol(a, x)) : []
    const core = term.kind === 'mul' ? makeMul(term.args.filter((a) => hasSymbol(a, x))) : term
    const value = cancel(makeMul([neg(rest), makePow(makeMul(consts), MINUS_ONE)]))
    if (core.kind === 'fn' && core.args.length === 1) {
      const pre = invert(core.name, value)
      if (pre !== undefined) {
        if (steps !== undefined) steps.push({ rule: `invert ${core.name}`, before: n, after: makeFn('List', pre) })
        const sols: Expr[] = []
        for (const v of pre) sols.push(...solveZero(makeAdd([core.args[0] as Expr, neg(v)]), x, steps, depth + 1))
        return keep(sols)
      }
    }
    if (core.kind === 'pow' && !hasSymbol(core.exp, x)) {
      // u^k = v gives u = v^(1/k) (principal root; for integer k the polynomial branch above finds all roots).
      if (steps !== undefined) steps.push({ rule: 'take a root', before: n, after: value })
      return keep(
        solveZero(makeAdd([core.base, neg(makePow(value, makePow(core.exp, MINUS_ONE)))]), x, steps, depth + 1),
      )
    }
    if (core.kind === 'pow' && !hasSymbol(core.base, x)) {
      // b^u = v gives u = log(v)/log(b), exactly k when v = b^k for a small rational k.
      const u =
        exactLog(core.base, value) ?? makeMul([makeFn('log', [value]), makePow(makeFn('log', [core.base]), MINUS_ONE)])
      if (steps !== undefined) steps.push({ rule: 'take a logarithm', before: n, after: u })
      return keep(solveZero(makeAdd([core.exp, neg(cancel(u))]), x, steps, depth + 1))
    }
  }
  // Two terms a*h(u) = b*h(v) with the same invertible h is beyond this solver.
  throw new WhiteboxError('not-implemented', 'solve: no method applies to this equation')
}

function solveImpl(eq: Expr, x: ExprSym, steps: Step[] | undefined): Expr[] {
  const f = asZero(eq)
  const sols = solveZero(f, x, steps, 0).map((s) => {
    const c = cancel(s)
    return c.key.length <= s.key.length ? c : s
  })
  return sortSolutions(dedupe(sols))
}

/**
 * Solve an equation for x (SymPy's `solve` for one equation): linear, quadratic and polynomial equations (exact
 * roots, factoring over the rationals first, so any degree whose factors have degree at most four), rational
 * equations (roots of the numerator that do not make the denominator vanish), products (each factor), polynomials
 * in one function of x such as exp(x) or sin(x), and simple transcendental equations a*h(u) + b = 0 for invertible h
 * (principal solutions: sin(x) = 0 gives 0 and pi, as SymPy). Solutions are real ones first in increasing order.
 *
 * @param eq - An equation `eq(lhs, rhs)`, or an expression equal to zero.
 * @param x - The unknown.
 * @returns The solutions.
 * @throws WhiteboxError with code `not-implemented` when no method applies, `value` for an identity.
 * @example
 * ```ts
 * import { parse, solve, sym, toText } from 'whitebox-engine/symbolic'
 * const x = sym('x')
 * solve(parse('x^2 - 5x + 6'), x).map(toText) // => ['2', '3']
 * solve(parse('Eq(exp(2*x) - 3*exp(x), -2)'), x).map(toText) // => ['0', 'log(2)']
 * solve(parse('sin(x) - 1/2'), x).map(toText) // => ['pi/6', '5*pi/6']
 * ```
 */
export function solve(eq: Expr, x: ExprSym): Expr[] {
  return bounded('solve', () => {
    return solveImpl(eq, x, undefined)
  })
}

/**
 * {@link solve} with its steps.
 *
 * @param eq - The equation.
 * @param x - The unknown.
 * @returns The solutions (as a `List` expression) and the steps.
 * @example
 * ```ts
 * import { parse, solveSteps, sym } from 'whitebox-engine/symbolic'
 * solveSteps(parse('2^x - 8'), sym('x')).steps[0]?.rule // => 'take a logarithm'
 * ```
 */
export function solveSteps(eq: Expr, x: ExprSym): Derivation {
  return bounded('solveSteps', () => {
    const steps: Step[] = []
    const sols = solveImpl(eq, x, steps)
    return { result: makeFn('List', sols), steps }
  })
}

/**
 * Solve a system of linear equations (SymPy's `solve` with a list of linear equations): Gauss-Jordan elimination
 * with exact expression arithmetic. An underdetermined system is solved for the pivot variables in terms of the
 * free ones, which are left out of the result.
 *
 * @param eqs - Equations (relations or expressions equal to zero), linear in `vars`.
 * @param vars - The unknowns.
 * @returns The solution, by variable name.
 * @throws WhiteboxError with code `value` when an equation is not linear or the system is inconsistent.
 * @example
 * ```ts
 * import { parse, solveLinearSystem, sym, toText } from 'whitebox-engine/symbolic'
 * const s = solveLinearSystem([parse('x + y - 3'), parse('Eq(x - y, 1)')], [sym('x'), sym('y')])
 * [toText(s['x'] ?? 0), toText(s['y'] ?? 0)] // => ['2', '1']
 * ```
 */
export function solveLinearSystem(eqs: readonly Expr[], vars: readonly ExprSym[]): Record<string, Expr> {
  return bounded('solveLinearSystem', () => {
    const m = vars.length
    const rows: Expr[][] = eqs.map((eq) => {
      const f = asZero(eq)
      const zero = vars.map(() => ZERO)
      const constant = substitute(
        f,
        vars.map((v, i) => [v, zero[i] as Expr] as const),
      )
      const coeffs = vars.map((v) => {
        const at1 = substitute(
          f,
          vars.map((w) => [w, w.key === v.key ? ONE : ZERO] as const),
        )
        return cancel(makeAdd([at1, neg(constant)]))
      })
      // Check linearity: f must equal sum coeff_i v_i + constant.
      const rebuilt = makeAdd([constant, ...coeffs.map((c, i) => makeMul([c, vars[i] as Expr]))])
      if (!isZero(makeAdd([f, neg(rebuilt)])))
        throw new WhiteboxError('value', 'solveLinearSystem: an equation is not linear')
      return [...coeffs, cancel(neg(constant))]
    })
    const n = rows.length
    const pivots: number[] = []
    let r = 0
    for (let c = 0; c < m && r < n; c++) {
      let piv = -1
      for (let i = r; i < n; i++) {
        if (!isZero((rows[i] as Expr[])[c] as Expr)) {
          piv = i
          break
        }
      }
      if (piv < 0) continue
      ;[rows[r], rows[piv]] = [rows[piv] as Expr[], rows[r] as Expr[]]
      const row = rows[r] as Expr[]
      const inv = makePow(row[c] as Expr, MINUS_ONE)
      for (let j = 0; j <= m; j++) row[j] = cancel(makeMul([row[j] as Expr, inv]))
      for (let i = 0; i < n; i++) {
        if (i === r) continue
        const other = rows[i] as Expr[]
        const f = other[c] as Expr
        if (isZero(f)) continue
        for (let j = 0; j <= m; j++) other[j] = cancel(makeAdd([other[j] as Expr, neg(makeMul([f, row[j] as Expr]))]))
      }
      pivots.push(c)
      r++
    }
    for (let i = r; i < n; i++) {
      if (!isZero((rows[i] as Expr[])[m] as Expr))
        throw new WhiteboxError('value', 'solveLinearSystem: the system is inconsistent')
    }
    const out: Record<string, Expr> = {}
    pivots.forEach((c, i) => {
      const row = rows[i] as Expr[]
      const terms: Expr[] = [row[m] as Expr]
      for (let j = 0; j < m; j++) {
        if (j !== c && !pivots.includes(j) && !isZero(row[j] as Expr))
          terms.push(neg(makeMul([row[j] as Expr, vars[j] as Expr])))
      }
      out[(vars[c] as ExprSym).name] = cancel(makeAdd(terms))
    })
    return out
  })
}
