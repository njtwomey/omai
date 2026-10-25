/**
 * Series expansions, as SymPy's `series`: Taylor and Laurent series about a point with a big-O remainder, Taylor
 * polynomials and coefficients, residues, Pade approximants (rational functions matching the series), and Landau's O
 * notation with `bigO` and `removeO`.
 */

import { MdmlError } from '../base/index.ts'
import { bounded } from './_budget.ts'
import {
  MINUS_ONE,
  makeAdd,
  makeMul,
  makeOrder,
  makePow,
  mapChildren,
  NEG_INF,
  numFromRational,
  OO,
  ZERO,
} from './_canon.ts'
import { makeFn } from './_functions.ts'
import { rat } from './_rational.ts'
import { cancel } from './_rational-functions.ts'
import { LOG_T, type S, seriesAt0, seriesToExpr } from './_series.ts'
import {
  asExpr,
  type Derivation,
  type Expr,
  type ExprLike,
  type ExprSym,
  type Step,
  substitute,
  sym,
  type WorkBudget,
} from './expr.ts'

/** Direction of approach: from above (`'+'`) or below (`'-'`). */
export type Direction = '+' | '-'

/** Where the expansion variable is and how it maps back to x. */
interface Frame {
  readonly t: ExprSym
  readonly sub: Expr
  readonly back: Expr
  readonly point: Expr
}

const T = sym('t_series__', { positive: true })

function frame(x: ExprSym, x0: Expr, dir: Direction): Frame {
  if (x0.key === OO.key) return { t: T, sub: makePow(T, MINUS_ONE), back: makePow(x, MINUS_ONE), point: OO }
  if (x0.key === NEG_INF.key)
    return {
      t: T,
      sub: makeMul([MINUS_ONE, makePow(T, MINUS_ONE)]),
      back: makeMul([MINUS_ONE, makePow(x, MINUS_ONE)]),
      point: NEG_INF,
    }
  const s = dir === '+' ? T : makeMul([MINUS_ONE, T])
  const back = dir === '+' ? makeAdd([x, makeMul([MINUS_ONE, x0])]) : makeAdd([x0, makeMul([MINUS_ONE, x])])
  return { t: T, sub: makeAdd([x0, s]), back, point: x0 }
}

function logBack(f: Frame): Expr {
  return makeFn('log', [f.back])
}

function expansion(e: Expr, x: ExprSym, x0: Expr, n: number, dir: Direction, steps?: Step[]): [S, Frame] {
  const f = frame(x, x0, dir)
  const inT = substitute(e, [[x, f.sub]])
  return [seriesAt0(inT, f.t, n, steps), f]
}

function backToX(s: S, f: Frame): Expr {
  const body = seriesToExpr(s, f.t)
  return substitute(body, [
    [f.t, f.back],
    [LOG_T, logBack(f)],
  ])
}

/**
 * The series of an expression about x0 to order n, with its big-O remainder (SymPy's `series`): Taylor series where
 * the expression is analytic, Laurent series at poles (1/sin(x) = 1/x + x/6 + ...), and expansions at oo in powers of
 * 1/x. Terms are exact; log(x) may appear in coefficients (x*log(x)). Expansions at a finite x0 are in powers of
 * (x - x0) from the right, or (x0 - x) with `dir = '-'`.
 *
 * @param e - The expression.
 * @param x - The variable.
 * @param x0 - The expansion point (default 0; `oo` for infinity).
 * @param n - The order of the remainder: terms up to (x - x0)^(n-1) (default 6).
 * @param dir - Direction (default `'+'`).
 * @param budget - Work limits (see {@link WorkBudget}).
 * @returns The series plus O((x - x0)^n).
 * @throws MdmlError with code `not-implemented` at essential singularities and branch points needing
 *   fractional powers (Puiseux series), and with code `convergence` when the work budget runs out (coefficients
 *   that grow too large to normalise); no partial series is returned, since its order would be wrong.
 * @example
 * ```ts
 * import { parse, series, sym, toText } from 'mdml-engine/symbolic'
 * toText(series(parse('sin(x)'), sym('x'))) // => 'x - x**3/6 + x**5/120 + O(x**6)'
 * toText(series(parse('1/sin(x)'), sym('x'), 0, 4)) // => '1/x + x/6 + 7*x**3/360 + O(x**4)'
 * series(parse('exp(sin(x))'), sym('x'), 0, 8, '+', { steps: 100 }) // => throws convergence
 * ```
 */
export function series(e: Expr, x: ExprSym, x0: ExprLike = 0, n = 6, dir: Direction = '+', budget?: WorkBudget): Expr {
  return bounded(
    'series',
    () => {
      const p = asExpr(x0)
      const [s, f] = expansion(e, x, p, n, dir)
      const order =
        p.key === OO.key || p.key === NEG_INF.key
          ? makeOrder(makePow(x, numFromRational(rat(BigInt(-n)))), x, p)
          : makeOrder(makePow(isZeroPoint(p) ? x : f.back, numFromRational(rat(BigInt(n)))), x, p)
      return makeAdd([backToX(s, f), order])
    },
    budget,
  )
}

function isZeroPoint(p: Expr): boolean {
  return p.kind === 'num' && p.n === 0n
}

/**
 * {@link series} with a step for every subexpression expanded.
 *
 * @param e - The expression.
 * @param x - The variable.
 * @param x0 - The expansion point.
 * @param n - The order.
 * @param dir - Direction.
 * @param budget - Work limits, as for series.
 * @returns The series and its steps.
 * @throws MdmlError with code `convergence` when the work budget runs out.
 * @example
 * ```ts
 * import { parse, seriesSteps, sym } from 'mdml-engine/symbolic'
 * seriesSteps(parse('exp(sin(x))'), sym('x'), 0, 4).steps.map((s) => s.rule) // => ['series of sin', 'series of exp']
 * ```
 */
export function seriesSteps(
  e: Expr,
  x: ExprSym,
  x0: ExprLike = 0,
  n = 6,
  dir: Direction = '+',
  budget?: WorkBudget,
): Derivation {
  return bounded('seriesSteps', () => seriesStepsBody(e, x, asExpr(x0), n, dir), budget)
}

function seriesStepsBody(e: Expr, x: ExprSym, x0: Expr, n: number, dir: Direction): Derivation {
  const steps: Step[] = []
  const p = x0
  const [s, f] = expansion(e, x, p, n, dir, steps)
  const back = (y: Expr): Expr =>
    substitute(y, [
      [f.t, f.back],
      [LOG_T, logBack(f)],
      [f.t, f.back],
    ])
  const shown = steps.map((st) => ({ rule: st.rule, before: back(st.before), after: back(st.after) }))
  return {
    result: makeAdd([
      backToX(s, f),
      makeOrder(makePow(isZeroPoint(p) ? x : f.back, numFromRational(rat(BigInt(n)))), x, p),
    ]),
    steps: shown,
  }
}

/**
 * The Taylor polynomial of degree n - 1 about x0 (the series without its remainder). With x0 = 0 this is the
 * Maclaurin polynomial.
 *
 * @param e - The expression (analytic at x0).
 * @param x - The variable.
 * @param x0 - The expansion point (default 0).
 * @param n - Number of terms by degree: powers up to (x - x0)^(n-1) (default 6).
 * @returns The polynomial in (x - x0).
 * @example
 * ```ts
 * import { parse, sym, taylor, toText } from 'mdml-engine/symbolic'
 * toText(taylor(parse('exp(x)'), sym('x'), 0, 4)) // => 'x**3/6 + x**2/2 + x + 1'
 * ```
 */
export function taylor(e: Expr, x: ExprSym, x0: ExprLike = 0, n = 6): Expr {
  return bounded('taylor', () => {
    const [s, f] = expansion(e, x, asExpr(x0), n, '+')
    if (s.v < 0) throw new MdmlError('value', 'taylor: the expression has a pole here; use series for Laurent terms')
    return backToX(s, f)
  })
}

/**
 * The coefficients a_k of (x - x0)^k for k from the lowest power present up to n - 1, with that lowest power.
 *
 * @param e - The expression.
 * @param x - The variable.
 * @param x0 - The expansion point (default 0).
 * @param n - The order (default 6).
 * @returns `{ start, coefficients }` with coefficients[i] multiplying (x - x0)^(start + i).
 * @example
 * ```ts
 * import { parse, seriesCoefficients, sym, toText } from 'mdml-engine/symbolic'
 * seriesCoefficients(parse('cos(x)'), sym('x'), 0, 5).coefficients.map(toText) // => ['1', '0', '-1/2', '0', '1/24']
 * ```
 */
export function seriesCoefficients(
  e: Expr,
  x: ExprSym,
  x0: ExprLike = 0,
  n = 6,
): { start: number; coefficients: Expr[] } {
  return bounded('seriesCoefficients', () => coefficientsBody(e, x, asExpr(x0), n))
}

function coefficientsBody(e: Expr, x: ExprSym, x0: Expr, n: number): { start: number; coefficients: Expr[] } {
  const [s] = expansion(e, x, x0, n, '+')
  if (s.c.some((c) => JSON.stringify(c.key).includes(LOG_T.key))) {
    throw new MdmlError('value', 'seriesCoefficients: the series has logarithmic terms')
  }
  const coefficients: Expr[] = []
  for (let k = s.v; k < n; k++) coefficients.push(s.c[k - s.v] ?? ZERO)
  return { start: s.v, coefficients }
}

/**
 * The residue at x0: the coefficient of 1/(x - x0) in the Laurent series.
 *
 * @param e - The expression.
 * @param x - The variable.
 * @param x0 - The point (default 0).
 * @returns The residue.
 * @example
 * ```ts
 * import { parse, residue, sym, toText } from 'mdml-engine/symbolic'
 * toText(residue(parse('exp(x)/x^2'), sym('x'))) // => '1'
 * ```
 */
export function residue(e: Expr, x: ExprSym, x0: ExprLike = 0): Expr {
  return bounded('residue', () => {
    const [s] = expansion(e, x, asExpr(x0), 0, '+')
    return s.v <= -1 ? (s.c[-1 - s.v] ?? ZERO) : ZERO
  })
}

/**
 * The Padé approximant [m/n] about x0: P(x)/Q(x) with deg P <= m, deg Q <= n, Q(x0) = 1, matching the Taylor series
 * through order m + n (as `scipy.interpolate.pade` on the Taylor coefficients).
 *
 * @param e - The expression (analytic at x0).
 * @param x - The variable.
 * @param m - Numerator degree.
 * @param n - Denominator degree.
 * @param x0 - The expansion point (default 0).
 * @returns The rational approximant.
 * @throws MdmlError with code `singular` when the approximant does not exist.
 * @example
 * ```ts
 * import { pade, parse, sym, toText } from 'mdml-engine/symbolic'
 * toText(pade(parse('exp(x)'), sym('x'), 1, 1)) // => '-(x + 2)/(x - 2)'
 * ```
 */
export function pade(e: Expr, x: ExprSym, m: number, n: number, x0: ExprLike = 0): Expr {
  return bounded('pade', () => padeBody(e, x, m, n, asExpr(x0)))
}

function padeBody(e: Expr, x: ExprSym, m: number, n: number, p: Expr): Expr {
  const { start, coefficients } = coefficientsBody(e, x, p, m + n + 1)
  if (start < 0) throw new MdmlError('value', 'pade: the expression has a pole at the expansion point')
  const c = (k: number): Expr => (k < start ? ZERO : (coefficients[k - start] ?? ZERO))
  // Solve sum_{j=1..n} q_j c_{k-j} = -c_k for k = m+1..m+n.
  const rows: Expr[][] = []
  const rhs: Expr[] = []
  for (let k = m + 1; k <= m + n; k++) {
    rows.push(Array.from({ length: n }, (_, j) => c(k - j - 1)))
    rhs.push(makeMul([MINUS_ONE, c(k)]))
  }
  const q = [numFromRational(rat(1n)), ...gauss(rows, rhs)]
  const pcoef = Array.from({ length: m + 1 }, (_, k) =>
    makeAdd(Array.from({ length: Math.min(k, n) + 1 }, (_, j) => makeMul([q[j] as Expr, c(k - j)]))),
  )
  const t = isZeroPoint(p) ? x : makeAdd([x, makeMul([MINUS_ONE, p])])
  const poly = (cs: readonly Expr[]): Expr =>
    makeAdd(cs.map((ci, k) => makeMul([ci, makePow(t, numFromRational(rat(BigInt(k))))])))
  return cancel(makeMul([poly(pcoef), makePow(poly(q), MINUS_ONE)]))
}

function gauss(rows: Expr[][], rhs: Expr[]): Expr[] {
  const n = rhs.length
  const m = rows.map((r, i) => [...r, rhs[i] as Expr])
  for (let c = 0; c < n; c++) {
    let piv = -1
    for (let i = c; i < n; i++) if (cancel((m[i] as Expr[])[c] as Expr).key !== ZERO.key) piv = piv < 0 ? i : piv
    if (piv < 0) throw new MdmlError('singular', 'pade: the approximant does not exist (singular system)')
    ;[m[c], m[piv]] = [m[piv] as Expr[], m[c] as Expr[]]
    const row = m[c] as Expr[]
    const inv = makePow(row[c] as Expr, MINUS_ONE)
    for (let j = c; j <= n; j++) row[j] = cancel(makeMul([row[j] as Expr, inv]))
    for (let i = 0; i < n; i++) {
      if (i === c) continue
      const other = m[i] as Expr[]
      const f = other[c] as Expr
      for (let j = c; j <= n; j++)
        other[j] = cancel(makeAdd([other[j] as Expr, makeMul([MINUS_ONE, f, row[j] as Expr])]))
    }
  }
  return m.map((r) => r[n] as Expr)
}

/**
 * Landau's O(e) as x tends to x0 (default 0): constants are dropped and a sum keeps its dominant term, so
 * O(x^2 + x^3) = O(x^2) at 0 and O(x^2 + x) = O(x^2) at oo. Sums with an O term absorb the terms it dominates.
 *
 * @param e - The expression.
 * @param x - The variable.
 * @param x0 - The limit point.
 * @returns The order term.
 * @example
 * ```ts
 * import { add, bigO, parse, sym, toText } from 'mdml-engine/symbolic'
 * const x = sym('x')
 * toText(add(parse('1 + x + x^3'), bigO(parse('x^2'), x))) // => '1 + x + O(x**2)'
 * ```
 */
export function bigO(e: Expr, x: ExprSym, x0: ExprLike = 0): Expr {
  return makeOrder(e, x, asExpr(x0))
}

/**
 * Drop big-O terms (SymPy's `removeO`).
 *
 * @param e - An expression, typically a series.
 * @returns The expression without its order terms.
 * @example
 * ```ts
 * import { parse, removeO, series, sym, toText } from 'mdml-engine/symbolic'
 * toText(removeO(series(parse('cos(x)'), sym('x'), 0, 4))) // => '1 - x**2/2'
 * ```
 */
export function removeO(e: Expr): Expr {
  if (e.kind === 'order') return ZERO
  if (e.kind === 'add') return makeAdd(e.args.filter((a) => a.kind !== 'order').map(removeO))
  return mapChildren(e, removeO)
}
