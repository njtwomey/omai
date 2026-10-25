// Helpers for the symbolic golden tests: the fixture output shape written by oracle/mdml_oracle/symbolic.py and
// value comparisons at its sample points. Not published.
import fc from 'fast-check'
import { z } from 'zod'
import { erf, gamma } from '../lib/numerics/special/index.ts'
import {
  add,
  cos,
  type Expr,
  evaluateComplex,
  exp,
  log,
  mul,
  type NumericFunctions,
  pow,
  rational,
  sin,
  sym,
} from '../lib/symbolic/index.ts'

/** Small integers and rationals. */
export const rationalArb: fc.Arbitrary<Expr> = fc
  .tuple(fc.integer({ min: -5, max: 5 }), fc.integer({ min: 1, max: 4 }))
  .map(([p, q]) => rational(p, q))

/** Polynomials in x and y with small rational coefficients, total degree up to `degree`, as expressions. */
export function polynomialArb(degree = 3, vars: readonly string[] = ['x', 'y']): fc.Arbitrary<Expr> {
  const monomials: [number, number][] = []
  for (let i = 0; i <= degree; i++) for (let j = 0; j + i <= degree; j++) monomials.push([i, j])
  const used = vars.length === 1 ? monomials.filter(([, j]) => j === 0) : monomials
  return fc
    .array(fc.tuple(fc.constantFrom(...used), fc.integer({ min: -4, max: 4 })), { minLength: 1, maxLength: 5 })
    .map((ts) => add(...ts.map(([[i, j], c]) => mul(c, pow(sym(vars[0] ?? 'x'), i), pow(sym(vars[1] ?? 'y'), j)))))
}

/** Products of small polynomial factors (so that factorisation has something to find). */
export function productArb(vars: readonly string[] = ['x', 'y']): fc.Arbitrary<Expr> {
  return fc.array(polynomialArb(2, vars), { minLength: 1, maxLength: 3 }).map((fs) => mul(...fs))
}

/** Expressions in x and y built from + * powers, sin, cos, exp and log of 1 + square (defined for real arguments). */
export const exprArb: fc.Arbitrary<Expr> = fc.letrec<{ e: Expr }>((tie) => ({
  e: fc.oneof(
    { depthSize: 'small', withCrossShrink: true },
    fc.constantFrom(sym('x'), sym('y')),
    rationalArb,
    fc.tuple(tie('e'), tie('e')).map(([a, b]) => add(a, b)),
    fc.tuple(tie('e'), tie('e')).map(([a, b]) => mul(a, b)),
    fc.tuple(tie('e'), fc.integer({ min: 0, max: 3 })).map(([a, k]) => pow(a, k)),
    tie('e').map((a) => sin(a)),
    tie('e').map((a) => cos(a)),
    tie('e').map((a) => exp(mul(rational(1, 2), a))),
    tie('e').map((a) => log(add(1, pow(a, 2)))),
  ),
})).e

import { item } from '../lib/tensor/index.ts'
import type { GoldenCase } from './golden.ts'

/** Numeric erf and gamma (from numerics) for evaluating results that contain them. */
export const NUMERIC_FUNCTIONS: NumericFunctions = {
  erf: (x) => item(erf(x)),
  gamma: (x) => item(gamma(x)),
  factorial: (x) => item(gamma(x + 1)),
}

const Num = z.union([z.number(), z.enum(['inf', '-inf', 'nan'])])

/** The output of `expr_case`: SymPy's text and LaTeX, and [re, im] values at the sample points. */
export interface SymbolicOutput {
  readonly text: string
  readonly latex: string
  readonly values: readonly (readonly [number | string, number | string])[]
  readonly names: readonly string[]
}

const OutputSchema: z.ZodType<SymbolicOutput> = z.object({
  text: z.string(),
  latex: z.string(),
  values: z.array(z.tuple([Num, Num])),
  names: z.array(z.string()),
})

/** Parse a case's output. */
export function symbolicOutput(c: GoldenCase): SymbolicOutput {
  return OutputSchema.parse(c.output)
}

const POINTS = [
  [0.37, 0.71, 1.23, 1.9, 2.6],
  [0.53, 1.07, 0.29, 1.61, 2.2],
  [1.31, 0.43, 0.87, 2.3, 0.61],
  [0.77, 1.49, 0.23, 1.13, 1.97],
]

/** Sample values of the i-th symbol, as `point_row` in the oracle. */
export function pointRow(i: number): number[] {
  return (POINTS[i % POINTS.length] as number[]).map((v) => v + 0.1 * Math.floor(i / POINTS.length))
}

function decode(v: number | string): number {
  return typeof v === 'number' ? v : v === 'inf' ? Number.POSITIVE_INFINITY : v === '-inf' ? -Infinity : Number.NaN
}

/** Values of an expression at the sample points of `names` (complex, principal branches). */
export function valuesAt(e: Expr, names: readonly string[]): [number, number][] {
  return pointRow(0).map((_, k) => {
    const env: Record<string, number> = {}
    names.forEach((n, i) => {
      env[n] = pointRow(i)[k] as number
    })
    try {
      return evaluateComplex(e, env, NUMERIC_FUNCTIONS)
    } catch {
      return [Number.NaN, Number.NaN]
    }
  })
}

/** Assert that e has the reference values at every sample point where the reference is finite. */
export function expectValues(e: Expr, out: SymbolicOutput, rtol: number, atol: number, label: string): void {
  // Relations and connectives have truth values, which numpy and mdml-js do not share a numeric form for.
  if (e.kind === 'rel' || e.kind === 'logic') return
  const got = valuesAt(e, out.names)
  let compared = 0
  out.values.forEach(([re, im], k) => {
    const want: [number, number] = [decode(re), decode(im)]
    if (!want.every(Number.isFinite)) return
    const [gr, gi] = got[k] as [number, number]
    const scale = Math.max(Math.hypot(want[0], want[1]), 1)
    const err = Math.hypot(gr - want[0], gi - want[1])
    if (!(err <= atol + rtol * scale)) {
      throw new Error(`${label}: value at point ${k} is [${gr}, ${gi}], expected [${want[0]}, ${want[1]}]`)
    }
    compared++
  })
  if (compared === 0 && out.values.length > 0 && out.values.some(([re]) => Number.isFinite(decode(re)))) {
    throw new Error(`${label}: no comparable points`)
  }
}

/**
 * Assert that e - reference is the same constant at every sample point (antiderivatives agree up to a constant).
 * With `realOn`, only points where that expression (the integrand) is real are used: off its real domain the two
 * antiderivatives may sit on different sides of a branch cut.
 */
export function expectSameUpToConstant(e: Expr, out: SymbolicOutput, tol: number, label: string, realOn?: Expr): void {
  const got = valuesAt(e, out.names)
  const domain = realOn === undefined ? undefined : valuesAt(realOn, out.names)
  const diffs: [number, number][] = []
  out.values.forEach(([re, im], k) => {
    const want: [number, number] = [decode(re), decode(im)]
    const g = got[k] as [number, number]
    if (!want.every(Number.isFinite) || !g.every(Number.isFinite)) return
    const d = domain?.[k]
    if (d !== undefined && (!d.every(Number.isFinite) || Math.abs(d[1]) > 1e-12)) return
    diffs.push([g[0] - want[0], g[1] - want[1]])
  })
  if (diffs.length < 2) throw new Error(`${label}: fewer than two comparable points`)
  const [d0r, d0i] = diffs[0] as [number, number]
  for (const [dr, di] of diffs) {
    if (Math.hypot(dr - d0r, di - d0i) > tol * Math.max(1, Math.hypot(d0r, d0i))) {
      throw new Error(`${label}: antiderivatives differ by a non-constant (${dr}, ${di} vs ${d0r}, ${d0i})`)
    }
  }
}
