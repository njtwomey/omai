/**
 * Symbolic differentiation, as SymPy's `diff`: derivatives with respect to one or more variables and to any order,
 * gradients, Jacobians and Hessians as symbolic matrices, and implicit differentiation (`idiff`). The `Steps` variants
 * record the rule applied at each node (sum, product, power, chain rule) for worked derivations.
 */

import { MdmlError } from '../base/index.ts'
import { bounded } from './_budget.ts'
import {
  asCoeffMul,
  HALF,
  MINUS_ONE,
  makeAdd,
  makeDerivative,
  makeIntegral,
  makeMul,
  makeOrder,
  makePiecewise,
  makePow,
  numFromRational,
  ONE,
  PI,
  TWO,
  ZERO,
} from './_canon.ts'
import { makeFn } from './_functions.ts'
import { hasSymbol } from './_node.ts'
import { KNOWN_FUNCTIONS } from './_order.ts'
import { rat } from './_rational.ts'
import { cancel } from './_rational-functions.ts'
import { type Derivation, type Expr, type ExprSym, type Step, substitute, sym } from './expr.ts'
import type { Matrix } from './matrices.ts'

/** A differentiation variable: a symbol (or its name), optionally with an order, as `[x, 2]`. */
export type DiffSpec = ExprSym | string | readonly [ExprSym | string, number]

const neg = (a: Expr): Expr => makeMul([MINUS_ONE, a])
const sq = (a: Expr): Expr => makePow(a, TWO)
const inv = (a: Expr): Expr => makePow(a, MINUS_ONE)

/** The derivative of a known function with respect to its (single) argument u, or undefined. */
function outer(name: string, u: Expr, args: readonly Expr[]): Expr | undefined {
  switch (name) {
    case 'exp':
      return makeFn('exp', [u])
    case 'log':
      return inv(u)
    case 'sin':
      return makeFn('cos', [u])
    case 'cos':
      return neg(makeFn('sin', [u]))
    case 'tan':
      return makeAdd([sq(makeFn('tan', [u])), ONE])
    case 'cot':
      return makeAdd([neg(sq(makeFn('cot', [u]))), MINUS_ONE])
    case 'sec':
      return makeMul([makeFn('tan', [u]), makeFn('sec', [u])])
    case 'csc':
      return neg(makeMul([makeFn('cot', [u]), makeFn('csc', [u])]))
    case 'asin':
      return inv(makePow(makeAdd([ONE, neg(sq(u))]), HALF))
    case 'acos':
      return neg(inv(makePow(makeAdd([ONE, neg(sq(u))]), HALF)))
    case 'atan':
      return inv(makeAdd([sq(u), ONE]))
    case 'acot':
      return neg(inv(makeAdd([sq(u), ONE])))
    case 'sinh':
      return makeFn('cosh', [u])
    case 'cosh':
      return makeFn('sinh', [u])
    case 'tanh':
      return makeAdd([ONE, neg(sq(makeFn('tanh', [u])))])
    case 'coth':
      return neg(inv(sq(makeFn('sinh', [u]))))
    case 'asinh':
      return inv(makePow(makeAdd([sq(u), ONE]), HALF))
    case 'acosh':
      return inv(makeMul([makePow(makeAdd([u, MINUS_ONE]), HALF), makePow(makeAdd([u, ONE]), HALF)]))
    case 'atanh':
      return inv(makeAdd([ONE, neg(sq(u))]))
    case 'Abs':
      return makeFn('sign', [u])
    case 'sign':
    case 'floor':
    case 'ceiling':
      return ZERO
    case 'erf':
      return makeMul([TWO, makePow(PI, numFromRational(rat(-1n, 2n))), makeFn('exp', [neg(sq(u))])])
    case 'gamma':
      return makeMul([makeFn('gamma', [u]), makeFn('polygamma', [ZERO, u])])
    case 'factorial':
      return makeMul([makeFn('factorial', [u]), makeFn('polygamma', [ZERO, makeAdd([u, ONE])])])
    default:
      void args
      return undefined
  }
}

const RULE_NAMES: Readonly<Record<string, string>> = {
  exp: 'exponential rule',
  log: 'logarithm rule',
}

function d(e: Expr, x: ExprSym, steps: Step[] | undefined): Expr {
  const out = dRaw(e, x, steps)
  return out
}

function record(steps: Step[] | undefined, rule: string, e: Expr, x: ExprSym, result: Expr): Expr {
  if (steps !== undefined) steps.push({ rule, before: makeDerivative(e, [[x, 1]]), after: result })
  return result
}

function dRaw(e: Expr, x: ExprSym, steps: Step[] | undefined): Expr {
  if (!hasSymbol(e, x)) return ZERO
  switch (e.kind) {
    case 'sym':
      return ONE
    case 'add':
      return record(steps, 'sum rule', e, x, makeAdd(e.args.map((t) => d(t, x, steps))))
    case 'mul': {
      const [c, rest] = asCoeffMul(e)
      if (!(c.kind === 'num' && c.n === 1n && c.d === 1n)) {
        return record(steps, 'constant multiple', e, x, makeMul([c, d(rest, x, steps)]))
      }
      const fs = e.args
      const terms = fs.map((f, i) => makeMul([d(f, x, steps), ...fs.filter((_, j) => j !== i)]))
      return record(steps, 'product rule', e, x, makeAdd(terms))
    }
    case 'pow': {
      const { base, exp } = e
      if (!hasSymbol(exp, x)) {
        const db = d(base, x, steps)
        const v = makeMul([exp, makePow(base, makeAdd([exp, MINUS_ONE])), db])
        return record(steps, base.key === x.key ? 'power rule' : 'power rule with chain rule', e, x, v)
      }
      if (!hasSymbol(base, x)) {
        const v = makeMul([e, makeFn('log', [base]), d(exp, x, steps)])
        return record(steps, 'exponential rule', e, x, v)
      }
      const v = makeMul([
        e,
        makeAdd([makeMul([d(exp, x, steps), makeFn('log', [base])]), makeMul([exp, d(base, x, steps), inv(base)])]),
      ])
      return record(steps, 'logarithmic differentiation', e, x, v)
    }
    case 'fn': {
      if (e.name === 'atan2' && e.args.length === 2) {
        const [y, z] = e.args as [Expr, Expr]
        const r2 = inv(makeAdd([sq(y), sq(z)]))
        const v = makeAdd([makeMul([z, r2, d(y, x, steps)]), neg(makeMul([y, r2, d(z, x, steps)]))])
        return record(steps, 'chain rule', e, x, v)
      }
      if (e.name === 'polygamma' && e.args.length === 2) {
        const [n, u] = e.args as [Expr, Expr]
        if (!hasSymbol(n, x)) {
          return record(
            steps,
            'chain rule',
            e,
            x,
            makeMul([makeFn('polygamma', [makeAdd([n, ONE]), u]), d(u, x, steps)]),
          )
        }
      }
      const u = e.args[0] as Expr
      const f = e.args.length === 1 ? outer(e.name, u, e.args) : undefined
      if (f === undefined) {
        if (KNOWN_FUNCTIONS.has(e.name) && e.args.length === 1) {
          throw new MdmlError('not-implemented', `diff: no derivative rule for ${e.name}`)
        }
        return record(steps, 'undefined function', e, x, makeDerivative(e, [[x, 1]]))
      }
      const du = d(u, x, steps)
      const rule = u.key === x.key ? (RULE_NAMES[e.name] ?? `derivative of ${e.name}`) : 'chain rule'
      return record(steps, rule, e, x, makeMul([f, du]))
    }
    case 'piecewise':
      return record(steps, 'piecewise', e, x, makePiecewise(e.pieces.map(([v, c]) => [d(v, x, steps), c] as const)))
    case 'derivative':
      return makeDerivative(e.expr, [...e.variables, [x, 1]])
    case 'integral': {
      if (e.bounds === undefined) {
        if (e.variable.key === x.key) return record(steps, 'fundamental theorem', e, x, e.expr)
        return record(steps, 'differentiate under the integral', e, x, makeIntegral(d(e.expr, x, steps), e.variable))
      }
      const [a, b] = e.bounds
      const at = (v: Expr): Expr => substitute(e.expr, [[e.variable, v]])
      const inner = hasSymbol(e.expr, x) ? makeIntegral(d(e.expr, x, steps), e.variable, e.bounds) : ZERO
      const v = makeAdd([makeMul([at(b), d(b, x, steps)]), neg(makeMul([at(a), d(a, x, steps)])), inner])
      return record(steps, 'Leibniz integral rule', e, x, v)
    }
    case 'order':
      return makeOrder(d(e.expr, x, steps), e.variable, e.point)
    default:
      throw new MdmlError('not-implemented', `diff: cannot differentiate a ${e.kind}`)
  }
}

function specs(vars: readonly DiffSpec[]): [ExprSym, number][] {
  const out: [ExprSym, number][] = []
  for (const v of vars) {
    if (Array.isArray(v)) {
      const [s, k] = v as readonly [ExprSym | string, number]
      out.push([typeof s === 'string' ? sym(s) : s, k])
    } else out.push([typeof v === 'string' ? sym(v) : (v as ExprSym), 1])
  }
  if (out.some(([, k]) => !Number.isInteger(k) || k < 0)) {
    throw new MdmlError('value', 'diff: orders must be non-negative integers')
  }
  return out
}

/**
 * Differentiate with respect to one or more variables in turn (SymPy's `diff`): `diff(f, x)`, `diff(f, x, y)` for
 * a mixed partial, `diff(f, [x, 3])` for a third derivative. Sum, product, power, chain and logarithmic rules, the
 * derivatives of every known function, piecewise expressions, Leibniz's rule for integrals; undefined functions give
 * unevaluated derivatives. `Abs` differentiates to `sign` (the derivative for real arguments).
 *
 * @param e - The expression.
 * @param vars - Variables (symbols or names), each optionally with an order.
 * @returns The derivative.
 * @example
 * ```ts
 * import { diff, parse, toText } from 'mdml-engine/symbolic'
 * toText(diff(parse('x^2*sin(x)'), 'x')) // => 'x**2*cos(x) + 2*x*sin(x)'
 * toText(diff(parse('x^3*y^2'), 'x', 'y')) // => '6*x**2*y'
 * ```
 */
export function diff(e: Expr, ...vars: readonly DiffSpec[]): Expr {
  return bounded('diff', () => {
    let out = e
    for (const [x, k] of specs(vars)) for (let i = 0; i < k; i++) out = d(out, x, undefined)
    return out
  })
}

/**
 * {@link diff} with the rule applied at every node (constant, sum, product, power, chain rules and so on); each step
 * takes the unevaluated derivative of a subexpression to its result.
 *
 * @param e - The expression.
 * @param vars - Variables, each optionally with an order.
 * @returns The derivative and its steps.
 * @example
 * ```ts
 * import { diffSteps, parse } from 'mdml-engine/symbolic'
 * diffSteps(parse('sin(x^2)'), 'x').steps.map((s) => s.rule) // => ['power rule', 'chain rule']
 * ```
 */
export function diffSteps(e: Expr, ...vars: readonly DiffSpec[]): Derivation {
  return bounded('diffSteps', () => {
    const steps: Step[] = []
    let out = e
    for (const [x, k] of specs(vars)) for (let i = 0; i < k; i++) out = d(out, x, steps)
    return { result: out, steps }
  })
}

/**
 * The gradient: the vector of first partial derivatives.
 *
 * @param e - A scalar expression.
 * @param vars - The variables.
 * @returns ∂e/∂v for each variable.
 * @example
 * ```ts
 * import { gradient, parse, toText } from 'mdml-engine/symbolic'
 * gradient(parse('x^2*y'), ['x', 'y']).map(toText) // => ['2*x*y', 'x**2']
 * ```
 */
export function gradient(e: Expr, vars: readonly (ExprSym | string)[]): Expr[] {
  return bounded('gradient', () => {
    return vars.map((v) => diff(e, v))
  })
}

/**
 * The Jacobian matrix J[i][j] = ∂f_i/∂v_j.
 *
 * @param fs - The component functions.
 * @param vars - The variables.
 * @returns The matrix.
 * @example
 * ```ts
 * import { jacobian, parse, toText } from 'mdml-engine/symbolic'
 * jacobian([parse('x*y'), parse('x + y')], ['x', 'y']).map((r) => r.map(toText)) // => [['y', 'x'], ['1', '1']]
 * ```
 */
export function jacobian(fs: readonly Expr[], vars: readonly (ExprSym | string)[]): Matrix {
  return bounded('jacobian', () => {
    return fs.map((f) => gradient(f, vars))
  })
}

/**
 * The Hessian matrix of second partial derivatives.
 *
 * @param e - A scalar expression.
 * @param vars - The variables.
 * @returns H[i][j] = ∂²e/∂v_i∂v_j.
 * @example
 * ```ts
 * import { hessian, parse, toText } from 'mdml-engine/symbolic'
 * hessian(parse('x^2*y + y^3'), ['x', 'y']).map((r) => r.map(toText)) // => [['2*y', '2*x'], ['2*x', '6*y']]
 * ```
 */
export function hessian(e: Expr, vars: readonly (ExprSym | string)[]): Matrix {
  return bounded('hessian', () => {
    const g = gradient(e, vars)
    return g.map((gi) => vars.map((v) => diff(gi, v)))
  })
}

function implicit(eq: Expr, y: ExprSym, x: ExprSym, n: number, steps: Step[] | undefined): Expr {
  const f = eq.kind === 'rel' && eq.op === '==' ? makeAdd([eq.lhs, neg(eq.rhs)]) : eq
  const fx = d(f, x, steps)
  const fy = d(f, y, steps)
  const dydx = cancel(neg(makeMul([fx, inv(fy)])))
  if (steps !== undefined)
    steps.push({ rule: 'implicit differentiation: -F_x/F_y', before: makeDerivative(y, [[x, 1]]), after: dydx })
  let cur = dydx
  for (let k = 2; k <= n; k++) {
    // d/dx of an expression in x and y(x): ∂/∂x + ∂/∂y · dy/dx.
    const next = cancel(makeAdd([d(cur, x, steps), makeMul([d(cur, y, steps), dydx])]))
    if (steps !== undefined)
      steps.push({
        rule: 'differentiate again with dy/dx substituted',
        before: makeDerivative(y, [[x, k]]),
        after: next,
      })
    cur = next
  }
  return cur
}

/**
 * Implicit differentiation (SymPy's `idiff`): d^n y / dx^n for y defined implicitly by eq = 0 (or an `eq(lhs, rhs)`
 * relation), as a rational expression in x and y.
 *
 * @param eq - The equation F(x, y) = 0.
 * @param y - The dependent variable.
 * @param x - The independent variable.
 * @param n - The order (default 1).
 * @returns The derivative in lowest terms.
 * @example
 * ```ts
 * import { implicitDiff, parse, sym, toText } from 'mdml-engine/symbolic'
 * toText(implicitDiff(parse('x^2 + y^2 - 4'), sym('y'), sym('x'))) // => '-x/y'
 * ```
 */
export function implicitDiff(eq: Expr, y: ExprSym, x: ExprSym, n = 1): Expr {
  return bounded('implicitDiff', () => {
    return implicit(eq, y, x, n, undefined)
  })
}

/**
 * {@link implicitDiff} with its steps.
 *
 * @param eq - The equation.
 * @param y - The dependent variable.
 * @param x - The independent variable.
 * @param n - The order.
 * @returns The derivative and its steps.
 * @example
 * ```ts
 * import { implicitDiffSteps, parse, sym } from 'mdml-engine/symbolic'
 * implicitDiffSteps(parse('x*y - 1'), sym('y'), sym('x')).steps.length > 0 // => true
 * ```
 */
export function implicitDiffSteps(eq: Expr, y: ExprSym, x: ExprSym, n = 1): Derivation {
  return bounded('implicitDiffSteps', () => {
    const steps: Step[] = []
    const result = implicit(eq, y, x, n, steps)
    return { result, steps }
  })
}
