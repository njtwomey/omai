// Standard optimisation test problems, written with tensor operations so they are differentiable. The oracle case
// files define the same functions (oracle/cases/optim/...); keep the two in step. Not published.
import {
  add,
  cos,
  div,
  exp,
  matmul,
  mul,
  sin,
  sliceAxis,
  sqrt,
  square,
  stack,
  sub,
  sum,
  tensor,
  type Value,
} from '../lib/tensor/index.ts'

const at = (x: Value, i: number): Value => sliceAxis(x, 0, i, i + 1)

/** Rosenbrock: sum 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2, minimum 0 at (1, ..., 1). */
export function rosen(x: Value): Value {
  const n = x.shape[0] ?? 0
  const head = sliceAxis(x, 0, 0, n - 1)
  const tail = sliceAxis(x, 0, 1, n)
  return sum(add(mul(square(sub(tail, square(head))), 100), square(sub(1, head))))
}

/** Beale's function, minimum 0 at (3, 0.5). */
export function beale(v: Value): Value {
  const x = at(v, 0)
  const y = at(v, 1)
  const xy = mul(x, y)
  return sum(
    add(
      add(square(add(sub(1.5, x), xy)), square(add(sub(2.25, x), mul(xy, y)))),
      square(add(sub(2.625, x), mul(mul(xy, y), y))),
    ),
  )
}

/** Himmelblau's function, four minima of 0 (one at (3, 2)). */
export function himmelblau(v: Value): Value {
  const x = at(v, 0)
  const y = at(v, 1)
  return sum(add(square(sub(add(square(x), y), 11)), square(sub(add(x, square(y)), 7))))
}

/** Ackley's function in n dimensions, global minimum 0 at the origin. */
export function ackley(x: Value): Value {
  const n = x.shape[0] ?? 1
  const a = mul(exp(mul(sqrt(mul(sum(square(x)), 1 / n)), -0.2)), -20)
  const b = mul(exp(mul(sum(cos(mul(x, 2 * Math.PI))), 1 / n)), -1)
  return add(add(a, b), 20 + Math.E)
}

/** Rastrigin's function, global minimum 0 at the origin. */
export function rastrigin(x: Value): Value {
  const n = x.shape[0] ?? 1
  return add(sum(sub(square(x), mul(cos(mul(x, 2 * Math.PI)), 10))), 10 * n)
}

/** The Branin-Hoo function on [-5, 10] x [0, 15], global minimum 0.397887 (at three points). */
export function branin(v: Value): Value {
  const x = at(v, 0)
  const y = at(v, 1)
  const b = 5.1 / (4 * Math.PI * Math.PI)
  const c = 5 / Math.PI
  const t = 1 / (8 * Math.PI)
  const inner = add(sub(y, mul(square(x), b)), sub(mul(x, c), 6))
  return sum(add(square(inner), add(mul(cos(x), 10 * (1 - t)), 10)))
}

/** The six-hump camel function, global minimum -1.0316 at (0.0898, -0.7126) and (-0.0898, 0.7126). */
export function sixHumpCamel(v: Value): Value {
  const x = at(v, 0)
  const y = at(v, 1)
  const x2 = square(x)
  const y2 = square(y)
  const a = mul(add(sub(4, mul(x2, 2.1)), mul(square(x2), 1 / 3)), x2)
  const b = mul(x, y)
  const c = mul(sub(mul(y2, 4), 4), y2)
  return sum(add(add(a, b), c))
}

/** A convex quadratic 0.5 x^T A x - b^T x with A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]], b = [1, 2, 3]. */
export function quadratic(x: Value): Value {
  const a = tensor([
    [4, 1, 0],
    [1, 3, 1],
    [0, 1, 2],
  ])
  const b = tensor([1, 2, 3])
  return sub(mul(sum(mul(x, matmul(a, x))), 0.5), sum(mul(b, x)))
}

/** A sinusoidal valley sin(x) + cos(y) + 0.1 (x^2 + y^2), several local minima. */
export function wavy(v: Value): Value {
  const x = at(v, 0)
  const y = at(v, 1)
  return sum(add(add(sin(x), cos(y)), mul(add(square(x), square(y)), 0.1)))
}

/** (x - 1)^2 + (y - 2.5)^2, the objective of scipy's constrained-minimisation tutorial. */
export function shifted(v: Value): Value {
  return sum(square(sub(v, tensor([1, 2.5]))))
}

/** cos(14.5 x - 0.3) + (x + 0.2) x, scipy's basinhopping example in one dimension (minimum near x = -0.195). */
export function hop1(x: Value): Value {
  return sum(add(cos(sub(mul(x, 14.5), 0.3)), mul(add(x, 0.2), x)))
}

/** cos(14.5 x0 - 0.3) + (x1 + 0.2) x1 + (x0 + 0.2) x0, scipy's basinhopping example in two dimensions. */
export function hop2(v: Value): Value {
  const a = at(v, 0)
  const b = at(v, 1)
  return sum(add(add(cos(sub(mul(a, 14.5), 0.3)), mul(add(b, 0.2), b)), mul(add(a, 0.2), a)))
}

/** The problems by name, as the oracle names them. */
export const PROBLEMS: Readonly<Record<string, (x: Value) => Value>> = {
  rosen,
  beale,
  himmelblau,
  ackley,
  rastrigin,
  branin,
  sixHumpCamel,
  quadratic,
  wavy,
  shifted,
  hop1,
  hop2,
}

/** ZDT1 (Zitzler, Deb and Thiele 2000) on [0, 1]^n: a convex Pareto front f2 = 1 - sqrt(f1). */
export function zdt1(x: Value): Value {
  const n = x.shape[0] ?? 2
  const f1 = sum(at(x, 0))
  const g = add(mul(sum(sliceAxis(x, 0, 1, n)), 9 / (n - 1)), 1)
  return stack([f1, mul(g, sub(1, sqrt(div(f1, g))))])
}

/** ZDT2 on [0, 1]^n: a concave Pareto front f2 = 1 - f1^2. */
export function zdt2(x: Value): Value {
  const n = x.shape[0] ?? 2
  const f1 = sum(at(x, 0))
  const g = add(mul(sum(sliceAxis(x, 0, 1, n)), 9 / (n - 1)), 1)
  return stack([f1, mul(g, sub(1, square(div(f1, g))))])
}

/** The vector (multiobjective) problems by name. */
export const VECTOR_PROBLEMS: Readonly<Record<string, (x: Value) => Value>> = { zdt1, zdt2 }
