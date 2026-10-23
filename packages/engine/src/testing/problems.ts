// Standard smooth test problems for the optim tests, the same as oracle/whitebox_oracle/optim_problems.py. Not published.
import { add, matmul, mul, pow, sliceAxis, square, sub, sum, tensor, type Value } from '../lib/tensor/index.ts'

const at = (x: Value, i: number): Value => sliceAxis(x, 0, i, i + 1)

/** The Rosenbrock function in n dimensions, sum 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2. */
export function rosen(x: Value): Value {
  const n = x.shape[0] ?? 0
  const head = sliceAxis(x, 0, 0, n - 1)
  const tail = sliceAxis(x, 0, 1, n)
  return add(sum(mul(square(sub(tail, square(head))), 100)), sum(square(sub(1, head))))
}

/** Beale's function. */
export function beale(v: Value): Value {
  const x = at(v, 0)
  const y = at(v, 1)
  const term = (c: number, k: number): Value => square(add(sub(c, x), mul(x, pow(y, k))))
  return sum(add(add(term(1.5, 1), term(2.25, 2)), term(2.625, 3)))
}

/** Himmelblau's function. */
export function himmelblau(v: Value): Value {
  const x = at(v, 0)
  const y = at(v, 1)
  return sum(add(square(sub(add(square(x), y), 11)), square(sub(add(x, square(y)), 7))))
}

/** The quadratic 0.5 x'Ax - b'x of the fixtures. */
export const QUADRATIC_A: Value = tensor([
  [4, 1, 0.5],
  [1, 3, -0.5],
  [0.5, -0.5, 2],
])
/** The linear term of {@link quadratic}. */
export const QUADRATIC_B: Value = tensor([1, -2, 0.5])

/** 0.5 x'Ax - b'x. */
export function quadratic(x: Value): Value {
  return sub(mul(sum(mul(x, matmul(QUADRATIC_A, x))), 0.5), sum(mul(QUADRATIC_B, x)))
}

/** Powell's singular function. */
export function powell(v: Value): Value {
  const [a, b, c, d] = [0, 1, 2, 3].map((i) => at(v, i)) as [Value, Value, Value, Value]
  return sum(
    add(
      add(square(add(a, mul(b, 10))), mul(square(sub(c, d)), 5)),
      add(pow(sub(b, mul(c, 2)), 4), mul(pow(sub(a, d), 4), 10)),
    ),
  )
}

/** Wood's four-variable function. */
export function wood(v: Value): Value {
  const [a, b, c, d] = [0, 1, 2, 3].map((i) => at(v, i)) as [Value, Value, Value, Value]
  return sum(
    add(
      add(
        add(mul(square(sub(b, square(a))), 100), square(sub(1, a))),
        add(mul(square(sub(d, square(c))), 90), square(sub(1, c))),
      ),
      add(mul(add(square(sub(b, 1)), square(sub(d, 1))), 10.1), mul(mul(sub(b, 1), sub(d, 1)), 19.8)),
    ),
  )
}

/** The problems by name, as the fixtures know them. */
export const PROBLEMS: Readonly<Record<string, (x: Value) => Value>> = {
  rosen,
  rosen5: rosen,
  beale,
  himmelblau,
  quadratic,
  powell,
  wood,
}

/** The problem of a fixture case by name. */
export function problem(name: string): (x: Value) => Value {
  const f = PROBLEMS[name]
  if (f === undefined) throw new Error(`unknown test problem ${name}`)
  return f
}
