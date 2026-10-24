import { grad, hessian } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import { qr } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import { type DictSpace, dict, real } from '../space/index.ts'
import {
  add,
  asTensor,
  cos,
  exp,
  item,
  matmul,
  mean,
  mul,
  reshape,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  transpose,
  type Value,
} from '../tensor/index.ts'

/**
 * An optimisation test function with its gradient and Hessian (by automatic differentiation), known minimisers and
 * minimum, and the usual search box. `f` takes a point `[d]` (giving a scalar) or a batch `[n, d]` (giving `[n]`).
 *
 * @concept optimisation-test-functions
 */
export interface TestFunction {
  readonly name: string
  readonly dim: number
  readonly f: (x: Value) => Value
  readonly gradient: (x: TensorLike) => Value
  readonly hessian: (x: TensorLike) => Value
  /** Every global minimiser. */
  readonly minimizers: readonly (readonly number[])[]
  readonly minimum: number
  readonly bounds: readonly (readonly [number, number])[]
}

/** The `j`-th coordinate of a point `[d]` or of every row of `[n, d]`. */
function coord(x: Value, j: number): Value {
  const last = x.shape.length - 1
  const s = sliceAxis(x, last, j, j + 1)
  return last === 0 ? reshape(s, []) : reshape(s, [x.shape[0] ?? 0])
}

function sumLast(v: Value): Value {
  return sum(v, v.shape.length - 1)
}

function testFunction(
  name: string,
  dim: number,
  f: (x: Value) => Value,
  minimizers: readonly (readonly number[])[],
  minimum: number,
  bounds: readonly (readonly [number, number])[],
): TestFunction {
  const scalar = (x: Value): Value => f(x)
  return {
    name,
    dim,
    f,
    gradient: (x) => grad(scalar)(asTensor(x)),
    hessian: (x) => hessian(scalar)(x),
    minimizers,
    minimum,
    bounds,
  }
}

const box = (d: number, lo: number, hi: number): [number, number][] => Array.from({ length: d }, () => [lo, hi])

/**
 * The Rosenbrock function `sum 100 (x_{i+1} - x_i²)² + (1 - x_i)²`, minimised at all ones with value 0.
 *
 * @param dim - The dimension (default 2).
 * @returns The test function.
 * @example
 * ```ts
 * import { rosenbrock } from 'whitebox-engine/datasets'
 * import { item, tensor } from 'whitebox-engine/tensor'
 * item(rosenbrock().f(tensor([0, 0]))) // => 1
 * ```
 */
export function rosenbrock(dim = 2): TestFunction {
  if (dim < 2) throw new WhiteboxError('value', 'rosenbrock: needs at least two dimensions')
  const f = (x: Value): Value => {
    const last = x.shape.length - 1
    const head = sliceAxis(x, last, 0, dim - 1)
    const tail = sliceAxis(x, last, 1, dim)
    return sumLast(add(mul(square(sub(tail, square(head))), 100), square(sub(1, head))))
  }
  return testFunction('rosenbrock', dim, f, [new Array<number>(dim).fill(1)], 0, box(dim, -2, 2))
}

/**
 * Himmelblau's function `(x² + y - 11)² + (x + y² - 7)²`, with four global minima of value 0.
 *
 * @returns The test function.
 * @example
 * ```ts
 * import { himmelblau } from 'whitebox-engine/datasets'
 * himmelblau().minimizers.length // => 4
 * ```
 */
export function himmelblau(): TestFunction {
  const f = (x: Value): Value => {
    const a = coord(x, 0)
    const b = coord(x, 1)
    return add(square(sub(add(square(a), b), 11)), square(sub(add(a, square(b)), 7)))
  }
  return testFunction(
    'himmelblau',
    2,
    f,
    [
      [3, 2],
      [-2.805118086952745, 3.1313125182505726],
      [-3.779310253377747, -3.2831859912861696],
      [3.5844283403304917, -1.8481265269644034],
    ],
    0,
    box(2, -5, 5),
  )
}

/**
 * Beale's function `(1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²`, minimised at `(3, 0.5)` with value 0.
 *
 * @returns The test function.
 * @example
 * ```ts
 * import { beale } from 'whitebox-engine/datasets'
 * import { item, tensor } from 'whitebox-engine/tensor'
 * item(beale().f(tensor([3, 0.5]))) // => 0
 * ```
 */
export function beale(): TestFunction {
  const f = (x: Value): Value => {
    const a = coord(x, 0)
    const b = coord(x, 1)
    const t = (c: number, p: number) =>
      square(add(sub(c, a), mul(a, p === 1 ? b : p === 2 ? square(b) : mul(square(b), b))))
    return add(add(t(1.5, 1), t(2.25, 2)), t(2.625, 3))
  }
  return testFunction('beale', 2, f, [[3, 0.5]], 0, box(2, -4.5, 4.5))
}

/**
 * The Rastrigin function `10 d + sum (x_i² - 10 cos(2π x_i))`, highly multimodal, minimised at the origin with value 0.
 *
 * @param dim - The dimension (default 2).
 * @returns The test function.
 * @example
 * ```ts
 * import { rastrigin } from 'whitebox-engine/datasets'
 * import { item, tensor } from 'whitebox-engine/tensor'
 * item(rastrigin(3).f(tensor([0, 0, 0]))) // => 0
 * ```
 */
export function rastrigin(dim = 2): TestFunction {
  const f = (x: Value): Value => add(sumLast(sub(square(x), mul(cos(mul(x, 2 * Math.PI)), 10))), 10 * dim)
  return testFunction('rastrigin', dim, f, [new Array<number>(dim).fill(0)], 0, box(dim, -5.12, 5.12))
}

/**
 * The Ackley function `-20 exp(-0.2 sqrt(mean x²)) - exp(mean cos 2πx) + 20 + e`, minimised at the origin with value 0.
 *
 * @param dim - The dimension (default 2).
 * @returns The test function.
 * @example
 * ```ts
 * import { ackley } from 'whitebox-engine/datasets'
 * import { item, tensor } from 'whitebox-engine/tensor'
 * Math.abs(item(ackley().f(tensor([0, 0])))) < 1e-12 // => true
 * ```
 */
export function ackley(dim = 2): TestFunction {
  const f = (x: Value): Value => {
    const last = x.shape.length - 1
    const r = sqrt(add(mean(square(x), last), 1e-300))
    return add(sub(mul(exp(mul(r, -0.2)), -20), exp(mean(cos(mul(x, 2 * Math.PI)), last))), 20 + Math.E)
  }
  return testFunction('ackley', dim, f, [new Array<number>(dim).fill(0)], 0, box(dim, -32.768, 32.768))
}

/**
 * The Branin-Hoo function `(x2 - b x1² + c x1 - 6)² + 10 (1 - t) cos x1 + 10` (`b = 5.1 / 4π²`, `c = 5 / π`,
 * `t = 1 / 8π`) on `[-5, 10] × [0, 15]`, with three global minima of value `0.397887...`, the classic
 * hyperparameter-optimisation benchmark.
 *
 * @returns The test function.
 * @example
 * ```ts
 * import { branin } from 'whitebox-engine/datasets'
 * branin().minimizers.length // => 3
 * ```
 */
export function branin(): TestFunction {
  const b = 5.1 / (4 * Math.PI * Math.PI)
  const c = 5 / Math.PI
  const t = 1 / (8 * Math.PI)
  const f = (x: Value): Value => {
    const x1 = coord(x, 0)
    const x2 = coord(x, 1)
    return add(add(square(sub(add(sub(x2, mul(square(x1), b)), mul(x1, c)), 6)), mul(cos(x1), 10 * (1 - t))), 10)
  }
  return testFunction(
    'branin',
    2,
    f,
    [
      [-Math.PI, 12.275],
      [Math.PI, 2.275],
      [9.42478, 2.475],
    ],
    0.39788735772973816,
    [
      [-5, 10],
      [0, 15],
    ],
  )
}

/** Options of {@link quadraticBowl}. */
export interface QuadraticOptions {
  readonly dim?: number
  /** The condition number of the Hessian (default 10). */
  readonly condition?: number
  /** A key for a random rotation of the axes (default: axis-aligned). */
  readonly key?: Key
  /** The minimiser (default the origin). */
  readonly center?: readonly number[]
}

/**
 * A convex quadratic bowl `½ (x - c)ᵀ A (x - c)` whose Hessian `A` has eigenvalues log-spaced from 1 to `condition`
 * (rotated by a random orthogonal matrix when a key is given), for studying how conditioning slows first-order
 * methods. The truth includes `A`.
 *
 * @param options - Dimension, condition number, rotation key and centre.
 * @returns The test function and its Hessian matrix.
 * @example
 * ```ts
 * import { quadraticBowl } from 'whitebox-engine/datasets'
 * import { item, tensor } from 'whitebox-engine/tensor'
 * item(quadraticBowl({ dim: 2, condition: 4 }).f(tensor([1, 1]))) // => 2.5
 * ```
 */
export function quadraticBowl(options: QuadraticOptions = {}): TestFunction & { readonly matrix: Tensor } {
  const d = options.dim ?? 2
  const kappa = options.condition ?? 10
  const eig = Array.from({ length: d }, (_, i) => (d === 1 ? 1 : kappa ** (i / (d - 1))))
  let a: Tensor
  if (options.key === undefined) a = tensor(eig.map((e, i) => eig.map((_, j) => (i === j ? e : 0))))
  else {
    const q = qr(normal(options.key, [d, d])).q
    a = matmul(mul(q, tensor(eig)), transpose(q)) as Tensor
  }
  const centre = options.center ?? new Array<number>(d).fill(0)
  const c = tensor(centre as number[])
  const f = (x: Value): Value => {
    const r = sub(x, c)
    const ar = x.shape.length === 1 ? matmul(a, r) : matmul(r, a)
    return mul(sumLast(mul(r, ar)), 0.5)
  }
  return { ...testFunction('quadraticBowl', d, f, [centre], 0, box(d, -5, 5)), matrix: a }
}

/** A synthetic tuning problem: a space of hyperparameters and a (possibly noisy, multi-fidelity) loss. */
export interface TuningSurface {
  readonly name: string
  readonly space: DictSpace
  /** The expected loss of a configuration at a budget (the noise-free surface). */
  readonly mean: (params: Readonly<Record<string, number>>, budget?: number) => number
  /** One noisy evaluation. */
  readonly evaluate: (params: Readonly<Record<string, number>>, key: Key, budget?: number) => number
  readonly optimum: Readonly<Record<string, number>>
  readonly optimalValue: number
  readonly noise: number
  readonly maxBudget: number
}

/**
 * A synthetic validation-loss surface over a log learning rate and a log weight decay, shaped like real tuning
 * problems: a tilted valley in `log10(lr)` with an interaction term, an optimum at `lr = 3e-3`, `wd = 1e-4`, a
 * learning curve in the budget (epochs) `loss = asymptote + 1.5 / budget^0.7`, and Gaussian evaluation noise. Its
 * optimum is known, so tuners and multi-fidelity schedulers can be scored.
 *
 * @param noise - The standard deviation of evaluation noise (default 0.02).
 * @returns The surface.
 * @example
 * ```ts
 * import { learningRateSurface } from 'whitebox-engine/datasets'
 * const s = learningRateSurface()
 * s.mean(s.optimum, s.maxBudget) === s.optimalValue // => true
 * ```
 */
export function learningRateSurface(noise = 0.02): TuningSurface {
  const lr0 = Math.log10(3e-3)
  const wd0 = Math.log10(1e-4)
  const maxBudget = 100
  const curve = (budget: number) => 1.5 / budget ** 0.7
  const asymptote = (p: Readonly<Record<string, number>>) => {
    const u = Math.log10(p['learningRate'] ?? 1e-3) - lr0
    const v = Math.log10(p['weightDecay'] ?? 1e-4) - wd0
    return 0.25 + 0.3 * u * u + 0.05 * v * v + 0.08 * u * v + 0.02 * u ** 4
  }
  const mean = (p: Readonly<Record<string, number>>, budget = maxBudget) => asymptote(p) + curve(budget)
  const optimum = { learningRate: 3e-3, weightDecay: 1e-4 }
  return {
    name: 'learningRateSurface',
    space: dict({
      learningRate: real(1e-5, 1, { log: true, label: 'learning rate' }),
      weightDecay: real(1e-7, 1e-1, { log: true, label: 'weight decay' }),
    }),
    mean,
    evaluate: (p, key, budget = maxBudget) =>
      mean(p, budget) + noise * (normal(foldIn(key, Math.round(budget)), [1]).data[0] ?? 0),
    optimum,
    optimalValue: mean(optimum, maxBudget),
    noise,
    maxBudget,
  }
}

/**
 * The Branin function as a tuning surface over its box (`x1`, `x2`), with optional evaluation noise.
 *
 * @param noise - Evaluation noise standard deviation (default 0).
 * @returns The surface.
 * @example
 * ```ts
 * import { braninSurface } from 'whitebox-engine/datasets'
 * braninSurface().optimalValue // => 0.39788735772973816
 * ```
 */
export function braninSurface(noise = 0): TuningSurface {
  const fn = branin()
  const mean = (p: Readonly<Record<string, number>>) => item(fn.f(tensor([p['x1'] ?? 0, p['x2'] ?? 0])))
  return {
    name: 'braninSurface',
    space: dict({ x1: real(-5, 10), x2: real(0, 15) }),
    mean,
    evaluate: (p, key) => mean(p) + noise * (normal(key, [1]).data[0] ?? 0),
    optimum: { x1: Math.PI, x2: 2.275 },
    optimalValue: fn.minimum,
    noise,
    maxBudget: 1,
  }
}
