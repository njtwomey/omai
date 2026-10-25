/**
 * Multiobjective optimisation basics: Pareto dominance, non-dominated sorting into fronts (NSGA-II's fast sort), Pareto
 * fronts, crowding distances, the hypervolume indicator and each point's contribution to it, and scalarisations that
 * turn several objectives into one (weighted sum, Chebyshev, augmented Chebyshev, epsilon-constraint).
 */

import { MdmlError } from '../base/index.ts'
import {
  add,
  asTensor,
  max,
  mul,
  primalOf,
  sliceAxis,
  sub,
  sum,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { crowdingRows, dominatesRow, hv, sortRows } from './_pareto.ts'
import type { InequalityConstraint } from './minimize/cobyla.ts'
import type { Objective } from './minimize/objective.ts'
import type { Tree } from './transform.ts'

/** A vector objective: the m objectives of a multiobjective problem as one vector of a parameter tree. */
export type VectorObjective = (params: Tree) => Value

/** Rows of a points matrix [n, m] as number arrays. */
function rowsOf(points: TensorLike): number[][] {
  const t = asTensor(points)
  if (t.shape.length === 1 && t.shape[0] === 0) return []
  if (t.shape.length !== 2)
    throw new MdmlError('shape', `expected a matrix of points [n, m], got [${t.shape.join(', ')}]`)
  const [n = 0, m = 0] = t.shape
  const flat = toFlat(primalOf(t))
  return Array.from({ length: n }, (_, i) => flat.slice(i * m, (i + 1) * m))
}

/**
 * Whether objective vector a Pareto-dominates b (minimisation): a is no worse in every objective and better in one.
 *
 * @param a - First objective vector.
 * @param b - Second objective vector.
 * @returns `true` when a dominates b.
 * @concept pareto-optimality
 * @example
 * ```ts
 * import { dominates } from 'mdml-engine/optim'
 * dominates([1, 2], [2, 2]) // => true
 * dominates([1, 3], [2, 2]) // => false
 * ```
 */
export function dominates(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  return dominatesRow(a, b)
}

/**
 * Deb's fast non-dominated sorting (NSGA-II): the points split into fronts, the first being the Pareto front, each
 * later front non-dominated once the earlier ones are removed.
 *
 * @param points - Objective vectors as a matrix [n, m] (minimisation).
 * @returns The fronts as lists of row indices, in increasing order within each front.
 * @concept pareto-optimality
 * @example
 * ```ts
 * import { nonDominatedSort } from 'mdml-engine/optim'
 * nonDominatedSort([[1, 4], [2, 2], [3, 3], [4, 1], [4, 4]]) // => [[0, 1, 3], [2], [4]]
 * ```
 */
export function nonDominatedSort(points: TensorLike): number[][] {
  return sortRows(rowsOf(points))
}

/**
 * The indices of the Pareto-optimal (non-dominated) points.
 *
 * @param points - Objective vectors as a matrix [n, m] (minimisation).
 * @returns Row indices of the first front, ascending.
 * @concept pareto-optimality
 * @example
 * ```ts
 * import { paretoFront } from 'mdml-engine/optim'
 * paretoFront([[1, 4], [2, 2], [3, 3], [4, 1]]) // => [0, 1, 3]
 * ```
 */
export function paretoFront(points: TensorLike): number[] {
  const rows = rowsOf(points)
  return rows.map((_, i) => i).filter((i) => !rows.some((r, j) => j !== i && dominates(r, rows[i] as number[])))
}

/**
 * Crowding distances of the points of one front (Deb et al. 2002): for each objective, the gap between a point's
 * neighbours divided by the objective's range, summed; boundary points get infinity. Larger is less crowded.
 *
 * @param points - The front's objective vectors [n, m].
 * @returns One distance per point.
 * @concept pareto-optimality
 * @example
 * ```ts
 * import { crowdingDistance } from 'mdml-engine/optim'
 * Array.from(crowdingDistance([[0, 4], [1, 2], [3, 1], [4, 0]])) // => [Infinity, 1.5, 1.25, Infinity]
 * ```
 */
export function crowdingDistance(points: TensorLike): Float64Array {
  return crowdingRows(rowsOf(points))
}

/**
 * The hypervolume indicator: the volume of objective space dominated by the points and bounded by a reference point
 * (minimisation). Exact: a sweep in two dimensions, the WFG algorithm (While, Bradstreet and Barone 2012) above.
 *
 * @param points - Objective vectors [n, m].
 * @param reference - The reference point (worse than every point of interest); points not dominating it are ignored.
 * @returns The hypervolume.
 * @concept hypervolume
 * @example
 * ```ts
 * import { hypervolume } from 'mdml-engine/optim'
 * hypervolume([[1, 3], [2, 2], [3, 1]], [4, 4]) // => 6
 * ```
 */
export function hypervolume(points: TensorLike, reference: readonly number[]): number {
  const rows = rowsOf(points)
  if (rows.length > 0 && (rows[0]?.length ?? 0) !== reference.length)
    throw new MdmlError('shape', `hypervolume: reference has ${reference.length} objectives, points ${rows[0]?.length}`)
  const inside = rows.filter((r) => r.every((v, k) => v < (reference[k] ?? 0)))
  const front = inside.filter(
    (r, i) => !inside.some((q, j) => j !== i && (dominates(q, r) || (j < i && q.every((v, k) => v === r[k])))),
  )
  return hv(front, reference)
}

/**
 * Exclusive hypervolume contributions: how much the hypervolume falls when each point is removed (the selection
 * criterion of SMS-EMOA). Dominated points contribute 0.
 *
 * @param points - Objective vectors [n, m].
 * @param reference - The reference point.
 * @returns One contribution per point.
 * @concept hypervolume
 * @example
 * ```ts
 * import { hypervolumeContributions } from 'mdml-engine/optim'
 * Array.from(hypervolumeContributions([[1, 3], [2, 2], [3, 1]], [4, 4])) // => [1, 1, 1]
 * ```
 */
export function hypervolumeContributions(points: TensorLike, reference: readonly number[]): Float64Array {
  const rows = rowsOf(points)
  const all = hypervolume(rows, reference)
  return Float64Array.from(
    rows,
    (_, i) =>
      all -
      hypervolume(
        rows.filter((_, j) => j !== i),
        reference,
      ),
  )
}

const weightsOf = (w: TensorLike): Value => asTensor(w)

/**
 * The weighted-sum scalarisation sum_i w_i f_i(x): its minimisers are Pareto optimal for positive weights, but only
 * points on the convex part of the front can be reached.
 *
 * @param f - The vector objective.
 * @param weights - One weight per objective.
 * @returns A scalar (differentiable) objective.
 * @concept scalarisation
 * @example
 * ```ts
 * import { weightedSum } from 'mdml-engine/optim'
 * import { item, square, stack, sub, sum, type Value } from 'mdml-engine/tensor'
 * const f = (x: unknown) => stack([sum(square(x as Value)), sum(square(sub(x as Value, 2)))])
 * item(weightedSum(f, [0.5, 0.5])([1])) // => 1
 * ```
 */
export function weightedSum(f: VectorObjective, weights: TensorLike): Objective {
  const w = weightsOf(weights)
  return (x) => sum(mul(f(x), w))
}

/**
 * The (weighted) Chebyshev scalarisation max_i w_i (f_i(x) - z_i) from an ideal point z: every Pareto-optimal point,
 * convex front or not, minimises it for some weights.
 *
 * @param f - The vector objective.
 * @param weights - One weight per objective.
 * @param ideal - The ideal (utopia) point z (default 0).
 * @returns A scalar objective (differentiable almost everywhere).
 * @concept scalarisation
 * @example
 * ```ts
 * import { chebyshev } from 'mdml-engine/optim'
 * import { item, stack, sum, type Value } from 'mdml-engine/tensor'
 * const f = (x: unknown) => stack([sum(x as Value), sum(x as Value)])
 * item(chebyshev(f, [1, 2])([3])) // => 6
 * ```
 */
export function chebyshev(f: VectorObjective, weights: TensorLike, ideal: TensorLike = 0): Objective {
  const w = weightsOf(weights)
  const z = asTensor(ideal)
  return (x) => max(mul(sub(f(x), z), w))
}

/**
 * The augmented Chebyshev scalarisation max_i w_i (f_i - z_i) + rho sum_i w_i (f_i - z_i), whose minimisers are
 * properly Pareto optimal (not merely weakly).
 *
 * @param f - The vector objective.
 * @param weights - One weight per objective.
 * @param ideal - The ideal point (default 0).
 * @param rho - The augmentation coefficient (default 1e-4).
 * @returns A scalar objective.
 * @concept scalarisation
 * @example
 * ```ts
 * import { augmentedChebyshev } from 'mdml-engine/optim'
 * import { item, stack, sum, type Value } from 'mdml-engine/tensor'
 * const f = (x: unknown) => stack([sum(x as Value), sum(x as Value)])
 * item(augmentedChebyshev(f, [1, 2], 0, 0.5)([2])) // => 7
 * ```
 */
export function augmentedChebyshev(
  f: VectorObjective,
  weights: TensorLike,
  ideal: TensorLike = 0,
  rho = 1e-4,
): Objective {
  const w = weightsOf(weights)
  const z = asTensor(ideal)
  return (x) => {
    const g = mul(sub(f(x), z), w)
    return add(max(g), mul(sum(g), rho))
  }
}

/**
 * The epsilon-constraint scalarisation: minimise one objective subject to every other objective staying below its
 * bound, f_j(x) <= eps_j. The constraints use the c(x) >= 0 convention of {@link cobyla} and `differentialEvolution`.
 *
 * @param f - The vector objective.
 * @param index - The objective to minimise.
 * @param epsilons - Upper bounds for the objectives, one per objective (the entry at `index` is ignored).
 * @returns The scalar objective and its inequality constraints.
 * @concept scalarisation
 * @example
 * ```ts
 * import { cobyla, epsilonConstraint } from 'mdml-engine/optim'
 * import { square, stack, sub, sum, toFlat, type Value } from 'mdml-engine/tensor'
 * // f1 = x^2, f2 = (x - 2)^2; minimise f1 with f2 <= 1: x = 1
 * const f = (x: unknown) => stack([sum(square(x as Value)), sum(square(sub(x as Value, 2)))])
 * const { objective, constraints } = epsilonConstraint(f, 0, [0, 1])
 * Math.round((toFlat(cobyla(objective, [2], { constraints, rhoend: 1e-9 }).x as Value)[0] ?? 0) * 1e6) / 1e6 // => 1
 * ```
 */
export function epsilonConstraint(
  f: VectorObjective,
  index: number,
  epsilons: readonly number[],
): { readonly objective: Objective; readonly constraints: InequalityConstraint[] } {
  const pick = (v: Value, i: number): Value => sum(sliceAxis(v, 0, i, i + 1))
  const constraints: InequalityConstraint[] = []
  for (const [j, eps] of epsilons.entries()) {
    if (j === index) continue
    constraints.push((x) => sub(tensor(eps), pick(f(x), j)))
  }
  return { objective: (x) => pick(f(x), index), constraints }
}
