/**
 * NSGA-II (Deb et al. 2002), the standard evolutionary algorithm for approximating a Pareto front: tournaments by rank
 * and crowding distance, simulated binary crossover and polynomial mutation within bounds, and elitist survival, as a
 * traced algorithm whose populations can be shown generation by generation.
 */

import { MdmlError } from '../base/index.ts'
import { Generator, type Key, latinHypercube } from '../random/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat, toVector } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, type RunStatus, runAlgorithm } from '../trace/index.ts'
import { crowdingRows, dominatesRow, hv, sortRows } from './_pareto.ts'
import { type BoundArrays, finiteBounds } from './minimize/_bounds.ts'
import { configOf } from './minimize/_shared.ts'
import type { InequalityConstraint } from './minimize/cobyla.ts'
import type { Bounds } from './minimize/objective.ts'
import { flatObjective, type Parameters } from './minimize/objective.ts'
import type { VectorObjective } from './multiobjective.ts'
import type { ParamTree } from './transform.ts'

/** Options of {@link sbxCrossover}. */
export interface SbxOptions {
  /** The distribution index eta_c: larger keeps children nearer their parents (default 15). */
  readonly eta?: number
  /** The probability of crossing a pair at all (default 0.9). */
  readonly probability?: number
  /** The probability of crossing each variable of a crossed pair (default 0.5). */
  readonly variableProbability?: number
}

/**
 * Simulated binary crossover (Deb and Agrawal 1995) with bounds, as in Deb's NSGA-II code: each variable of a crossed
 * pair is recombined with a spread factor drawn from a polynomial distribution that respects the box. Takes the parents
 * as tensor-likes and returns tensors (migration, D-009: they were `ArrayLike<number>` and `Float64Array`s).
 *
 * @param key - The random key.
 * @param a - First parent: a vector.
 * @param b - Second parent: a vector.
 * @param bounds - Box bounds of the variables.
 * @param options - Distribution index and probabilities.
 * @returns The two children, `[n]` tensors.
 * @throws MdmlError with code `shape` unless a parent has rank 1, and `not-differentiable` for a traced one.
 * @concept genetic-operators
 * @example
 * ```ts
 * import { sbxCrossover } from 'mdml-engine/optim'
 * import { key } from 'mdml-engine/random'
 * import { toFlat } from 'mdml-engine/tensor'
 * const [c1, c2] = sbxCrossover(key(0), [0, 1], [1, 0], [[0, 1], [0, 1]], { probability: 1 })
 * toFlat(c1).every((v) => v >= 0 && v <= 1) && toFlat(c2).every((v) => v >= 0 && v <= 1) // => true
 * ```
 */
export function sbxCrossover(
  key: Key,
  a: TensorLike,
  b: TensorLike,
  bounds: Bounds,
  options: SbxOptions = {},
): [Tensor<'float64', [number]>, Tensor<'float64', [number]>] {
  const pa = toVector(a, 'sbxCrossover')
  const pb = toVector(b, 'sbxCrossover')
  const [c1, c2] = sbx(new Generator(key), pa, pb, finiteBounds(bounds, pa.length, 'sbxCrossover'), options)
  return [fromStorage(c1, [c1.length], 'float64'), fromStorage(c2, [c2.length], 'float64')]
}

function sbx(
  rng: Generator,
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  box: BoundArrays,
  options: SbxOptions,
): [Float64Array, Float64Array] {
  const n = a.length
  const eta = options.eta ?? 15
  const pc = options.probability ?? 0.9
  const pv = options.variableProbability ?? 0.5
  const c1 = Float64Array.from(a)
  const c2 = Float64Array.from(b)
  const u = rng.uniform([1 + 3 * n]).data
  if ((u[0] ?? 1) > pc) return [c1, c2]
  for (let i = 0; i < n; i++) {
    if ((u[1 + 3 * i] ?? 1) > pv) continue
    const p1 = a[i] ?? 0
    const p2 = b[i] ?? 0
    if (Math.abs(p1 - p2) <= 1e-14) continue
    const y1 = Math.min(p1, p2)
    const y2 = Math.max(p1, p2)
    const yl = box.lower[i] ?? 0
    const yu = box.upper[i] ?? 0
    const r = u[2 + 3 * i] ?? 0
    const betaq = (beta: number): number => {
      const alpha = 2 - beta ** -(eta + 1)
      return r <= 1 / alpha ? (r * alpha) ** (1 / (eta + 1)) : (1 / (2 - r * alpha)) ** (1 / (eta + 1))
    }
    let x1 = 0.5 * (y1 + y2 - betaq(1 + (2 * (y1 - yl)) / (y2 - y1)) * (y2 - y1))
    let x2 = 0.5 * (y1 + y2 + betaq(1 + (2 * (yu - y2)) / (y2 - y1)) * (y2 - y1))
    x1 = Math.min(Math.max(x1, yl), yu)
    x2 = Math.min(Math.max(x2, yl), yu)
    if ((u[3 + 3 * i] ?? 0) <= 0.5) {
      c1[i] = x2
      c2[i] = x1
    } else {
      c1[i] = x1
      c2[i] = x2
    }
  }
  return [c1, c2]
}

/** Options of {@link polynomialMutation}. */
export interface PolynomialMutationOptions {
  /** The distribution index eta_m: larger makes smaller perturbations (default 20). */
  readonly eta?: number
  /** The probability of mutating each variable (default 1 / n). */
  readonly probability?: number
}

/**
 * Polynomial mutation with bounds (Deb and Goyal 1996, the bounded form of Deb's NSGA-II code): each variable moves by a
 * polynomially distributed fraction of its range, never leaving the box. Takes the individual as a tensor-like and
 * returns a tensor (migration, D-009: was `ArrayLike<number>` in, `Float64Array` out).
 *
 * @param key - The random key.
 * @param x - The individual: a vector.
 * @param bounds - Box bounds.
 * @param options - Distribution index and per-variable probability.
 * @returns The mutated individual, an `[n]` tensor.
 * @throws MdmlError with code `shape` unless `x` has rank 1, and `not-differentiable` for a traced one.
 * @concept genetic-operators
 * @example
 * ```ts
 * import { polynomialMutation } from 'mdml-engine/optim'
 * import { key } from 'mdml-engine/random'
 * import { toFlat } from 'mdml-engine/tensor'
 * const y = toFlat(polynomialMutation(key(1), [0.5, 0.5], [[0, 1], [0, 1]], { probability: 1 }))
 * y.every((v) => v >= 0 && v <= 1) && (y[0] !== 0.5 || y[1] !== 0.5) // => true
 * ```
 */
export function polynomialMutation(
  key: Key,
  x: TensorLike,
  bounds: Bounds,
  options: PolynomialMutationOptions = {},
): Tensor<'float64', [number]> {
  const individual = toVector(x, 'polynomialMutation')
  const y = mutate(
    new Generator(key),
    individual,
    finiteBounds(bounds, individual.length, 'polynomialMutation'),
    options,
  )
  return fromStorage(y, [y.length], 'float64')
}

function mutate(
  rng: Generator,
  x: ArrayLike<number>,
  box: BoundArrays,
  options: PolynomialMutationOptions,
): Float64Array {
  const n = x.length
  const eta = options.eta ?? 20
  const pm = options.probability ?? 1 / n
  const u = rng.uniform([2 * n]).data
  const y = Float64Array.from(x)
  const pow = 1 / (eta + 1)
  for (let i = 0; i < n; i++) {
    if ((u[2 * i] ?? 1) > pm) continue
    const yl = box.lower[i] ?? 0
    const yu = box.upper[i] ?? 0
    const v = y[i] ?? 0
    const d1 = (v - yl) / (yu - yl)
    const d2 = (yu - v) / (yu - yl)
    const r = u[2 * i + 1] ?? 0
    let dq: number
    if (r <= 0.5) {
      const val = 2 * r + (1 - 2 * r) * (1 - d1) ** (eta + 1)
      dq = val ** pow - 1
    } else {
      const val = 2 * (1 - r) + 2 * (r - 0.5) * (1 - d2) ** (eta + 1)
      dq = 1 - val ** pow
    }
    y[i] = Math.min(Math.max(v + dq * (yu - yl), yl), yu)
  }
  return y
}

/** Options of {@link nsga2}. */
export interface Nsga2Options extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** Box bounds of the variables (required, finite). */
  readonly bounds: Bounds
  /** Population size (default 100; rounded up to an even number). */
  readonly popSize?: number
  /** Simulated binary crossover settings. */
  readonly crossover?: SbxOptions
  /** Polynomial mutation settings. */
  readonly mutation?: PolynomialMutationOptions
  /** Inequality constraints c(x) >= 0, handled by Deb's constrained domination. */
  readonly constraints?: readonly InequalityConstraint[]
  /** A reference point: the trace then records the hypervolume of the first front each generation. */
  readonly reference?: readonly number[]
  /** Generations (default 250). */
  readonly maxIterations?: number
}

/** One individual: variables, objectives and total constraint violation. */
interface Individual {
  readonly x: Float64Array
  readonly f: Float64Array
  readonly cv: number
}

/** State of an NSGA-II run. */
export interface Nsga2State {
  readonly population: readonly Individual[]
  /** Non-domination rank and crowding distance of each member. */
  readonly rank: Int32Array
  readonly crowding: Float64Array
  readonly nfev: number
  readonly generation: number
}

/** Result of {@link nsga2}. */
export interface Nsga2Result {
  /** The final population's variables [N, n] and objectives [N, m]. */
  readonly population: Tensor
  readonly objectives: Tensor
  /** Indices of the first (non-dominated, feasible) front within the population. */
  readonly front: readonly number[]
  /** The Pareto set as parameter trees, and the Pareto front [k, m]. */
  readonly paretoSet: readonly ParamTree[]
  readonly paretoFront: Tensor
  readonly generations: number
  readonly nfev: number
  readonly status: RunStatus
}

function rankAndCrowd(pop: readonly Individual[]): { rank: Int32Array; crowding: Float64Array; fronts: number[][] } {
  const rows = pop.map((p) => Array.from(p.f))
  const fronts = sortRows(rows, (i, j) => constrainedDominates(pop[i] as Individual, pop[j] as Individual))
  const rank = new Int32Array(pop.length)
  const crowding = new Float64Array(pop.length)
  for (const [r, front] of fronts.entries()) {
    const d = crowdingRows(front.map((i) => rows[i] as number[]))
    for (const [k, i] of front.entries()) {
      rank[i] = r
      crowding[i] = d[k] ?? 0
    }
  }
  return { rank, crowding, fronts }
}

/** Deb's constrained domination: feasible beats infeasible, less violation beats more, else Pareto dominance. */
function constrainedDominates(a: Individual, b: Individual): boolean {
  if (a.cv <= 0 && b.cv > 0) return true
  if (a.cv > 0 && b.cv > 0) return a.cv < b.cv
  if (a.cv > 0) return false
  return dominatesRow(a.f, b.f)
}

/**
 * NSGA-II (Deb, Pratap, Agarwal and Meyarivan 2002) as a traced algorithm: binary tournaments by rank and crowding
 * distance, simulated binary crossover and polynomial mutation, then elitist survival of the best N of parents and
 * offspring by non-dominated sorting and crowding. Constraints use constrained domination. Each step is one
 * generation; the trace records the size of the first front, its hypervolume (given a reference point) and the
 * objectives.
 *
 * @param f - The vector objective (minimise every component).
 * @param x0 - A point fixing the parameter structure (it joins the initial population).
 * @param options - Bounds, population, operators, constraints and reference point.
 * @returns The algorithm.
 * @concept nsga-ii
 * @example
 * ```ts
 * import { nsga2Algorithm } from 'mdml-engine/optim'
 * import { square, stack, sub, sum, type Value } from 'mdml-engine/tensor'
 * const f = (x: unknown) => stack([sum(square(x as Value)), sum(square(sub(x as Value, 2)))])
 * nsga2Algorithm(f, [0], { bounds: [[-5, 5]] }).name // => 'nsga2'
 * ```
 */
export function nsga2Algorithm(f: VectorObjective, x0: Parameters, options: Nsga2Options): Algorithm<Nsga2State> {
  const shape = flatObjective(() => fromStorage(new Float64Array([0]), [], 'float64'), x0)
  const n = shape.n
  if (n === 0) throw new MdmlError('value', 'nsga2: no parameters')
  const box = finiteBounds(options.bounds, n, 'nsga2')
  const size = 2 * Math.ceil((options.popSize ?? 100) / 2)
  if (size < 4) throw new MdmlError('value', 'nsga2: popSize must be at least 4')
  const constraints = options.constraints ?? []
  const evaluate = (x: Float64Array): Individual => {
    const tree = shape.unravel(x)
    let cv = 0
    for (const c of constraints) for (const v of toFlat(c(tree))) cv += Math.max(0, -v)
    return { x, f: Float64Array.from(toFlat(f(tree))), cv }
  }
  const reference = options.reference
  const frontRows = (s: Nsga2State): number[][] =>
    s.population.filter((p, i) => s.rank[i] === 0 && p.cv <= 0).map((p) => Array.from(p.f))
  return {
    name: 'nsga2',
    config: configOf({
      size,
      eta_c: options.crossover?.eta ?? 15,
      eta_m: options.mutation?.eta ?? 20,
      constraints: constraints.length,
    }),
    schema: channels({
      metrics: {
        frontSize: { description: 'members of the first front' },
        hypervolume: { description: 'hypervolume of the first front from the reference point', direction: 'higher' },
        feasible: { description: 'feasible members' },
      },
      tensors: {
        objectives: { role: 'state', description: 'objective vectors of the population [N, m]' },
        population: { role: 'param', description: 'variables of the population [N, n]' },
      },
    }),
    init: (key) => {
      const lhs = latinHypercube(key, size, n).data
      const members = Array.from({ length: size }, (_, k) =>
        k === 0
          ? Float64Array.from(shape.x0, (v, i) => Math.min(Math.max(v, box.lower[i] ?? 0), box.upper[i] ?? 0))
          : Float64Array.from(
              { length: n },
              (_, i) => (box.lower[i] ?? 0) + (lhs[k * n + i] ?? 0) * ((box.upper[i] ?? 0) - (box.lower[i] ?? 0)),
            ),
      )
      const population = members.map(evaluate)
      const { rank, crowding } = rankAndCrowd(population)
      return { population, rank, crowding, nfev: size, generation: 0 }
    },
    step: (s, ctx) => {
      const rng = new Generator(ctx.key)
      const picks = rng.integers(0, size, [2 * size]).data
      const better = (i: number, j: number): number =>
        (s.rank[i] ?? 0) < (s.rank[j] ?? 0) ||
        ((s.rank[i] ?? 0) === (s.rank[j] ?? 0) && (s.crowding[i] ?? 0) > (s.crowding[j] ?? 0))
          ? i
          : j
      const offspring: Individual[] = []
      for (let k = 0; k < size; k += 2) {
        const pa = s.population[better(picks[2 * k] ?? 0, picks[2 * k + 1] ?? 0)] as Individual
        const pb = s.population[better(picks[2 * k + 2] ?? 0, picks[2 * k + 3] ?? 0)] as Individual
        const [c1, c2] = sbx(rng, pa.x, pb.x, box, options.crossover ?? {})
        offspring.push(evaluate(mutate(rng, c1, box, options.mutation ?? {})))
        offspring.push(evaluate(mutate(rng, c2, box, options.mutation ?? {})))
      }
      const merged = [...s.population, ...offspring]
      const { fronts } = rankAndCrowd(merged)
      const next: Individual[] = []
      for (const front of fronts) {
        if (next.length + front.length <= size) {
          for (const i of front) next.push(merged[i] as Individual)
          continue
        }
        const d = crowdingRows(front.map((i) => Array.from((merged[i] as Individual).f)))
        const order = front.map((_, k) => k).sort((a, b) => (d[b] ?? 0) - (d[a] ?? 0) || a - b)
        for (const k of order.slice(0, size - next.length)) next.push(merged[front[k] ?? 0] as Individual)
        break
      }
      const { rank, crowding } = rankAndCrowd(next)
      return { population: next, rank, crowding, nfev: s.nfev + size, generation: s.generation + 1 }
    },
    observe: (s) => {
      const m = s.population[0]?.f.length ?? 0
      const front = frontRows(s)
      const objectives = new Float64Array(size * m)
      const variables = new Float64Array(size * n)
      for (const [k, p] of s.population.entries()) {
        objectives.set(p.f, k * m)
        variables.set(p.x, k * n)
      }
      const feasibleRows = front.filter((r) => reference === undefined || r.every((v, k) => v < (reference[k] ?? 0)))
      return {
        metrics: {
          frontSize: front.length,
          feasible: s.population.filter((p) => p.cv <= 0).length,
          ...(reference === undefined
            ? {}
            : {
                hypervolume: hv(
                  feasibleRows.filter(
                    (r, i) =>
                      !feasibleRows.some(
                        (q, j) => j !== i && (dominatesRow(q, r) || (j < i && q.every((v, k) => v === r[k]))),
                      ),
                  ),
                  reference,
                ),
              }),
        },
        tensors: {
          objectives: { role: 'state', value: objectives, shape: [size, m] },
          population: { role: 'param', value: variables, shape: [size, n] },
        },
      }
    },
  }
}

/**
 * Approximate the Pareto front of a multiobjective problem by NSGA-II. `maxIterations` counts generations (default
 * 250).
 *
 * @param f - The vector objective (minimise every component).
 * @param x0 - A point fixing the parameter structure.
 * @param options - Bounds (required), population, operators, constraints, reference point, key or seed and run options.
 * @returns The final population, its first front and the Pareto set as parameter trees.
 * @concept nsga-ii
 * @example
 * ```ts
 * import { nsga2 } from 'mdml-engine/optim'
 * import { square, stack, sub, sum, toArray, type Value } from 'mdml-engine/tensor'
 * // Schaffer's problem: f1 = x^2, f2 = (x - 2)^2, Pareto set [0, 2]
 * const f = (x: unknown) => stack([sum(square(x as Value)), sum(square(sub(x as Value, 2)))])
 * const r = nsga2(f, [0], { bounds: [[-5, 5]], popSize: 20, maxIterations: 30, seed: 0 })
 * r.paretoSet.every((x) => { const v = (toArray(x as Value) as number[])[0] ?? -1; return v >= -1e-3 && v <= 2.001 }) // => true
 * ```
 */
export function nsga2(f: VectorObjective, x0: Parameters, options: Nsga2Options): Nsga2Result {
  const shape = flatObjective(() => fromStorage(new Float64Array([0]), [], 'float64'), x0)
  const { maxIterations, ...run } = options
  const result = runAlgorithm(nsga2Algorithm(f, x0, options), { ...run, maxSteps: maxIterations ?? 250 })
  const pop = result.state.population
  const n = shape.n
  const m = pop[0]?.f.length ?? 0
  const front = pop.map((_, i) => i).filter((i) => result.state.rank[i] === 0 && (pop[i] as Individual).cv <= 0)
  const vars = new Float64Array(pop.length * n)
  const objs = new Float64Array(pop.length * m)
  for (const [k, p] of pop.entries()) {
    vars.set(p.x, k * n)
    objs.set(p.f, k * m)
  }
  const frontObjs = new Float64Array(front.length * m)
  for (const [k, i] of front.entries()) frontObjs.set((pop[i] as Individual).f, k * m)
  return {
    population: fromStorage(vars, [pop.length, n], 'float64'),
    objectives: fromStorage(objs, [pop.length, m], 'float64'),
    front,
    paretoSet: front.map((i) => shape.unravel((pop[i] as Individual).x)),
    paretoFront: fromStorage(frontObjs, [front.length, m], 'float64'),
    generations: result.steps,
    nfev: result.state.nfev,
    status: result.status,
  }
}
