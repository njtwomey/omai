/**
 * Weight learning for Markov logic networks. The generative objective is the pseudo-log-likelihood (Besag 1975), the
 * sum over ground atoms of the log conditional probability of the atom given the rest of the world, which is a
 * logistic regression whose features are the changes in each formula's true-grounding count when the atom flips: smooth
 * and convex in the weights, consistent, and needing no partition function. `mlnPseudoLikelihoodProblem` states it as a
 * `SmoothProblem` for the solver slot (`smoothSolvers`, L-BFGS by default). For tiny domains the exact likelihood and
 * the exact conditional likelihood are also stated, as the truth the approximations are measured against. The
 * discriminative learner is the voted perceptron of Singla and Domingos (2005): the weights move by the difference
 * between the formulas' counts in the training world and in the MAP world (or their exact expectation) given the
 * evidence, and the average of the weights over the steps is returned.
 */

import { type Json, OmaiError } from '../base/index.ts'
import {
  analyticObjective,
  type GradientPath,
  gradientContract,
  loggedSolver,
  type SmoothProblem,
  type SolverPath,
  smoothSolverConfig,
  solverContract,
} from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { type GroundMln, mlnCounts, mlnMap, type World, withWeights, worldVector } from './markov-logic.ts'

const logistic = (z: number): number => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)))
const softplus = (z: number): number => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)))

function checkLearnable(ground: GroundMln, who: string): void {
  if (ground.weights.some((w) => !Number.isFinite(w)))
    throw new OmaiError('value', `${who}: hard formulas have no weight to learn`)
}

function startWeights(ground: GroundMln, initial: readonly number[] | undefined, who: string): number[] {
  if (initial === undefined) return new Array<number>(ground.weights.length).fill(0)
  if (initial.length !== ground.weights.length)
    throw new OmaiError('value', `${who}: expected ${ground.weights.length} initial weights, got ${initial.length}`)
  return [...initial]
}

// ------------------------------------------------------------------------------------------------ pseudo-likelihood

/** The rows of the pseudo-likelihood's logistic regression: a count-change vector, a label and a multiplicity. */
interface PseudoRows {
  readonly features: number
  readonly d: Float64Array
  readonly label: Uint8Array
  readonly count: Float64Array
  /** The (world, atom) pairs summed over. */
  readonly total: number
}

function pseudoRows(ground: GroundMln, worlds: readonly World[]): PseudoRows {
  const f = ground.weights.length
  const merged = new Map<string, { d: number[]; label: number; count: number }>()
  for (const record of worlds) {
    const world = worldVector(ground, record)
    for (let a = 0; a < ground.atoms.length; a++) {
      const d = new Array<number>(f).fill(0)
      const label = world[a] as number
      for (const gi of ground.occurrences[a] ?? []) {
        const g = ground.formulas[gi]
        if (g === undefined) continue
        world[a] = 1
        const v1 = g.holds(world)
        world[a] = 0
        const v0 = g.holds(world)
        d[g.source] = (d[g.source] as number) + (v1 ? 1 : 0) - (v0 ? 1 : 0)
      }
      world[a] = label
      const k = `${d.join(',')}|${label}`
      const hit = merged.get(k)
      if (hit === undefined) merged.set(k, { d, label, count: 1 })
      else hit.count++
    }
  }
  const rows = [...merged.values()]
  const flat = new Float64Array(rows.length * f)
  rows.forEach((r, i) => {
    flat.set(r.d, i * f)
  })
  return {
    features: f,
    d: flat,
    label: Uint8Array.from(rows, (r) => r.label),
    count: Float64Array.from(rows, (r) => r.count),
    total: worlds.length * ground.atoms.length,
  }
}

/** Options of the likelihood objectives. */
export interface MlnObjectiveOptions {
  /** A ridge penalty `l2/2 |w|^2` on the mean negative log likelihood (default 0). */
  readonly l2?: number
  /** The starting weights (default zeros). */
  readonly initial?: readonly number[]
}

/**
 * The pseudo-log-likelihood of observed worlds as a smooth problem for the solver slot: the mean over worlds and ground
 * atoms of `-log P(x_l | rest)`, a logistic regression with value, gradient and Hessian, plus a ridge penalty. Each
 * ground atom's conditional is `sigmoid(sum_i w_i (n_i(x with the atom true) - n_i(x with it false)))`, so no partition
 * function is needed. Closed under minimisation by any solver of `smoothSolvers`; the objective is convex.
 *
 * @param ground - The ground network (every formula must have a finite weight; the weights given are ignored).
 * @param worlds - Complete worlds (atoms not listed are false).
 * @param options - The ridge penalty and the starting weights.
 * @returns The problem; its vector is the weight of each first-order formula.
 * @throws OmaiError with code `value` for hard formulas, no worlds or an unknown atom.
 * @example
 * ```ts
 * import { groundMln, mlnPseudoLikelihoodProblem } from 'omai-engine/logic'
 * const g = groundMln({ domain: ['a'], formulas: [{ weight: 0, formula: 'p(X)' }] })
 * mlnPseudoLikelihoodProblem(g, [{ 'p(a)': true }]).x0 // => [0]
 * ```
 */
export function mlnPseudoLikelihoodProblem(
  ground: GroundMln,
  worlds: readonly World[],
  options: MlnObjectiveOptions = {},
): SmoothProblem {
  checkLearnable(ground, 'mlnPseudoLikelihoodProblem')
  if (worlds.length === 0) throw new OmaiError('value', 'mlnPseudoLikelihoodProblem: no worlds')
  const l2 = options.l2 ?? 0
  const rows = pseudoRows(ground, worlds)
  const { features: f, d, label, count, total } = rows
  const m = label.length
  const logit = (w: ArrayLike<number>, r: number): number => {
    let z = 0
    for (let j = 0; j < f; j++) z += (d[r * f + j] as number) * (w[j] as number)
    return z
  }
  return {
    objective: analyticObjective({
      value: (w) => {
        let v = 0
        for (let r = 0; r < m; r++) {
          const z = logit(w, r)
          v += (count[r] as number) * (softplus(z) - (label[r] as number) * z)
        }
        v /= total
        for (let j = 0; j < f; j++) v += 0.5 * l2 * (w[j] as number) ** 2
        return v
      },
      gradient: (w) => {
        const g = new Float64Array(f)
        for (let r = 0; r < m; r++) {
          const e = ((count[r] as number) * (logistic(logit(w, r)) - (label[r] as number))) / total
          for (let j = 0; j < f; j++) g[j] = (g[j] as number) + e * (d[r * f + j] as number)
        }
        for (let j = 0; j < f; j++) g[j] = (g[j] as number) + l2 * (w[j] as number)
        return g
      },
      hessian: (w) => {
        const h = new Float64Array(f * f)
        for (let r = 0; r < m; r++) {
          const p = logistic(logit(w, r))
          const e = ((count[r] as number) * p * (1 - p)) / total
          for (let j = 0; j < f; j++) {
            const dj = d[r * f + j] as number
            if (dj === 0) continue
            for (let k = 0; k < f; k++) h[j * f + k] = (h[j * f + k] as number) + e * dj * (d[r * f + k] as number)
          }
        }
        for (let j = 0; j < f; j++) h[j * f + j] = (h[j * f + j] as number) + l2
        return h
      },
    }),
    x0: startWeights(ground, options.initial, 'mlnPseudoLikelihoodProblem'),
  }
}

// ------------------------------------------------------------------------------------------------ exact likelihoods

const LEARNING_ENUMERATION_LIMIT = 18

/** The count vector of every assignment to some atoms, the others as in a base world. */
function countMatrix(ground: GroundMln, base: Uint8Array, atoms: readonly number[]): Float64Array {
  const f = ground.weights.length
  const out = new Float64Array(2 ** atoms.length * f)
  const world = Uint8Array.from(base)
  for (let m = 0; m < 2 ** atoms.length; m++) {
    for (let j = 0; j < atoms.length; j++) world[atoms[j] as number] = (m >> j) & 1
    out.set(mlnCounts(ground, world), m * f)
  }
  return out
}

/** The mean negative log likelihood of a table of count rows (`rows x f`) with the data's row index per example. */
function softmaxObjective(
  f: number,
  tables: readonly Float64Array[],
  data: readonly number[],
  l2: number,
): SmoothProblem['objective'] {
  const n = tables.length
  const probabilities = (w: ArrayLike<number>, t: Float64Array): { p: Float64Array; logZ: number } => {
    const rows = t.length / f
    const s = new Float64Array(rows)
    let max = Number.NEGATIVE_INFINITY
    for (let r = 0; r < rows; r++) {
      let z = 0
      for (let j = 0; j < f; j++) z += (t[r * f + j] as number) * (w[j] as number)
      s[r] = z
      if (z > max) max = z
    }
    let sum = 0
    for (let r = 0; r < rows; r++) sum += Math.exp((s[r] as number) - max)
    const p = s.map((z) => Math.exp(z - max) / sum)
    return { p, logZ: max + Math.log(sum) }
  }
  return analyticObjective({
    value: (w) => {
      let v = 0
      tables.forEach((t, e) => {
        const { logZ } = probabilities(w, t)
        let z = 0
        for (let j = 0; j < f; j++) z += (t[(data[e] as number) * f + j] as number) * (w[j] as number)
        v += logZ - z
      })
      v /= n
      for (let j = 0; j < f; j++) v += 0.5 * l2 * (w[j] as number) ** 2
      return v
    },
    gradient: (w) => {
      const g = new Float64Array(f)
      tables.forEach((t, e) => {
        const { p } = probabilities(w, t)
        for (let r = 0; r < p.length; r++)
          for (let j = 0; j < f; j++) g[j] = (g[j] as number) + ((p[r] as number) * (t[r * f + j] as number)) / n
        for (let j = 0; j < f; j++) g[j] = (g[j] as number) - (t[(data[e] as number) * f + j] as number) / n
      })
      for (let j = 0; j < f; j++) g[j] = (g[j] as number) + l2 * (w[j] as number)
      return g
    },
    hessian: (w) => {
      const h = new Float64Array(f * f)
      tables.forEach((t) => {
        const { p } = probabilities(w, t)
        const mean = new Float64Array(f)
        for (let r = 0; r < p.length; r++)
          for (let j = 0; j < f; j++) mean[j] = (mean[j] as number) + (p[r] as number) * (t[r * f + j] as number)
        for (let r = 0; r < p.length; r++)
          for (let j = 0; j < f; j++)
            for (let k = 0; k < f; k++)
              h[j * f + k] =
                (h[j * f + k] as number) +
                ((p[r] as number) *
                  ((t[r * f + j] as number) - (mean[j] as number)) *
                  ((t[r * f + k] as number) - (mean[k] as number))) /
                  n
      })
      for (let j = 0; j < f; j++) h[j * f + j] = (h[j * f + j] as number) + l2
      return h
    },
  })
}

/**
 * The exact negative log likelihood of observed worlds as a smooth problem, by enumerating all worlds for the partition
 * function: `log Z(w) - w . mean_n(x)`, with gradient `E_w[n] - mean_n(x)` and the covariance of the counts as Hessian.
 * Only for tiny domains (at most 18 atoms); the maximum-likelihood estimate the pseudo-likelihood approximates.
 *
 * @param ground - The ground network.
 * @param worlds - Complete worlds.
 * @param options - The ridge penalty and the starting weights.
 * @returns The problem.
 * @throws OmaiError with code `value` for hard formulas, no worlds or too many atoms.
 * @example
 * ```ts
 * import { groundMln, mlnLikelihoodProblem } from 'omai-engine/logic'
 * const g = groundMln({ domain: ['a'], formulas: [{ weight: 0, formula: 'p(X)' }] })
 * mlnLikelihoodProblem(g, [{ 'p(a)': true }]).x0 // => [0]
 * ```
 */
export function mlnLikelihoodProblem(
  ground: GroundMln,
  worlds: readonly World[],
  options: MlnObjectiveOptions = {},
): SmoothProblem {
  checkLearnable(ground, 'mlnLikelihoodProblem')
  if (worlds.length === 0) throw new OmaiError('value', 'mlnLikelihoodProblem: no worlds')
  if (ground.atoms.length > LEARNING_ENUMERATION_LIMIT)
    throw new OmaiError('value', `mlnLikelihoodProblem: ${ground.atoms.length} atoms is too many to enumerate`)
  const f = ground.weights.length
  const all = countMatrix(ground, new Uint8Array(ground.atoms.length), [...Array(ground.atoms.length).keys()])
  // Every example shares the table of all worlds; its data row is the world's index.
  const data = worlds.map((w) => {
    const v = worldVector(ground, w)
    return v.reduce((m, bit, i) => m | (bit << i), 0)
  })
  return {
    objective: likelihoodObjective(f, all, data, options.l2 ?? 0),
    x0: startWeights(ground, options.initial, 'mlnLikelihoodProblem'),
  }
}

/** The mean negative log likelihood when all examples share one table: the data rows are averaged. */
function likelihoodObjective(
  f: number,
  table: Float64Array,
  data: readonly number[],
  l2: number,
): SmoothProblem['objective'] {
  const mean = new Float64Array(f)
  for (const row of data)
    for (let j = 0; j < f; j++) mean[j] = (mean[j] as number) + (table[row * f + j] as number) / data.length
  const rows = table.length / f
  const fit = (w: ArrayLike<number>): { p: Float64Array; logZ: number; expected: Float64Array } => {
    const s = new Float64Array(rows)
    let max = Number.NEGATIVE_INFINITY
    for (let r = 0; r < rows; r++) {
      let z = 0
      for (let j = 0; j < f; j++) z += (table[r * f + j] as number) * (w[j] as number)
      s[r] = z
      if (z > max) max = z
    }
    let sum = 0
    for (let r = 0; r < rows; r++) sum += Math.exp((s[r] as number) - max)
    const p = s.map((z) => Math.exp(z - max) / sum)
    const expected = new Float64Array(f)
    for (let r = 0; r < rows; r++)
      for (let j = 0; j < f; j++)
        expected[j] = (expected[j] as number) + (p[r] as number) * (table[r * f + j] as number)
    return { p, logZ: max + Math.log(sum), expected }
  }
  return analyticObjective({
    value: (w) => {
      let v = fit(w).logZ
      for (let j = 0; j < f; j++) v += -(mean[j] as number) * (w[j] as number) + 0.5 * l2 * (w[j] as number) ** 2
      return v
    },
    gradient: (w) => {
      const { expected } = fit(w)
      return Float64Array.from(expected, (e, j) => e - (mean[j] as number) + l2 * (w[j] as number))
    },
    hessian: (w) => {
      const { p, expected } = fit(w)
      const h = new Float64Array(f * f)
      for (let r = 0; r < rows; r++)
        for (let j = 0; j < f; j++)
          for (let k = 0; k < f; k++)
            h[j * f + k] =
              (h[j * f + k] as number) +
              (p[r] as number) *
                ((table[r * f + j] as number) - (expected[j] as number)) *
                ((table[r * f + k] as number) - (expected[k] as number))
      for (let j = 0; j < f; j++) h[j * f + j] = (h[j * f + j] as number) + l2
      return h
    },
  })
}

// ------------------------------------------------------------------------------------------------ conditional models

/** Evidence/query split of training worlds: the query predicates' atoms are the unknowns. */
interface ConditionalModel {
  readonly queryAtoms: readonly number[]
  /** Per example: the evidence as a world with the query atoms false, and the data assignment's index. */
  readonly examples: readonly { readonly base: Uint8Array; readonly truth: number; readonly world: Uint8Array }[]
}

function conditionalModel(
  ground: GroundMln,
  worlds: readonly World[],
  query: readonly string[],
  who: string,
): ConditionalModel {
  const names = new Set(query)
  const queryAtoms = ground.atoms
    .map((_a, i) => i)
    .filter((i) => names.has((ground.atoms[i] as string).split('(')[0] as string))
  if (queryAtoms.length === 0) throw new OmaiError('value', `${who}: no atom belongs to a query predicate`)
  return {
    queryAtoms,
    examples: worlds.map((record) => {
      const world = worldVector(ground, record)
      const base = Uint8Array.from(world)
      let truth = 0
      queryAtoms.forEach((a, j) => {
        base[a] = 0
        truth |= (world[a] as number) << j
      })
      return { base, truth, world }
    }),
  }
}

/**
 * The exact conditional negative log likelihood as a smooth problem: the mean over examples of `-log P(query atoms |
 * evidence atoms)`, normalised by enumerating the query atoms' assignments. The objective the discriminative
 * perceptron approximates; only for few query atoms (at most 14).
 *
 * @param ground - The ground network.
 * @param worlds - Complete worlds; the atoms of the query predicates are the unknowns, the rest the evidence.
 * @param query - The names of the query predicates.
 * @param options - The ridge penalty and the starting weights.
 * @returns The problem.
 * @throws OmaiError with code `value` for hard formulas, no worlds, no query atoms or more than 14.
 * @example
 * ```ts
 * import { groundMln, mlnConditionalLikelihoodProblem } from 'omai-engine/logic'
 * const g = groundMln({ domain: ['a'], formulas: [{ weight: 0, formula: 'e(X) -> q(X)' }] })
 * mlnConditionalLikelihoodProblem(g, [{ 'e(a)': true, 'q(a)': true }], ['q']).x0 // => [0]
 * ```
 */
export function mlnConditionalLikelihoodProblem(
  ground: GroundMln,
  worlds: readonly World[],
  query: readonly string[],
  options: MlnObjectiveOptions = {},
): SmoothProblem {
  checkLearnable(ground, 'mlnConditionalLikelihoodProblem')
  if (worlds.length === 0) throw new OmaiError('value', 'mlnConditionalLikelihoodProblem: no worlds')
  const model = conditionalModel(ground, worlds, query, 'mlnConditionalLikelihoodProblem')
  if (model.queryAtoms.length > 14)
    throw new OmaiError('value', 'mlnConditionalLikelihoodProblem: too many query atoms to enumerate')
  const tables = model.examples.map((e) => countMatrix(ground, e.base, model.queryAtoms))
  return {
    objective: softmaxObjective(
      ground.weights.length,
      tables,
      model.examples.map((e) => e.truth),
      options.l2 ?? 0,
    ),
    x0: startWeights(ground, options.initial, 'mlnConditionalLikelihoodProblem'),
  }
}

// ------------------------------------------------------------------------------------------------ generative learning

/** The paths of {@link mlnPseudoLikelihoodAlgorithm}: the solver's, and the weights. */
export type MlnLearningPath = SolverPath | GradientPath | 'mln/weights'

/** The log contract of {@link mlnPseudoLikelihoodAlgorithm}. */
export const mlnLearningContract: Contract<unknown, MlnLearningPath> = defineContract({
  ...solverContract,
  ...gradientContract,
  'mln/weights': tensor({ role: 'param', shape: ['formulas'], description: 'the weight of each first-order formula' }),
})

/** Options of the weight learners. */
export interface MlnLearnOptions extends MlnObjectiveOptions {
  /** How the objective is minimised: a value of `smoothSolvers.space` (default L-BFGS). */
  readonly solver?: SpaceValue
  /** The most solver steps (default 500). */
  readonly maxSteps?: number
  /** The run's key (for the stochastic solver). */
  readonly key?: Key
}

/** The default solver of the weight learners: L-BFGS to a tight gradient tolerance. */
export const mlnDefaultSolver: SpaceValue = smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-9 })

/**
 * Generative weight learning as a traced algorithm over the solver slot: the pseudo-log-likelihood of
 * {@link mlnPseudoLikelihoodProblem} minimised by any solver of `smoothSolvers`, logging the solver's objective,
 * gradient norm and step and the weights (`mln/weights`) each step.
 *
 * @param ground - The ground network.
 * @param worlds - Complete training worlds.
 * @param options - The ridge penalty, the starting weights, the solver and its key.
 * @returns The algorithm.
 * @throws OmaiError with code `value` as {@link mlnPseudoLikelihoodProblem}.
 * @example
 * ```ts
 * import { groundMln, mlnPseudoLikelihoodAlgorithm } from 'omai-engine/logic'
 * const g = groundMln({ domain: ['a', 'b'], formulas: [{ weight: 0, formula: 'p(X)' }] })
 * mlnPseudoLikelihoodAlgorithm(g, [{ 'p(a)': true, 'p(b)': true }]).name // => 'lbfgs'
 * ```
 */
export function mlnPseudoLikelihoodAlgorithm(
  ground: GroundMln,
  worlds: readonly World[],
  options: MlnLearnOptions = {},
): Algorithm<unknown, MlnLearningPath> {
  const problem = mlnPseudoLikelihoodProblem(ground, worlds, options)
  const inner = loggedSolver<'mln/weights'>(
    problem,
    options.solver ?? mlnDefaultSolver,
    options.key ?? makeKey(0),
    (x, ctx) => {
      ctx.log('mln/weights', x)
    },
  )
  return { ...inner, schema: contractSchema(mlnLearningContract) }
}

/** What a weight learner returns. */
export interface MlnFit {
  /** The learned weight of each first-order formula. */
  readonly weights: readonly number[]
  /** The objective at the weights (the mean negative log likelihood, with the penalty). */
  readonly objective: number
  /** Whether the solver stopped on its tolerance. */
  readonly converged: boolean
  readonly steps: number
}

/**
 * Minimise any of the learning objectives (or a problem of the caller's) with the solver of the slot.
 *
 * @param problem - The smooth problem.
 * @param options - The solver (default L-BFGS), step budget and key.
 * @returns The weights, the objective and whether the solver converged.
 * @example
 * ```ts
 * import { fitMlnProblem, groundMln, mlnConditionalLikelihoodProblem } from 'omai-engine/logic'
 * const g = groundMln({ domain: ['a', 'b'], formulas: [{ weight: 0, formula: 'e(X) -> q(X)' }] })
 * const worlds = [{ 'e(a)': true, 'q(a)': true, 'e(b)': true }, { 'e(a)': true, 'e(b)': true, 'q(b)': true }]
 * const problem = mlnConditionalLikelihoodProblem(g, worlds, ['q'], { l2: 0.1 })
 * fitMlnProblem(problem).weights.length // => 1
 * ```
 */
export function fitMlnProblem(problem: SmoothProblem, options: MlnLearnOptions = {}): MlnFit {
  const solver = options.solver ?? mlnDefaultSolver
  const key = options.key ?? makeKey(0)
  const alg = loggedSolver(problem, solver, key)
  const run = runAlgorithm(alg as unknown as Algorithm<unknown>, { key, maxSteps: options.maxSteps ?? 500 })
  const r = run.state as { x: Float64Array; value: number }
  return { weights: Array.from(r.x), objective: r.value, converged: run.status === 'converged', steps: run.steps }
}

/**
 * Learn the weights of a network's formulas by maximising the pseudo-likelihood of observed worlds, with the solver as
 * a slot. The estimator is consistent: with enough worlds drawn from a network of this structure it recovers the
 * network's weights.
 *
 * @param ground - The ground network (its weights are only the structure; learning starts from `initial` or zeros).
 * @param worlds - Complete training worlds.
 * @param options - The ridge penalty, starting weights, solver, step budget and key.
 * @returns The weights, the objective and whether it converged.
 * @throws OmaiError with code `value` as {@link mlnPseudoLikelihoodProblem}.
 * @example
 * ```ts
 * import { groundMln, learnMlnWeights } from 'omai-engine/logic'
 * const g = groundMln({ domain: ['a', 'b', 'c', 'd'], formulas: [{ weight: 0, formula: 'p(X)' }] })
 * const world = { 'p(a)': true, 'p(b)': true, 'p(c)': true }   // three of four true: the weight is log(3)
 * Math.abs((learnMlnWeights(g, [world]).weights[0] ?? 0) - Math.log(3)) < 1e-6 // => true
 * ```
 */
export function learnMlnWeights(ground: GroundMln, worlds: readonly World[], options: MlnLearnOptions = {}): MlnFit {
  return fitMlnProblem(mlnPseudoLikelihoodProblem(ground, worlds, options), options)
}

/**
 * Maximum-likelihood weights by the exact likelihood of {@link mlnLikelihoodProblem} (tiny domains), with the solver as
 * a slot.
 *
 * @param ground - The ground network.
 * @param worlds - Complete training worlds.
 * @param options - As {@link learnMlnWeights}.
 * @returns The weights, the objective and whether it converged.
 * @throws OmaiError with code `value` as {@link mlnLikelihoodProblem}.
 * @example
 * ```ts
 * import { groundMln, learnMlnWeightsExact } from 'omai-engine/logic'
 * const g = groundMln({ domain: ['a', 'b', 'c', 'd'], formulas: [{ weight: 0, formula: 'p(X)' }] })
 * Math.abs((learnMlnWeightsExact(g, [{ 'p(a)': true, 'p(b)': true, 'p(c)': true }]).weights[0] ?? 0) - Math.log(3)) < 1e-6 // => true
 * ```
 */
export function learnMlnWeightsExact(
  ground: GroundMln,
  worlds: readonly World[],
  options: MlnLearnOptions = {},
): MlnFit {
  return fitMlnProblem(mlnLikelihoodProblem(ground, worlds, options), options)
}

// ------------------------------------------------------------------------------------------------ voted perceptron

/** The paths of {@link mlnPerceptronAlgorithm}. */
export type MlnPerceptronPath = 'mln/weights' | 'mln/voted' | 'mln/error' | 'mln/update'

/** The log contract of {@link mlnPerceptronAlgorithm}. */
export const mlnPerceptronContract: Contract<MlnPerceptronState, MlnPerceptronPath> = defineContract({
  'mln/weights': tensor({ role: 'param', shape: ['formulas'], description: 'the current weights' }),
  'mln/voted': tensor({
    role: 'output',
    shape: ['formulas'],
    description: 'the average of the weights over the steps',
  }),
  'mln/error': scalar({
    description: 'query atoms wrong per example at the start of the step (in the MAP world, or in expectation)',
    direction: 'lower',
  }),
  'mln/update': scalar({ description: 'the norm of the weight update of the step' }),
})

/** The state of {@link mlnPerceptronAlgorithm}. */
export interface MlnPerceptronState {
  readonly weights: number[]
  /** The sum of the weights after each step, for the average. */
  readonly sum: number[]
  readonly iteration: number
  /** Query atoms wrong per example before the last update. */
  readonly error: number
  /** The norm of the last update. */
  readonly update: number
}

/** Options of {@link mlnPerceptronAlgorithm}. */
export interface MlnPerceptronOptions {
  /** The names of the query predicates: their atoms are inferred, all others are evidence. */
  readonly query: readonly string[]
  /** The step size of the weight update (default 0.1). */
  readonly learningRate?: number
  /** Where the formulas' expected counts come from: the MAP world (the voted perceptron, default) or exact enumeration. */
  readonly expectation?: 'map' | 'exact'
  /** The starting weights (default zeros). */
  readonly initial?: readonly number[]
}

/**
 * The voted perceptron for discriminative Markov logic weight learning (Singla and Domingos 2005), one pass over the
 * examples per step. For each training world the evidence atoms are fixed and the query atoms inferred: the MAP world by
 * `mlnMap` (the perceptron), or the exact expected counts by enumeration (`expectation: 'exact'`, which is gradient
 * ascent on the conditional likelihood). The weights move along the mean difference between the formulas' counts in the
 * data and in the inferred state, and the average of the weights over the steps (the vote) is `voted`.
 *
 * @param ground - The ground network.
 * @param worlds - Complete training worlds.
 * @param options - The query predicates, learning rate, expectation and starting weights.
 * @returns The algorithm; the voted weights are `sum / iteration` of its state ({@link votedWeights}).
 * @throws OmaiError with code `value` for hard formulas, no worlds or a query that matches no atom.
 * @example
 * ```ts
 * import { groundMln, mlnPerceptronAlgorithm, votedWeights } from 'omai-engine/logic'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = groundMln({ domain: ['a', 'b'], formulas: [{ weight: 0, formula: 'e(X) -> q(X)' }] })
 * const worlds = [{ 'e(a)': true, 'q(a)': true }, { 'e(b)': true, 'q(b)': true }]
 * const s = runAlgorithm(mlnPerceptronAlgorithm(g, worlds, { query: ['q'], expectation: 'exact' }), { seed: 0, maxSteps: 50 }).state
 * votedWeights(s)[0] > 0 // => true
 * ```
 */
export function mlnPerceptronAlgorithm(
  ground: GroundMln,
  worlds: readonly World[],
  options: MlnPerceptronOptions,
): Algorithm<MlnPerceptronState, MlnPerceptronPath> {
  checkLearnable(ground, 'mlnPerceptronAlgorithm')
  if (worlds.length === 0) throw new OmaiError('value', 'mlnPerceptronAlgorithm: no worlds')
  const model = conditionalModel(ground, worlds, options.query, 'mlnPerceptronAlgorithm')
  const rate = options.learningRate ?? 0.1
  const expectation = options.expectation ?? 'map'
  if (expectation === 'exact' && model.queryAtoms.length > 14)
    throw new OmaiError('value', 'mlnPerceptronAlgorithm: too many query atoms for the exact expectation')
  const f = ground.weights.length
  const dataCounts = model.examples.map((e) => mlnCounts(ground, e.world))
  const start = startWeights(ground, options.initial, 'mlnPerceptronAlgorithm')
  const evidenceOf = (base: Uint8Array): World =>
    Object.fromEntries(
      ground.atoms.flatMap((a, i) => (model.queryAtoms.includes(i) ? [] : [[a, (base[i] as number) === 1]])),
    )
  const evidence = model.examples.map((e) => evidenceOf(e.base))
  const tables = expectation === 'exact' ? model.examples.map((e) => countMatrix(ground, e.base, model.queryAtoms)) : []
  return {
    name: 'mln-voted-perceptron',
    config: { query: options.query, learningRate: rate, expectation, examples: worlds.length } as Json,
    schema: contractSchema(mlnPerceptronContract),
    init: () => ({ weights: start, sum: new Array<number>(f).fill(0), iteration: 0, error: Number.NaN, update: 0 }),
    step: (s) => {
      const g = withWeights(ground, s.weights)
      const direction = new Array<number>(f).fill(0)
      let wrong = 0
      model.examples.forEach((e, i) => {
        if (expectation === 'map') {
          const r = mlnMap(g, { evidence: evidence[i] as World })
          const inferred = r.world === null ? e.world : worldVector(g, r.world)
          const c = mlnCounts(g, inferred)
          for (const a of model.queryAtoms) if (inferred[a] !== e.world[a]) wrong++
          for (let j = 0; j < f; j++)
            direction[j] = (direction[j] as number) + ((dataCounts[i]?.[j] ?? 0) - (c[j] ?? 0))
        } else {
          const t = tables[i] as Float64Array
          const rows = t.length / f
          const score = new Float64Array(rows)
          let max = Number.NEGATIVE_INFINITY
          for (let r = 0; r < rows; r++) {
            let z = 0
            for (let j = 0; j < f; j++) z += (t[r * f + j] as number) * (s.weights[j] as number)
            score[r] = z
            if (z > max) max = z
          }
          let sum = 0
          for (let r = 0; r < rows; r++) sum += Math.exp((score[r] as number) - max)
          for (let r = 0; r < rows; r++) {
            const p = Math.exp((score[r] as number) - max) / sum
            for (let j = 0; j < f; j++) direction[j] = (direction[j] as number) - p * (t[r * f + j] as number)
            model.queryAtoms.forEach((a, q) => {
              const on = (r >> q) & 1
              wrong += p * (e.world[a] === on ? 0 : 1)
            })
          }
          for (let j = 0; j < f; j++) direction[j] = (direction[j] as number) + (dataCounts[i]?.[j] ?? 0)
        }
      })
      const step = direction.map((d) => (rate * d) / worlds.length)
      const weights = s.weights.map((w, j) => w + (step[j] as number))
      return {
        weights,
        sum: s.sum.map((v, j) => v + (weights[j] as number)),
        iteration: s.iteration + 1,
        error: wrong / worlds.length,
        update: Math.hypot(...step),
      }
    },
    observe: (s) => ({
      metrics: { 'mln/error': Number.isNaN(s.error) ? model.queryAtoms.length : s.error, 'mln/update': s.update },
      tensors: {
        'mln/weights': { role: 'param', value: s.weights },
        'mln/voted': { role: 'output', value: votedWeights(s) },
      },
    }),
    serialise: (s) => stateToJson({ ...s, error: Number.isNaN(s.error) ? -1 : s.error }),
    deserialise: (j) => {
      const s = stateFromJson(j) as MlnPerceptronState
      return { ...s, error: s.error < 0 ? Number.NaN : s.error }
    },
  }
}

/**
 * The voted (averaged) weights of a perceptron state: the mean of the weights after each step, the current weights
 * before the first.
 *
 * @param s - The state of {@link mlnPerceptronAlgorithm}.
 * @returns One weight per first-order formula.
 * @example
 * ```ts
 * import { votedWeights } from 'omai-engine/logic'
 * votedWeights({ weights: [3], sum: [4], iteration: 2, error: 0, update: 0 }) // => [2]
 * ```
 */
export function votedWeights(s: MlnPerceptronState): number[] {
  return s.iteration === 0 ? [...s.weights] : s.sum.map((v) => v / s.iteration)
}

/**
 * Discriminative weight learning by the voted perceptron, run for a number of passes.
 *
 * @param ground - The ground network.
 * @param worlds - Complete training worlds.
 * @param options - As {@link mlnPerceptronAlgorithm}, plus the number of passes (default 50).
 * @returns The voted weights, the last weights and the training error of the last pass.
 * @throws OmaiError with code `value` as {@link mlnPerceptronAlgorithm}.
 * @example
 * ```ts
 * import { groundMln, learnMlnDiscriminative } from 'omai-engine/logic'
 * const g = groundMln({ domain: ['a', 'b'], formulas: [{ weight: 0, formula: 'e(X) -> q(X)' }] })
 * const worlds = [{ 'e(a)': true, 'q(a)': true }, { 'e(b)': true, 'q(b)': true }]
 * learnMlnDiscriminative(g, worlds, { query: ['q'], expectation: 'exact', passes: 30 }).voted[0] > 0 // => true
 * ```
 */
export function learnMlnDiscriminative(
  ground: GroundMln,
  worlds: readonly World[],
  options: MlnPerceptronOptions & { readonly passes?: number },
): { readonly voted: readonly number[]; readonly weights: readonly number[]; readonly error: number } {
  const { passes = 50, ...rest } = options
  const { state } = runAlgorithm(
    mlnPerceptronAlgorithm(ground, worlds, rest) as unknown as Algorithm<MlnPerceptronState>,
    {
      seed: 0,
      maxSteps: passes,
    },
  )
  return { voted: votedWeights(state), weights: state.weights, error: state.error }
}
