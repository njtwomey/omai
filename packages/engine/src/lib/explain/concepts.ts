/**
 * Concept-based explanations (TCAV): a concept activation vector is the direction in a hidden layer that separates
 * examples of a human concept from random examples; a class's TCAV score is the share of its examples whose output
 * rises along that direction, tested against CAVs of random examples for significance. Probing classifiers measure
 * how well a layer encodes a concept at all.
 */
import { MdmlError } from '../base/index.ts'
import { matrixView } from '../learn/validation.ts'
import { solve } from '../numerics/linalg/index.ts'
import { tests } from '../stats/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type DifferentiableModel, type Target, targetGradient } from './gradients.ts'

/** A linear probe: weights, bias and its training accuracy. */
export interface LinearProbe {
  readonly weights: readonly number[]
  readonly bias: number
  readonly accuracy: number
}

/**
 * A linear probing classifier (Alain and Bengio 2016): L2-regularised logistic regression of binary labels on a
 * layer's activations, by Newton's method (iteratively reweighted least squares), reporting its training accuracy:
 * how linearly decodable the labels are from the layer.
 *
 * @param activations - Activations `[n, p]`.
 * @param labels - 0/1 labels.
 * @param options - l2 (default 1e-2) and iterations (default 25).
 * @returns The probe.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * const probe = explain.linearProbe([[0, 1], [0, 2], [1, 0], [2, 0]], [1, 1, 0, 0])
 * probe.accuracy // => 1
 * ```
 */
export function linearProbe(
  activations: TensorLike,
  labels: readonly number[],
  options: { readonly l2?: number; readonly iterations?: number } = {},
): LinearProbe {
  const m = matrixView(activations, 'linearProbe')
  if (labels.length !== m.n) throw new MdmlError('shape', 'linearProbe: one label per row')
  const p = m.d + 1
  const lambda = options.l2 ?? 1e-2
  let w = new Float64Array(p)
  const row = (i: number, j: number): number => (j === m.d ? 1 : (m.data[i * m.d + j] ?? 0))
  for (let it = 0; it < (options.iterations ?? 25); it++) {
    const g = new Float64Array(p)
    const h = new Float64Array(p * p)
    for (let i = 0; i < m.n; i++) {
      let z = 0
      for (let j = 0; j < p; j++) z += (w[j] ?? 0) * row(i, j)
      const mu = 1 / (1 + Math.exp(-z))
      const r = mu - (labels[i] ?? 0)
      const s = mu * (1 - mu)
      for (let a = 0; a < p; a++) {
        g[a] = (g[a] ?? 0) + r * row(i, a)
        for (let b = 0; b < p; b++) h[a * p + b] = (h[a * p + b] ?? 0) + s * row(i, a) * row(i, b)
      }
    }
    for (let a = 0; a < m.d; a++) {
      g[a] = (g[a] ?? 0) + lambda * (w[a] ?? 0)
      h[a * p + a] = (h[a * p + a] ?? 0) + lambda
    }
    h[p * p - 1] = (h[p * p - 1] ?? 0) + 1e-9
    const step = toFlat(solve(fromStorage(h, [p, p], 'float64'), fromStorage(g, [p], 'float64')) as Tensor)
    w = Float64Array.from(w, (v, j) => v - (step[j] ?? 0))
    if (step.reduce((s, v) => s + Math.abs(v), 0) < 1e-10) break
  }
  let correct = 0
  for (let i = 0; i < m.n; i++) {
    let z = 0
    for (let j = 0; j < p; j++) z += (w[j] ?? 0) * row(i, j)
    if ((z > 0 ? 1 : 0) === (labels[i] ?? 0)) correct++
  }
  return { weights: Array.from(w.subarray(0, m.d)), bias: w[m.d] ?? 0, accuracy: correct / m.n }
}

/**
 * A concept activation vector (Kim et al. 2018): the unit normal of a linear probe separating a concept's
 * activations (label 1) from random ones (label 0), pointing towards the concept.
 *
 * @param concept - The concept examples' activations `[n, p]`.
 * @param random - Random examples' activations `[m, p]`.
 * @param options - l2 of the probe (default 1e-2).
 * @returns The unit vector and the probe's accuracy.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * const cav = explain.conceptActivationVector([[2, 0], [3, 0.1]], [[0, 0.1], [0.1, -0.1]])
 * cav.vector[0] > 0.9 // => true
 * ```
 */
export function conceptActivationVector(
  concept: TensorLike,
  random: TensorLike,
  options: { readonly l2?: number } = {},
): { readonly vector: readonly number[]; readonly accuracy: number } {
  const c = matrixView(concept, 'conceptActivationVector')
  const r = matrixView(random, 'conceptActivationVector', { nFeatures: c.d })
  const data = new Float64Array((c.n + r.n) * c.d)
  data.set(c.data, 0)
  data.set(r.data, c.n * c.d)
  const labels = [...new Array<number>(c.n).fill(1), ...new Array<number>(r.n).fill(0)]
  const probe = linearProbe(fromStorage(data, [c.n + r.n, c.d], 'float64'), labels, options)
  const norm = Math.sqrt(probe.weights.reduce((s, v) => s + v * v, 0)) || 1
  return { vector: probe.weights.map((v) => v / norm), accuracy: probe.accuracy }
}

/**
 * Directional derivatives of a target output along a concept vector at a layer: for each example, the gradient of
 * the output with respect to the layer's activations dotted with the vector.
 *
 * @param tail - The layer's activations to the outputs.
 * @param activations - The examples' activations `[n, p]`.
 * @param vector - The concept vector `[p]`.
 * @param options - target (default 0).
 * @returns One derivative per example.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { matmul } from 'mdml-engine/tensor'
 * explain.directionalDerivatives((h) => matmul(h, [[1], [-2]]), [[0, 0], [1, 1]], [0, 1]) // => [-2, -2]
 * ```
 */
export function directionalDerivatives(
  tail: DifferentiableModel,
  activations: TensorLike,
  vector: readonly number[],
  options: { readonly target?: Target } = {},
): number[] {
  const m = matrixView(activations, 'directionalDerivatives', { nFeatures: vector.length })
  const g = toFlat(targetGradient(tail, fromStorage(m.data, [m.n, m.d], 'float64'), options.target ?? 0))
  return Array.from({ length: m.n }, (_, i) => vector.reduce((s, v, j) => s + v * (g[i * m.d + j] ?? 0), 0))
}

/** Options of {@link tcav}. */
export interface TcavOptions {
  readonly target?: Target
  /** The probe's L2 weight (default 1e-2). */
  readonly l2?: number
}

/** A TCAV test: the concept's scores over runs, the random baseline's, and the t-test between them. */
export interface TcavResult {
  /** The TCAV score (share of positive directional derivatives) of each concept-vs-random run. */
  readonly scores: readonly number[]
  /** The scores of random-vs-random CAVs, the null. */
  readonly randomScores: readonly number[]
  readonly mean: number
  /** The two-sided p-value of the t test between scores and random scores. */
  readonly pValue: number
  /** The CAVs' probe accuracies. */
  readonly accuracies: readonly number[]
}

/**
 * Testing with concept activation vectors (Kim et al. 2018; captum TCAV): for each random set, a CAV separating the
 * concept from it, and the class's TCAV score along it (the share of class examples whose target output increases
 * in the concept's direction); the same with a second random set in place of the concept gives the null scores, and a
 * two-sided t test says whether the concept's scores differ from chance.
 *
 * @param tail - The layer's activations to the outputs.
 * @param classActivations - Activations of the class's examples `[n, p]`.
 * @param conceptActivations - Activations of the concept's examples `[c, p]`.
 * @param randomActivations - Activations of two or more random sets, each `[r, p]`.
 * @param options - target and l2.
 * @returns The scores, random scores, mean, p-value and probe accuracies.
 * @example
 * ```ts
 * import * as explain from 'mdml-engine/explain'
 * import { matmul } from 'mdml-engine/tensor'
 * // The output is the first unit: a concept lying along it scores 1 in every run.
 * const tail = (h) => matmul(h, [[1], [0]])
 * const randoms = [[[0, 1], [0.1, -1]], [[-0.1, 0.5], [0, -0.4]], [[0.1, 0.2], [-0.1, -0.3]]]
 * explain.tcav(tail, [[0, 0], [1, 1]], [[3, 0], [4, 0.2]], randoms).scores // => [1, 1, 1]
 * ```
 */
export function tcav(
  tail: DifferentiableModel,
  classActivations: TensorLike,
  conceptActivations: TensorLike,
  randomActivations: readonly TensorLike[],
  options: TcavOptions = {},
): TcavResult {
  if (randomActivations.length < 2) throw new MdmlError('value', 'tcav: needs at least two random sets')
  const probeOptions = options.l2 === undefined ? {} : { l2: options.l2 }
  const score = (vector: readonly number[]): number => {
    const dd = directionalDerivatives(
      tail,
      classActivations,
      vector,
      options.target === undefined ? {} : { target: options.target },
    )
    return dd.filter((v) => v > 0).length / Math.max(1, dd.length)
  }
  const accuracies: number[] = []
  const scores = randomActivations.map((r) => {
    const cav = conceptActivationVector(conceptActivations, r, probeOptions)
    accuracies.push(cav.accuracy)
    return score(cav.vector)
  })
  const randomScores = randomActivations.map((r, i) => {
    const other = randomActivations[(i + 1) % randomActivations.length] as TensorLike
    return score(conceptActivationVector(r, other, probeOptions).vector)
  })
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length
  const varies = (v: readonly number[]): boolean => v.some((x) => x !== v[0])
  const pValue =
    !varies(scores) && !varies(randomScores)
      ? scores[0] === randomScores[0]
        ? 1
        : 0
      : tests.ttestInd(scores, randomScores, { equalVar: false }).pValue
  return { scores, randomScores, mean, pValue, accuracies }
}
