/**
 * Sanity checks for explanations (Adebayo et al. 2018): an explainer that is about the model must change when the
 * model does. In the model randomisation test the trained parameters are replaced by random ones, layer by layer from
 * the output down (cascading) or one layer at a time (independent), and the explanation of the same input is compared
 * with the trained model's; an explainer whose map survives full randomisation is explaining the input, not the model.
 * The data randomisation test retrains on permuted labels and compares in the same way. The cascade is an `Algorithm`
 * with one layer randomised per step, so the similarity falling layer by layer can be watched.
 */
import type { Json } from '../base/index.ts'
import { OmaiError } from '../base/index.ts'
import { foldIn, type Key, normal, permutation } from '../random/index.ts'
import { asTensor, mul, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'
import { explanationSimilarity, type SimilarityMetric } from './evaluation.ts'

/** An explainer of a model given as per-layer parameters (input to output): the attribution map of one fixed input. */
export type ParameterExplainer = (params: readonly Tensor[]) => readonly number[]

/** Options of {@link randomisationTestAlgorithm}. */
export interface RandomisationOptions {
  /** `cascading` (default) randomises from the output layer down, keeping earlier randomisations; `independent` one layer at a time. */
  readonly order?: 'cascading' | 'independent'
  /** How the maps are compared (default Spearman's correlation of absolute values, as the paper). */
  readonly metric?: SimilarityMetric
  /** The key of the random replacements. */
  readonly key: Key
}

/** The state of {@link randomisationTestAlgorithm}. */
export interface RandomisationState {
  /** The parameters now (the trained ones with the randomised layers replaced). */
  readonly params: readonly Tensor[]
  /** The layers randomised so far, in the order done. */
  readonly randomised: readonly number[]
  /** The similarity to the trained model's map after each step. */
  readonly similarity: readonly number[]
  /** The current map. */
  readonly map: readonly number[]
  readonly step: number
}

/** The paths of {@link randomisationTestAlgorithm}. */
export type RandomisationPath = 'sanity/layer' | 'sanity/similarity' | 'sanity/map'

/** What a randomisation test logs. */
export const randomisationContract: Contract<RandomisationState, RandomisationPath> = defineContract({
  'sanity/layer': scalar({ description: 'the layer randomised at this step' }),
  'sanity/similarity': scalar({ description: 'the similarity of the map to the trained model’s' }),
  'sanity/map': tensorSource({
    role: 'output',
    shape: ['features'],
    description: 'the attribution map of the randomised model',
  }),
})

/** A layer's replacement: random normal draws with the layer's own standard deviation about zero. */
function randomLike(layer: Tensor, key: Key): Tensor {
  const flat = toFlat(layer)
  const sd = Math.sqrt(flat.reduce((a, b) => a + b * b, 0) / Math.max(1, flat.length)) || 1
  return mul(normal(key, [...layer.shape]), sd) as Tensor
}

/**
 * The model randomisation test as an algorithm: step `i` randomises the next layer (the last layer first) and
 * explains the same input again. Logs the layer, the similarity to the trained model's explanation and the map. A
 * sound explainer's similarity falls towards zero as layers are randomised; one that stays near 1 fails the check.
 *
 * @param explain - The explainer: from the parameters, the map of one fixed input.
 * @param params - The trained parameters, one tensor per layer from input to output.
 * @param options - Order, metric and key.
 * @returns The algorithm, completing after the last layer.
 * @concept explanation-evaluation
 * @example
 * ```ts
 * import { randomisationTestAlgorithm } from 'omai-engine/explain'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import { tensor, toFlat } from 'omai-engine/tensor'
 * // An explainer that ignores the model: its map never changes.
 * const alg = randomisationTestAlgorithm(() => [3, 1, 2], [tensor([1, 2]), tensor([3, 4])], { key: key(0) })
 * runAlgorithm(alg, { maxSteps: 5 }).state.similarity // => [1, 1]
 * ```
 */
export function randomisationTestAlgorithm(
  explain: ParameterExplainer,
  params: readonly TensorLike[],
  options: RandomisationOptions,
): Algorithm<RandomisationState, RandomisationPath> {
  const trained = params.map((p) => asTensor(p) as Tensor)
  const order = options.order ?? 'cascading'
  const baseline = explain(trained)
  const layers = trained.length
  if (layers === 0) throw new OmaiError('value', 'randomisationTestAlgorithm: give at least one layer')
  return {
    name: 'randomisationTest',
    config: { layers, order, metric: options.metric ?? 'spearman' } as Json,
    init: () => ({ params: trained, randomised: [], similarity: [], map: baseline, step: 0 }),
    step: (s, ctx) => {
      const layer = layers - 1 - s.step
      const replaced = randomLike(trained[layer] as Tensor, foldIn(options.key, layer))
      const base = order === 'cascading' ? s.params : trained
      const next = base.map((p, i) => (i === layer ? replaced : p))
      const map = explain(next)
      const similarity = explanationSimilarity(
        baseline,
        map,
        options.metric === undefined ? {} : { metric: options.metric },
      )
      ctx.log('sanity/layer', layer)
      ctx.log('sanity/similarity', similarity)
      ctx.log('sanity/map', map)
      return {
        params: order === 'cascading' ? next : s.params,
        randomised: [...s.randomised, layer],
        similarity: [...s.similarity, similarity],
        map,
        step: s.step + 1,
      }
    },
    done: (s) => (s.step >= layers ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { similarity: s.similarity[s.similarity.length - 1] ?? 1 },
      tensors: { map: { role: 'output', value: s.map } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as RandomisationState,
  }
}

/** The outcome of a randomisation test run to the end. */
export interface RandomisationResult {
  /** The layers in the order randomised. */
  readonly layers: readonly number[]
  /** The similarity after each. */
  readonly similarity: readonly number[]
}

/**
 * Run the model randomisation test to the last layer.
 *
 * @param explain - The explainer.
 * @param params - The trained parameters, input to output.
 * @param options - Order, metric and key.
 * @returns The layers randomised and the similarity after each.
 * @concept explanation-evaluation
 * @example
 * ```ts
 * import { modelRandomisationTest } from 'omai-engine/explain'
 * import { key } from 'omai-engine/random'
 * import { tensor } from 'omai-engine/tensor'
 * modelRandomisationTest(() => [1, 2, 3], [tensor([1]), tensor([2])], { key: key(1) }).layers // => [1, 0]
 * ```
 */
export function modelRandomisationTest(
  explain: ParameterExplainer,
  params: readonly TensorLike[],
  options: RandomisationOptions,
): RandomisationResult {
  const alg = randomisationTestAlgorithm(explain, params, options)
  let s = alg.init(options.key)
  const ctx = { t: 0, key: options.key, log: () => undefined, wants: () => false, emit: () => undefined } as never
  while (alg.done?.(s, s.step) === undefined) s = alg.step(s, ctx)
  return { layers: s.randomised, similarity: s.similarity }
}

/**
 * The data randomisation test: retrain on labels permuted at random and compare the explanation of the same input with
 * the trained model's. An explainer that explains the learned relation changes when the relation is destroyed.
 *
 * @param train - Fits a model to labels and returns the explanation of the fixed input.
 * @param labels - The true labels.
 * @param options - The key and the metric.
 * @returns The similarity between the explanation after training on true and on permuted labels.
 * @concept explanation-evaluation
 * @example
 * ```ts
 * import { dataRandomisationTest } from 'omai-engine/explain'
 * import { key } from 'omai-engine/random'
 * // A "model" whose explanation is the label mean pattern: permuting labels changes it.
 * dataRandomisationTest((y) => [y[0] ?? 0, y[1] ?? 0, y[2] ?? 0], [1, 2, 3], { key: key(3) }) <= 1 // => true
 * ```
 */
export function dataRandomisationTest(
  train: (labels: readonly number[]) => readonly number[],
  labels: readonly number[],
  options: { readonly key: Key; readonly metric?: SimilarityMetric },
): number {
  const order = Array.from(permutation(options.key, labels.length).data)
  const permuted = order.map((i) => labels[i] as number)
  return explanationSimilarity(
    train(labels),
    train(permuted),
    options.metric === undefined ? {} : { metric: options.metric },
  )
}
