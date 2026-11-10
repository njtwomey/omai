/**
 * The linear-chain CRF as a `StructuredOracle`: the model of `crf.ts` offered to every solver of `structuredSolvers`.
 * Feature counts, expected feature counts and Viterbi decoding (cost-augmented with the Hamming loss when asked) come
 * from the functions of `crf.ts` and `inference/chain.ts` on plain inputs, and from the typed arrays of `compiled.ts`
 * (through the same `forwardBackward` and `viterbi`) on a compiled dataset, which is the way to train on thousands of
 * sequences.
 */

import { OmaiError } from '../../base/index.ts'
import { ddot } from '../../numerics/linalg/index.ts'
import { forwardBackward, viterbi } from '../inference/chain.ts'
import {
  addCompiledExpectedCounts,
  addCompiledFeatureCounts,
  type CompiledChainInput,
  chainLabelIds,
  chainVocabulary,
  compiledChain,
  compiledScore,
  isCompiledChainInput,
} from './compiled.ts'
import {
  type CrfInput,
  crfChain,
  crfExpectedFeatureCounts,
  crfFeatureCounts,
  type LinearChainCrf,
  withCrfParameters,
} from './crf.ts'
import type { StructuredOracle } from './oracle.ts'

/** The input of the chain oracle: item (and edge) attributes, or the same compiled to ids. */
export type ChainInput = CrfInput | CompiledChainInput

/** The output of the chain oracle: one label per item, as names or ids. */
export type ChainLabels = readonly (string | number)[]

const asFloat = (w: ArrayLike<number>): Float64Array => (w instanceof Float64Array ? w : Float64Array.from(w))

/**
 * The oracle of a linear-chain CRF. The parameter layout is that of `crfParameters` (state, transition, bigram
 * weights). Inputs may be plain (`CrfInput`, decoded labels come back as names) or compiled (`compileChainData`,
 * decoded labels come back as ids and the fast typed-array path is used); labels given to `features`, `clamped`,
 * `decode`'s `against` and `loss` may be names or ids. The loss is the Hamming distance, and `decode` with `against`
 * adds 1 to the score of every label that differs from the gold one at a position. `gradientInto` is provided, so the
 * objective of `structuredObjective` accumulates without a parameter-sized vector per sequence.
 *
 * @param crf - The model: its labels, attributes and bigram attributes define the layout (its weights are not used).
 * @returns The oracle.
 * @concept conditional-random-fields
 * @index structure Linear-chain CRF as a structured oracle: counts, expectations, Viterbi, Hamming loss.
 * @example
 * ```ts
 * import { chainOracle, linearChainCrf } from 'omai-engine/pgm'
 * const oracle = chainOracle(linearChainCrf(['A', 'B'], ['f']))
 * oracle.decode([2, -2, 0, 0, 0, 0], [{ f: 1 }, { f: 1 }]).y // => ['A', 'A']
 * ```
 */
export function chainOracle(crf: LinearChainCrf): StructuredOracle<ChainInput, ChainLabels> {
  const vocabulary = chainVocabulary(crf)
  const layout = vocabulary.layout
  const ids = (y: ChainLabels): number[] => chainLabelIds(vocabulary, y)
  const model = (w: ArrayLike<number>): LinearChainCrf => withCrfParameters(crf, w)
  const checkLength = (n: number, y: ChainLabels, who: string): void => {
    if (y.length !== n) throw new OmaiError('shape', `${who}: one label per item`)
  }
  return {
    size: layout.size,
    features: (x, y) => {
      if (!isCompiledChainInput(x)) return crfFeatureCounts(crf, x, y)
      checkLength(x.length, y, 'chainOracle.features')
      const out = new Float64Array(layout.size)
      addCompiledFeatureCounts(layout, x, ids(y), out)
      return out
    },
    expectations: (w, x) => {
      if (!isCompiledChainInput(x)) return crfExpectedFeatureCounts(model(w), x)
      const counts = new Float64Array(layout.size)
      const fb = forwardBackward(compiledChain(layout, asFloat(w), x))
      addCompiledExpectedCounts(layout, x, fb.posteriors, fb.pairwise, counts)
      return { counts, logPartition: fb.logLikelihood }
    },
    gradientInto: (w, x, y, gradient) => {
      if (!isCompiledChainInput(x)) {
        const free = crfExpectedFeatureCounts(model(w), x)
        const observed = crfFeatureCounts(crf, x, y)
        for (let q = 0; q < gradient.length; q++)
          gradient[q] = (gradient[q] ?? 0) + (free.counts[q] ?? 0) - (observed[q] ?? 0)
        return free.logPartition - ddot(asFloat(w), observed)
      }
      checkLength(x.length, y, 'chainOracle.gradientInto')
      const wf = asFloat(w)
      const labels = ids(y)
      const fb = forwardBackward(compiledChain(layout, wf, x))
      addCompiledExpectedCounts(layout, x, fb.posteriors, fb.pairwise, gradient)
      addCompiledFeatureCounts(layout, x, labels, gradient, -1)
      return fb.logLikelihood - compiledScore(layout, wf, x, labels)
    },
    decode: (w, x, options) => {
      const compiled = isCompiledChainInput(x)
      const T = compiled ? x.length : 'unigram' in x ? x.unigram.length : x.length
      const against = options?.against === undefined ? undefined : ids(options.against)
      if (against !== undefined) checkLength(T, against, 'chainOracle.decode')
      let chain = compiled ? compiledChain(layout, asFloat(w), x, against) : crfChain(model(w), x)
      if (!compiled && against !== undefined) {
        const gold = against
        const base = chain
        chain = {
          ...base,
          logEmission: base.logEmission.map((row, t) => row.map((v, k) => v + (k === gold[t] ? 0 : 1))),
        }
      }
      const v = viterbi(chain)
      return { y: compiled ? v.path : v.path.map((k) => crf.labels[k] ?? ''), score: v.logProbability }
    },
    loss: (gold, predicted) => {
      checkLength(gold.length, predicted, 'chainOracle.loss')
      const g = ids(gold)
      const p = ids(predicted)
      let wrong = 0
      for (let t = 0; t < g.length; t++) if (g[t] !== p[t]) wrong++
      return wrong
    },
  }
}
