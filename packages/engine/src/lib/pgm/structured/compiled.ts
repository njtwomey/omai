/**
 * A dataset of a linear-chain CRF compiled once for fast training. The attribute strings of every sequence are resolved
 * to integer feature ids against the model's vocabularies and kept in typed arrays (compressed sparse rows: the
 * attributes of item t sit between `itemStart[t]` and `itemStart[t + 1]`), so an objective or gradient evaluation does
 * no string or map lookups; the chain potentials are then built from the typed arrays and handed to the library's
 * `forwardBackward` and `viterbi`. The numbers equal those of the string-keyed functions of `crf.ts` (the emissions are
 * summed in the same order).
 */

import { OmaiError } from '../../base/index.ts'
import type { ChainModel } from '../inference/chain.ts'
import type { CrfInput, LabelledSequence, LinearChainCrf } from './crf.ts'

/**
 * The shape of a linear-chain CRF's parameter vector: state weights [attribute][label], then plain transitions
 * [from][to] (none for a model without label-pair transitions), then bigram weights [attribute][from][to], the order of
 * `crfParameters`.
 */
export interface ChainLayout {
  readonly labels: number
  readonly attributes: number
  readonly bigramAttributes: number
  readonly stateSize: number
  readonly transitionSize: number
  readonly bigramSize: number
  /** The number of parameters. */
  readonly size: number
}

/**
 * The parameter layout of a CRF.
 *
 * @param crf - The model (its vocabularies; the weights are not used).
 * @returns The layout.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { chainLayout, linearChainCrf } from 'omai-engine/pgm'
 * chainLayout(linearChainCrf(['A', 'B'], ['f', 'g'])).size // => 8
 * ```
 */
export function chainLayout(crf: LinearChainCrf): ChainLayout {
  const stateSize = crf.stateWeights.length
  const transitionSize = crf.transitionWeights.length
  const bigramSize = crf.bigramWeights.length
  return {
    labels: crf.labels.length,
    attributes: crf.attributes.length,
    bigramAttributes: crf.bigramAttributes.length,
    stateSize,
    transitionSize,
    bigramSize,
    size: stateSize + transitionSize + bigramSize,
  }
}

/** The name-to-id maps of a CRF's vocabularies, built once for a whole dataset. */
export interface ChainVocabulary {
  readonly layout: ChainLayout
  readonly attributes: ReadonlyMap<string, number>
  readonly bigramAttributes: ReadonlyMap<string, number>
  readonly labels: ReadonlyMap<string, number>
}

/**
 * The vocabulary maps of a CRF.
 *
 * @param crf - The model.
 * @returns The maps from attribute, bigram attribute and label names to ids.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { chainVocabulary, linearChainCrf } from 'omai-engine/pgm'
 * chainVocabulary(linearChainCrf(['A', 'B'], ['f', 'g'])).attributes.get('g') // => 1
 * ```
 */
export function chainVocabulary(crf: LinearChainCrf): ChainVocabulary {
  const index = (names: readonly string[]): Map<string, number> => new Map(names.map((a, i) => [a, i]))
  return {
    layout: chainLayout(crf),
    attributes: index(crf.attributes),
    bigramAttributes: index(crf.bigramAttributes),
    labels: index(crf.labels),
  }
}

/**
 * The input of one sequence with its attributes resolved to ids: compressed sparse rows over the items (and over the
 * edges, for bigram attributes). Attributes outside the vocabulary are dropped, as they are at test time in CRFsuite.
 */
export interface CompiledChainInput {
  readonly kind: 'compiledChain'
  /** The number of items. */
  readonly length: number
  /** The attributes of item t are `attribute[itemStart[t] .. itemStart[t + 1])` with `value` alongside. */
  readonly itemStart: Int32Array
  readonly attribute: Int32Array
  readonly value: Float64Array
  /** The same for the edge into item t (t >= 1); absent when the model has no bigram attributes. */
  readonly edgeStart?: Int32Array
  readonly edgeAttribute?: Int32Array
  readonly edgeValue?: Float64Array
  /** The vocabulary sizes it was compiled against (checked when it is used). */
  readonly attributes: number
  readonly bigramAttributes: number
}

/** A sequence of a CRF's dataset, compiled: the input as ids and the labels as label ids. */
export type CompiledChainExample = { readonly x: CompiledChainInput; readonly y: readonly number[] }

/**
 * Whether an input is a compiled one.
 *
 * @param x - A CRF input, compiled or not.
 * @returns True for {@link CompiledChainInput}.
 * @example
 * ```ts
 * import { isCompiledChainInput } from 'omai-engine/pgm'
 * isCompiledChainInput([{ f: 1 }]) // => false
 * ```
 */
export function isCompiledChainInput(x: unknown): x is CompiledChainInput {
  return typeof x === 'object' && x !== null && (x as { kind?: unknown }).kind === 'compiledChain'
}

/**
 * Compile the input of one sequence against a vocabulary.
 *
 * @param vocabulary - The model's vocabulary maps ({@link chainVocabulary}).
 * @param x - The item (and edge) attributes.
 * @returns The compiled input.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { chainVocabulary, compileChainInput, linearChainCrf } from 'omai-engine/pgm'
 * const c = compileChainInput(chainVocabulary(linearChainCrf(['A', 'B'], ['f', 'g'])), [{ g: 2 }, { f: 1, h: 1 }])
 * [Array.from(c.itemStart), Array.from(c.attribute), Array.from(c.value)] // => [[0, 1, 2], [1, 0], [2, 1]]
 * ```
 */
export function compileChainInput(vocabulary: ChainVocabulary, x: CrfInput): CompiledChainInput {
  const items = 'unigram' in x ? x.unigram : x
  const edges = 'unigram' in x ? x.bigram : undefined
  const withEdges = vocabulary.layout.bigramAttributes > 0 && edges !== undefined
  const T = items.length
  const itemStart = new Int32Array(T + 1)
  const attribute: number[] = []
  const value: number[] = []
  const edgeStart = new Int32Array(T + 1)
  const edgeAttribute: number[] = []
  const edgeValue: number[] = []
  for (let t = 0; t < T; t++) {
    for (const [a, v] of Object.entries(items[t] ?? {})) {
      const i = vocabulary.attributes.get(a)
      if (i === undefined) continue
      attribute.push(i)
      value.push(v)
    }
    itemStart[t + 1] = attribute.length
    if (withEdges && t > 0)
      for (const [b, v] of Object.entries(edges?.[t] ?? {})) {
        const i = vocabulary.bigramAttributes.get(b)
        if (i === undefined) continue
        edgeAttribute.push(i)
        edgeValue.push(v)
      }
    edgeStart[t + 1] = edgeAttribute.length
  }
  return {
    kind: 'compiledChain',
    length: T,
    itemStart,
    attribute: Int32Array.from(attribute),
    value: Float64Array.from(value),
    ...(withEdges
      ? { edgeStart, edgeAttribute: Int32Array.from(edgeAttribute), edgeValue: Float64Array.from(edgeValue) }
      : {}),
    attributes: vocabulary.layout.attributes,
    bigramAttributes: vocabulary.layout.bigramAttributes,
  }
}

/**
 * Label ids of a labelling given as names or ids.
 *
 * @param vocabulary - The model's vocabulary maps.
 * @param y - The labels.
 * @returns The label ids.
 * @throws OmaiError with code `value` for an unknown label.
 * @example
 * ```ts
 * import { chainLabelIds, chainVocabulary, linearChainCrf } from 'omai-engine/pgm'
 * chainLabelIds(chainVocabulary(linearChainCrf(['A', 'B'], ['f'])), ['B', 0]) // => [1, 0]
 * ```
 */
export function chainLabelIds(vocabulary: ChainVocabulary, y: readonly (string | number)[]): number[] {
  return y.map((l) => {
    const i = typeof l === 'number' ? l : vocabulary.labels.get(l)
    if (i === undefined || !(i >= 0 && i < vocabulary.layout.labels))
      throw new OmaiError('value', `crf: unknown label ${String(l)}`)
    return i
  })
}

/**
 * Compile a labelled dataset once: every sequence's attributes to ids in typed arrays and its labels to label ids.
 * Training on the result (the oracle of `chainOracle` accepts it) does no string lookups per evaluation.
 *
 * @param crf - The model (its vocabularies).
 * @param data - Labelled sequences.
 * @returns The compiled sequences, in the same order.
 * @throws OmaiError with code `shape` for a different number of labels and items, `value` for an unknown label.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { compileChainData, linearChainCrf } from 'omai-engine/pgm'
 * const c = compileChainData(linearChainCrf(['A', 'B'], ['f']), [{ x: [{ f: 1 }, { f: 2 }], y: ['A', 'B'] }])
 * [c[0]?.x.length, c[0]?.y] // => [2, [0, 1]]
 * ```
 */
export function compileChainData(crf: LinearChainCrf, data: readonly LabelledSequence[]): CompiledChainExample[] {
  const vocabulary = chainVocabulary(crf)
  return data.map(({ x, y }) => {
    const input = compileChainInput(vocabulary, x)
    if (y.length !== input.length) throw new OmaiError('shape', 'compileChainData: one label per item')
    return { x: input, y: chainLabelIds(vocabulary, y) }
  })
}

/** Check that a compiled input belongs to a layout. */
export function checkCompiled(layout: ChainLayout, x: CompiledChainInput): void {
  if (x.attributes !== layout.attributes || x.bigramAttributes !== layout.bigramAttributes)
    throw new OmaiError('shape', 'compiled chain input: compiled against a different vocabulary')
}

/**
 * The potentials of a compiled sequence under a weight vector, as a chain for `forwardBackward` and `viterbi`: the
 * unary log potentials by one pass over the typed arrays, the transition matrix (plain transitions) and, for items
 * whose edge has bigram attributes, that edge's own matrix.
 *
 * @param layout - The parameter layout.
 * @param w - The weights, in `crfParameters` order.
 * @param x - The compiled input.
 * @param cost - Labels (ids) to add 1 to the unary potential of every other label, for cost-augmented decoding.
 * @returns The chain.
 * @example
 * ```ts
 * import { chainLayout, compileChainData, compiledChain, linearChainCrf } from 'omai-engine/pgm'
 * const crf = linearChainCrf(['A', 'B'], ['f'])
 * const c = compileChainData(crf, [{ x: [{ f: 2 }], y: ['A'] }])
 * compiledChain(chainLayout(crf), Float64Array.of(1, -1, 0, 0, 0, 0), c[0]!.x).logEmission // => [[2, -2]]
 * ```
 */
export function compiledChain(
  layout: ChainLayout,
  w: ArrayLike<number>,
  x: CompiledChainInput,
  cost?: readonly number[],
): ChainModel {
  checkCompiled(layout, x)
  const K = layout.labels
  const T = x.length
  const logEmission: number[][] = new Array(T)
  for (let t = 0; t < T; t++) {
    const row = new Array<number>(K).fill(0)
    for (let p = x.itemStart[t] ?? 0; p < (x.itemStart[t + 1] ?? 0); p++) {
      const base = (x.attribute[p] ?? 0) * K
      const v = x.value[p] ?? 0
      for (let k = 0; k < K; k++) row[k] = (row[k] ?? 0) + v * (w[base + k] ?? 0)
    }
    if (cost !== undefined) for (let k = 0; k < K; k++) if (k !== cost[t]) row[k] = (row[k] ?? 0) + 1
    logEmission[t] = row
  }
  const plain = Array.from({ length: K }, (_, i) =>
    Array.from({ length: K }, (_, j) => (layout.transitionSize > 0 ? (w[layout.stateSize + i * K + j] ?? 0) : 0)),
  )
  const logInitial = new Array<number>(K).fill(0)
  if (x.edgeStart === undefined || T < 2) return { logInitial, logTransition: plain, logEmission }
  const bigramAt = layout.stateSize + layout.transitionSize
  const logTransition = Array.from({ length: T - 1 }, (_, e) => {
    const t = e + 1
    if ((x.edgeStart?.[t] ?? 0) === (x.edgeStart?.[t + 1] ?? 0)) return plain
    const m = plain.map((r) => r.slice())
    for (let p = x.edgeStart?.[t] ?? 0; p < (x.edgeStart?.[t + 1] ?? 0); p++) {
      const base = bigramAt + (x.edgeAttribute?.[p] ?? 0) * K * K
      const v = x.edgeValue?.[p] ?? 0
      for (let i = 0; i < K; i++) {
        const row = m[i] as number[]
        for (let j = 0; j < K; j++) row[j] = (row[j] ?? 0) + v * (w[base + (i * K + j)] ?? 0)
      }
    }
    return m
  })
  return { logInitial, logTransition, logEmission }
}

/**
 * Add `scale` times the feature counts phi(x, y) of a labelling (the state, transition and bigram features it switches
 * on, counted with the attribute values) to a vector, for a compiled sequence.
 *
 * @param layout - The parameter layout.
 * @param x - The compiled input.
 * @param y - The label ids.
 * @param into - The vector to add to (length `layout.size`).
 * @param scale - The multiplier (default 1; -1 subtracts).
 * @example
 * ```ts
 * import { addCompiledFeatureCounts, chainLayout, compileChainData, linearChainCrf } from 'omai-engine/pgm'
 * const crf = linearChainCrf(['A', 'B'], ['f'])
 * const [c] = compileChainData(crf, [{ x: [{ f: 2 }, { f: 1 }], y: ['A', 'B'] }])
 * const out = new Float64Array(6)
 * addCompiledFeatureCounts(chainLayout(crf), c!.x, c!.y, out)
 * Array.from(out) // => [2, 1, 0, 1, 0, 0]
 * ```
 */
export function addCompiledFeatureCounts(
  layout: ChainLayout,
  x: CompiledChainInput,
  y: readonly number[],
  into: Float64Array,
  scale = 1,
): void {
  const K = layout.labels
  const bigramAt = layout.stateSize + layout.transitionSize
  for (let t = 0; t < x.length; t++) {
    const yt = y[t] ?? 0
    for (let p = x.itemStart[t] ?? 0; p < (x.itemStart[t + 1] ?? 0); p++) {
      const at = (x.attribute[p] ?? 0) * K + yt
      into[at] = (into[at] ?? 0) + scale * (x.value[p] ?? 0)
    }
    if (t === 0) continue
    const prev = y[t - 1] ?? 0
    if (layout.transitionSize > 0) {
      const at = layout.stateSize + prev * K + yt
      into[at] = (into[at] ?? 0) + scale
    }
    if (x.edgeStart === undefined) continue
    for (let p = x.edgeStart[t] ?? 0; p < (x.edgeStart[t + 1] ?? 0); p++) {
      const at = bigramAt + ((x.edgeAttribute?.[p] ?? 0) * K + prev) * K + yt
      into[at] = (into[at] ?? 0) + scale * (x.edgeValue?.[p] ?? 0)
    }
  }
}

/**
 * Add `scale` times the expected feature counts E[phi(x, Y)] of a compiled sequence to a vector, from the posteriors of
 * `forwardBackward` on its chain: the unigram marginals for the state features, the pairwise marginals for the
 * transition and bigram features.
 *
 * @param layout - The parameter layout.
 * @param x - The compiled input.
 * @param posteriors - The marginals p(y_t), `[T][K]`.
 * @param pairwise - The pairwise marginals of consecutive labels, `[T - 1][K][K]`.
 * @param into - The vector to add to (length `layout.size`).
 * @param scale - The multiplier (default 1).
 * @example
 * ```ts
 * import { addCompiledExpectedCounts, chainLayout, compileChainData, linearChainCrf } from 'omai-engine/pgm'
 * const crf = linearChainCrf(['A', 'B'], ['f'])
 * const [c] = compileChainData(crf, [{ x: [{ f: 1 }], y: ['A'] }])
 * const out = new Float64Array(6)
 * addCompiledExpectedCounts(chainLayout(crf), c!.x, [[0.25, 0.75]], [], out)
 * Array.from(out) // => [0.25, 0.75, 0, 0, 0, 0]
 * ```
 */
export function addCompiledExpectedCounts(
  layout: ChainLayout,
  x: CompiledChainInput,
  posteriors: readonly (readonly number[])[],
  pairwise: readonly (readonly (readonly number[])[])[],
  into: Float64Array,
  scale = 1,
): void {
  const K = layout.labels
  const bigramAt = layout.stateSize + layout.transitionSize
  for (let t = 0; t < x.length; t++) {
    const post = posteriors[t] ?? []
    for (let p = x.itemStart[t] ?? 0; p < (x.itemStart[t + 1] ?? 0); p++) {
      const base = (x.attribute[p] ?? 0) * K
      const v = scale * (x.value[p] ?? 0)
      for (let k = 0; k < K; k++) into[base + k] = (into[base + k] ?? 0) + v * (post[k] ?? 0)
    }
    if (t === 0) continue
    const xi = pairwise[t - 1] ?? []
    if (layout.transitionSize > 0)
      for (let i = 0; i < K; i++)
        for (let j = 0; j < K; j++) {
          const at = layout.stateSize + i * K + j
          into[at] = (into[at] ?? 0) + scale * (xi[i]?.[j] ?? 0)
        }
    if (x.edgeStart === undefined) continue
    for (let p = x.edgeStart[t] ?? 0; p < (x.edgeStart[t + 1] ?? 0); p++) {
      const base = bigramAt + (x.edgeAttribute?.[p] ?? 0) * K * K
      const v = scale * (x.edgeValue?.[p] ?? 0)
      for (let i = 0; i < K; i++)
        for (let j = 0; j < K; j++) into[base + i * K + j] = (into[base + i * K + j] ?? 0) + v * (xi[i]?.[j] ?? 0)
    }
  }
}

/**
 * The unnormalised score w . phi(x, y) of a labelling of a compiled sequence, without forming the feature vector.
 *
 * @param layout - The parameter layout.
 * @param w - The weights, in `crfParameters` order.
 * @param x - The compiled input.
 * @param y - The label ids.
 * @returns The score.
 * @example
 * ```ts
 * import { chainLayout, compileChainData, compiledScore, linearChainCrf } from 'omai-engine/pgm'
 * const crf = linearChainCrf(['A', 'B'], ['f'])
 * const [c] = compileChainData(crf, [{ x: [{ f: 1 }, { f: 1 }], y: ['A', 'B'] }])
 * compiledScore(chainLayout(crf), [1, -1, 0, 3, 0, 0], c!.x, c!.y) // => 3
 * ```
 */
export function compiledScore(
  layout: ChainLayout,
  w: ArrayLike<number>,
  x: CompiledChainInput,
  y: readonly number[],
): number {
  const K = layout.labels
  const bigramAt = layout.stateSize + layout.transitionSize
  let s = 0
  for (let t = 0; t < x.length; t++) {
    const yt = y[t] ?? 0
    for (let p = x.itemStart[t] ?? 0; p < (x.itemStart[t + 1] ?? 0); p++)
      s += (x.value[p] ?? 0) * (w[(x.attribute[p] ?? 0) * K + yt] ?? 0)
    if (t === 0) continue
    const prev = y[t - 1] ?? 0
    if (layout.transitionSize > 0) s += w[layout.stateSize + prev * K + yt] ?? 0
    if (x.edgeStart === undefined) continue
    for (let p = x.edgeStart[t] ?? 0; p < (x.edgeStart[t + 1] ?? 0); p++)
      s += (x.edgeValue?.[p] ?? 0) * (w[bigramAt + ((x.edgeAttribute?.[p] ?? 0) * K + prev) * K + yt] ?? 0)
  }
  return s
}
