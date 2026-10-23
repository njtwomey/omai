import { WhiteboxError } from '../../base/index.ts'
import type { Tensor, TensorLike } from '../../tensor/index.ts'
import { checkRows, concrete, metaFor } from './_coerce.ts'
import type { DatasetMeta } from './common.ts'

/**
 * A collection of documents, as raw text, as token-id sequences into a vocabulary, or both, with optional labels.
 */
export interface CorpusData {
  readonly kind: 'corpus'
  /** Raw text, one string per document. */
  readonly documents?: readonly string[]
  /** Token ids per document, indices into `vocabulary`. */
  readonly tokens?: readonly (readonly number[])[]
  /** A class per document `[n]` (int32). */
  readonly labels?: Tensor<'int32'>
  readonly classes?: readonly string[]
  readonly vocabulary?: readonly string[]
  readonly meta: DatasetMeta
}

/** Token sequences with one tag per token (part of speech, named entities, chunks). */
export interface SequenceLabellingData {
  readonly kind: 'sequence-labelling'
  /** The tokens of each sentence. */
  readonly tokens: readonly (readonly string[])[]
  /** Tag ids per token, the same lengths as `tokens`. */
  readonly tags: readonly (readonly number[])[]
  readonly tagNames: readonly string[]
  readonly meta: DatasetMeta
}

/** Aligned pairs of texts (translations, paraphrases, questions and answers). */
export interface ParallelData {
  readonly kind: 'parallel'
  readonly source: readonly string[]
  readonly target: readonly string[]
  readonly sourceLanguage?: string
  readonly targetLanguage?: string
  readonly meta: DatasetMeta
}

/** Options of {@link corpus}. */
export interface CorpusOptions {
  readonly documents?: readonly string[]
  readonly tokens?: readonly (readonly number[])[]
  readonly labels?: TensorLike
  readonly classes?: readonly string[]
  readonly vocabulary?: readonly string[]
}

/**
 * A corpus from documents and/or token sequences, checking that the per-document fields agree and that token ids lie
 * in the vocabulary.
 *
 * @param options - Documents, tokens, labels, classes and vocabulary.
 * @param meta - Name, description and provenance.
 * @returns The corpus.
 * @throws WhiteboxError with code `value` when there are neither documents nor tokens, or a token id is out of range.
 * @example
 * ```ts
 * import { corpus } from 'whitebox-engine/data'
 * corpus({ tokens: [[0, 1], [1]], vocabulary: ['a', 'b'] }).tokens?.length // => 2
 * ```
 */
export function corpus(options: CorpusOptions, meta: Partial<DatasetMeta> = {}): CorpusData {
  const n = options.documents?.length ?? options.tokens?.length
  if (n === undefined) throw new WhiteboxError('value', 'corpus: give documents or tokens')
  const labels = options.labels === undefined ? undefined : concrete(options.labels, 'int32', 'corpus')
  checkRows('corpus', n, { documents: options.documents, tokens: options.tokens, labels })
  const v = options.vocabulary?.length
  if (v !== undefined && options.tokens !== undefined) {
    for (const doc of options.tokens) {
      for (const t of doc) {
        if (!(t >= 0 && t < v)) throw new WhiteboxError('value', `corpus: token id ${t} outside a vocabulary of ${v}`)
      }
    }
  }
  return {
    kind: 'corpus',
    ...(options.documents === undefined ? {} : { documents: options.documents }),
    ...(options.tokens === undefined ? {} : { tokens: options.tokens }),
    ...(labels === undefined ? {} : { labels }),
    ...(options.classes === undefined ? {} : { classes: options.classes }),
    ...(options.vocabulary === undefined ? {} : { vocabulary: options.vocabulary }),
    meta: metaFor('corpus', meta),
  }
}

/**
 * A sequence-labelling dataset, checking that every sentence has one tag per token.
 *
 * @param tokens - Tokens per sentence.
 * @param tags - Tag ids per token.
 * @param tagNames - The tag set.
 * @param meta - Name, description and provenance.
 * @returns The dataset.
 * @throws WhiteboxError with code `shape` when a sentence's tags and tokens differ in length.
 * @example
 * ```ts
 * import { sequenceLabelling } from 'whitebox-engine/data'
 * sequenceLabelling([['the', 'cat']], [[0, 1]], ['DET', 'NOUN']).tagNames.length // => 2
 * ```
 */
export function sequenceLabelling(
  tokens: readonly (readonly string[])[],
  tags: readonly (readonly number[])[],
  tagNames: readonly string[],
  meta: Partial<DatasetMeta> = {},
): SequenceLabellingData {
  checkRows('sequenceLabelling', tokens.length, { tags })
  tokens.forEach((sentence, i) => {
    if ((tags[i]?.length ?? -1) !== sentence.length)
      throw new WhiteboxError(
        'shape',
        `sequenceLabelling: sentence ${i} has ${sentence.length} tokens and ${tags[i]?.length} tags`,
      )
  })
  return { kind: 'sequence-labelling', tokens, tags, tagNames, meta: metaFor('sequence-labelling', meta) }
}

/**
 * A parallel corpus of aligned text pairs.
 *
 * @param source - Source texts.
 * @param target - Target texts, aligned with the sources.
 * @param meta - Languages, name and provenance.
 * @returns The dataset.
 * @throws WhiteboxError with code `shape` when the sides differ in length.
 * @example
 * ```ts
 * import { parallel } from 'whitebox-engine/data'
 * parallel(['one', 'two'], ['un', 'deux'], { sourceLanguage: 'en', targetLanguage: 'fr' }).target[1] // => 'deux'
 * ```
 */
export function parallel(
  source: readonly string[],
  target: readonly string[],
  meta: Partial<DatasetMeta> & { readonly sourceLanguage?: string; readonly targetLanguage?: string } = {},
): ParallelData {
  checkRows('parallel', source.length, { target })
  const { sourceLanguage, targetLanguage, ...rest } = meta
  return {
    kind: 'parallel',
    source,
    target,
    ...(sourceLanguage === undefined ? {} : { sourceLanguage }),
    ...(targetLanguage === undefined ? {} : { targetLanguage }),
    meta: metaFor('parallel', rest),
  }
}
