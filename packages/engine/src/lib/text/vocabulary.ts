/**
 * Token vocabularies: building one from tokenised documents with frequency cut-offs, a size limit and special tokens
 * first, or from a fixed list, and mapping between tokens and ids with an unknown token for words outside it.
 */

import { OmaiError } from '../base/index.ts'
import { compareCodePoints } from './_classes.ts'

/**
 * A token vocabulary: tokens in id order, the id of each, and their training counts (0 for special tokens), with an
 * optional unknown token that out-of-vocabulary tokens map to.
 */
export interface Vocabulary {
  readonly tokens: readonly string[]
  readonly index: ReadonlyMap<string, number>
  readonly counts: readonly number[]
  readonly specials: readonly string[]
  readonly unk?: string
}

/** Options of {@link buildVocabulary}. */
export interface VocabularyOptions {
  /** Drop tokens seen fewer times (default 1). */
  readonly minCount?: number
  /** At most this many tokens, special tokens included (the most frequent are kept). */
  readonly maxSize?: number
  /** Special tokens, placed first in this order (torchtext's `specials`, HF's special tokens). */
  readonly specials?: readonly string[]
  /** The unknown token (added to the specials if missing). */
  readonly unk?: string
  /**
   * The order of the other tokens: `frequency` (descending, ties by code point, torchtext's default), `alphabetical`
   * (code point order, scikit-learn's) or `first` (first occurrence).
   */
  readonly order?: 'frequency' | 'alphabetical' | 'first'
}

/**
 * Build a vocabulary from tokenised documents: count tokens, drop the rare ones, keep the most frequent up to the size
 * limit, put the special tokens first and order the rest.
 *
 * @param documents - Token lists (or one token list).
 * @param options - Count threshold, size limit, specials, unknown token and order.
 * @returns The vocabulary.
 * @throws OmaiError with code `value` for repeated special tokens or a size limit below the number of specials.
 * @concept vocabulary
 * @example
 * ```ts
 * import { buildVocabulary } from 'omai-engine/text'
 * const v = buildVocabulary([['a', 'b', 'a'], ['c', 'b', 'a']], { specials: ['<pad>'], unk: '<unk>', maxSize: 4 })
 * [v.tokens, v.counts] // => [['<pad>', '<unk>', 'a', 'b'], [0, 0, 3, 2]]
 * ```
 */
export function buildVocabulary(
  documents: readonly (readonly string[])[] | readonly string[],
  options: VocabularyOptions = {},
): Vocabulary {
  const docs = (
    documents.length > 0 && typeof documents[0] === 'string' ? [documents] : documents
  ) as readonly (readonly string[])[]
  const specials = [...(options.specials ?? [])]
  if (options.unk !== undefined && !specials.includes(options.unk)) specials.push(options.unk)
  if (new Set(specials).size !== specials.length) throw new OmaiError('value', 'buildVocabulary: repeated specials')
  const counts = new Map<string, number>()
  for (const d of docs) for (const t of d) counts.set(t, (counts.get(t) ?? 0) + 1)
  for (const s of specials) counts.delete(s)
  const min = options.minCount ?? 1
  let entries = [...counts].filter(([, c]) => c >= min)
  const byFrequency = (a: readonly [string, number], b: readonly [string, number]): number =>
    b[1] - a[1] || compareCodePoints(a[0], b[0])
  if (options.maxSize !== undefined) {
    const room = options.maxSize - specials.length
    if (room < 0) throw new OmaiError('value', 'buildVocabulary: maxSize is smaller than the specials')
    if (entries.length > room) {
      const keep = new Set(
        [...entries]
          .sort(byFrequency)
          .slice(0, room)
          .map(([t]) => t),
      )
      entries = entries.filter(([t]) => keep.has(t))
    }
  }
  const order = options.order ?? 'frequency'
  if (order === 'frequency') entries.sort(byFrequency)
  else if (order === 'alphabetical') entries.sort((a, b) => compareCodePoints(a[0], b[0]))
  const tokens = [...specials, ...entries.map(([t]) => t)]
  return {
    tokens,
    index: new Map(tokens.map((t, i) => [t, i])),
    counts: [...specials.map(() => 0), ...entries.map(([, c]) => c)],
    specials,
    ...(options.unk === undefined ? {} : { unk: options.unk }),
  }
}

/**
 * A vocabulary from a fixed token list (ids in list order).
 *
 * @param tokens - The tokens.
 * @param options - Special tokens among them and the unknown token.
 * @returns The vocabulary.
 * @throws OmaiError with code `value` for repeated tokens.
 * @concept vocabulary
 * @example
 * ```ts
 * import { vocabularyFromTokens } from 'omai-engine/text'
 * vocabularyFromTokens(['[UNK]', 'a', 'b'], { unk: '[UNK]' }).index.get('b') // => 2
 * ```
 */
export function vocabularyFromTokens(
  tokens: readonly string[],
  options: { readonly specials?: readonly string[]; readonly unk?: string } = {},
): Vocabulary {
  if (new Set(tokens).size !== tokens.length) throw new OmaiError('value', 'vocabularyFromTokens: repeated tokens')
  const specials = [...(options.specials ?? [])]
  if (options.unk !== undefined && !specials.includes(options.unk)) specials.push(options.unk)
  return {
    tokens: [...tokens],
    index: new Map(tokens.map((t, i) => [t, i])),
    counts: tokens.map(() => 0),
    specials,
    ...(options.unk === undefined ? {} : { unk: options.unk }),
  }
}

/**
 * Token ids of a token list; out-of-vocabulary tokens map to the unknown token.
 *
 * @param vocab - The vocabulary.
 * @param tokens - The tokens.
 * @returns The ids.
 * @throws OmaiError with code `value` for an unknown token when the vocabulary has no unknown token.
 * @concept vocabulary
 * @example
 * ```ts
 * import { buildVocabulary, encodeTokens } from 'omai-engine/text'
 * encodeTokens(buildVocabulary([['a', 'b']], { unk: '<unk>' }), ['b', 'z']) // => [2, 0]
 * ```
 */
export function encodeTokens(vocab: Vocabulary, tokens: readonly string[]): number[] {
  const unk = vocab.unk === undefined ? undefined : vocab.index.get(vocab.unk)
  return tokens.map((t) => {
    const id = vocab.index.get(t) ?? unk
    if (id === undefined) throw new OmaiError('value', `encodeTokens: ${t} is not in the vocabulary`)
    return id
  })
}

/**
 * The tokens of a list of ids, optionally without special tokens.
 *
 * @param vocab - The vocabulary.
 * @param ids - The ids.
 * @param options - Whether to drop special tokens.
 * @returns The tokens.
 * @throws OmaiError with code `value` for an id outside the vocabulary.
 * @concept vocabulary
 * @example
 * ```ts
 * import { buildVocabulary, decodeTokens } from 'omai-engine/text'
 * decodeTokens(buildVocabulary([['a', 'b']], { specials: ['<s>'] }), [0, 1, 2], { skipSpecials: true }) // => ['a', 'b']
 * ```
 */
export function decodeTokens(
  vocab: Vocabulary,
  ids: readonly number[],
  options: { readonly skipSpecials?: boolean } = {},
): string[] {
  const specials = new Set(vocab.specials)
  const out: string[] = []
  for (const id of ids) {
    const t = vocab.tokens[id]
    if (t === undefined) throw new OmaiError('value', `decodeTokens: id ${id} is out of range`)
    if (options.skipSpecials === true && specials.has(t)) continue
    out.push(t)
  }
  return out
}
