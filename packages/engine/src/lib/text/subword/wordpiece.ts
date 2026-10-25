/**
 * WordPiece, the subword tokeniser of BERT: words are encoded by greedy longest-match-first lookup in a vocabulary,
 * with continuation pieces marked by a prefix. Models from a vocabulary, encoding as HF tokenizers does, and vocabulary
 * training.
 */

import { MdmlError } from '../../base/index.ts'
import type { TracePolicy, Tracer } from '../../trace/index.ts'
import { compareCodePoints } from '../_classes.ts'
import { type BpeTrainingOptions, type SubwordToken, trainBpe } from './bpe.ts'

/** A WordPiece model (HF tokenizers' `models.WordPiece`). */
export interface WordPieceModel {
  readonly vocab: ReadonlyMap<string, number>
  /** The unknown token (default `[UNK]`). */
  readonly unkToken: string
  /** The prefix of word-internal pieces (default `##`). */
  readonly continuingSubwordPrefix: string
  /** Longer words become the unknown token (default 100 characters). */
  readonly maxInputCharsPerWord: number
}

/**
 * A WordPiece model from a vocabulary.
 *
 * @param vocab - Token to id (or a token list, ids in order).
 * @param options - Unknown token, prefix and word length limit.
 * @returns The model.
 * @throws MdmlError with code `value` when the unknown token is not in the vocabulary.
 * @concept wordpiece
 * @example
 * ```ts
 * import { wordPieceModel } from 'mdml-engine/text/subword'
 * wordPieceModel(['[UNK]', 'un', '##aff', '##able']).vocab.get('##able') // => 3
 * ```
 */
export function wordPieceModel(
  vocab: ReadonlyMap<string, number> | readonly string[],
  options: {
    readonly unkToken?: string
    readonly continuingSubwordPrefix?: string
    readonly maxInputCharsPerWord?: number
  } = {},
): WordPieceModel {
  const map = Array.isArray(vocab)
    ? new Map((vocab as readonly string[]).map((t, i) => [t, i]))
    : new Map(vocab as ReadonlyMap<string, number>)
  const unkToken = options.unkToken ?? '[UNK]'
  if (!map.has(unkToken))
    throw new MdmlError('value', `wordPieceModel: the unknown token ${unkToken} is not in the vocabulary`)
  return {
    vocab: map,
    unkToken,
    continuingSubwordPrefix: options.continuingSubwordPrefix ?? '##',
    maxInputCharsPerWord: options.maxInputCharsPerWord ?? 100,
  }
}

/**
 * Greedy longest-match-first WordPiece encoding of one word (Wu et al. 2016; HF's WordPiece.tokenize): repeatedly
 * take the longest vocabulary piece from the current position (with the prefix after the first piece); a word with an
 * unmatched remainder, or longer than the limit, becomes a single unknown token.
 *
 * @param model - The model.
 * @param word - The pre-tokenised word.
 * @returns The pieces with ids and offsets in the word.
 * @concept wordpiece
 * @example
 * ```ts
 * import { wordPieceModel, wordPieceTokenize } from 'mdml-engine/text/subword'
 * const m = wordPieceModel(['[UNK]', 'un', '##aff', '##able', 'a'])
 * [wordPieceTokenize(m, 'unaffable').map((t) => t.token), wordPieceTokenize(m, 'unx').map((t) => t.token)] // => [['un', '##aff', '##able'], ['[UNK]']]
 * ```
 */
export function wordPieceTokenize(model: WordPieceModel, word: string): SubwordToken[] {
  if (word.length === 0) return []
  const unk = { token: model.unkToken, id: model.vocab.get(model.unkToken) ?? 0, start: 0, end: word.length }
  if (Array.from(word).length > model.maxInputCharsPerWord) return [unk]
  const out: SubwordToken[] = []
  let start = 0
  while (start < word.length) {
    let end = word.length
    let found: SubwordToken | undefined
    while (start < end) {
      const piece = (start > 0 ? model.continuingSubwordPrefix : '') + word.slice(start, end)
      const id = model.vocab.get(piece)
      if (id !== undefined) {
        found = { token: piece, id, start, end }
        break
      }
      const last = word.codePointAt(end - 1) ?? 0
      // Step back one code point (a low surrogate ends a pair).
      end -= last >= 0xdc00 && last <= 0xdfff && end >= 2 ? 2 : 1
    }
    if (found === undefined) return [unk]
    out.push(found)
    start = found.end
  }
  return out
}

/** Options of WordPiece training. */
export interface WordPieceTrainingOptions extends Omit<BpeTrainingOptions, 'endOfWordSuffix' | 'byteLevel'> {
  /** The unknown token, added to the specials (default `[UNK]`). */
  readonly unkToken?: string
  /**
   * How pairs are chosen: `frequency` (default; HF's WordPieceTrainer, which trains BPE with the `##` prefix) or
   * `likelihood` (Schuster and Nakajima 2012, the original WordPiece: the pair maximising count(ab) / (count(a)
   * count(b)), ties by the earliest pair).
   */
  readonly scoring?: 'frequency' | 'likelihood'
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/**
 * Train a WordPiece vocabulary on word counts. With `frequency` scoring this is HF's WordPieceTrainer: BPE training
 * with the `##` continuing-subword prefix, whose vocabulary becomes the WordPiece vocabulary.
 *
 * @param wordCounts - Pre-tokenised words with their counts.
 * @param options - Vocabulary size, frequency floor, specials, alphabet limits, prefix, scoring.
 * @returns The model.
 * @concept wordpiece
 * @example
 * ```ts
 * import { trainWordPiece, wordPieceTokenize } from 'mdml-engine/text/subword'
 * const m = trainWordPiece(new Map([['hugging', 10], ['hug', 5], ['hugs', 6]]), { vocabSize: 14 })
 * wordPieceTokenize(m, 'hugs').map((t) => t.token) // => ['hug', '##s']
 * ```
 */
export function trainWordPiece(
  wordCounts: ReadonlyMap<string, number> | Iterable<readonly [string, number]>,
  options: WordPieceTrainingOptions = {},
): WordPieceModel {
  const unkToken = options.unkToken ?? '[UNK]'
  const prefix = options.continuingSubwordPrefix ?? '##'
  const specials = [...(options.specialTokens ?? [])]
  if (!specials.includes(unkToken)) specials.push(unkToken)
  if (options.scoring === 'likelihood')
    return trainLikelihood([...wordCounts], { ...options, specialTokens: specials }, unkToken, prefix)
  const bpe = trainBpe(wordCounts, { ...options, specialTokens: specials, continuingSubwordPrefix: prefix })
  return wordPieceModel(bpe.vocab, { unkToken, continuingSubwordPrefix: prefix })
}

/** WordPiece training by the likelihood score, with a direct scan of the pairs each step. */
function trainLikelihood(
  entries: readonly (readonly [string, number])[],
  options: WordPieceTrainingOptions,
  unkToken: string,
  prefix: string,
): WordPieceModel {
  const vocabSize = options.vocabSize ?? 30000
  const tokens: string[] = []
  const seen = new Set<string>()
  const add = (t: string): void => {
    if (!seen.has(t)) {
      seen.add(t)
      tokens.push(t)
    }
  }
  for (const s of options.specialTokens ?? []) add(s)
  const words = entries.map(([w, c]) => ({
    symbols: Array.from(w).map((ch, k) => (k > 0 ? prefix + ch : ch)),
    count: c,
  }))
  const alphabet = [...new Set(words.flatMap((w) => w.symbols))].sort(compareCodePoints)
  for (const a of alphabet) add(a)
  const minFrequency = options.minFrequency ?? 0
  // Each step merges one pair everywhere, so steps are bounded by the number of symbols.
  const cap = words.reduce((n, w) => n + w.symbols.length, 0)
  for (let step = 0; step < cap && tokens.length < vocabSize; step++) {
    const unit = new Map<string, number>()
    const pairs = new Map<string, number>()
    for (const w of words) {
      for (const [k, s] of w.symbols.entries()) {
        unit.set(s, (unit.get(s) ?? 0) + w.count)
        const next = w.symbols[k + 1]
        if (next !== undefined) {
          const key = `${s}\u0000${next}`
          pairs.set(key, (pairs.get(key) ?? 0) + w.count)
        }
      }
    }
    let best: string | undefined
    let bestScore = -1
    for (const [key, c] of pairs) {
      if (c < minFrequency || c < 1) continue
      const [a = '', b = ''] = key.split('\u0000')
      const score = c / ((unit.get(a) ?? 1) * (unit.get(b) ?? 1))
      if (score > bestScore) {
        bestScore = score
        best = key
      }
    }
    if (best === undefined) break
    const [a = '', b = ''] = best.split('\u0000')
    const merged = a + (b.startsWith(prefix) ? b.slice(prefix.length) : b)
    add(merged)
    for (const w of words) {
      const next: string[] = []
      for (let k = 0; k < w.symbols.length; k++) {
        if (w.symbols[k] === a && w.symbols[k + 1] === b) {
          next.push(merged)
          k++
        } else next.push(w.symbols[k] ?? '')
      }
      w.symbols = next
    }
  }
  return wordPieceModel(tokens, { unkToken, continuingSubwordPrefix: prefix })
}
