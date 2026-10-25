/**
 * Loading HF tokenizers' `tokenizer.json` files: the added tokens, normaliser, pre-tokeniser, model (BPE, WordPiece,
 * Unigram or WordLevel), post-processor, decoder, truncation and padding are read into a ready tokenizer pipeline.
 */

import { WhiteboxError } from '../base/index.ts'
import { bpeModel } from './subword/bpe.ts'
import { unigramModel } from './subword/unigram.ts'
import { wordPieceModel } from './subword/wordpiece.ts'
import type { SplitBehavior } from './tokenise.ts'
import type {
  AddedToken,
  PaddingOptions,
  PostProcessorSpec,
  TemplatePiece,
  Tokenizer,
  TokenizerModel,
  TruncationOptions,
} from './tokenizer.ts'
import { tokenizer } from './tokenizer.ts'
import {
  type DecoderSpec,
  hfRegex,
  type NormalizerSpec,
  type PreTokenizerSpec,
  type TokenizerPattern,
} from './tokenizer-components.ts'

type Obj = Readonly<Record<string, unknown>>

const obj = (x: unknown, where: string): Obj => {
  if (x === null || typeof x !== 'object' || Array.isArray(x))
    throw new WhiteboxError('schema', `loadTokenizer: ${where} must be an object`)
  return x as Obj
}
const str = (x: unknown, fallback?: string): string => (typeof x === 'string' ? x : (fallback ?? ''))
const num = (x: unknown, fallback = 0): number => (typeof x === 'number' ? x : fallback)
const bool = (x: unknown, fallback: boolean): boolean => (typeof x === 'boolean' ? x : fallback)
const list = (x: unknown): readonly unknown[] => (Array.isArray(x) ? x : [])

function notImplemented(kind: string, type: string): never {
  throw new WhiteboxError('not-implemented', `loadTokenizer: the ${kind} ${type} is not supported`)
}

function pattern(x: unknown): TokenizerPattern {
  const o = obj(x, 'pattern')
  if (typeof o['String'] === 'string') return o['String']
  if (typeof o['Regex'] === 'string') return hfRegex(o['Regex'])
  throw new WhiteboxError('schema', 'loadTokenizer: a pattern is a String or a Regex')
}

const BEHAVIORS: Readonly<Record<string, SplitBehavior>> = {
  Removed: 'removed',
  Isolated: 'isolated',
  MergedWithPrevious: 'mergedWithPrevious',
  MergedWithNext: 'mergedWithNext',
  Contiguous: 'contiguous',
}
const behavior = (x: unknown): SplitBehavior => BEHAVIORS[str(x, 'Isolated')] ?? 'isolated'
const scheme = (x: unknown): 'always' | 'first' | 'never' => {
  const s = str(x, 'always')
  return s === 'first' || s === 'never' ? s : 'always'
}

function normalizer(x: unknown): NormalizerSpec {
  const o = obj(x, 'normalizer')
  const type = str(o['type'])
  switch (type) {
    case 'NFC':
    case 'NFD':
    case 'NFKC':
    case 'NFKD':
    case 'Lowercase':
    case 'StripAccents':
    case 'Nmt':
    case 'ByteLevel':
      return { type }
    case 'Strip':
      return { type, left: bool(o['strip_left'], true), right: bool(o['strip_right'], true) }
    case 'Replace':
      return { type, pattern: pattern(o['pattern']), content: str(o['content']) }
    case 'Prepend':
      return { type, prepend: str(o['prepend']) }
    case 'BertNormalizer':
      return {
        type,
        cleanText: bool(o['clean_text'], true),
        handleChineseChars: bool(o['handle_chinese_chars'], true),
        lowercase: bool(o['lowercase'], true),
        ...(typeof o['strip_accents'] === 'boolean' ? { stripAccents: o['strip_accents'] } : {}),
      }
    case 'Sequence':
      return { type, normalizers: list(o['normalizers']).map(normalizer) }
    default:
      return notImplemented('normalizer', type)
  }
}

function preTokenizer(x: unknown): PreTokenizerSpec {
  const o = obj(x, 'pre_tokenizer')
  const type = str(o['type'])
  switch (type) {
    case 'Whitespace':
    case 'WhitespaceSplit':
    case 'BertPreTokenizer':
      return { type }
    case 'ByteLevel':
      return { type, addPrefixSpace: bool(o['add_prefix_space'], true), useRegex: bool(o['use_regex'], true) }
    case 'Metaspace':
      return {
        type,
        replacement: str(o['replacement'], '▁'),
        prependScheme: scheme(o['prepend_scheme']),
        split: bool(o['split'], true),
      }
    case 'Punctuation':
      return { type, behavior: behavior(o['behavior']) }
    case 'Digits':
      return { type, individualDigits: bool(o['individual_digits'], false) }
    case 'Split':
      return {
        type,
        pattern: pattern(o['pattern']),
        behavior: behavior(o['behavior']),
        invert: bool(o['invert'], false),
      }
    case 'CharDelimiterSplit':
      return { type, delimiter: str(o['delimiter']) }
    case 'FixedLength':
      return { type, length: num(o['length'], 5) }
    case 'Sequence':
      return { type, pretokenizers: list(o['pretokenizers']).map(preTokenizer) }
    default:
      return notImplemented('pre-tokenizer', type)
  }
}

function decoder(x: unknown): DecoderSpec {
  const o = obj(x, 'decoder')
  const type = str(o['type'])
  switch (type) {
    case 'ByteLevel':
    case 'Fuse':
    case 'ByteFallback':
      return { type }
    case 'WordPiece':
      return { type, prefix: str(o['prefix'], '##'), cleanup: bool(o['cleanup'], true) }
    case 'Metaspace':
      return { type, replacement: str(o['replacement'], '▁'), prependScheme: scheme(o['prepend_scheme']) }
    case 'BPEDecoder':
      return { type, suffix: str(o['suffix'], '</w>') }
    case 'Replace':
      return { type, pattern: pattern(o['pattern']), content: str(o['content']) }
    case 'Strip':
      return { type, content: str(o['content'], ' '), start: num(o['start']), stop: num(o['stop']) }
    case 'CTC':
      return {
        type,
        padToken: str(o['pad_token'], '<pad>'),
        wordDelimiterToken: str(o['word_delimiter_token'], '|'),
        cleanup: bool(o['cleanup'], true),
      }
    case 'Sequence':
      return { type, decoders: list(o['decoders']).map(decoder) }
    default:
      return notImplemented('decoder', type)
  }
}

const pairOf = (x: unknown): readonly [string, number] => {
  const a = list(x)
  return [str(a[0]), num(a[1])]
}

function templatePieces(x: unknown): TemplatePiece[] {
  return list(x).map((p) => {
    const o = obj(p, 'template piece')
    if (o['Sequence'] !== undefined) {
      const s = obj(o['Sequence'], 'Sequence')
      return { sequence: str(s['id'], 'A') === 'B' ? 'B' : 'A', typeId: num(s['type_id']) }
    }
    const s = obj(o['SpecialToken'], 'SpecialToken')
    return { special: str(s['id']), typeId: num(s['type_id']) }
  })
}

function postProcessor(x: unknown): PostProcessorSpec {
  const o = obj(x, 'post_processor')
  const type = str(o['type'])
  switch (type) {
    case 'TemplateProcessing': {
      const specials: Record<string, { ids: number[]; tokens: string[] }> = {}
      for (const [k, v] of Object.entries(obj(o['special_tokens'] ?? {}, 'special_tokens'))) {
        const s = obj(v, 'special token')
        specials[k] = { ids: list(s['ids']).map((i) => num(i)), tokens: list(s['tokens']).map((t) => str(t)) }
      }
      return { type, single: templatePieces(o['single']), pair: templatePieces(o['pair']), specialTokens: specials }
    }
    case 'BertProcessing':
      return { type, cls: pairOf(o['cls']), sep: pairOf(o['sep']) }
    case 'RobertaProcessing':
      return {
        type,
        cls: pairOf(o['cls']),
        sep: pairOf(o['sep']),
        trimOffsets: bool(o['trim_offsets'], true),
        addPrefixSpace: bool(o['add_prefix_space'], true),
      }
    case 'ByteLevel':
      return { type, trimOffsets: bool(o['trim_offsets'], true), addPrefixSpace: bool(o['add_prefix_space'], true) }
    case 'Sequence':
      return { type, processors: list(o['processors']).map(postProcessor) }
    default:
      return notImplemented('post-processor', type)
  }
}

function vocabMap(x: unknown): Map<string, number> {
  return new Map(Object.entries(obj(x, 'vocab')).map(([k, v]) => [k, num(v)]))
}

function model(x: unknown): TokenizerModel {
  const o = obj(x, 'model')
  const type = str(o['type'], 'BPE')
  switch (type) {
    case 'BPE': {
      const merges = list(o['merges']).map((m): [string, string] => {
        if (Array.isArray(m)) return [str(m[0]), str(m[1])]
        const s = str(m)
        const i = s.indexOf(' ')
        return [s.slice(0, i), s.slice(i + 1)]
      })
      const opt = <T>(key: string, value: T | undefined): Record<string, T> =>
        value === undefined || value === null ? {} : { [key]: value }
      return {
        type,
        model: bpeModel(vocabMap(o['vocab']), merges, {
          ...opt('unkToken', typeof o['unk_token'] === 'string' ? o['unk_token'] : undefined),
          ...opt(
            'continuingSubwordPrefix',
            typeof o['continuing_subword_prefix'] === 'string' ? o['continuing_subword_prefix'] : undefined,
          ),
          ...opt('endOfWordSuffix', typeof o['end_of_word_suffix'] === 'string' ? o['end_of_word_suffix'] : undefined),
          ...opt('dropout', typeof o['dropout'] === 'number' ? o['dropout'] : undefined),
          fuseUnk: bool(o['fuse_unk'], false),
          byteFallback: bool(o['byte_fallback'], false),
          ignoreMerges: bool(o['ignore_merges'], false),
        }),
      }
    }
    case 'WordPiece':
      return {
        type,
        model: wordPieceModel(vocabMap(o['vocab']), {
          unkToken: str(o['unk_token'], '[UNK]'),
          continuingSubwordPrefix: str(o['continuing_subword_prefix'], '##'),
          maxInputCharsPerWord: num(o['max_input_chars_per_word'], 100),
        }),
      }
    case 'Unigram': {
      const pieces = list(o['vocab']).map((p): [string, number] => [str(list(p)[0]), num(list(p)[1])])
      const unkId = typeof o['unk_id'] === 'number' ? o['unk_id'] : undefined
      return {
        type,
        model: unigramModel(pieces, {
          ...(unkId === undefined ? {} : { unkId }),
          byteFallback: bool(o['byte_fallback'], false),
        }),
      }
    }
    case 'WordLevel':
      return {
        type,
        vocab: vocabMap(o['vocab']),
        ...(typeof o['unk_token'] === 'string' ? { unkToken: o['unk_token'] } : {}),
      }
    default:
      return notImplemented('model', type)
  }
}

function truncation(x: unknown): TruncationOptions | undefined {
  if (x === null || x === undefined) return undefined
  const o = obj(x, 'truncation')
  const strategies: Readonly<Record<string, TruncationOptions['strategy']>> = {
    LongestFirst: 'longest_first',
    OnlyFirst: 'only_first',
    OnlySecond: 'only_second',
  }
  return {
    maxLength: num(o['max_length'], 512),
    stride: num(o['stride']),
    strategy: strategies[str(o['strategy'], 'LongestFirst')] ?? 'longest_first',
    direction: str(o['direction'], 'Right') === 'Left' ? 'left' : 'right',
  }
}

function padding(x: unknown): PaddingOptions | undefined {
  if (x === null || x === undefined) return undefined
  const o = obj(x, 'padding')
  const strategy = o['strategy']
  const fixed =
    strategy !== null && typeof strategy === 'object' ? num((strategy as Obj)['Fixed'], Number.NaN) : Number.NaN
  return {
    ...(Number.isFinite(fixed) ? { length: fixed } : {}),
    padId: num(o['pad_id']),
    padTypeId: num(o['pad_type_id']),
    padToken: str(o['pad_token'], '[PAD]'),
    direction: str(o['direction'], 'Right') === 'Left' ? 'left' : 'right',
    ...(typeof o['pad_to_multiple_of'] === 'number' ? { padToMultipleOf: o['pad_to_multiple_of'] } : {}),
  }
}

/**
 * Load an HF tokenizers `tokenizer.json` (the format `Tokenizer.to_str()` and `save` write): added tokens,
 * normaliser, pre-tokeniser, model (BPE, WordPiece, Unigram, WordLevel), post-processor, decoder, truncation and
 * padding. Regular expressions are translated to JavaScript ({@link hfRegex}).
 *
 * @param json - The file's text or its parsed object.
 * @returns The tokenizer.
 * @throws WhiteboxError with code `schema` for malformed input and `not-implemented` for components not supported
 * (the Precompiled normaliser, UnicodeScripts).
 * @concept tokenizer-pipeline
 * @example
 * ```ts
 * import { loadTokenizer, encode } from 'whitebox-engine/text'
 * const tok = loadTokenizer({ pre_tokenizer: { type: 'Whitespace' }, model: { type: 'WordLevel', vocab: { '[UNK]': 0, hi: 1 }, unk_token: '[UNK]' } })
 * encode(tok, 'hi you').ids // => [1, 0]
 * ```
 */
export function loadTokenizer(json: string | Readonly<Record<string, unknown>>): Tokenizer {
  let parsed: unknown = json
  if (typeof json === 'string') {
    try {
      parsed = JSON.parse(json)
    } catch {
      throw new WhiteboxError('schema', 'loadTokenizer: not valid JSON')
    }
  }
  const o = obj(parsed, 'tokenizer')
  const added: AddedToken[] = list(o['added_tokens']).map((t) => {
    const a = obj(t, 'added token')
    return {
      content: str(a['content']),
      id: num(a['id']),
      special: bool(a['special'], false),
      singleWord: bool(a['single_word'], false),
      lstrip: bool(a['lstrip'], false),
      rstrip: bool(a['rstrip'], false),
      normalized: bool(a['normalized'], !bool(a['special'], false)),
    }
  })
  const t = truncation(o['truncation'])
  const p = padding(o['padding'])
  return tokenizer({
    model: model(o['model']),
    addedTokens: added,
    ...(o['normalizer'] === null || o['normalizer'] === undefined ? {} : { normalizer: normalizer(o['normalizer']) }),
    ...(o['pre_tokenizer'] === null || o['pre_tokenizer'] === undefined
      ? {}
      : { preTokenizer: preTokenizer(o['pre_tokenizer']) }),
    ...(o['post_processor'] === null || o['post_processor'] === undefined
      ? {}
      : { postProcessor: postProcessor(o['post_processor']) }),
    ...(o['decoder'] === null || o['decoder'] === undefined ? {} : { decoder: decoder(o['decoder']) }),
    ...(t === undefined ? {} : { truncation: t }),
    ...(p === undefined ? {} : { padding: p }),
  })
}
