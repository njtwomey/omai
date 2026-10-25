/**
 * The tokenizer pipeline of HF tokenizers: added tokens, normaliser, pre-tokeniser, subword model, post-processor
 * (special tokens and type ids) and decoder. `encode` produces ids, tokens, offsets into the input, masks and word
 * indices; `encodeBatch` adds padding and truncation; `decode` turns ids back into text.
 */

import { MdmlError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import { UNI_SPACE, UNI_WORD } from './_classes.ts'
import { type AlignedText, alignedText, originalSpan, sliceAligned } from './normalise.ts'
import { type BpeModel, bpeTokenize, bytesToUnicode, type SubwordToken } from './subword/bpe.ts'
import { type UnigramModel, unigramTokenize } from './subword/unigram.ts'
import { type WordPieceModel, wordPieceTokenize } from './subword/wordpiece.ts'
import {
  applyNormalizer,
  applyPreTokenizer,
  type DecoderSpec,
  decodeChain,
  type NormalizerSpec,
  type PreTokenizerSpec,
} from './tokenizer-components.ts'

/** The model of a tokenizer pipeline. */
export type TokenizerModel =
  | { readonly type: 'BPE'; readonly model: BpeModel }
  | { readonly type: 'WordPiece'; readonly model: WordPieceModel }
  | { readonly type: 'Unigram'; readonly model: UnigramModel }
  | { readonly type: 'WordLevel'; readonly vocab: ReadonlyMap<string, number>; readonly unkToken?: string }

/** A token added to a tokenizer's vocabulary, matched in the text before the model runs (HF's AddedToken). */
export interface AddedToken {
  readonly content: string
  readonly id: number
  /** Special tokens are skipped when decoding with `skipSpecialTokens`. */
  readonly special?: boolean
  /** Only match as a whole word. */
  readonly singleWord?: boolean
  /** Absorb whitespace on the left. */
  readonly lstrip?: boolean
  /** Absorb whitespace on the right. */
  readonly rstrip?: boolean
  /** Match in the normalised text (default: not for special tokens). */
  readonly normalized?: boolean
}

/** A piece of a post-processing template: an input sequence or a special token, with its type id. */
export type TemplatePiece =
  | { readonly sequence: 'A' | 'B'; readonly typeId: number }
  | { readonly special: string; readonly typeId: number }

/** A post-processor of a tokenizer pipeline (adds special tokens, sets type ids, trims offsets). */
export type PostProcessorSpec =
  | {
      readonly type: 'TemplateProcessing'
      readonly single: readonly TemplatePiece[]
      readonly pair: readonly TemplatePiece[]
      readonly specialTokens: Readonly<
        Record<string, { readonly ids: readonly number[]; readonly tokens: readonly string[] }>
      >
    }
  | {
      readonly type: 'BertProcessing'
      readonly cls: readonly [string, number]
      readonly sep: readonly [string, number]
    }
  | {
      readonly type: 'RobertaProcessing'
      readonly cls: readonly [string, number]
      readonly sep: readonly [string, number]
      readonly trimOffsets?: boolean
      readonly addPrefixSpace?: boolean
    }
  | { readonly type: 'ByteLevel'; readonly trimOffsets?: boolean; readonly addPrefixSpace?: boolean }
  | { readonly type: 'Sequence'; readonly processors: readonly PostProcessorSpec[] }

/** Truncation of encodings (HF's TruncationParams). */
export interface TruncationOptions {
  readonly maxLength: number
  /** Overlap between overflowing windows (default 0). */
  readonly stride?: number
  readonly strategy?: 'longest_first' | 'only_first' | 'only_second'
  readonly direction?: 'right' | 'left'
}

/** Padding of encodings (HF's PaddingParams); without `length`, a batch pads to its longest encoding. */
export interface PaddingOptions {
  readonly length?: number
  readonly padId?: number
  readonly padTypeId?: number
  readonly padToken?: string
  readonly direction?: 'right' | 'left'
  readonly padToMultipleOf?: number
}

/** An HF-style tokenizer pipeline: normaliser, pre-tokeniser, model, post-processor and decoder. */
export interface Tokenizer {
  readonly normalizer?: NormalizerSpec
  readonly preTokenizer?: PreTokenizerSpec
  readonly model: TokenizerModel
  readonly postProcessor?: PostProcessorSpec
  readonly decoder?: DecoderSpec
  readonly addedTokens?: readonly AddedToken[]
  readonly truncation?: TruncationOptions
  readonly padding?: PaddingOptions
}

/**
 * The result of encoding (HF's Encoding): ids, tokens, `[start, end)` offsets in the input (UTF-16 code units; added
 * special tokens have (0, 0)), type ids, masks, the word and sequence of each token, and overflowing windows.
 */
export interface Encoding {
  readonly ids: number[]
  readonly tokens: string[]
  readonly offsets: [number, number][]
  readonly typeIds: number[]
  readonly attentionMask: number[]
  readonly specialTokensMask: number[]
  readonly wordIds: (number | null)[]
  readonly sequenceIds: (number | null)[]
  readonly overflowing: Encoding[]
}

/**
 * An encoding being built: HF keeps the range of each input sequence rather than a sequence id per token (ranges are
 * cleared by truncation; without ranges every token, padding included, belongs to sequence 0).
 */
interface Work extends Omit<Encoding, 'sequenceIds' | 'overflowing'> {
  readonly ranges: ReadonlyMap<number, readonly [number, number]>
  readonly overflowing: Work[]
}

function emptyEncoding(): Work {
  return {
    ids: [],
    tokens: [],
    offsets: [],
    typeIds: [],
    attentionMask: [],
    specialTokensMask: [],
    wordIds: [],
    ranges: new Map(),
    overflowing: [],
  }
}

/** The public encoding: sequence ids from the ranges (HF's get_sequence_ids). */
function finish(w: Work): Encoding {
  const { ranges, overflowing, ...rest } = w
  const sequenceIds: (number | null)[] = w.ids.map(() => (ranges.size === 0 ? 0 : null))
  for (const [id, [a, b]] of ranges) for (let i = a; i < b; i++) sequenceIds[i] = id
  return { ...rest, sequenceIds, overflowing: overflowing.map(finish) }
}

/** HF's set_sequence_id: the whole encoding (and its overflowing windows) is sequence `id`. */
function withSequence(e: Work, id: number): Work {
  return {
    ...e,
    ranges: new Map([[id, [0, e.ids.length] as const]]),
    overflowing: e.overflowing.map((o) => withSequence(o, id)),
  }
}

/** One split of the input: aligned text, and tokens already fixed (added tokens). */
interface Split {
  readonly text: AlignedText
  readonly tokens?: readonly SubwordToken[]
}

const STARTS_WORD = new RegExp(`^[${UNI_WORD}]`, 'u')
const ENDS_WORD = new RegExp(`[${UNI_WORD}]$`, 'u')
const SPACE_END = new RegExp(`[${UNI_SPACE}]*$`, 'u')
const SPACE_START = new RegExp(`^[${UNI_SPACE}]*`, 'u')

/** Split a piece on added tokens (leftmost-longest, then single-word, lstrip and rstrip, as HF's AddedVocabulary). */
function splitAdded(split: Split, added: readonly AddedToken[]): Split[] {
  if (split.tokens !== undefined || added.length === 0) return [split]
  const s = split.text.text
  // Leftmost-longest, non-overlapping matches.
  const matches: { start: number; end: number; token: AddedToken }[] = []
  let at = 0
  while (at < s.length) {
    let best: AddedToken | undefined
    for (const t of added)
      if (
        t.content.length > 0 &&
        s.startsWith(t.content, at) &&
        (best === undefined || t.content.length > best.content.length)
      )
        best = t
    if (best === undefined) {
      at++
      continue
    }
    matches.push({ start: at, end: at + best.content.length, token: best })
    at += best.content.length
  }
  const out: Split[] = []
  let last = 0
  for (const m of matches) {
    let { start, end } = m
    if (m.token.singleWord === true) {
      const leftOk = start === 0 || !ENDS_WORD.test(s.slice(0, start))
      const rightOk = end === s.length || !STARTS_WORD.test(s.slice(end))
      if (!leftOk || !rightOk) continue
    }
    if (m.token.lstrip === true) start = Math.max(SPACE_END.exec(s.slice(0, start))?.index ?? start, last)
    if (m.token.rstrip === true) end += SPACE_START.exec(s.slice(end))?.[0].length ?? 0
    if (last < start) out.push({ text: sliceAligned(split.text, last, start) })
    const text = sliceAligned(split.text, start, end)
    out.push({ text, tokens: [{ token: text.text, id: m.token.id, start: 0, end: text.text.length }] })
    last = end
  }
  if (last < s.length) out.push({ text: sliceAligned(split.text, last, s.length) })
  return out.length === 0 ? [split] : out
}

/** The tokens of one pre-tokenised piece under the model. */
function modelTokens(
  model: TokenizerModel,
  word: string,
  options: { readonly dropout?: number; readonly key?: Key },
): SubwordToken[] {
  switch (model.type) {
    case 'BPE':
      return bpeTokenize(model.model, word, options)
    case 'WordPiece':
      return wordPieceTokenize(model.model, word)
    case 'Unigram':
      return unigramTokenize(model.model, word)
    case 'WordLevel': {
      const id = model.vocab.get(word) ?? (model.unkToken === undefined ? undefined : model.vocab.get(model.unkToken))
      if (id === undefined) throw new MdmlError('value', `encode: ${word} is not in the vocabulary`)
      return [{ token: model.vocab.has(word) ? word : (model.unkToken ?? word), id, start: 0, end: word.length }]
    }
  }
}

/** Options of {@link encode}. */
export interface EncodeOptions {
  /** Add the post-processor's special tokens (default true). */
  readonly addSpecialTokens?: boolean
  /** BPE dropout probability, with `key`. */
  readonly dropout?: number
  readonly key?: Key
}

/** Encode one sequence up to the post-processor: added tokens, normaliser, pre-tokeniser, model. */
function encodeSequence(tok: Tokenizer, text: string, typeId: number, options: EncodeOptions): Work {
  const added = tok.addedTokens ?? []
  const raw = added.filter((t) => !(t.normalized ?? t.special !== true))
  const normalized = added.filter((t) => t.normalized ?? t.special !== true)
  let splits: Split[] = splitAdded({ text: alignedText(text) }, raw)
  splits = splits.flatMap((sp) => {
    if (sp.tokens !== undefined) return [sp]
    const t = tok.normalizer === undefined ? sp.text : applyNormalizer(tok.normalizer, sp.text)
    return splitAdded({ text: t }, normalized)
  })
  splits = splits.filter((sp) => sp.text.text.length > 0)
  if (tok.preTokenizer !== undefined) {
    const pre = tok.preTokenizer
    splits = splits.flatMap((sp) =>
      sp.tokens !== undefined ? [sp] : applyPreTokenizer(pre, [sp.text]).map((t) => ({ text: t })),
    )
  }
  const enc = emptyEncoding()
  for (const [w, sp] of splits.entries()) {
    const toks = sp.tokens ?? modelTokens(tok.model, sp.text.text, options)
    for (const t of toks) {
      enc.ids.push(t.id)
      enc.tokens.push(t.token)
      enc.offsets.push(originalSpan(sp.text, t.start, t.end))
      enc.typeIds.push(typeId)
      enc.attentionMask.push(1)
      enc.specialTokensMask.push(0)
      enc.wordIds.push(w)
    }
  }
  return enc
}

function sliceEncoding(e: Work, a: number, b: number): Work {
  return {
    ids: e.ids.slice(a, b),
    tokens: e.tokens.slice(a, b),
    offsets: e.offsets.slice(a, b),
    typeIds: e.typeIds.slice(a, b),
    attentionMask: e.attentionMask.slice(a, b),
    specialTokensMask: e.specialTokensMask.slice(a, b),
    wordIds: e.wordIds.slice(a, b),
    ranges: new Map(),
    overflowing: [],
  }
}

/** HF's Encoding::truncate: keep `maxLength` tokens and move the rest into overflowing windows that overlap by `stride`. */
function truncateEncoding(e: Work, maxLength: number, stride: number, direction: 'right' | 'left'): Work {
  const n = e.ids.length
  if (maxLength >= n) return e
  if (maxLength === 0) return { ...emptyEncoding(), overflowing: [e] }
  if (stride >= maxLength) throw new MdmlError('value', 'truncate: stride must be smaller than the maximum length')
  const step = maxLength - stride
  const ranges: [number, number][] = []
  if (direction === 'right') {
    for (let start = 0; start < n; start += step) {
      const stop = Math.min(start + maxLength, n)
      ranges.push([start, stop])
      if (stop === n) break
    }
  } else {
    for (let s = n - 1; s >= 0; s -= step) {
      const stop = s + 1
      const start = Math.max(0, stop - maxLength)
      ranges.push([start, stop])
      if (start === 0) break
    }
  }
  const [first, ...rest] = ranges
  const main = sliceEncoding(e, first?.[0] ?? 0, first?.[1] ?? 0)
  return { ...main, overflowing: rest.map(([a, b]) => sliceEncoding(e, a, b)) }
}

function truncatePair(a: Work, b: Work | undefined, t: TruncationOptions, maxLength: number): [Work, Work | undefined] {
  const stride = t.stride ?? 0
  const dir = t.direction ?? 'right'
  if (maxLength <= 0)
    return [truncateEncoding(a, 0, stride, dir), b === undefined ? undefined : truncateEncoding(b, 0, stride, dir)]
  const total = a.ids.length + (b?.ids.length ?? 0)
  if (total <= maxLength) return [a, b]
  const remove = total - maxLength
  const strategy = t.strategy ?? 'longest_first'
  if (strategy === 'longest_first') {
    if (b === undefined) return [truncateEncoding(a, total - remove, stride, dir), undefined]
    let n1 = a.ids.length
    let n2 = b.ids.length
    const swap = n1 > n2
    if (swap) [n1, n2] = [n2, n1]
    n2 = n1 > maxLength ? n1 : Math.max(n1, maxLength - n1)
    if (n1 + n2 > maxLength) {
      n1 = Math.floor(maxLength / 2)
      n2 = n1 + (maxLength % 2)
    }
    if (swap) [n1, n2] = [n2, n1]
    return [truncateEncoding(a, n1, stride, dir), truncateEncoding(b, n2, stride, dir)]
  }
  const second = strategy === 'only_second'
  const target = second ? b : a
  if (target === undefined) throw new MdmlError('value', 'truncate: the second sequence is missing')
  if (target.ids.length <= remove) throw new MdmlError('value', 'truncate: the sequence is too short to truncate')
  const cut = truncateEncoding(target, target.ids.length - remove, stride, dir)
  return second ? [a, cut] : [cut, b]
}

/** HF's Encoding::merge_with: concatenate, combining overflowing windows. */
function mergeEncodings(a: Work, b: Work): Work {
  const overflowing: Work[] = []
  // As HF: the clones keep their own overflowing windows, so windows of windows appear for pairs.
  for (const ao of a.overflowing) {
    overflowing.push(mergeEncodings(ao, b))
    for (const bo of b.overflowing) overflowing.push(mergeEncodings(ao, bo))
  }
  for (const bo of b.overflowing) overflowing.push(mergeEncodings(a, bo))
  return {
    ids: [...a.ids, ...b.ids],
    tokens: [...a.tokens, ...b.tokens],
    offsets: [...a.offsets, ...b.offsets],
    typeIds: [...a.typeIds, ...b.typeIds],
    attentionMask: [...a.attentionMask, ...b.attentionMask],
    specialTokensMask: [...a.specialTokensMask, ...b.specialTokensMask],
    wordIds: [...a.wordIds, ...b.wordIds],
    ranges: new Map([
      ...a.ranges,
      ...[...b.ranges].map(([k, [x, y]]) => [k, [x + a.ids.length, y + a.ids.length] as const] as const),
    ]),
    overflowing,
  }
}

function specialEncoding(ids: readonly number[], tokens: readonly string[], typeId: number): Work {
  return {
    ids: [...ids],
    tokens: [...tokens],
    offsets: ids.map(() => [0, 0]),
    typeIds: ids.map(() => typeId),
    attentionMask: ids.map(() => 1),
    specialTokensMask: ids.map(() => 1),
    wordIds: ids.map(() => null),
    ranges: new Map(),
    overflowing: [],
  }
}

const withTypes = (e: Work, typeId: number): Work => ({
  ...e,
  typeIds: e.typeIds.map(() => typeId),
  overflowing: e.overflowing.map((o) => withTypes(o, typeId)),
})

const SPACE_CHAR = bytesToUnicode()[32] ?? 'Ġ'
const RUST_WS = new RegExp(`[${UNI_SPACE}]`, 'u')

/** HF's process_offsets: move offsets past leading and trailing (byte-level) spaces of each token. */
function trimOffsets(e: Work, addPrefixSpace: boolean): Work {
  const offsets = e.offsets.map(([a, b], i): [number, number] => {
    const chars = Array.from(e.tokens[i] ?? '')
    const isSpace = (c: string): boolean => c === SPACE_CHAR || RUST_WS.test(c)
    let lead = 0
    while (lead < chars.length && isSpace(chars[lead] ?? '')) lead++
    let trail = 0
    while (trail < chars.length && isSpace(chars[chars.length - 1 - trail] ?? '')) trail++
    let start = a
    let end = b
    if (lead > 0) {
      if ((i === 0 || a === 0) && addPrefixSpace && lead === 1) lead = 0
      start = Math.min(a + lead, b)
    }
    if (trail > 0 && end >= trail) end = Math.max(end - trail, start)
    return [start, end]
  })
  return { ...e, offsets, overflowing: e.overflowing.map((o) => trimOffsets(o, addPrefixSpace)) }
}

/** The number of special tokens a post-processor adds. */
function addedCount(p: PostProcessorSpec | undefined, pair: boolean): number {
  if (p === undefined) return 0
  switch (p.type) {
    case 'TemplateProcessing':
      return (pair ? p.pair : p.single).reduce(
        (n, piece) => n + ('special' in piece ? (p.specialTokens[piece.special]?.ids.length ?? 0) : 0),
        0,
      )
    case 'BertProcessing':
    case 'RobertaProcessing':
      return pair ? 4 - (p.type === 'BertProcessing' ? 1 : 0) : 2
    case 'ByteLevel':
      return 0
    case 'Sequence':
      return p.processors.reduce((n, q) => n + addedCount(q, pair), 0)
  }
}

/** Apply a post-processor to the encoding(s) of one input, returning one encoding. */
function postProcess(p: PostProcessorSpec | undefined, encs: Work[], addSpecial: boolean): Work[] {
  if (p === undefined) return encs
  switch (p.type) {
    case 'ByteLevel':
      return p.trimOffsets === false ? encs : encs.map((e) => trimOffsets(e, p.addPrefixSpace !== false))
    case 'TemplateProcessing': {
      const template = encs.length > 1 ? p.pair : p.single
      const out: Work[] = []
      for (const piece of template) {
        if ('sequence' in piece) {
          const i = piece.sequence === 'A' ? 0 : 1
          const e = encs[i]
          // HF sets the template's type id on the encoding itself; its overflowing windows keep their own.
          if (e !== undefined) out.push(withSequence({ ...e, typeIds: e.typeIds.map(() => piece.typeId) }, i))
        } else if (addSpecial) {
          const sp = p.specialTokens[piece.special]
          if (sp === undefined) throw new MdmlError('value', `postProcess: missing special token ${piece.special}`)
          out.push(specialEncoding(sp.ids, sp.tokens, piece.typeId))
        }
      }
      return out
    }
    case 'BertProcessing':
    case 'RobertaProcessing': {
      const roberta = p.type === 'RobertaProcessing'
      let es = encs
      if (roberta && p.trimOffsets !== false) es = es.map((e) => trimOffsets(e, p.addPrefixSpace !== false))
      es = es.map((e, i) => withTypes(e, roberta ? 0 : i))
      if (!addSpecial) return es
      const [cls, clsId] = p.cls
      const [sep, sepId] = p.sep
      const out: Work[] = []
      for (const [i, e] of es.entries()) {
        const t = roberta ? 0 : i
        if (i === 0) out.push(specialEncoding([clsId], [cls], 0), e, specialEncoding([sepId], [sep], 0))
        else {
          if (roberta) out.push(specialEncoding([sepId], [sep], t))
          out.push(e, specialEncoding([sepId], [sep], t))
        }
      }
      return out
    }
    case 'Sequence':
      return p.processors.reduce((es, q) => postProcess(q, es, addSpecial), encs)
  }
}

/** Pad one encoding (and its overflowing windows) to a length. */
function padEncoding(e: Work, length: number, p: PaddingOptions): Work {
  const overflowing = e.overflowing.map((o) => padEncoding(o, length, p))
  const n = length - e.ids.length
  if (n <= 0) return { ...e, overflowing }
  const fill = <T>(v: T): T[] => Array.from({ length: n }, () => v)
  const join = <T>(xs: T[], pad: T[]): T[] => (p.direction === 'left' ? [...pad, ...xs] : [...xs, ...pad])
  return {
    ids: join(e.ids, fill(p.padId ?? 0)),
    tokens: join(e.tokens, fill(p.padToken ?? '[PAD]')),
    offsets: join(
      e.offsets,
      Array.from({ length: n }, (): [number, number] => [0, 0]),
    ),
    typeIds: join(e.typeIds, fill(p.padTypeId ?? 0)),
    attentionMask: join(e.attentionMask, fill(0)),
    specialTokensMask: join(e.specialTokensMask, fill(1)),
    wordIds: join(e.wordIds, fill<number | null>(null)),
    ranges:
      p.direction === 'left'
        ? new Map([...e.ranges].map(([k, [x, y]]) => [k, [x + n, y + n] as const] as const))
        : e.ranges,
    overflowing,
  }
}

function padTo(length: number, p: PaddingOptions): number {
  const m = p.padToMultipleOf
  return m !== undefined && m > 0 && length % m > 0 ? length + m - (length % m) : length
}

/**
 * Encode a text (or a pair of texts) with a tokenizer pipeline, as HF tokenizers' `Tokenizer.encode`: added tokens are
 * cut out first, the rest is normalised, pre-tokenised and tokenised by the model; then the pair is truncated (leaving
 * room for special tokens), post-processed (special tokens, type ids, offset trimming) and padded to a fixed length.
 *
 * @param tok - The tokenizer.
 * @param text - The text.
 * @param pair - An optional second text.
 * @param options - Whether to add special tokens, and BPE dropout.
 * @returns The encoding.
 * @throws MdmlError with code `value` for impossible truncation or unknown tokens without an unknown token.
 * @concept tokenizer-pipeline
 * @example
 * ```ts
 * import { tokenizer, encode } from 'mdml-engine/text'
 * import { wordPieceModel } from 'mdml-engine/text/subword'
 * const tok = tokenizer({
 *   normalizer: { type: 'BertNormalizer' },
 *   preTokenizer: { type: 'BertPreTokenizer' },
 *   model: { type: 'WordPiece', model: wordPieceModel(['[UNK]', '[CLS]', '[SEP]', 'hello', 'world', '!']) },
 *   postProcessor: { type: 'BertProcessing', cls: ['[CLS]', 1], sep: ['[SEP]', 2] },
 * })
 * const e = encode(tok, 'Hello World!')
 * [e.tokens, e.offsets] // => [['[CLS]', 'hello', 'world', '!', '[SEP]'], [[0, 0], [0, 5], [6, 11], [11, 12], [0, 0]]]
 * ```
 */
export function encode(tok: Tokenizer, text: string, pair?: string, options: EncodeOptions = {}): Encoding {
  return finish(encodeWork(tok, text, pair, options))
}

function encodeWork(tok: Tokenizer, text: string, pair: string | undefined, options: EncodeOptions): Work {
  const addSpecial = options.addSpecialTokens !== false
  const a = encodeSequence(tok, text, 0, options)
  const b = pair === undefined ? undefined : encodeSequence(tok, pair, 1, options)
  let [ta, tb] = [a, b]
  if (tok.truncation !== undefined) {
    const n = addSpecial ? addedCount(tok.postProcessor, b !== undefined) : 0
    ;[ta, tb] = truncatePair(a, b, tok.truncation, tok.truncation.maxLength - n)
  }
  const inputs = tb === undefined ? [ta] : [ta, tb]
  // HF's PostProcessor::process marks each input as its sequence first; without a processor only pairs are marked.
  const marked = tok.postProcessor !== undefined || inputs.length > 1 ? inputs.map(withSequence) : inputs
  const parts = postProcess(tok.postProcessor, marked, addSpecial)
  let merged =
    parts.length === 1
      ? (parts[0] ?? emptyEncoding())
      : parts.reduce((acc, e) => mergeEncodings(acc, e), emptyEncoding())
  if (tok.padding?.length !== undefined)
    merged = padEncoding(merged, padTo(tok.padding.length, tok.padding), tok.padding)
  return merged
}

/**
 * Encode a batch (HF's `encode_batch`): each input encoded on its own, then padded to the longest (or the fixed
 * length) when the tokenizer pads.
 *
 * @param tok - The tokenizer.
 * @param inputs - Texts, or `[text, pair]` tuples.
 * @param options - Encoding options.
 * @returns The encodings.
 * @concept tokenizer-pipeline
 * @example
 * ```ts
 * import { tokenizer, encodeBatch } from 'mdml-engine/text'
 * const tok = tokenizer({
 *   preTokenizer: { type: 'WhitespaceSplit' },
 *   model: { type: 'WordLevel', vocab: new Map([['[PAD]', 0], ['a', 1], ['b', 2]]), unkToken: '[PAD]' },
 *   padding: { padToken: '[PAD]' },
 * })
 * encodeBatch(tok, ['a b a', 'b']).map((e) => [e.ids, e.attentionMask]) // => [[[1, 2, 1], [1, 1, 1]], [[2, 0, 0], [1, 0, 0]]]
 * ```
 */
export function encodeBatch(
  tok: Tokenizer,
  inputs: readonly (string | readonly [string, string])[],
  options: EncodeOptions = {},
): Encoding[] {
  const { padding, ...unpadded } = tok
  const encs = inputs.map((x) =>
    typeof x === 'string' ? encodeWork(unpadded, x, undefined, options) : encodeWork(unpadded, x[0], x[1], options),
  )
  const p = padding
  if (p === undefined) return encs.map(finish)
  let longest = 0
  for (const e of encs) if (e.ids.length > longest) longest = e.ids.length
  const length = padTo(p.length ?? longest, p)
  return encs.map((e) => finish(padEncoding(e, length, p)))
}

/** The token of each id: added tokens, then the model's vocabulary. */
function idToToken(tok: Tokenizer, id: number): string | undefined {
  for (const t of tok.addedTokens ?? []) if (t.id === id) return t.content
  const m = tok.model
  switch (m.type) {
    case 'Unigram':
      return m.model.pieces[id]?.[0]
    case 'BPE':
    case 'WordPiece':
    case 'WordLevel': {
      const vocab = m.type === 'WordLevel' ? m.vocab : m.model.vocab
      for (const [t, i] of vocab) if (i === id) return t
      return undefined
    }
  }
}

/**
 * Decode ids back to text (HF's `Tokenizer.decode`): ids to tokens (unknown ids dropped, special added tokens skipped
 * by default), then the decoder (or a space join).
 *
 * @param tok - The tokenizer.
 * @param ids - The ids.
 * @param options - Whether to skip special tokens (default true).
 * @returns The text.
 * @concept tokenizer-pipeline
 * @example
 * ```ts
 * import { tokenizer, decode } from 'mdml-engine/text'
 * import { wordPieceModel } from 'mdml-engine/text/subword'
 * const tok = tokenizer({
 *   model: { type: 'WordPiece', model: wordPieceModel(['[UNK]', '[CLS]', 'un', '##able']) },
 *   decoder: { type: 'WordPiece' },
 *   addedTokens: [{ content: '[CLS]', id: 1, special: true }],
 * })
 * decode(tok, [1, 2, 3]) // => 'unable'
 * ```
 */
export function decode(
  tok: Tokenizer,
  ids: readonly number[],
  options: { readonly skipSpecialTokens?: boolean } = {},
): string {
  const specials = new Set((tok.addedTokens ?? []).filter((t) => t.special === true).map((t) => t.content))
  const tokens: string[] = []
  for (const id of ids) {
    const t = idToToken(tok, id)
    if (t === undefined) continue
    if (options.skipSpecialTokens !== false && specials.has(t)) continue
    tokens.push(t)
  }
  return tok.decoder === undefined ? tokens.join(' ') : decodeChain(tok.decoder, tokens).join('')
}

/**
 * A tokenizer pipeline from its parts (checked: added token ids must be distinct).
 *
 * @param parts - Normaliser, pre-tokeniser, model, post-processor, decoder, added tokens, truncation, padding.
 * @returns The tokenizer.
 * @throws MdmlError with code `value` for repeated added token ids or a non-positive truncation length.
 * @concept tokenizer-pipeline
 * @example
 * ```ts
 * import { tokenizer, encode } from 'mdml-engine/text'
 * const tok = tokenizer({ preTokenizer: { type: 'Whitespace' }, model: { type: 'WordLevel', vocab: new Map([['[UNK]', 0], ['hi', 1]]), unkToken: '[UNK]' } })
 * encode(tok, 'hi there').tokens // => ['hi', '[UNK]']
 * ```
 */
export function tokenizer(parts: Tokenizer): Tokenizer {
  const ids = (parts.addedTokens ?? []).map((t) => t.id)
  if (new Set(ids).size !== ids.length) throw new MdmlError('value', 'tokenizer: added token ids must be distinct')
  if (parts.truncation !== undefined && !(parts.truncation.maxLength >= 0))
    throw new MdmlError('value', 'tokenizer: truncation needs a maximum length >= 0')
  return { ...parts }
}
