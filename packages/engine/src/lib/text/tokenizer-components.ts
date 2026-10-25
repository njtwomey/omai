/**
 * The interchangeable stages of an HF tokenizers pipeline as described in a `tokenizer.json`: normalisers applied to
 * aligned text, pre-tokenisers that split it into pieces, decoders that turn tokens back into text, and conversion of
 * the file's Oniguruma regular expressions to JavaScript.
 */

import { WhiteboxError } from '../base/index.ts'
import { UNI_SPACE, UNI_WORD } from './_classes.ts'
import {
  type AlignedText,
  filterAligned,
  mapAligned,
  type NormalForm,
  normaliseAligned,
  prependAligned,
  replaceAligned,
  sliceAligned,
  stripAligned,
} from './normalise.ts'
import { bytesToUnicode, fromByteLevel } from './subword/bpe.ts'
import { isBertPunctuation, PRE_TOKENIZER_PATTERNS, type SplitBehavior, splitSpans } from './tokenise.ts'

/** A pattern of a tokenizer component: a literal string or a regular expression. */
export type TokenizerPattern = string | RegExp

/** A normaliser of an HF-style tokenizer pipeline (the `normalizer` of a `tokenizer.json`). */
export type NormalizerSpec =
  | { readonly type: NormalForm | 'Lowercase' | 'StripAccents' | 'Nmt' | 'ByteLevel' }
  | { readonly type: 'Strip'; readonly left?: boolean; readonly right?: boolean }
  | { readonly type: 'Replace'; readonly pattern: TokenizerPattern; readonly content: string }
  | { readonly type: 'Prepend'; readonly prepend: string }
  | {
      readonly type: 'BertNormalizer'
      readonly cleanText?: boolean
      readonly handleChineseChars?: boolean
      /** Default: follow `lowercase`. */
      readonly stripAccents?: boolean
      readonly lowercase?: boolean
    }
  | { readonly type: 'Sequence'; readonly normalizers: readonly NormalizerSpec[] }

const NONSPACING = /\p{Mn}/u
const MARK = /\p{M}/u
const OTHER = /\p{C}/u
const RUST_SPACE = new RegExp(`[${UNI_SPACE}]`, 'u')

function isChinese(c: string): boolean {
  const cp = c.codePointAt(0) ?? 0
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2a700 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) ||
    (cp >= 0x2b920 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x2f800 && cp <= 0x2fa1f)
  )
}

const NMT_DROP = (cp: number): boolean =>
  (cp >= 0x1 && cp <= 0x8) || cp === 0xb || (cp >= 0xe && cp <= 0x1f) || cp === 0x7f || cp === 0x8f || cp === 0x9f
const NMT_SPACE = new Set([
  0x9, 0xa, 0xc, 0xd, 0x1680, 0x200b, 0x200c, 0x200d, 0x200e, 0x200f, 0x2028, 0x2029, 0x2581, 0xfeff, 0xfffd,
])

/** GPT-2's byte alphabet applied to aligned text: each UTF-8 byte of a character becomes one character. */
function byteLevelAligned(a: AlignedText): AlignedText {
  const table = bytesToUnicode()
  const encoder = new TextEncoder()
  return mapAligned(a, (ch) => Array.from(encoder.encode(ch), (b) => table[b] ?? '').join(''))
}

/**
 * Apply a normaliser to aligned text, as HF tokenizers' normalisers do (alignments to the original are kept).
 *
 * @param spec - The normaliser.
 * @param text - The aligned text.
 * @returns The normalised aligned text.
 * @throws WhiteboxError with code `value` for an unknown normaliser.
 * @concept tokenizer-pipeline
 * @example
 * ```ts
 * import { alignedText, applyNormalizer } from 'whitebox-engine/text'
 * applyNormalizer({ type: 'BertNormalizer', lowercase: true }, alignedText('Héllo\tWORLD 中文')).text // => 'hello world  中  文 '
 * ```
 */
export function applyNormalizer(spec: NormalizerSpec, text: AlignedText): AlignedText {
  switch (spec.type) {
    case 'NFC':
    case 'NFD':
    case 'NFKC':
    case 'NFKD':
      return normaliseAligned(text, spec.type)
    case 'Lowercase':
      return mapAligned(text, (c) => c.toLowerCase())
    case 'StripAccents':
      return filterAligned(text, (c) => !MARK.test(c))
    case 'Nmt':
      return mapAligned(text, (c) => {
        const cp = c.codePointAt(0) ?? 0
        return NMT_DROP(cp) ? '' : NMT_SPACE.has(cp) ? ' ' : c
      })
    case 'ByteLevel':
      return text.text.length === 0 ? text : byteLevelAligned(text)
    case 'Strip':
      return stripAligned(text, { left: spec.left !== false, right: spec.right !== false })
    case 'Replace':
      return replaceAligned(text, spec.pattern, spec.content)
    case 'Prepend':
      return prependAligned(text, spec.prepend)
    case 'BertNormalizer': {
      let t = text
      if (spec.cleanText !== false) {
        t = mapAligned(t, (c) => {
          const cp = c.codePointAt(0) ?? 0
          if (cp === 0 || cp === 0xfffd) return ''
          if (c !== '\t' && c !== '\n' && c !== '\r' && OTHER.test(c)) return ''
          return c === '\t' || c === '\n' || c === '\r' || RUST_SPACE.test(c) ? ' ' : c
        })
      }
      if (spec.handleChineseChars !== false) t = mapAligned(t, (c) => (isChinese(c) ? ` ${c} ` : c))
      const lowercase = spec.lowercase !== false
      if (spec.stripAccents ?? lowercase) t = filterAligned(normaliseAligned(t, 'NFD'), (c) => !NONSPACING.test(c))
      if (lowercase) t = mapAligned(t, (c) => c.toLowerCase())
      return t
    }
    case 'Sequence':
      return spec.normalizers.reduce((t, n) => applyNormalizer(n, t), text)
    default:
      throw new WhiteboxError('value', `applyNormalizer: unknown normalizer ${(spec as { type: string }).type}`)
  }
}

/** A pre-tokeniser of an HF-style tokenizer pipeline (the `pre_tokenizer` of a `tokenizer.json`). */
export type PreTokenizerSpec =
  | { readonly type: 'Whitespace' | 'WhitespaceSplit' | 'BertPreTokenizer' }
  | {
      readonly type: 'ByteLevel'
      readonly addPrefixSpace?: boolean
      readonly useRegex?: boolean
      readonly trimOffsets?: boolean
    }
  | {
      readonly type: 'Metaspace'
      readonly replacement?: string
      readonly prependScheme?: 'always' | 'first' | 'never'
      readonly split?: boolean
    }
  | { readonly type: 'Punctuation'; readonly behavior?: SplitBehavior }
  | { readonly type: 'Digits'; readonly individualDigits?: boolean }
  | {
      readonly type: 'Split'
      readonly pattern: TokenizerPattern
      readonly behavior: SplitBehavior
      readonly invert?: boolean
    }
  | { readonly type: 'CharDelimiterSplit'; readonly delimiter: string }
  | { readonly type: 'FixedLength'; readonly length: number }
  | { readonly type: 'Sequence'; readonly pretokenizers: readonly PreTokenizerSpec[] }

const NUMERIC = /\p{N}/u

function splitAligned(
  a: AlignedText,
  delimiter: Parameters<typeof splitSpans>[1],
  behavior: SplitBehavior,
  invert = false,
): AlignedText[] {
  return splitSpans(a.text, delimiter, behavior, invert).map(([s, e]) => sliceAligned(a, s, e))
}

/**
 * Apply a pre-tokeniser to aligned pieces, as HF tokenizers' pre-tokenisers do: each piece is split further (and, for
 * ByteLevel, mapped to the byte alphabet; for Metaspace, spaces become the replacement character).
 *
 * @param spec - The pre-tokeniser.
 * @param pieces - The pieces so far.
 * @param first - Whether the first piece starts the input (for Metaspace's `first` scheme).
 * @returns The new pieces (empty ones dropped).
 * @throws WhiteboxError with code `value` for an unknown pre-tokeniser.
 * @concept tokenizer-pipeline
 * @example
 * ```ts
 * import { alignedText, applyPreTokenizer } from 'whitebox-engine/text'
 * applyPreTokenizer({ type: 'ByteLevel', addPrefixSpace: true }, [alignedText('Hello world')]).map((p) => p.text) // => ['ĠHello', 'Ġworld']
 * ```
 */
export function applyPreTokenizer(spec: PreTokenizerSpec, pieces: readonly AlignedText[], first = true): AlignedText[] {
  const each = (f: (p: AlignedText, i: number) => AlignedText[]): AlignedText[] =>
    pieces.flatMap((p, i) => f(p, i)).filter((p) => p.text.length > 0)
  switch (spec.type) {
    case 'Whitespace':
      return each((p) => splitAligned(p, PRE_TOKENIZER_PATTERNS.whitespace, 'removed', true))
    case 'WhitespaceSplit':
      return each((p) => splitAligned(p, (c) => RUST_SPACE.test(c), 'removed'))
    case 'BertPreTokenizer':
      return each((p) =>
        splitAligned(p, (c) => RUST_SPACE.test(c), 'removed').flatMap((q) =>
          splitAligned(q, isBertPunctuation, 'isolated'),
        ),
      )
    case 'ByteLevel':
      return each((p) => {
        const q = spec.addPrefixSpace !== false && !p.text.startsWith(' ') ? prependAligned(p, ' ') : p
        const parts = spec.useRegex === false ? [q] : splitAligned(q, PRE_TOKENIZER_PATTERNS.gpt2, 'isolated')
        return parts.map(byteLevelAligned)
      })
    case 'Metaspace': {
      const rep = spec.replacement ?? '▁'
      const scheme = spec.prependScheme ?? 'always'
      return each((p, i) => {
        let q = replaceAligned(p, ' ', rep)
        if (
          !q.text.startsWith(rep) &&
          (scheme === 'always' || (scheme === 'first' && first && i === 0 && (p.alignments[0]?.[0] ?? 0) === 0))
        )
          q = prependAligned(q, rep)
        return spec.split === false ? [q] : splitAligned(q, rep, 'mergedWithNext')
      })
    }
    case 'Punctuation':
      return each((p) => splitAligned(p, isBertPunctuation, spec.behavior ?? 'isolated'))
    case 'Digits':
      return each((p) =>
        splitAligned(p, (c) => NUMERIC.test(c), spec.individualDigits === true ? 'isolated' : 'contiguous'),
      )
    case 'Split':
      return each((p) => splitAligned(p, spec.pattern, spec.behavior, spec.invert === true))
    case 'CharDelimiterSplit':
      return each((p) => splitAligned(p, spec.delimiter, 'removed'))
    case 'FixedLength':
      return each((p) => {
        const out: AlignedText[] = []
        const chars = Array.from(p.text)
        let at = 0
        for (let i = 0; i < chars.length; i += spec.length) {
          const len = chars.slice(i, i + spec.length).join('').length
          out.push(sliceAligned(p, at, at + len))
          at += len
        }
        return out
      })
    case 'Sequence':
      return spec.pretokenizers.reduce<AlignedText[]>((ps, s) => applyPreTokenizer(s, ps, first), [...pieces])
    default:
      throw new WhiteboxError('value', `applyPreTokenizer: unknown pre-tokenizer ${(spec as { type: string }).type}`)
  }
}

/** A decoder of an HF-style tokenizer pipeline (the `decoder` of a `tokenizer.json`). */
export type DecoderSpec =
  | { readonly type: 'ByteLevel' | 'Fuse' | 'ByteFallback' }
  | { readonly type: 'WordPiece'; readonly prefix?: string; readonly cleanup?: boolean }
  | { readonly type: 'Metaspace'; readonly replacement?: string; readonly prependScheme?: 'always' | 'first' | 'never' }
  | { readonly type: 'BPEDecoder'; readonly suffix?: string }
  | { readonly type: 'Replace'; readonly pattern: TokenizerPattern; readonly content: string }
  | { readonly type: 'Strip'; readonly content: string; readonly start: number; readonly stop: number }
  | {
      readonly type: 'CTC'
      readonly padToken?: string
      readonly wordDelimiterToken?: string
      readonly cleanup?: boolean
    }
  | { readonly type: 'Sequence'; readonly decoders: readonly DecoderSpec[] }

/** WordPiece's decoding clean-up of spaces before punctuation and clitics. */
function cleanup(s: string): string {
  return s
    .replaceAll(' .', '.')
    .replaceAll(' ?', '?')
    .replaceAll(' !', '!')
    .replaceAll(' ,', ',')
    .replaceAll(" ' ", "'")
    .replaceAll(" n't", "n't")
    .replaceAll(" 'm", "'m")
    .replaceAll(' do not', " don't")
    .replaceAll(" 's", "'s")
    .replaceAll(" 've", "'ve")
    .replaceAll(" 're", "'re")
}

/**
 * Apply a decoder's chain step to tokens (HF's `decode_chain`): the decoded pieces, which a final join concatenates.
 *
 * @param spec - The decoder.
 * @param tokens - The tokens (or the pieces of the previous decoder).
 * @returns The decoded pieces.
 * @throws WhiteboxError with code `value` for an unknown decoder.
 * @concept tokenizer-pipeline
 * @example
 * ```ts
 * import { decodeChain } from 'whitebox-engine/text'
 * decodeChain({ type: 'WordPiece' }, ['un', '##aff', '##able', 'ok', '.']).join('') // => 'unaffable ok.'
 * ```
 */
export function decodeChain(spec: DecoderSpec, tokens: readonly string[]): string[] {
  switch (spec.type) {
    case 'ByteLevel':
      return [fromByteLevel(tokens.join(''))]
    case 'Fuse':
      return [tokens.join('')]
    case 'ByteFallback': {
      const out: string[] = []
      let bytes: number[] = []
      const flush = (): void => {
        if (bytes.length === 0) return
        try {
          out.push(new TextDecoder('utf-8', { fatal: true }).decode(Uint8Array.from(bytes)))
        } catch {
          for (let i = 0; i < bytes.length; i++) out.push('�')
        }
        bytes = []
      }
      for (const t of tokens) {
        const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(t)
        if (m !== null) bytes.push(Number.parseInt(m[1] ?? '0', 16))
        else {
          flush()
          out.push(t)
        }
      }
      flush()
      return out
    }
    case 'WordPiece': {
      const prefix = spec.prefix ?? '##'
      return tokens.map((t, i) => {
        let s = i === 0 ? t : t.startsWith(prefix) ? t.slice(prefix.length) : ` ${t}`
        if (spec.cleanup !== false) s = cleanup(s)
        return s
      })
    }
    case 'Metaspace': {
      const rep = spec.replacement ?? '▁'
      const scheme = spec.prependScheme ?? 'always'
      return tokens.map((t, i) => {
        let out = ''
        for (const c of t) out += c === rep ? (i === 0 && scheme !== 'never' ? '' : ' ') : c
        return out
      })
    }
    case 'BPEDecoder': {
      const suffix = spec.suffix ?? '</w>'
      return tokens.map((t, i) => t.replaceAll(suffix, i === tokens.length - 1 ? '' : ' '))
    }
    case 'Replace':
      return tokens.map((t) =>
        t.replace(
          typeof spec.pattern === 'string'
            ? new RegExp(spec.pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gu')
            : new RegExp(spec.pattern.source, `${spec.pattern.flags.replace('g', '')}g`),
          () => spec.content,
        ),
      )
    case 'Strip':
      return tokens.map((t) => {
        const chars = Array.from(t)
        let a = 0
        for (let i = 0; i < Math.min(spec.start, chars.length) && chars[i] === spec.content; i++) a = i + 1
        let b = chars.length
        for (let i = 0; i < spec.stop && b - 1 >= 0 && chars[chars.length - i - 1] === spec.content; i++)
          b = chars.length - i - 1
        return chars.slice(a, Math.max(a, b)).join('')
      })
    case 'CTC': {
      const pad = spec.padToken ?? '<pad>'
      const delim = spec.wordDelimiterToken ?? '|'
      const out: string[] = []
      let prev: string | undefined
      for (const t of tokens) {
        if (t === prev) continue
        prev = t
        let r = t.replaceAll(pad, '')
        if (spec.cleanup !== false) r = cleanup(r).replaceAll(delim, ' ')
        if (r.length > 0) out.push(r)
      }
      return out
    }
    case 'Sequence':
      return spec.decoders.reduce<string[]>((ts, d) => decodeChain(d, ts), [...tokens])
    default:
      throw new WhiteboxError('value', `decodeChain: unknown decoder ${(spec as { type: string }).type}`)
  }
}

/**
 * A regular expression of an HF `tokenizer.json` (Oniguruma or Rust syntax) as a JavaScript one: Unicode classes for
 * `\w`, `\s`, `\d` and their negations, possessive quantifiers made greedy (equivalent for the usual pre-tokeniser
 * patterns, whose possessive parts cannot backtrack usefully).
 *
 * @param source - The pattern.
 * @returns The regular expression (global, Unicode).
 * @throws WhiteboxError with code `value` for syntax JavaScript lacks (atomic groups).
 * @concept tokenizer-pipeline
 * @example
 * ```ts
 * import { hfRegex } from 'whitebox-engine/text'
 * 'Hi  there'.match(hfRegex('\\s+(?!\\S)|\\w+'))?.length // => 3
 * ```
 */
export function hfRegex(source: string): RegExp {
  if (source.includes('(?>')) throw new WhiteboxError('value', 'hfRegex: atomic groups are not supported')
  let out = ''
  let inClass = false
  for (let i = 0; i < source.length; i++) {
    const c = source[i] ?? ''
    if (c === '\\') {
      const n = source[i + 1] ?? ''
      i++
      const cls: Record<string, [string, string]> = {
        w: [`[${UNI_WORD}]`, UNI_WORD],
        s: ['\\p{White_Space}', '\\p{White_Space}'],
        S: ['\\P{White_Space}', '\\P{White_Space}'],
        d: ['\\p{Nd}', '\\p{Nd}'],
        D: ['\\P{Nd}', '\\P{Nd}'],
        W: [`[^${UNI_WORD}]`, `[^${UNI_WORD}]`],
      }
      const sub = cls[n]
      out += sub === undefined ? `\\${n}` : inClass ? sub[1] : sub[0]
      continue
    }
    if (c === '[' && !inClass) inClass = true
    else if (c === ']' && inClass) inClass = false
    if (!inClass && (c === '+' || c === '*' || c === '?' || c === '}') && source[i + 1] === '+') {
      out += c
      i++
      continue
    }
    out += c
  }
  try {
    return new RegExp(out, 'gu')
  } catch {
    throw new WhiteboxError('value', `hfRegex: cannot translate ${source}`)
  }
}
