import { UNI_SPACE, UNI_WORD } from './_classes.ts'
import { CP1252_HIGH, HTML_ENTITIES } from './_entities.data.ts'

// nltk's casual tokenizer (Christopher Potts' "happyfuntokenizing"), compiled there with the `regex` module and the
// VERBOSE, IGNORECASE and UNICODE flags. Each piece below is the original with whitespace and comments removed.
const W = UNI_WORD
const S = UNI_SPACE
const D = '\\p{Nd}'
const ALPHA = `(?:(?![${D}_])[${W}])`
const MODIFIER = '[\\u{1f3fb}-\\u{1f3ff}]'
const NOT_SPACE = `[^${S}]`

const EMOTICONS = "(?:[<>]?[:;=8][\\-o*']?[)\\](\\[dDpP/:}{@|\\\\]|[)\\](\\[dDpP/:}{@|\\\\][\\-o*']?[:;=8][<>]?|</?3)"
const PARENS1 = `\\([^${S}()]{0,255}?\\([^${S}()]{1,255}\\)[^${S}()]{0,255}?\\)`
const PARENS0 = `\\([^${S}]{1,255}?\\)`
const URLS =
  `(?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\\-]{1,255}[.](?:[a-z]{2,13})/)` +
  `(?:[^${S}()<>{}\\[\\]]+|${PARENS1}|${PARENS0})+` +
  `(?:${PARENS1}|${PARENS0}|[^${S}\`!()\\[\\]{};:'".,<>?«»“”‘’])` +
  `|(?:(?<!@)[a-z0-9]+(?:[.\\-][a-z0-9]+){0,126}[.](?:[a-z]{2,13})(?:(?<=[${W}])(?![${W}])|(?<![${W}])(?=[${W}]))/?(?!@))`
const PHONE = `(?:(?:\\+?[01][ *\\-.)]*)?(?:[(]?${D}{3}[ *\\-.)]*)?${D}{3}[ *\\-.)]*${D}{4})`
const FLAGS =
  '(?:[\\u{1F1E6}-\\u{1F1FF}]{2}' +
  '|\\u{1F3F4}\\u{E0067}\\u{E0062}\\u{E0065}\\u{E006E}\\u{E0067}\\u{E007F}' +
  '|\\u{1F3F4}\\u{E0067}\\u{E0062}\\u{E0073}\\u{E0063}\\u{E0074}\\u{E007F}' +
  '|\\u{1F3F4}\\u{E0067}\\u{E0062}\\u{E0077}\\u{E006C}\\u{E0073}\\u{E007F})'
const REST = [
  '<[^>\\p{White_Space}]+>',
  '[\\-]+>|<[\\-]+',
  `(?:@[${W}_]+)`,
  `(?:#+[${W}_]+[${W}'_\\-]*[${W}_]+)`,
  `[${W}.+\\-]{1,64}@[${W}\\-]{1,63}\\.(?:[${W}\\-]\\.?){1,251}[${W}\\-]`,
  `${NOT_SPACE}(?:${MODIFIER}?(?:\\u200d${NOT_SPACE}${MODIFIER}?)+|${MODIFIER})`,
  FLAGS,
  `(?:${ALPHA}(?:${ALPHA}|['\\-_])+${ALPHA})|(?:[+\\-]?${D}+[,/.:\\-]${D}+[+\\-]?)|(?:[${W}_]+)|(?:\\.(?:(?:(?![\\r\\n])[${S}])*\\.){1,})|(?:${NOT_SPACE})`,
]

let compiled: { word: RegExp; phone: RegExp; emoticon: RegExp } | undefined
function patterns(): { word: RegExp; phone: RegExp; emoticon: RegExp } {
  compiled ??= {
    word: new RegExp([URLS, EMOTICONS, ...REST].join('|'), 'giu'),
    phone: new RegExp([URLS, PHONE, EMOTICONS, ...REST].join('|'), 'giu'),
    emoticon: new RegExp(EMOTICONS, 'iu'),
  }
  return compiled
}

let entities: Map<string, number> | undefined
let cp1252: Map<number, number> | undefined

/** Python's int(s, base) on the bodies nltk passes it (optional sign, underscores between digits). */
function parseInteger(body: string, hex: boolean): number | undefined {
  const re = hex ? /^[+-]?(?:0x_?)?[0-9a-f]+(?:_[0-9a-f]+)*$/i : /^[+-]?[0-9]+(?:_[0-9]+)*$/
  if (!re.test(body)) return undefined
  const clean = body.replace(/_/g, '').replace(/^([+-]?)0x/i, '$1')
  return Number.parseInt(clean, hex ? 16 : 10)
}

/**
 * Replace HTML entities by the characters they name, as nltk's casual tokenizer does: named HTML 4 entities, decimal
 * `&#nnn;` and hexadecimal `&#xhh;` references (0x80 to 0x9F read as Windows-1252); entities that cannot be converted
 * are removed (or kept with `removeIllegal: false`), and names in `keep` are left alone.
 *
 * @param text - The text.
 * @param options - Entity names to keep, and whether to remove illegal entities (default true).
 * @returns The text with entities replaced.
 * @concept tokenisation
 * @example
 * ```ts
 * import { replaceHtmlEntities } from 'whitebox-engine/text'
 * replaceHtmlEntities('Price: &pound;100 &#8364;5 &#x41; &#128; &bogus;') // => 'Price: £100 €5 A € '
 * ```
 */
export function replaceHtmlEntities(
  text: string,
  options: { readonly keep?: readonly string[]; readonly removeIllegal?: boolean } = {},
): string {
  entities ??= new Map(HTML_ENTITIES)
  cp1252 ??= new Map(CP1252_HIGH)
  const names = entities
  const high = cp1252
  const remove = options.removeIllegal !== false
  return text.replace(
    /&(#?(x?))([^&;\p{White_Space}]+);/gu,
    (all: string, numeric: string, x: string, body: string) => {
      let n: number | undefined
      if (numeric.length > 0) {
        n = parseInteger(body, x.length > 0)
        if (n !== undefined && n >= 0x80 && n <= 0x9f) {
          const c = high.get(n)
          return c === undefined ? (remove ? '' : all) : String.fromCodePoint(c)
        }
      } else {
        if (options.keep?.includes(body) === true) return all
        n = names.get(body)
      }
      if (n !== undefined && n >= 0 && n <= 0x10ffff) return String.fromCodePoint(n)
      return remove ? '' : all
    },
  )
}

/**
 * Remove Twitter handles (`@name`, up to 15 word characters, not inside an e-mail address), replacing each by a space.
 *
 * @param text - The text.
 * @returns The text without handles.
 * @example
 * ```ts
 * import { removeHandles } from 'whitebox-engine/text'
 * removeHandles('@remy: hi a@b.com') // => ' : hi a@b.com'
 * ```
 */
export function removeHandles(text: string): string {
  return text.replace(/(?<![A-Za-z0-9_!@#$%&*])@(([A-Za-z0-9_]){15}(?!@)|([A-Za-z0-9_]){1,14}(?![A-Za-z0-9_]*@))/g, ' ')
}

/**
 * Shorten runs of three or more repeated characters to three (`waaaaayyyy` to `waaayyy`).
 *
 * @param text - The text.
 * @returns The shortened text.
 * @example
 * ```ts
 * import { reduceLengthening } from 'whitebox-engine/text'
 * reduceLengthening('waaaaayyyy!!!!') // => 'waaayyy!!!'
 * ```
 */
export function reduceLengthening(text: string): string {
  return text.replace(/([^\n])\1{2,}/gu, '$1$1$1')
}

/** Options of {@link tweetTokenize} (nltk's TweetTokenizer). */
export interface TweetOptions {
  /** Keep case (default true); otherwise lowercase everything except emoticons. */
  readonly preserveCase?: boolean
  /** Shorten character runs to three (default false). */
  readonly reduceLen?: boolean
  /** Remove handles (default false). */
  readonly stripHandles?: boolean
  /** Recognise phone numbers as single tokens (default true). */
  readonly matchPhoneNumbers?: boolean
}

/**
 * nltk's TweetTokenizer: HTML entities replaced, then URLs, phone numbers, emoticons, HTML tags, arrows, handles,
 * hashtags, e-mail addresses, emoji sequences with skin tones and ZWJ joins, flags, words with apostrophes and dashes,
 * numbers, ellipses and single characters, matched in that order of preference.
 *
 * @param text - The text.
 * @param options - Case, lengthening, handles and phone numbers.
 * @returns The tokens.
 * @concept tokenisation
 * @example
 * ```ts
 * import { tweetTokenize } from 'whitebox-engine/text'
 * tweetTokenize('This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--') // => ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->', '<--']
 * tweetTokenize('@remy: This is waaaaayyyy too much for you!!!!!!', { stripHandles: true, reduceLen: true }) // => [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']
 * ```
 */
export function tweetTokenize(text: string, options: TweetOptions = {}): string[] {
  let s = replaceHtmlEntities(text)
  if (options.stripHandles === true) s = removeHandles(s)
  if (options.reduceLen === true) s = reduceLengthening(s)
  const safe = s.replace(/([^\p{L}\p{N}])\1{3,}/gu, '$1$1$1')
  const { word, phone, emoticon } = patterns()
  const words = Array.from(safe.matchAll(options.matchPhoneNumbers === false ? word : phone), (m) => m[0])
  if (options.preserveCase === false) return words.map((w) => (emoticon.test(w) ? w : w.toLowerCase()))
  return words
}
