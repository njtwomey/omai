/**
 * Phonetic encodings, which map words that sound alike to the same code for fuzzy name matching: American Soundex and
 * Metaphone, as jellyfish computes them.
 */

const SOUNDEX_CODES: Readonly<Record<string, string>> = {
  B: '1',
  F: '1',
  P: '1',
  V: '1',
  C: '2',
  G: '2',
  J: '2',
  K: '2',
  Q: '2',
  S: '2',
  X: '2',
  Z: '2',
  D: '3',
  T: '3',
  L: '4',
  M: '5',
  N: '5',
  R: '6',
}

/**
 * American Soundex as jellyfish computes it: the first letter (after upper-casing and NFKD), then the digit codes of
 * the following consonants, collapsing repeats (also across H and W but not across vowels), padded or cut to four.
 *
 * @param word - The word.
 * @returns The four-character code (empty for an empty word).
 * @concept phonetic-encoding
 * @example
 * ```ts
 * import { soundex } from 'mdml-engine/text'
 * ['Robert', 'Rupert', 'Tymczak', 'Pfister', 'Ashcraft'].map(soundex) // => ['R163', 'R163', 'T522', 'P236', 'A261']
 * ```
 */
export function soundex(word: string): string {
  if (word.length === 0) return ''
  const v = Array.from(word.toUpperCase().normalize('NFKD'))
  const first = v[0] ?? ''
  const result = [first]
  let last = SOUNDEX_CODES[first] ?? '*'
  for (const letter of v.slice(1)) {
    const sub = SOUNDEX_CODES[letter] ?? '*'
    if (sub !== '*') {
      if (sub !== last) {
        result.push(sub)
        if (result.length === 4) break
      }
      last = sub
    } else if (letter !== 'H' && letter !== 'W') last = '*'
  }
  while (result.length < 4) result.push('0')
  return result.join('')
}

const isVowel = (c: string | undefined): boolean => c === 'A' || c === 'E' || c === 'I' || c === 'O' || c === 'U'
const isIey = (c: string | undefined): boolean => c === 'I' || c === 'E' || c === 'Y'

/**
 * Metaphone (Philips 1990) as jellyfish implements it: a consonant skeleton of the pronunciation, keeping initial
 * vowels, with the usual rules for C, G, GH, PH, SH, TH (as `0`), silent initial letters and doubled letters.
 *
 * @param word - The word (or words separated by spaces).
 * @returns The code, in upper case.
 * @concept phonetic-encoding
 * @example
 * ```ts
 * import { metaphone } from 'mdml-engine/text'
 * ['Thompson', 'knight', 'Schmidt', 'Xavier', 'phonetic'].map(metaphone) // => ['0MPSN', 'NT', 'SXMTT', 'SFR', 'FNTK']
 * ```
 */
export function metaphone(word: string): string {
  if (word.length === 0) return ''
  const upper = word.toUpperCase()
  const v = Array.from(upper.normalize('NFKD'))
  if (['KN', 'GN', 'PN', 'WR', 'AE'].some((p) => upper.startsWith(p))) v.shift()
  const out: string[] = []
  let i = 0
  while (i < v.length) {
    const c = v[i] ?? ''
    const next = v[i + 1] ?? '*'
    const nextnext = v[i + 2] ?? '*'
    if (c === next && c !== 'C') {
      i++
      continue
    }
    switch (c) {
      case 'A':
      case 'E':
      case 'I':
      case 'O':
      case 'U':
        if (i === 0 || v[i - 1] === ' ') out.push(c)
        break
      case 'B':
        if (i === 0 || v[i - 1] !== 'M' || next !== '*') out.push('B')
        break
      case 'C':
        if ((next === 'I' && nextnext === 'A') || next === 'H') {
          i++
          out.push('X')
        } else if (isIey(next)) {
          i++
          out.push('S')
        } else out.push('K')
        break
      case 'D':
        if (next === 'G' && isIey(nextnext)) {
          i += 2
          out.push('J')
        } else out.push('T')
        break
      case 'F':
      case 'J':
      case 'L':
      case 'M':
      case 'N':
      case 'R':
        out.push(c)
        break
      case 'G':
        if (isIey(next)) out.push('J')
        else if ((next === 'H' && !isVowel(nextnext)) || (next === 'N' && nextnext === '*')) i++
        else out.push('K')
        break
      case 'H':
        if (i === 0 || isVowel(next) || !isVowel(v[i - 1])) out.push('H')
        break
      case 'K':
        if (i === 0 || v[i - 1] !== 'C') out.push('K')
        break
      case 'P':
        if (next === 'H') {
          i++
          out.push('F')
        } else out.push('P')
        break
      case 'Q':
        out.push('K')
        break
      case 'S':
        if (next === 'H') {
          i++
          out.push('X')
        } else if (next === 'I' && (nextnext === 'O' || nextnext === 'A')) {
          i += 2
          out.push('X')
        } else out.push('S')
        break
      case 'T':
        if (next === 'I' && (nextnext === 'O' || nextnext === 'A')) out.push('X')
        else if (next === 'H') {
          i++
          out.push('0')
        } else if (next !== 'C' || nextnext !== 'H') out.push('T')
        break
      case 'V':
        out.push('F')
        break
      case 'W':
        if (i === 0 && next === 'H') {
          i++
          out.push('W')
        } else if (isVowel(next)) out.push('W')
        break
      case 'X':
        if (i === 0) out.push(next === 'H' || (next === 'I' && (nextnext === 'O' || nextnext === 'A')) ? 'X' : 'S')
        else out.push('K', 'S')
        break
      case 'Y':
        if (isVowel(next)) out.push('Y')
        break
      case 'Z':
        out.push('S')
        break
      case ' ':
        if (out.length > 0 && out[out.length - 1] !== ' ') out.push(' ')
        break
      default:
        break
    }
    i++
  }
  return out.join('')
}
