/**
 * Stemming: stripping suffixes to reduce related word forms to a common stem. The Porter stemmer exactly as nltk
 * implements it (three variants), with a trace of every rule that fired, and the Snowball English (Porter2) stemmer.
 */

import { MdmlError } from '../base/index.ts'

// ---------------------------------------------------------------------------------------------------------------
// Porter (1980), as nltk's PorterStemmer implements it in its three modes.

/** The variant of the Porter stemmer: nltk's extensions (default), Martin Porter's later ones, or the 1980 paper. */
export type PorterMode = 'nltk' | 'martin' | 'original'

/** One rule that changed the word, in a {@link porterStemTrace}. */
export interface PorterStep {
  /** The step of the algorithm (`1a`, `1b`, `1c`, `2`, `3`, `4`, `5a`, `5b`, or `irregular`). */
  readonly step: string
  /** The rule as `suffix -> replacement`. */
  readonly rule: string
  readonly before: string
  readonly after: string
}

const VOWELS = new Set(['a', 'e', 'i', 'o', 'u'])

const IRREGULAR: Readonly<Record<string, readonly string[]>> = {
  sky: ['sky', 'skies'],
  die: ['dying'],
  lie: ['lying'],
  tie: ['tying'],
  news: ['news'],
  inning: ['innings', 'inning'],
  outing: ['outings', 'outing'],
  canning: ['cannings', 'canning'],
  howe: ['howe'],
  proceed: ['proceed'],
  exceed: ['exceed'],
  succeed: ['succeed'],
}
const IRREGULAR_POOL = new Map(Object.entries(IRREGULAR).flatMap(([k, vs]) => vs.map((v) => [v, k] as const)))

/** A Porter rule: a suffix (`*d` for a double consonant), its replacement and an optional condition on the stem. */
type Rule = readonly [string, string, ((stem: string) => boolean) | undefined]

class Porter {
  readonly steps: PorterStep[] = []
  readonly mode: PorterMode
  constructor(mode: PorterMode) {
    this.mode = mode
  }

  isConsonant(word: string, index: number): boolean {
    let i = index
    const ch = word[i] ?? ''
    if (VOWELS.has(ch)) return false
    if (ch !== 'y') return true
    let negate = false
    while (i > 0 && word[i] === 'y') {
      negate = !negate
      i--
    }
    return !VOWELS.has(word[i] ?? '') !== negate
  }

  flags(word: string): boolean[] {
    const out: boolean[] = []
    for (let i = 0; i < word.length; i++) {
      const ch = word[i] ?? ''
      out.push(VOWELS.has(ch) ? false : ch === 'y' ? (i === 0 ? true : !(out[i - 1] ?? false)) : true)
    }
    return out
  }

  /** The measure m of a stem: the number of vowel-consonant sequences in its [C](VC)^m[V] form. */
  measure(stem: string): number {
    const f = this.flags(stem)
    let m = 0
    for (let i = 1; i < f.length; i++) if (f[i] === true && f[i - 1] === false) m++
    return m
  }

  containsVowel(stem: string): boolean {
    return this.flags(stem).some((c) => !c)
  }

  endsDoubleConsonant(word: string): boolean {
    const n = word.length
    return n >= 2 && word[n - 1] === word[n - 2] && this.isConsonant(word, n - 1)
  }

  endsCvc(word: string): boolean {
    const n = word.length
    return (
      (n >= 3 &&
        this.isConsonant(word, n - 3) &&
        !this.isConsonant(word, n - 2) &&
        this.isConsonant(word, n - 1) &&
        !['w', 'x', 'y'].includes(word[n - 1] ?? '')) ||
      (this.mode === 'nltk' && n === 2 && !this.isConsonant(word, 0) && this.isConsonant(word, 1))
    )
  }

  record(step: string, rule: string, before: string, after: string): string {
    if (after !== before) this.steps.push({ step, rule, before, after })
    return after
  }

  applyRules(step: string, word: string, rules: readonly Rule[]): string {
    for (const [suffix, replacement, condition] of rules) {
      if (suffix === '*d' && this.endsDoubleConsonant(word)) {
        const stem = word.slice(0, -2)
        return condition === undefined || condition(stem)
          ? this.record(step, `*d -> ${replacement}`, word, stem + replacement)
          : word
      }
      if (word.endsWith(suffix)) {
        const stem = suffix === '' ? word : word.slice(0, -suffix.length)
        return condition === undefined || condition(stem)
          ? this.record(step, `${suffix || '()'} -> ${replacement || '()'}`, word, stem + replacement)
          : word
      }
    }
    return word
  }

  step1a(word: string): string {
    if (this.mode === 'nltk' && word.endsWith('ies') && word.length === 4)
      return this.record('1a', 'ies -> ie', word, `${word.slice(0, -3)}ie`)
    return this.applyRules('1a', word, [
      ['sses', 'ss', undefined],
      ['ies', 'i', undefined],
      ['ss', 'ss', undefined],
      ['s', '', undefined],
    ])
  }

  step1b(word: string): string {
    if (this.mode === 'nltk' && word.endsWith('ied')) {
      return word.length === 4
        ? this.record('1b', 'ied -> ie', word, `${word.slice(0, -3)}ie`)
        : this.record('1b', 'ied -> i', word, `${word.slice(0, -3)}i`)
    }
    if (word.endsWith('eed')) {
      const stem = word.slice(0, -3)
      return this.measure(stem) > 0 ? this.record('1b', 'eed -> ee', word, `${stem}ee`) : word
    }
    let intermediate: string | undefined
    for (const suffix of ['ed', 'ing']) {
      if (word.endsWith(suffix)) {
        const stem = word.slice(0, -suffix.length)
        if (this.containsVowel(stem)) {
          intermediate = this.record('1b', `${suffix} -> ()`, word, stem)
          break
        }
      }
    }
    if (intermediate === undefined) return word
    const s = intermediate
    const last = s[s.length - 1] ?? ''
    return this.applyRules('1b', s, [
      ['at', 'ate', undefined],
      ['bl', 'ble', undefined],
      ['iz', 'ize', undefined],
      ['*d', last, () => !['l', 's', 'z'].includes(last)],
      ['', 'e', (stem) => this.measure(stem) === 1 && this.endsCvc(stem)],
    ])
  }

  step1c(word: string): string {
    const condition =
      this.mode === 'nltk'
        ? (stem: string): boolean => stem.length > 1 && this.isConsonant(stem, stem.length - 1)
        : (stem: string): boolean => this.containsVowel(stem)
    return this.applyRules('1c', word, [['y', 'i', condition]])
  }

  step2(word: string): string {
    const m0 = (stem: string): boolean => this.measure(stem) > 0
    if (this.mode === 'nltk' && word.endsWith('alli') && m0(word.slice(0, -4)))
      return this.step2(this.record('2', 'alli -> al', word, `${word.slice(0, -4)}al`))
    const rules: Rule[] = [
      ['ational', 'ate', m0],
      ['tional', 'tion', m0],
      ['enci', 'ence', m0],
      ['anci', 'ance', m0],
      ['izer', 'ize', m0],
      this.mode === 'original' ? ['abli', 'able', m0] : ['bli', 'ble', m0],
      ['alli', 'al', m0],
      ['entli', 'ent', m0],
      ['eli', 'e', m0],
      ['ousli', 'ous', m0],
      ['ization', 'ize', m0],
      ['ation', 'ate', m0],
      ['ator', 'ate', m0],
      ['alism', 'al', m0],
      ['iveness', 'ive', m0],
      ['fulness', 'ful', m0],
      ['ousness', 'ous', m0],
      ['aliti', 'al', m0],
      ['iviti', 'ive', m0],
      ['biliti', 'ble', m0],
    ]
    if (this.mode === 'nltk') {
      rules.push(['fulli', 'ful', m0])
      rules.push(['logi', 'log', () => m0(word.slice(0, -3))])
    }
    if (this.mode === 'martin') rules.push(['logi', 'log', m0])
    return this.applyRules('2', word, rules)
  }

  step3(word: string): string {
    const m0 = (stem: string): boolean => this.measure(stem) > 0
    return this.applyRules('3', word, [
      ['icate', 'ic', m0],
      ['ative', '', m0],
      ['alize', 'al', m0],
      ['iciti', 'ic', m0],
      ['ical', 'ic', m0],
      ['ful', '', m0],
      ['ness', '', m0],
    ])
  }

  step4(word: string): string {
    const m1 = (stem: string): boolean => this.measure(stem) > 1
    const rules: Rule[] = ['al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement', 'ment', 'ent'].map(
      (s): Rule => [s, '', m1],
    )
    rules.push(['ion', '', (stem) => this.measure(stem) > 1 && ['s', 't'].includes(stem[stem.length - 1] ?? '')])
    for (const s of ['ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize']) rules.push([s, '', m1])
    return this.applyRules('4', word, rules)
  }

  step5a(word: string): string {
    if (!word.endsWith('e')) return word
    const stem = word.slice(0, -1)
    const m = this.measure(stem)
    if (m > 1 || (m === 1 && !this.endsCvc(stem))) return this.record('5a', 'e -> ()', word, stem)
    return word
  }

  step5b(word: string): string {
    return this.applyRules('5b', word, [['ll', 'l', () => this.measure(word.slice(0, -1)) > 1]])
  }

  stem(word: string, lowercase: boolean): string {
    let s = lowercase ? word.toLowerCase() : word
    const irregular = this.mode === 'nltk' ? IRREGULAR_POOL.get(s) : undefined
    if (irregular !== undefined) return this.record('irregular', `${s} -> ${irregular}`, s, irregular)
    if (this.mode !== 'original' && word.length <= 2) return s
    s = this.step1a(s)
    s = this.step1b(s)
    s = this.step1c(s)
    s = this.step2(s)
    s = this.step3(s)
    s = this.step4(s)
    s = this.step5a(s)
    return this.step5b(s)
  }
}

/** Options of the Porter stemmer. */
export interface PorterOptions {
  /** The variant (default `nltk`, nltk's default). */
  readonly mode?: PorterMode
  /** Lowercase the word first (default true). */
  readonly lowercase?: boolean
}

function checkMode(mode: PorterMode): void {
  if (mode !== 'nltk' && mode !== 'martin' && mode !== 'original')
    throw new MdmlError('value', `porterStem: unknown mode ${String(mode)}`)
}

/**
 * The Porter stemmer (Porter 1980) exactly as nltk's PorterStemmer: strip suffixes in five steps, each rule guarded by
 * the measure of the remaining stem. Modes: `nltk` (default; irregular forms, the two-letter cvc rule and a few
 * extra rules), `martin` (the extensions of Porter's own later implementations) and `original` (the paper).
 *
 * @param word - The word.
 * @param options - Mode and lowercasing.
 * @returns The stem.
 * @throws MdmlError with code `value` for an unknown mode.
 * @concept stemming
 * @example
 * ```ts
 * import { porterStem } from 'mdml-engine/text'
 * ['caresses', 'ponies', 'relational', 'generalizations', 'dying'].map((w) => porterStem(w)) // => ['caress', 'poni', 'relat', 'gener', 'die']
 * porterStem('ponies', { mode: 'original' }) // => 'poni'
 * ```
 */
export function porterStem(word: string, options: PorterOptions = {}): string {
  const mode = options.mode ?? 'nltk'
  checkMode(mode)
  return new Porter(mode).stem(word, options.lowercase !== false)
}

/**
 * The Porter stemmer with a record of every rule that fired: the step, the rule and the word before and after.
 *
 * @param word - The word.
 * @param options - Mode and lowercasing.
 * @returns The stem and the rules applied in order.
 * @concept stemming
 * @example
 * ```ts
 * import { porterStemTrace } from 'mdml-engine/text'
 * porterStemTrace('generalizations').steps.map((s) => `${s.step}: ${s.rule}`) // => ['1a: s -> ()', '2: ization -> ize', '3: alize -> al', '4: al -> ()']
 * ```
 */
export function porterStemTrace(
  word: string,
  options: PorterOptions = {},
): { readonly stem: string; readonly steps: readonly PorterStep[] } {
  const mode = options.mode ?? 'nltk'
  checkMode(mode)
  const p = new Porter(mode)
  const stem = p.stem(word, options.lowercase !== false)
  return { stem, steps: p.steps }
}

// ---------------------------------------------------------------------------------------------------------------
// Snowball English (Porter2), as nltk's SnowballStemmer('english').

const SB_VOWELS = 'aeiouy'
const isVowel = (c: string | undefined): boolean => c !== undefined && c.length === 1 && SB_VOWELS.includes(c)
const DOUBLES = ['bb', 'dd', 'ff', 'gg', 'mm', 'nn', 'pp', 'rr', 'tt']
const LI_ENDING = 'cdeghkmnrt'
const STEP1A = ['sses', 'ied', 'ies', 'us', 'ss', 's']
const STEP1B = ['eedly', 'ingly', 'edly', 'eed', 'ing', 'ed']
const STEP2 = [
  'ization',
  'ational',
  'fulness',
  'ousness',
  'iveness',
  'tional',
  'biliti',
  'lessli',
  'entli',
  'ation',
  'alism',
  'aliti',
  'ousli',
  'iviti',
  'fulli',
  'enci',
  'anci',
  'abli',
  'izer',
  'ator',
  'alli',
  'bli',
  'ogi',
  'li',
]
const STEP3 = ['ational', 'tional', 'alize', 'icate', 'iciti', 'ative', 'ical', 'ness', 'ful']
const STEP4 = [
  'ement',
  'ance',
  'ence',
  'able',
  'ible',
  'ment',
  'ant',
  'ent',
  'ism',
  'ate',
  'iti',
  'ous',
  'ive',
  'ize',
  'ion',
  'al',
  'er',
  'ic',
]
const SPECIAL: Readonly<Record<string, string>> = {
  skis: 'ski',
  skies: 'sky',
  dying: 'die',
  lying: 'lie',
  tying: 'tie',
  idly: 'idl',
  gently: 'gentl',
  ugly: 'ugli',
  early: 'earli',
  only: 'onli',
  singly: 'singl',
  sky: 'sky',
  news: 'news',
  howe: 'howe',
  atlas: 'atlas',
  cosmos: 'cosmos',
  bias: 'bias',
  andes: 'andes',
  inning: 'inning',
  innings: 'inning',
  outing: 'outing',
  outings: 'outing',
  canning: 'canning',
  cannings: 'canning',
  herring: 'herring',
  herrings: 'herring',
  earring: 'earring',
  earrings: 'earring',
  proceed: 'proceed',
  proceeds: 'proceed',
  proceeded: 'proceed',
  proceeding: 'proceed',
  exceed: 'exceed',
  exceeds: 'exceed',
  exceeded: 'exceed',
  exceeding: 'exceed',
  succeed: 'succeed',
  succeeds: 'succeed',
  succeeded: 'succeed',
  succeeding: 'succeed',
}

/** The regions R1 and R2: what follows the first non-vowel after a vowel, and the same within R1. */
function regions(word: string): [string, string] {
  let r1 = ''
  let r2 = ''
  for (let i = 1; i < word.length; i++) {
    if (!isVowel(word[i]) && isVowel(word[i - 1])) {
      r1 = word.slice(i + 1)
      break
    }
  }
  for (let i = 1; i < r1.length; i++) {
    if (!isVowel(r1[i]) && isVowel(r1[i - 1])) {
      r2 = r1.slice(i + 1)
      break
    }
  }
  return [r1, r2]
}

/** Python's s[:-k] for k >= 1. */
const cut = (s: string, k: number): string => s.slice(0, Math.max(0, s.length - k))
const replaceSuffix = (s: string, suffix: string, by: string): string => cut(s, suffix.length) + by

/**
 * The Snowball English stemmer (Porter2) exactly as nltk's SnowballStemmer('english'): Porter's algorithm revised with
 * the regions R1 and R2, an exception list, apostrophe handling and the `-li` and `-ogi` rules.
 *
 * @param word - The word.
 * @param options - Stop words to leave unstemmed (default none, as nltk's default).
 * @returns The stem.
 * @concept stemming
 * @example
 * ```ts
 * import { snowballStem } from 'mdml-engine/text'
 * ['generously', 'running', "dog's", 'communication', 'skies'].map((w) => snowballStem(w)) // => ['generous', 'run', 'dog', 'communic', 'sky']
 * ```
 */
export function snowballStem(word: string, options: { readonly stopWords?: ReadonlySet<string> } = {}): string {
  let w = word.toLowerCase()
  if (options.stopWords?.has(w) === true || w.length <= 2) return w
  const special = SPECIAL[w]
  if (special !== undefined) return special
  w = w.replace(/[’‘‛]/g, "'")
  if (w.startsWith("'")) w = w.slice(1)
  if (w.startsWith('y')) w = `Y${w.slice(1)}`
  const chars = w.split('')
  for (let i = 1; i < chars.length; i++) if (isVowel(chars[i - 1]) && chars[i] === 'y') chars[i] = 'Y'
  w = chars.join('')
  let r1: string
  let r2 = ''
  if (w.startsWith('gener') || w.startsWith('commun') || w.startsWith('arsen')) {
    r1 = w.startsWith('commun') ? w.slice(6) : w.slice(5)
    for (let i = 1; i < r1.length; i++) {
      if (!isVowel(r1[i]) && isVowel(r1[i - 1])) {
        r2 = r1.slice(i + 1)
        break
      }
    }
  } else [r1, r2] = regions(w)
  const drop = (k: number): void => {
    w = cut(w, k)
    r1 = cut(r1, k)
    r2 = cut(r2, k)
  }
  /** Replace a suffix in the word and in each region that contains it (else the region becomes `otherwise`). */
  const swap = (suffix: string, by: string, otherwise1 = '', otherwise2 = ''): void => {
    w = replaceSuffix(w, suffix, by)
    r1 = r1.length >= suffix.length ? replaceSuffix(r1, suffix, by) : otherwise1
    r2 = r2.length >= suffix.length ? replaceSuffix(r2, suffix, by) : otherwise2
  }
  for (const suffix of ["'s'", "'s", "'"]) {
    if (w.endsWith(suffix)) {
      drop(suffix.length)
      break
    }
  }
  for (const suffix of STEP1A) {
    if (!w.endsWith(suffix)) continue
    if (suffix === 'sses') drop(2)
    else if (suffix === 'ied' || suffix === 'ies') drop(cut(w, suffix.length).length > 1 ? 2 : 1)
    else if (suffix === 's') {
      if (cut(w, 2).split('').some(isVowel)) drop(1)
    }
    break
  }
  for (const suffix of STEP1B) {
    if (!w.endsWith(suffix)) continue
    if (suffix === 'eed' || suffix === 'eedly') {
      if (r1.endsWith(suffix)) swap(suffix, 'ee')
    } else if (cut(w, suffix.length).split('').some(isVowel)) {
      drop(suffix.length)
      if (w.endsWith('at') || w.endsWith('bl') || w.endsWith('iz')) {
        w += 'e'
        r1 += 'e'
        if (w.length > 5 || r1.length >= 3) r2 += 'e'
      } else if (DOUBLES.some((d) => w.endsWith(d))) drop(1)
      else if (
        (r1 === '' &&
          w.length >= 3 &&
          !isVowel(w[w.length - 1]) &&
          !'wxY'.includes(w[w.length - 1] ?? '') &&
          isVowel(w[w.length - 2]) &&
          !isVowel(w[w.length - 3])) ||
        (r1 === '' && w.length === 2 && isVowel(w[0]) && !isVowel(w[1]))
      ) {
        w += 'e'
        if (r1.length > 0) r1 += 'e'
        if (r2.length > 0) r2 += 'e'
      }
    }
    break
  }
  if (w.length > 2 && 'yY'.includes(w[w.length - 1] ?? '') && !isVowel(w[w.length - 2])) {
    w = `${cut(w, 1)}i`
    r1 = r1.length >= 1 ? `${cut(r1, 1)}i` : ''
    r2 = r2.length >= 1 ? `${cut(r2, 1)}i` : ''
  }
  for (const suffix of STEP2) {
    if (!w.endsWith(suffix)) continue
    if (r1.endsWith(suffix)) {
      if (suffix === 'tional' || suffix === 'entli' || suffix === 'fulli' || suffix === 'lessli') drop(2)
      else if (suffix === 'enci' || suffix === 'anci' || suffix === 'abli') swap('i', 'e')
      else if (suffix === 'izer' || suffix === 'ization') swap(suffix, 'ize')
      else if (suffix === 'ational' || suffix === 'ation' || suffix === 'ator') swap(suffix, 'ate', '', 'e')
      else if (suffix === 'alism' || suffix === 'aliti' || suffix === 'alli') swap(suffix, 'al')
      else if (suffix === 'fulness') drop(4)
      else if (suffix === 'ousli' || suffix === 'ousness') swap(suffix, 'ous')
      else if (suffix === 'iveness' || suffix === 'iviti') swap(suffix, 'ive', '', 'e')
      else if (suffix === 'biliti' || suffix === 'bli') swap(suffix, 'ble')
      else if (suffix === 'ogi' && w[w.length - 4] === 'l') drop(1)
      else if (suffix === 'li' && LI_ENDING.includes(w[w.length - 3] ?? '-')) drop(2)
    }
    break
  }
  for (const suffix of STEP3) {
    if (!w.endsWith(suffix)) continue
    if (r1.endsWith(suffix)) {
      if (suffix === 'tional') drop(2)
      else if (suffix === 'ational') swap(suffix, 'ate')
      else if (suffix === 'alize') drop(3)
      else if (suffix === 'icate' || suffix === 'iciti' || suffix === 'ical') swap(suffix, 'ic')
      else if (suffix === 'ful' || suffix === 'ness') drop(suffix.length)
      else if (suffix === 'ative' && r2.endsWith(suffix)) drop(5)
    }
    break
  }
  for (const suffix of STEP4) {
    if (!w.endsWith(suffix)) continue
    if (r2.endsWith(suffix)) {
      if (suffix === 'ion') {
        if ('st'.includes(w[w.length - 4] ?? '-')) drop(3)
      } else drop(suffix.length)
    }
    break
  }
  if (r2.endsWith('l') && w[w.length - 2] === 'l') w = cut(w, 1)
  else if (r2.endsWith('e')) w = cut(w, 1)
  else if (r1.endsWith('e')) {
    if (
      w.length >= 4 &&
      (isVowel(w[w.length - 2]) ||
        'wxY'.includes(w[w.length - 2] ?? '-') ||
        !isVowel(w[w.length - 3]) ||
        isVowel(w[w.length - 4]))
    )
      w = cut(w, 1)
  }
  return w.replace(/Y/g, 'y')
}
