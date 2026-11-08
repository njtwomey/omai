/**
 * Sentence splitting: Punkt, the unsupervised method nltk uses, which learns abbreviations, collocations and sentence
 * starters from raw text and then decides whether each period ends a sentence. Training, sentence spans, and a
 * `splitSentences` front door that also accepts a fixed abbreviation list.
 */

import { OmaiError } from '../base/index.ts'
import { type Algorithm, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { PY_SPACE, UNI_SPACE, UNI_WORD } from './_classes.ts'
import type { Span } from './tokenise.ts'
import type { TokenizerPath } from './tokenizer-contract.ts'

// Punkt (Kiss and Strunk 2006) as nltk implements it: an unsupervised sentence boundary detector that learns
// abbreviations, collocations, sentence starters and orthographic context from raw text.

const BEG_UC = 1 << 1
const MID_UC = 1 << 2
const UNK_UC = 1 << 3
const BEG_LC = 1 << 4
const MID_LC = 1 << 5
const UNK_LC = 1 << 6
const ORTHO_UC = BEG_UC + MID_UC + UNK_UC
const ORTHO_LC = BEG_LC + MID_LC + UNK_LC
const ORTHO_MAP: Readonly<Record<string, number>> = {
  'initial upper': BEG_UC,
  'internal upper': MID_UC,
  'unknown upper': UNK_UC,
  'initial lower': BEG_LC,
  'internal lower': MID_LC,
  'unknown lower': UNK_LC,
}

/** What Punkt learns: abbreviation types, collocations (`first second` pairs), sentence starters, orthographic context. */
export interface PunktParameters {
  readonly abbrevTypes: ReadonlySet<string>
  /** Pairs of types joined by a space. */
  readonly collocations: ReadonlySet<string>
  readonly sentStarters: ReadonlySet<string>
  /** Bit flags of the cases a type was seen in (upper or lower, at sentence starts, inside, or unknown). */
  readonly orthoContext: ReadonlyMap<string, number>
}

const S = UNI_SPACE
const NON_WORD = `(?:[)";}\\]*:@'({\\[\\u2018\\u2019\\u201c\\u201d\\xab\\xbb?!])`
const MULTI_CHAR = `(?:-{2,}|\\.{2,}|(?:\\.[${S}]){2,}\\.)`
const WORD_START = '[^("`{\\[:;&#*@)}\\]\\-,]'
const WORD_RE = new RegExp(
  `(${MULTI_CHAR}|(?=${WORD_START})[^${S}]+?(?=[${S}]|$|${NON_WORD}|${MULTI_CHAR}|,(?=$|[${S}]|${NON_WORD}|${MULTI_CHAR}))|[^${S}])`,
  'gu',
)
const PERIOD_CONTEXT = new RegExp(`[.?!](?=(?<after>${NON_WORD}|[${S}]+(?<next>[^${S}]+)))`, 'gu')
const REALIGN = new RegExp(`^["')\\]}\\u2018\\u2019\\u201c\\u201d\\xab\\xbb]+?(?:[${S}]+|(?=--)|$)`, 'mu')
const ALPHA_CLASS = `(?:(?!\\p{Nd})[${UNI_WORD}])`
const NON_PUNCT = new RegExp(ALPHA_CLASS, 'u')
const RE_NUMERIC = /^-?[.,]?\p{Nd}[\p{Nd},.-]*\.?$/u
const RE_INITIAL = new RegExp(`^${ALPHA_CLASS}\\.$`, 'u')
const RE_ALPHA = new RegExp(`^${ALPHA_CLASS}+$`, 'u')
const RE_ELLIPSIS = /^\.\.+$/
const PY_STRIP = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, 'gu')
const PY_RSTRIP = new RegExp(`[${PY_SPACE}]+$`, 'u')
const UPPER = /^\p{Uppercase}/u
const LOWER = /^\p{Lowercase}/u

/** A token of Punkt with its annotations. */
interface PToken {
  readonly tok: string
  readonly type: string
  readonly periodFinal: boolean
  parastart: boolean
  linestart: boolean
  sentbreak: boolean
  abbr: boolean
  ellipsis: boolean
}

function token(tok: string, parastart = false, linestart = false): PToken {
  return {
    tok,
    type: tok.toLowerCase().replace(RE_NUMERIC, '##number##'),
    periodFinal: tok.endsWith('.'),
    parastart,
    linestart,
    sentbreak: false,
    abbr: false,
    ellipsis: false,
  }
}

const typeNoPeriod = (t: PToken): string => (t.type.length > 1 && t.type.endsWith('.') ? t.type.slice(0, -1) : t.type)
const typeNoSentPeriod = (t: PToken): string => (t.sentbreak ? typeNoPeriod(t) : t.type)
const firstCase = (t: PToken): 'lower' | 'upper' | 'none' =>
  LOWER.test(t.tok) ? 'lower' : UPPER.test(t.tok) ? 'upper' : 'none'
const isNumber = (t: PToken): boolean => t.type.startsWith('##number##')
const isInitial = (t: PToken): boolean => RE_INITIAL.test(t.tok)

/** Punkt's word tokens of a text, line by line, marking line and paragraph starts. */
function tokenizeWords(text: string): PToken[] {
  const out: PToken[] = []
  let parastart = false
  for (const line of text.split('\n')) {
    if (line.replace(PY_STRIP, '').length === 0) {
      parastart = true
      continue
    }
    const toks = Array.from(line.matchAll(WORD_RE), (m) => m[0])
    if (toks.length === 0) continue
    out.push(token(toks[0] ?? '', parastart, true))
    parastart = false
    for (const t of toks.slice(1)) out.push(token(t))
  }
  return out
}

function firstPass(tokens: PToken[], params: PunktParameters): void {
  for (const t of tokens) {
    if (t.tok === '.' || t.tok === '?' || t.tok === '!') t.sentbreak = true
    else if (RE_ELLIPSIS.test(t.tok)) t.ellipsis = true
    else if (t.periodFinal && !t.tok.endsWith('..')) {
      const stem = t.tok.slice(0, -1).toLowerCase()
      if (params.abbrevTypes.has(stem) || params.abbrevTypes.has(stem.split('-').at(-1) ?? '')) t.abbr = true
      else t.sentbreak = true
    }
  }
}

function orthoHeuristic(t: PToken, params: PunktParameters): boolean | 'unknown' {
  if (';:,.!?'.includes(t.tok) && t.tok.length === 1) return false
  const ctx = params.orthoContext.get(typeNoSentPeriod(t)) ?? 0
  if (UPPER.test(t.tok) && ctx & ORTHO_LC && !(ctx & MID_UC)) return true
  if (LOWER.test(t.tok) && (ctx & ORTHO_UC || !(ctx & BEG_LC))) return false
  return 'unknown'
}

function secondPass(tokens: PToken[], params: PunktParameters): void {
  for (let i = 0; i + 1 < tokens.length; i++) {
    const a = tokens[i] as PToken
    const b = tokens[i + 1] as PToken
    if (!a.periodFinal) continue
    const typ = typeNoPeriod(a)
    const next = typeNoSentPeriod(b)
    const initial = isInitial(a)
    if (params.collocations.has(`${typ} ${next}`)) {
      a.sentbreak = false
      a.abbr = true
      continue
    }
    if ((a.abbr || a.ellipsis) && !initial) {
      if (orthoHeuristic(b, params) === true) {
        a.sentbreak = true
        continue
      }
      if (UPPER.test(b.tok) && params.sentStarters.has(next)) {
        a.sentbreak = true
        continue
      }
    }
    if (initial || typ === '##number##') {
      const starter = orthoHeuristic(b, params)
      if (starter === false) {
        a.sentbreak = false
        a.abbr = true
        continue
      }
      if (starter === 'unknown' && initial && UPPER.test(b.tok) && !((params.orthoContext.get(next) ?? 0) & ORTHO_LC)) {
        a.sentbreak = false
        a.abbr = true
      }
    }
  }
}

/** Options of {@link trainPunkt}: nltk's PunktTrainer constants. */
export interface PunktTrainingOptions {
  /** Score threshold for abbreviations (default 0.3). */
  readonly abbrev?: number
  /** Ignore the length penalty of abbreviations (default false). */
  readonly ignoreAbbrevPenalty?: boolean
  /** Types seen this often are not rare abbreviations (default 5). */
  readonly abbrevBackoff?: number
  /** Log-likelihood threshold for collocations (default 7.88). */
  readonly collocation?: number
  /** Log-likelihood threshold for sentence starters (default 30). */
  readonly sentStarter?: number
  /** Consider every pair as a collocation candidate (default false). */
  readonly includeAllCollocs?: boolean
  /** Consider pairs after abbreviations as collocation candidates (default false). */
  readonly includeAbbrevCollocs?: boolean
  /** The least collocation count (default 1). */
  readonly minCollocFreq?: number
}

/** Dunning's log-likelihood as Punkt scores abbreviations (with its fixed alternative p2 = 0.99). */
function dunning(countA: number, countB: number, countAB: number, n: number): number {
  const p1 = countB / n
  const p2 = 0.99
  const nullHypo = countAB * Math.log(p1 + 1e-8) + (countA - countAB) * Math.log(1 - p1 + 1e-8)
  const altHypo = countAB * Math.log(p2) + (countA - countAB) * Math.log(1 - p2)
  return -2 * (nullHypo - altHypo)
}

/** Python's math.log, which raises (here: undefined) at zero and below. */
const pyLog = (x: number): number | undefined => (x > 0 ? Math.log(x) : undefined)

/** The collocation log-likelihood of Punkt (Dunning's ratio for two binomials). */
function colLogLikelihood(countA: number, countB: number, countAB: number, n: number): number {
  const p = countB / n
  const p1 = countAB / countA
  const p2 = n - countA === 0 ? 1 : (countB - countAB) / (n - countA)
  const lp = pyLog(p)
  const lq = pyLog(1 - p)
  // Python raises on log(0) only when it is evaluated; 0 * log(...) of a valid log is still a number.
  const s1 = lp === undefined || lq === undefined ? 0 : countAB * lp + (countA - countAB) * lq
  const s2 = lp === undefined || lq === undefined ? 0 : (countB - countAB) * lp + (n - countA - countB + countAB) * lq
  const s3 =
    countA === countAB || p1 <= 0 || p1 >= 1 ? 0 : countAB * Math.log(p1) + (countA - countAB) * Math.log(1 - p1)
  const s4 =
    countB === countAB || p2 <= 0 || p2 >= 1
      ? 0
      : (countB - countAB) * Math.log(p2) + (n - countA - countB + countAB) * Math.log(1 - p2)
  return -2 * (s1 + s2 - s3 - s4)
}

/**
 * The state of {@link trainPunktAlgorithm}: the abbreviation candidates scored so far (the types with the abbreviation
 * decision made for them), then the parameters once the later passes have run.
 */
export interface PunktTrainingState {
  /** The next candidate type to score. */
  next: number
  /** The abbreviation types so far, as a record of types. */
  readonly abbrevTypes: Record<string, number>
  /** The learned parameters, once the passes after scoring have run (as sorted lists and a record). */
  result: {
    readonly abbrevTypes: string[]
    readonly collocations: string[]
    readonly sentStarters: string[]
    readonly orthoContext: Record<string, number>
  } | null
}

/**
 * Punkt training (nltk's PunktTrainer) as a stepped algorithm: each of the first steps scores one abbreviation candidate
 * (a type with a final period, or a current abbreviation without one) by Dunning's log-likelihood with the length and
 * period factors and keeps or drops it, logging the score; a last step annotates a first pass, gathers the orthographic
 * context, finds rare abbreviations and keeps the sentence starters and collocations whose log-likelihoods pass their
 * thresholds. It is done (`converged`) after that step. The state serialises; read the parameters with
 * {@link punktParametersOf}.
 *
 * @param text - Training text (plain, with newlines; blank lines mark paragraphs).
 * @param options - nltk's training constants.
 * @returns The algorithm.
 * @concept sentence-segmentation
 * @example
 * ```ts
 * import { punktParametersOf, trainPunktAlgorithm } from 'omai-engine/text'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const text = 'Mr. Smith arrived. He saw Mr. Jones. Mr. Brown left. Then Mr. Smith left too. '.repeat(3)
 * const run = runAlgorithm(trainPunktAlgorithm(text), { maxSteps: 100 })
 * punktParametersOf(run.state).abbrevTypes.has('mr') // => true
 * ```
 */
export function trainPunktAlgorithm(
  text: string,
  options: PunktTrainingOptions = {},
): Algorithm<PunktTrainingState, TokenizerPath> {
  const ABBREV = options.abbrev ?? 0.3
  const BACKOFF = options.abbrevBackoff ?? 5
  const COLLOCATION = options.collocation ?? 7.88
  const SENT_STARTER = options.sentStarter ?? 30
  const MIN_COLLOC = options.minCollocFreq ?? 1
  const typeCounts = new Map<string, number>()
  const count = (t: string): number => typeCounts.get(t) ?? 0
  const tokens = tokenizeWords(text)
  let periodTokens = 0
  for (const t of tokens) {
    typeCounts.set(t.type, count(t.type) + 1)
    if (t.periodFinal) periodTokens++
  }
  let total = 0
  for (const c of typeCounts.values()) total += c
  // Abbreviation candidates (types with a final period, and current abbreviations without one).
  const candidates = [...new Set(tokens.map((t) => t.type))].filter((o) => NON_PUNCT.test(o) && o !== '##number##')
  return {
    name: 'text/punkt-training',
    config: { candidates: candidates.length, abbrev: ABBREV },
    init: () => ({ next: 0, abbrevTypes: {}, result: null }),
    step: (st, ctx) => {
      if (st.next < candidates.length) {
        const original = candidates[st.next] as string
        st.next += 1
        let typ = original
        let isAdd: boolean
        if (typ.endsWith('.')) {
          if (st.abbrevTypes[typ] !== undefined) return st
          typ = typ.slice(0, -1)
          isAdd = true
        } else {
          if (st.abbrevTypes[typ] === undefined) return st
          isAdd = false
        }
        const periods = typ.split('.').length
        const nonPeriods = Array.from(typ).length - periods + 1
        const withPeriod = count(`${typ}.`)
        const without = count(typ)
        const ll = dunning(withPeriod + without, periodTokens, withPeriod, total)
        const penalty = options.ignoreAbbrevPenalty === true ? 1 : nonPeriods ** -without
        const score = ll * Math.exp(-nonPeriods) * periods * penalty
        if (score >= ABBREV) {
          if (isAdd) st.abbrevTypes[typ] = 1
        } else if (!isAdd) delete st.abbrevTypes[typ]
        ctx.log('punkt/score', score)
        ctx.log('punkt/abbrev', st.abbrevTypes[typ] === undefined ? 0 : 1)
        return st
      }
      const abbrevTypes = new Set(Object.keys(st.abbrevTypes))
      const orthoContext = new Map<string, number>()
      const params: PunktParameters = { abbrevTypes, collocations: new Set(), sentStarters: new Set(), orthoContext }
      firstPass(tokens, params)
      // Orthographic context.
      let context: 'internal' | 'initial' | 'unknown' = 'internal'
      for (const t of tokens) {
        if (t.parastart && context !== 'unknown') context = 'initial'
        if (t.linestart && context === 'internal') context = 'unknown'
        const flag = ORTHO_MAP[`${context} ${firstCase(t)}`] ?? 0
        if (flag !== 0) {
          const typ = typeNoSentPeriod(t)
          orthoContext.set(typ, (orthoContext.get(typ) ?? 0) | flag)
        }
        if (t.sentbreak) context = isNumber(t) || isInitial(t) ? 'unknown' : 'initial'
        else context = t.ellipsis || t.abbr ? 'unknown' : 'internal'
      }
      let sentbreaks = 0
      for (const t of tokens) if (t.sentbreak) sentbreaks++
      const starterCounts = new Map<string, number>()
      const collocCounts = new Map<string, number>()
      for (let i = 0; i + 1 < tokens.length; i++) {
        const a = tokens[i] as PToken
        const b = tokens[i + 1] as PToken
        if (!a.periodFinal) continue
        // Rare abbreviations.
        if (!a.abbr && a.sentbreak) {
          const typ = typeNoSentPeriod(a)
          const c = count(typ) + count(typ.slice(0, -1))
          if (!abbrevTypes.has(typ) && c < BACKOFF) {
            if (',:;'.includes(b.tok.slice(0, 1)) && b.tok.length > 0) abbrevTypes.add(typeNoPeriod(a))
            else if (LOWER.test(b.tok)) {
              const ctxFlags = orthoContext.get(typeNoSentPeriod(b)) ?? 0
              if (ctxFlags & BEG_UC && !(ctxFlags & MID_UC)) abbrevTypes.add(typeNoPeriod(a))
            }
          }
        }
        if (a.sentbreak && !(isNumber(a) || isInitial(a)) && RE_ALPHA.test(b.tok))
          starterCounts.set(b.type, (starterCounts.get(b.type) ?? 0) + 1)
        const potential =
          (options.includeAllCollocs === true ||
            (options.includeAbbrevCollocs === true && a.abbr) ||
            (a.sentbreak && (isNumber(a) || isInitial(a)))) &&
          NON_PUNCT.test(a.type) &&
          NON_PUNCT.test(b.type)
        if (potential) {
          const key = `${typeNoPeriod(a)} ${typeNoSentPeriod(b)}`
          collocCounts.set(key, (collocCounts.get(key) ?? 0) + 1)
        }
      }
      const sentStarters = new Set<string>()
      for (const [typ, atBreak] of starterCounts) {
        if (typ.length === 0) continue
        const c = count(typ) + count(`${typ}.`)
        if (c < atBreak) continue
        const ll = colLogLikelihood(sentbreaks, c, atBreak, total)
        if (ll >= SENT_STARTER && total / sentbreaks > c / atBreak) sentStarters.add(typ)
      }
      const collocations = new Set<string>()
      for (const [key, c] of collocCounts) {
        const sp = key.indexOf(' ')
        const t1 = key.slice(0, sp)
        const t2 = key.slice(sp + 1)
        if (sentStarters.has(t2)) continue
        const c1 = count(t1) + count(`${t1}.`)
        const c2 = count(t2) + count(`${t2}.`)
        if (c1 > 1 && c2 > 1 && MIN_COLLOC < c && c <= Math.min(c1, c2)) {
          const ll = colLogLikelihood(c1, c2, c, total)
          if (ll >= COLLOCATION && total / c1 > c2 / c) collocations.add(key)
        }
      }
      st.result = {
        abbrevTypes: [...abbrevTypes],
        collocations: [...collocations],
        sentStarters: [...sentStarters],
        orthoContext: Object.fromEntries(orthoContext),
      }
      return st
    },
    done: (st) => (st.result === null ? undefined : 'converged'),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as PunktTrainingState,
  }
}

/**
 * The parameters of a finished Punkt training state.
 *
 * @param state - A finished state of {@link trainPunktAlgorithm}.
 * @returns The learned parameters.
 * @throws OmaiError with code `value` when training is not finished.
 * @concept sentence-segmentation
 * @example
 * ```ts
 * import { punktParametersOf, trainPunktAlgorithm } from 'omai-engine/text'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(trainPunktAlgorithm('Dr. Who came. He left. '.repeat(4)), { maxSteps: 100 })
 * punktParametersOf(run.state).sentStarters instanceof Set // => true
 * ```
 */
export function punktParametersOf(state: PunktTrainingState): PunktParameters {
  if (state.result === null) throw new OmaiError('value', 'punktParametersOf: training has not finished')
  return {
    abbrevTypes: new Set(state.result.abbrevTypes),
    collocations: new Set(state.result.collocations),
    sentStarters: new Set(state.result.sentStarters),
    orthoContext: new Map(Object.entries(state.result.orthoContext)),
  }
}

/**
 * Train Punkt on raw text (nltk's PunktTrainer): count types, score abbreviation candidates by Dunning's
 * log-likelihood with length and period factors, annotate a first pass, gather orthographic context, find rare
 * abbreviations, and keep sentence starters and collocations whose log-likelihoods pass their thresholds.
 * {@link trainPunktAlgorithm} run to the end.
 *
 * @param text - Training text (plain, with newlines; blank lines mark paragraphs).
 * @param options - nltk's training constants.
 * @returns The learned parameters.
 * @concept sentence-segmentation
 * @example
 * ```ts
 * import { trainPunkt } from 'omai-engine/text'
 * const text = 'Mr. Smith arrived. He saw Mr. Jones. Mr. Brown left. Then Mr. Smith left too. '.repeat(3)
 * trainPunkt(text).abbrevTypes.has('mr') // => true
 * ```
 */
export function trainPunkt(text: string, options: PunktTrainingOptions = {}): PunktParameters {
  return punktParametersOf(
    runAlgorithm(trainPunktAlgorithm(text, options), { maxSteps: Number.MAX_SAFE_INTEGER }).state,
  )
}

/** Punkt parameters with only an abbreviation list (a rule-based splitter), or nothing learned. */
function plainParameters(abbreviations: readonly string[] = []): PunktParameters {
  return {
    abbrevTypes: new Set(abbreviations.map((a) => a.toLowerCase().replace(/\.$/, ''))),
    collocations: new Set(),
    sentStarters: new Set(),
    orthoContext: new Map(),
  }
}

const ASCII_SPACE = new Set([' ', '\t', '\n', '\r', '\x0b', '\x0c'])

/**
 * Sentence spans as nltk's PunktSentenceTokenizer finds them: candidate boundaries at `.`, `?`, `!` followed by
 * punctuation or whitespace and a token; each context is annotated with the parameters (abbreviations, collocations,
 * sentence starters, orthographic heuristics), and closing quotes and brackets are moved to the sentence they end.
 *
 * @param text - The text.
 * @param params - Trained parameters ({@link trainPunkt}); default none, which splits at every sentence-ending period.
 * @param options - Whether to realign boundaries past closing punctuation (default true).
 * @returns The sentence spans.
 * @concept sentence-segmentation
 * @example
 * ```ts
 * import { sentenceSpans } from 'omai-engine/text'
 * const s = 'Hello there. "How are you?" she asked.'
 * sentenceSpans(s).map(([a, b]) => s.slice(a, b)) // => ['Hello there.', '"How are you?"', 'she asked.']
 * ```
 */
export function sentenceSpans(
  text: string,
  params: PunktParameters = plainParameters(),
  options: { readonly realignBoundaries?: boolean } = {},
): Span[] {
  const containsBreak = (context: string): boolean => {
    const toks = tokenizeWords(context)
    firstPass(toks, params)
    secondPass(toks, params)
    let found = false
    for (const t of toks) {
      if (found) return true
      if (t.sentbreak) found = true
    }
    return false
  }
  // Candidate contexts: the word before each candidate, the end character and what follows.
  const slices: [number, number][] = []
  let lastBreak = 0
  let prevSlice: [number, number] = [0, 0]
  let prevMatch: RegExpExecArray | undefined
  const decide = (m: RegExpExecArray, slice: [number, number]): void => {
    const after = m.groups?.['after'] ?? ''
    const context = text.slice(slice[0], slice[1]) + m[0] + after
    if (containsBreak(context)) {
      const end = m.index + m[0].length
      slices.push([lastBreak, end])
      const next = m.groups?.['next']
      lastBreak = next === undefined ? end : end + after.length - next.length
    }
  }
  for (const m of text.matchAll(PERIOD_CONTEXT)) {
    const before = text.slice(prevSlice[1], m.index)
    let afterSpace = 0
    for (let i = before.length - 1; i >= 0; i--) {
      if (ASCII_SPACE.has(before[i] ?? '')) {
        afterSpace = i
        break
      }
    }
    const start = afterSpace > 0 ? afterSpace + prevSlice[1] + 1 : prevSlice[0]
    const wordSlice: [number, number] = [start, m.index]
    if (prevMatch !== undefined && prevSlice[1] <= wordSlice[0]) decide(prevMatch, prevSlice)
    prevMatch = m as RegExpExecArray
    prevSlice = wordSlice
  }
  if (prevMatch !== undefined) decide(prevMatch, prevSlice)
  slices.push([lastBreak, text.replace(PY_RSTRIP, '').length])
  if (options.realignBoundaries === false) return slices
  const out: Span[] = []
  let realign = 0
  for (let i = 0; i < slices.length; i++) {
    const s1: [number, number] = [(slices[i]?.[0] ?? 0) + realign, slices[i]?.[1] ?? 0]
    const s2 = slices[i + 1]
    if (s2 === undefined) {
      if (s1[1] > s1[0]) out.push(s1)
      continue
    }
    const m = REALIGN.exec(text.slice(s2[0], s2[1]))
    if (m !== null) {
      out.push([s1[0], s2[0] + m[0].replace(PY_RSTRIP, '').length])
      realign = m[0].length
    } else {
      realign = 0
      if (s1[1] > s1[0]) out.push(s1)
    }
  }
  return out
}

/**
 * Split text into sentences: Punkt with trained parameters, with a list of abbreviations (a rule-based splitter that
 * never breaks after them), or with nothing learned.
 *
 * @param text - The text.
 * @param options - Trained parameters, or abbreviations (without or with their final period).
 * @returns The sentences.
 * @concept sentence-segmentation
 * @example
 * ```ts
 * import { splitSentences } from 'omai-engine/text'
 * splitSentences('Dr. Who arrived at 10 a.m. sharp. Then he left!', { abbreviations: ['dr', 'a.m'] }) // => ['Dr. Who arrived at 10 a.m. sharp.', 'Then he left!']
 * ```
 */
export function splitSentences(
  text: string,
  options: { readonly params?: PunktParameters; readonly abbreviations?: readonly string[] } = {},
): string[] {
  const params = options.params ?? plainParameters(options.abbreviations)
  return sentenceSpans(text, params).map(([a, b]) => text.slice(a, b))
}
