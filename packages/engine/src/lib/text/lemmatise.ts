/**
 * Rule-based lemmatisation: reducing an inflected word to its dictionary form. WordNet's morphy as nltk implements it,
 * applying detachment rules per part of speech and irregular-form exception lists against a lexicon of known lemmas.
 */

import { MdmlError } from '../base/index.ts'

/** A WordNet part of speech: noun, verb, adjective, satellite adjective, adverb. */
export type WordNetPos = 'n' | 'v' | 'a' | 's' | 'r'

/** WordNet's detachment rules (morphy), by part of speech: suffix and replacement, tried in this order. */
export const MORPHOLOGICAL_SUBSTITUTIONS: Readonly<Record<WordNetPos, readonly (readonly [string, string])[]>> = {
  n: [
    ['s', ''],
    ['ses', 's'],
    ['ves', 'f'],
    ['xes', 'x'],
    ['zes', 'z'],
    ['ches', 'ch'],
    ['shes', 'sh'],
    ['men', 'man'],
    ['ies', 'y'],
  ],
  v: [
    ['s', ''],
    ['ies', 'y'],
    ['es', 'e'],
    ['es', ''],
    ['ed', 'e'],
    ['ed', ''],
    ['ing', 'e'],
    ['ing', ''],
  ],
  a: [
    ['er', ''],
    ['est', ''],
    ['er', 'e'],
    ['est', 'e'],
  ],
  s: [
    ['er', ''],
    ['est', ''],
    ['er', 'e'],
    ['est', 'e'],
  ],
  r: [],
}

/**
 * A lexicon for lemmatisation: the parts of speech of each known lemma, and for each part of speech the irregular
 * forms with their lemmas (WordNet's exception lists).
 */
export interface Lexicon {
  readonly lemmas: ReadonlyMap<string, ReadonlySet<WordNetPos>>
  readonly exceptions?: Readonly<Partial<Record<WordNetPos, ReadonlyMap<string, readonly string[]>>>>
}

/**
 * Build a lexicon from lemma lists by part of speech and optional irregular forms.
 *
 * @param lemmas - Known lemmas of each part of speech.
 * @param exceptions - Irregular forms of each part of speech, mapping a form to its lemmas.
 * @returns The lexicon.
 * @concept lemmatisation
 * @example
 * ```ts
 * import { lexicon } from 'mdml-engine/text'
 * lexicon({ n: ['dog'], v: ['dog', 'run'] }).lemmas.get('dog')?.size // => 2
 * ```
 */
export function lexicon(
  lemmas: Readonly<Partial<Record<WordNetPos, readonly string[]>>>,
  exceptions: Readonly<Partial<Record<WordNetPos, Readonly<Record<string, readonly string[]>>>>> = {},
): Lexicon {
  const map = new Map<string, Set<WordNetPos>>()
  for (const [pos, words] of Object.entries(lemmas) as [WordNetPos, readonly string[]][]) {
    for (const w of words) {
      const set = map.get(w) ?? new Set<WordNetPos>()
      set.add(pos)
      map.set(w, set)
    }
  }
  const exc: Partial<Record<WordNetPos, ReadonlyMap<string, readonly string[]>>> = {}
  for (const [pos, table] of Object.entries(exceptions) as [WordNetPos, Readonly<Record<string, readonly string[]>>][])
    exc[pos] = new Map(Object.entries(table))
  return { lemmas: map, exceptions: exc }
}

function checkPos(pos: WordNetPos): void {
  if (!(pos in MORPHOLOGICAL_SUBSTITUTIONS)) throw new MdmlError('value', `lemmatize: unknown part of speech ${pos}`)
}

/**
 * WordNet's morphy as nltk implements it: the irregular forms of the word if it is an exception, otherwise the forms
 * made by one application of each matching detachment rule; then the word itself and those forms that are lemmas of
 * that part of speech in the lexicon, without repeats.
 *
 * @param form - The word form.
 * @param pos - The part of speech.
 * @param lex - The lexicon.
 * @returns The candidate lemmas, in order.
 * @throws MdmlError with code `value` for an unknown part of speech.
 * @concept lemmatisation
 * @example
 * ```ts
 * import { lexicon, morphy } from 'mdml-engine/text'
 * const lex = lexicon({ n: ['church', 'churche', 'axis', 'axe'] }, { n: { axes: ['axis', 'axe'] } })
 * [morphy('churches', 'n', lex), morphy('axes', 'n', lex)] // => [['churche', 'church'], ['axis', 'axe']]
 * ```
 */
export function morphy(form: string, pos: WordNetPos, lex: Lexicon): string[] {
  checkPos(pos)
  const irregular = lex.exceptions?.[pos]?.get(form)
  const forms =
    irregular === undefined
      ? MORPHOLOGICAL_SUBSTITUTIONS[pos]
          .filter(([old]) => form.endsWith(old))
          .map(([old, by]) => form.slice(0, form.length - old.length) + by)
      : [...irregular]
  const out: string[] = []
  for (const f of [form, ...forms]) if (lex.lemmas.get(f)?.has(pos) === true && !out.includes(f)) out.push(f)
  return out
}

/**
 * Rule-based lemmatisation. With a lexicon, nltk's WordNetLemmatizer: the shortest {@link morphy} candidate, or the
 * word itself when there is none. Without one, the detachment rule with the longest matching suffix (the earliest
 * in rule order among equals) applies once, after the built-in irregular forms of common English words; the word is
 * lowercased first in that case.
 *
 * @param word - The word.
 * @param options - The part of speech (default noun) and an optional lexicon.
 * @returns The lemma.
 * @throws MdmlError with code `value` for an unknown part of speech.
 * @concept lemmatisation
 * @example
 * ```ts
 * import { lemmatize } from 'mdml-engine/text'
 * ['churches', 'wolves', 'women', 'ponies', 'mice'].map((w) => lemmatize(w)) // => ['church', 'wolf', 'woman', 'pony', 'mouse']
 * [lemmatize('went', { pos: 'v' }), lemmatize('better', { pos: 'a' })] // => ['go', 'good']
 * ```
 */
export function lemmatize(
  word: string,
  options: { readonly pos?: WordNetPos; readonly lexicon?: Lexicon } = {},
): string {
  const pos = options.pos ?? 'n'
  checkPos(pos)
  if (options.lexicon !== undefined) {
    const lemmas = morphy(word, pos, options.lexicon)
    let best: string | undefined
    for (const l of lemmas) if (best === undefined || l.length < best.length) best = l
    return best ?? word
  }
  const w = word.toLowerCase()
  const irregular = IRREGULAR[pos === 's' ? 'a' : pos]?.[w]
  if (irregular !== undefined) return irregular
  let best: readonly [string, string] | undefined
  for (const rule of MORPHOLOGICAL_SUBSTITUTIONS[pos])
    if (w.endsWith(rule[0]) && w.length > rule[0].length && (best === undefined || rule[0].length > best[0].length))
      best = rule
  if (best === undefined) return w
  return w.slice(0, w.length - best[0].length) + best[1]
}

// Irregular forms of common English words (a small subset of WordNet's exception lists), used without a lexicon.
const IRREGULAR: Readonly<Partial<Record<WordNetPos, Readonly<Record<string, string>>>>> = {
  n: {
    men: 'man',
    women: 'woman',
    children: 'child',
    feet: 'foot',
    teeth: 'tooth',
    geese: 'goose',
    mice: 'mouse',
    lice: 'louse',
    oxen: 'ox',
    people: 'person',
    data: 'datum',
    criteria: 'criterion',
    phenomena: 'phenomenon',
    analyses: 'analysis',
    theses: 'thesis',
    crises: 'crisis',
    axes: 'axis',
    indices: 'index',
    matrices: 'matrix',
    vertices: 'vertex',
    knives: 'knife',
    lives: 'life',
    wives: 'wife',
    leaves: 'leaf',
    halves: 'half',
    selves: 'self',
    is: 'is',
    news: 'news',
    series: 'series',
    species: 'species',
    glass: 'glass',
    class: 'class',
    bus: 'bus',
    gas: 'gas',
  },
  v: {
    am: 'be',
    are: 'be',
    is: 'be',
    was: 'be',
    were: 'be',
    been: 'be',
    being: 'be',
    has: 'have',
    had: 'have',
    having: 'have',
    does: 'do',
    did: 'do',
    done: 'do',
    went: 'go',
    gone: 'go',
    goes: 'go',
    made: 'make',
    said: 'say',
    took: 'take',
    taken: 'take',
    came: 'come',
    saw: 'see',
    seen: 'see',
    knew: 'know',
    known: 'know',
    got: 'get',
    gotten: 'get',
    gave: 'give',
    given: 'give',
    found: 'find',
    thought: 'think',
    told: 'tell',
    became: 'become',
    left: 'leave',
    felt: 'feel',
    brought: 'bring',
    began: 'begin',
    begun: 'begin',
    kept: 'keep',
    held: 'hold',
    wrote: 'write',
    written: 'write',
    stood: 'stand',
    heard: 'hear',
    meant: 'mean',
    met: 'meet',
    ran: 'run',
    running: 'run',
    sat: 'sit',
    spoke: 'speak',
    spoken: 'speak',
    ate: 'eat',
    eaten: 'eat',
    bought: 'buy',
    taught: 'teach',
    caught: 'catch',
    fought: 'fight',
    sold: 'sell',
    built: 'build',
    sent: 'send',
    spent: 'spend',
    lost: 'lose',
    paid: 'pay',
    lying: 'lie',
    dying: 'die',
    tying: 'tie',
  },
  a: {
    better: 'good',
    best: 'good',
    worse: 'bad',
    worst: 'bad',
    more: 'many',
    most: 'many',
    less: 'little',
    least: 'little',
    further: 'far',
    furthest: 'far',
    farther: 'far',
    farthest: 'far',
    elder: 'old',
    eldest: 'old',
  },
  r: { better: 'well', best: 'well', worse: 'badly', worst: 'badly' },
}
