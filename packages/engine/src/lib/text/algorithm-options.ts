/**
 * The slots of the stepped text algorithms that vary as option tables: how a sentence is parsed (CYK on a grammar in
 * Chomsky normal form, converting it first or not, or Earley on any grammar) and how many trees are enumerated; how
 * strictly Punkt keeps abbreviations (nltk's defaults, strict, loose); and how often BPE encoding drops a merge. Each
 * table is one `union` space for controls and grids.
 */

import {
  bool,
  defaults,
  dict,
  int,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../space/index.ts'
import type { Algorithm } from '../trace/index.ts'
import {
  type CykState,
  cykAlgorithm,
  type EarleyState,
  earleyAlgorithm,
  type Grammar,
  type ParsePath,
  toChomskyNormalForm,
} from './parsing.ts'
import type { PunktTrainingOptions } from './sentences.ts'

function option<T, P>(
  description: string,
  properties: P,
  space: ReturnType<typeof dict>,
  build: (k: (name: string) => number | boolean) => T,
): Option<T, P> {
  return {
    description,
    properties,
    space,
    make: (config) => {
      const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }
      return build((name) => full[name] as number | boolean)
    },
  }
}

/** What a chart parser is built from: a grammar and the words, giving its stepped algorithm. */
export type ParserBuilder = (g: Grammar, tokens: readonly string[]) => Algorithm<CykState | EarleyState, ParsePath>

/** What a parsing strategy accepts, for validity rules and selectors. */
export interface ParserProperties {
  /** Whether it needs a grammar in Chomsky normal form as given. */
  readonly needsCnf: boolean
  /** Whether it handles empty productions. */
  readonly nullable: boolean
}

/**
 * The chart parsers as one slot: `cyk` (a span length per step; converts the grammar to Chomsky normal form first when
 * asked) and `earley` (an item per step; any grammar, nullable symbols too). A table's `make` gives a builder from a
 * grammar and words to the stepped algorithm.
 *
 * @concept parsing
 * @example
 * ```ts
 * import { grammar, parsingStrategies } from 'mdml-engine/text'
 * parsingStrategies.make({ kind: 'earley' })(grammar("S -> 'a'"), ['a']).name // => 'text/earley'
 * ```
 */
export const parsingStrategies: OptionTable<ParserBuilder, ParserProperties> = optionTable(
  {
    cyk: option<ParserBuilder, ParserProperties>(
      'fill the chart one span length at a time; Chomsky normal form',
      { needsCnf: true, nullable: false },
      dict({ convert: bool({ default: true, label: 'convert the grammar to Chomsky normal form' }) }),
      (k) => (g, tokens) =>
        cykAlgorithm(k('convert') === true ? toChomskyNormalForm(g) : g, tokens) as unknown as Algorithm<
          CykState | EarleyState,
          ParsePath
        >,
    ),
    earley: option<ParserBuilder, ParserProperties>(
      'predict, scan and complete one item at a time; any grammar',
      { needsCnf: false, nullable: true },
      dict({}),
      () => (g, tokens) => earleyAlgorithm(g, tokens) as unknown as Algorithm<CykState | EarleyState, ParsePath>,
    ),
  },
  { default: 'earley', label: 'parser', description: 'How a sentence is parsed with a context-free grammar.' },
)

function punkt(
  description: string,
  abbrev: number,
  backoff: number,
): Option<PunktTrainingOptions, Record<string, never>> {
  return option(
    description,
    {},
    dict({
      abbrev: real(0.01, 10, { default: abbrev, log: true, label: 'abbreviation score threshold' }),
      abbrevBackoff: int(1, 50, { default: backoff, label: 'rare abbreviation backoff' }),
      collocation: real(1, 30, { default: 7.88, label: 'collocation log-likelihood threshold' }),
      sentStarter: real(1, 100, { default: 30, label: 'sentence starter log-likelihood threshold' }),
    }),
    (k) => ({
      abbrev: k('abbrev') as number,
      abbrevBackoff: k('abbrevBackoff') as number,
      collocation: k('collocation') as number,
      sentStarter: k('sentStarter') as number,
    }),
  )
}

/**
 * How strictly Punkt training keeps abbreviations, as one slot: `nltk` (nltk's constants), `strict` (a higher score is
 * needed, fewer abbreviations) and `loose` (a lower one, more). Each has the thresholds as knobs.
 *
 * @concept sentence-segmentation
 * @example
 * ```ts
 * import { punktProfiles } from 'mdml-engine/text'
 * punktProfiles.make({ kind: 'strict' }).abbrev // => 1
 * ```
 */
export const punktProfiles: OptionTable<PunktTrainingOptions, Record<string, never>> = optionTable(
  {
    nltk: punkt("nltk's constants", 0.3, 5),
    strict: punkt('a high bar: fewer abbreviations are kept', 1, 3),
    loose: punkt('a low bar: more abbreviations are kept', 0.1, 8),
  },
  {
    default: 'nltk',
    label: 'Punkt training',
    description: 'How readily a type with a period is taken to be an abbreviation.',
  },
)

/**
 * How often BPE encoding skips a merge (BPE-dropout, Provilkov et al. 2020), as one slot: `none` (deterministic), `light`
 * (10%) and `heavy` (50%), the probability a knob. Dropout needs a key.
 *
 * @concept byte-pair-encoding
 * @example
 * ```ts
 * import { bpeDropouts } from 'mdml-engine/text'
 * bpeDropouts.make({ kind: 'light' }).dropout // => 0.1
 * ```
 */
export const bpeDropouts: OptionTable<{ readonly dropout: number }, Record<string, never>> = optionTable(
  {
    none: option('every merge applies', {}, dict({}), () => ({ dropout: 0 })),
    light: option(
      'each candidate merge is skipped with a small probability',
      {},
      dict({ dropout: real(0, 1, { default: 0.1, label: 'dropout' }) }),
      (k) => ({ dropout: k('dropout') as number }),
    ),
    heavy: option(
      'each candidate merge is skipped about half the time',
      {},
      dict({ dropout: real(0, 1, { default: 0.5, label: 'dropout' }) }),
      (k) => ({ dropout: k('dropout') as number }),
    ),
  },
  { default: 'none', label: 'BPE dropout', description: 'The probability that BPE encoding skips a candidate merge.' },
)
