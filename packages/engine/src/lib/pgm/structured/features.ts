/**
 * Feature extractors for linear-chain CRFs: the route from raw sequences to the attribute records a model scores. An
 * extractor is a template language plus its options (CRF++ templates, Wapiti patterns, CRFsuite attribute templates, a
 * function of your own, a hashed or prefixed version of any of them, or several combined) that turns the input of one
 * sequence into a `CrfInput`; `crfModelFor` builds the model with its attribute vocabulary and minimum-frequency cut-off
 * from training inputs, and `explain` says which template produced which feature, so a fitted model can be read back
 * as templates (`crfTopFeatures`, `crfTemplateWeights`).
 *
 * The extractor is generic in its input `I`. Sequences of tokens are the built-in case (`TokenTable`, one row per
 * token and one column per attribute); an input that is not a sequence (the nodes and edges of a graph, the candidate
 * segments of a semi-Markov model) plugs in the same way: write one function from the input to the attribute records of
 * its items and of its edges (`functionExtractor` does it for sequences, `featureRecord` turns feature strings into a
 * record, `featureIndex` and `FeatureValues` do the vocabulary), because every structured model of this folder scores
 * the same `ItemFeatures` records. Extraction happens once, before training, and the records are then reused by every
 * evaluation of the objective.
 */

import { OmaiError } from '../../base/index.ts'
import { type CoefficientTable, coefficientTable } from '../../exposition/index.ts'
import {
  type AttributeTemplate,
  type CrfppTemplate,
  crfppFeatures,
  expandAttributeTemplates,
  type FeatureIndex,
  featureIndex,
  featureRecord,
  fireFeatures,
  hashedColumn,
  parseAttributeTemplates,
  parseCrfppTemplate,
  parseWapitiPatterns,
  type Token,
  type TokenTable,
  type WapitiLine,
  wapitiFeatures,
} from '../../text/index.ts'
import {
  type CrfInput,
  type CrfObservation,
  crfBigrams,
  crfUnigrams,
  type ItemFeatures,
  type LabelledSequence,
  type LinearChainCrf,
  linearChainCrf,
} from './crf.ts'

/** The features of one item or edge: feature strings (each firing counts once) or an attribute record with values. */
export type FeatureValues = readonly string[] | Readonly<Record<string, number>>

/** One feature fired at a position, with the template that produced it. */
export interface FeatureOrigin {
  /** Whether the feature is an item (`unigram`) or an edge (`bigram`) feature. */
  readonly kind: 'unigram' | 'bigram'
  /** The extractor (or the part of a combination) that produced it. */
  readonly source: string
  /** The template: a CRF++ line, a Wapiti pattern line, an attribute template such as `w[-1]|w[0]`, or a name. */
  readonly template: string
  readonly feature: string
  readonly value: number
}

/**
 * A feature extractor: the input of one sequence (or any structured input) to the observation a CRF scores, and the
 * account of where each feature came from.
 *
 * @concept crf-features
 * @index structure A template language and its options as a map from an input to CRF attribute records.
 */
export interface FeatureExtractor<I> {
  readonly name: string
  /**
   * Whether the model needs plain label-pair transition weights beside the extracted features: CRFsuite's model always
   * has them, while CRF++ and Wapiti get theirs from a bigram line (a bare `B`) that fires on every edge.
   */
  readonly labelTransitions: boolean
  /** The item attributes (and edge attributes, where the language has them) of an input. */
  extract(input: I): CrfInput
  /** The features fired at a position (the edge into it, too, from the second position), each with its template. */
  explain(input: I, position: number): readonly FeatureOrigin[]
}

/** An input of the template extractors: a token-by-attribute table, as the columns of a CRF++ or Wapiti file. */
export type TemplateInput = TokenTable

const record = (values: FeatureValues): Record<string, number> =>
  Array.isArray(values) ? featureRecord(values as readonly string[]) : { ...(values as Record<string, number>) }

function crfppSource(template: string | CrfppTemplate): CrfppTemplate {
  return typeof template === 'string' ? parseCrfppTemplate(template) : template
}

/**
 * CRF++ templates (`text.parseCrfppTemplate`) as an extractor: unigram lines at every position and bigram lines at every
 * edge, as attributes counting their firings. Boundary values are CRF++'s `_B-k` and `_B+k`; label-pair transitions
 * come from the bigram lines, as CRF++ has them.
 *
 * @param template - The template text or its parse.
 * @returns The extractor.
 * @throws OmaiError with code `value` for a malformed template.
 * @concept crf-features
 * @index function CRF++ templates as a feature extractor over token tables.
 * @example
 * ```ts
 * import { crfppExtractor } from 'omai-engine/pgm'
 * crfppExtractor('U00:%x[0,0]\nU01:%x[-1,0]/%x[0,0]\nB').extract([['a'], ['b']]) // => { unigram: [{ 'U00:a': 1, 'U01:_B-1/a': 1 }, { 'U00:b': 1, 'U01:a/b': 1 }], bigram: [{}, { B: 1 }] }
 * ```
 */
export function crfppExtractor(template: string | CrfppTemplate): FeatureExtractor<TemplateInput> {
  const parsed = crfppSource(template)
  return {
    name: 'crfpp',
    labelTransitions: false,
    extract: (table) => {
      const f = crfppFeatures(parsed, table)
      return { unigram: f.unigram.map(featureRecord), bigram: f.bigram.map(featureRecord) }
    },
    explain: (table, position) =>
      fireFeatures(parsed, table, position).map((f) => ({
        kind: f.kind === 'U' ? 'unigram' : 'bigram',
        source: 'crfpp',
        template: f.template,
        feature: f.feature,
        value: 1,
      })),
  }
}

function wapitiSource(patterns: string | readonly WapitiLine[]): readonly WapitiLine[] {
  return typeof patterns === 'string' ? parseWapitiPatterns(patterns) : patterns
}

/**
 * The observation of a sequence under Wapiti patterns (`text.parseWapitiPatterns`): the expanded unigram features of
 * each item and bigram features of each edge, as attributes that count their firings (the Wapiti counterpart of
 * `crfppObservation`).
 *
 * @param patterns - The pattern text or its parse.
 * @param table - The token-by-attribute table of the sequence.
 * @returns The observation.
 * @throws OmaiError with code `value` for a malformed pattern or a column outside the table.
 * @concept crf-features
 * @example
 * ```ts
 * import { wapitiObservation } from 'omai-engine/pgm'
 * wapitiObservation('u:w=%x[0,0]\n*:cap=%t[0,0,"^\\u"]', [['a'], ['B']]) // => { unigram: [{ 'u:w=a': 1, '*:cap=false': 1 }, { 'u:w=B': 1, '*:cap=true': 1 }], bigram: [{}, { '*:cap=true': 1 }] }
 * ```
 */
export function wapitiObservation(patterns: string | readonly WapitiLine[], table: TokenTable): CrfObservation {
  const f = wapitiFeatures(wapitiSource(patterns), table)
  return { unigram: f.unigram.map(featureRecord), bigram: f.bigram.map(featureRecord) }
}

/**
 * Wapiti patterns as an extractor: `u` lines at every position, `b` lines at every edge, `*` lines at both; `%t` and
 * `%m` give regular-expression tests and matches. Boundary values are Wapiti's `_x-1` ... `_x+4`; label-pair
 * transitions come from the bigram lines, as in Wapiti.
 *
 * @param patterns - The pattern text or its parse.
 * @returns The extractor.
 * @throws OmaiError with code `value` for a malformed pattern.
 * @concept crf-features
 * @index function Wapiti patterns (with regular-expression tests) as a feature extractor.
 * @example
 * ```ts
 * import { wapitiExtractor } from 'omai-engine/pgm'
 * wapitiExtractor('u:w=%x[0,0]\nb').extract([['a'], ['b']]) // => { unigram: [{ 'u:w=a': 1 }, { 'u:w=b': 1 }], bigram: [{}, { b: 1 }] }
 * ```
 */
export function wapitiExtractor(patterns: string | readonly WapitiLine[]): FeatureExtractor<TemplateInput> {
  const lines = wapitiSource(patterns)
  return {
    name: 'wapiti',
    labelTransitions: false,
    extract: (table) => wapitiObservation(lines, table),
    explain: (table, position) => {
      const f = wapitiFeatures(lines, table)
      const origins: FeatureOrigin[] = []
      const unigramLines = lines.filter((l) => l.kind !== 'b')
      for (const [k, l] of unigramLines.entries())
        origins.push({
          kind: 'unigram',
          source: 'wapiti',
          template: l.source,
          feature: f.unigram[position]?.[k] ?? '',
          value: 1,
        })
      if (position > 0)
        for (const [k, l] of lines.filter((l) => l.kind !== 'u').entries())
          origins.push({
            kind: 'bigram',
            source: 'wapiti',
            template: l.source,
            feature: f.bigram[position]?.[k] ?? '',
            value: 1,
          })
      return origins
    },
  }
}

/** The input of an attribute-template extractor: tokens as records by column name, or rows of a token table. */
export type AttributeInput = readonly Token[] | TokenTable

/** Options of {@link attributeExtractor}. */
export interface AttributeExtractorOptions {
  /** The names of the columns of a token table (`w`, `pos`, ...); not needed when tokens are records. */
  readonly columns?: readonly string[]
  /** Cells outside the sequence: `skip` the template there (CRFsuite, default) or `pad` with `_B-k` / `_B+k` (CRF++). */
  readonly boundary?: 'skip' | 'pad'
  /** Add a `bias` attribute at every position. */
  readonly bias?: boolean
  /** Add `__BOS__` at the first and `__EOS__` at the last position (CRFsuite's crfutils). */
  readonly bosEos?: boolean
}

function tokensOf(input: AttributeInput, columns: readonly string[] | undefined): readonly Token[] {
  return input.map((row) => {
    if (!Array.isArray(row)) return row as Token
    if (columns === undefined)
      throw new OmaiError('value', 'attributeExtractor: rows of a token table need the names of their columns')
    return Object.fromEntries(columns.map((c, i) => [c, (row as readonly string[])[i] ?? '']))
  })
}

/**
 * CRFsuite-style attribute templates (`w[-1]|w[0]`, `text.expandAttributeTemplates`) as an extractor, with the bias
 * attribute, sentence markers (`__BOS__`, `__EOS__`) and boundary handling of crfutils. The model it asks for has
 * label-pair transition weights, as CRFsuite's.
 *
 * @param templates - The templates as text (one per line or `;`) or parsed.
 * @param options - Column names for table rows, boundary handling, bias and sentence markers.
 * @returns The extractor over tokens (records by column name) or token tables.
 * @throws OmaiError with code `value` for a malformed template, or a table input without column names.
 * @concept crf-features
 * @index function CRFsuite attribute templates (bias, BOS/EOS, boundary options) as a feature extractor.
 * @example
 * ```ts
 * import { attributeExtractor } from 'omai-engine/pgm'
 * const e = attributeExtractor('w[0]\nw[-1]|w[0]', { columns: ['w'], bias: true })
 * e.extract([['the'], ['cat']]) // => [{ bias: 1, 'w[0]=the': 1 }, { bias: 1, 'w[0]=cat': 1, 'w[-1]|w[0]=the|cat': 1 }]
 * ```
 */
export function attributeExtractor(
  templates: string | readonly AttributeTemplate[],
  options: AttributeExtractorOptions = {},
): FeatureExtractor<AttributeInput> {
  const parsed = typeof templates === 'string' ? parseAttributeTemplates(templates) : templates
  const expansion = {
    ...(options.boundary === undefined ? {} : { boundary: options.boundary }),
    ...(options.bias === undefined ? {} : { bias: options.bias }),
    ...(options.bosEos === undefined ? {} : { bosEos: options.bosEos }),
  }
  return {
    name: 'attributes',
    labelTransitions: true,
    extract: (input) => expandAttributeTemplates(tokensOf(input, options.columns), parsed, expansion),
    explain: (input, position) => {
      const tokens = tokensOf(input, options.columns)
      const origins: FeatureOrigin[] = []
      const add = (template: string, items: Record<string, number>[]): void => {
        for (const [feature, value] of Object.entries(items[position] ?? {}))
          origins.push({ kind: 'unigram', source: 'attributes', template, feature, value })
      }
      for (const t of parsed)
        add(
          t.map(([c, o]) => `${c}[${o}]`).join('|'),
          expandAttributeTemplates(tokens, [t], options.boundary === undefined ? {} : { boundary: options.boundary }),
        )
      if (options.bias === true) add('bias', expandAttributeTemplates(tokens, [], { bias: true }))
      if (options.bosEos === true) add('bosEos', expandAttributeTemplates(tokens, [], { bosEos: true }))
      return origins
    },
  }
}

/** What a function extractor computes: the features of the items and, optionally, of the edges of an input. */
export interface FunctionExtractorSpec<I> {
  /** The number of items of an input. */
  readonly length: (input: I) => number
  /** The features of item `t`. */
  readonly unigram: (input: I, t: number) => FeatureValues
  /** The features of the edge into item `t` (called for `t >= 1`): observation-dependent transitions. */
  readonly bigram?: (input: I, t: number) => FeatureValues
  /** Whether the model gets plain label-pair transition weights (default true). */
  readonly labelTransitions?: boolean
}

/**
 * An extractor from your own functions: one for the features of an item, optionally one for those of an edge. The
 * input can be anything with items (a sentence, a signal, the letters of a word), and the same shape of function is
 * what a non-sequence extractor writes for the nodes and edges of its structure.
 *
 * @param name - The extractor's name, shown as the template of its features.
 * @param spec - The functions.
 * @returns The extractor.
 * @concept crf-features
 * @index function An extractor from your own per-item (and per-edge) feature functions.
 * @example
 * ```ts
 * import { functionExtractor } from 'omai-engine/pgm'
 * const e = functionExtractor<string[]>('shape', {
 *   length: (w) => w.length,
 *   unigram: (w, t) => [`len=${(w[t] ?? '').length}`, (w[t] ?? '')[0] === (w[t] ?? '')[0]?.toUpperCase() ? 'cap' : 'low'],
 * })
 * e.extract(['The', 'cat']) // => [{ 'len=3': 1, cap: 1 }, { 'len=3': 1, low: 1 }]
 * ```
 */
export function functionExtractor<I>(name: string, spec: FunctionExtractorSpec<I>): FeatureExtractor<I> {
  const { bigram } = spec
  return {
    name,
    labelTransitions: spec.labelTransitions ?? true,
    extract: (input) => {
      const n = spec.length(input)
      const unigram = Array.from({ length: n }, (_, t) => record(spec.unigram(input, t)))
      if (bigram === undefined) return unigram
      return { unigram, bigram: Array.from({ length: n }, (_, t) => (t === 0 ? {} : record(bigram(input, t)))) }
    },
    explain: (input, position) => [
      ...Object.entries(record(spec.unigram(input, position))).map(([feature, value]) => ({
        kind: 'unigram' as const,
        source: name,
        template: name,
        feature,
        value,
      })),
      ...(bigram === undefined || position === 0
        ? []
        : Object.entries(record(bigram(input, position))).map(([feature, value]) => ({
            kind: 'bigram' as const,
            source: name,
            template: name,
            feature,
            value,
          }))),
    ],
  }
}

function addInto(target: Record<string, number>, from: ItemFeatures): void {
  for (const [f, v] of Object.entries(from)) target[f] = (target[f] ?? 0) + v
}

/**
 * Several extractors over the same input as one: the item (and edge) attributes of each are merged, adding the values of
 * a feature that two of them produce. Give the parts distinct feature names (CRF++ identifiers such as `U00:` already
 * do; `prefixedExtractor` namespaces the rest). The label-pair transitions are wanted if any part wants them.
 *
 * @param extractors - The extractors (at least one).
 * @returns The combined extractor.
 * @throws OmaiError with code `value` for none, and `shape` at extraction for parts disagreeing on the length of an
 * input.
 * @concept crf-features
 * @index function Several feature extractors merged into one over the same input.
 * @example
 * ```ts
 * import { attributeExtractor, combineExtractors, crfppExtractor } from 'omai-engine/pgm'
 * const e = combineExtractors([crfppExtractor('U00:%x[0,0]\nB'), attributeExtractor('w[0]', { columns: ['w'] })])
 * e.extract([['a'], ['b']]) // => { unigram: [{ 'U00:a': 1, 'w[0]=a': 1 }, { 'U00:b': 1, 'w[0]=b': 1 }], bigram: [{}, { B: 1 }] }
 * ```
 */
export function combineExtractors<I>(extractors: readonly FeatureExtractor<I>[]): FeatureExtractor<I> {
  if (extractors.length === 0) throw new OmaiError('value', 'combineExtractors: no extractors')
  return {
    name: extractors.map((e) => e.name).join('+'),
    labelTransitions: extractors.some((e) => e.labelTransitions),
    extract: (input) => {
      const parts = extractors.map((e) => e.extract(input))
      const n = crfUnigrams(parts[0] as CrfInput).length
      const unigram: Record<string, number>[] = Array.from({ length: n }, () => ({}))
      const bigram: Record<string, number>[] = Array.from({ length: n }, () => ({}))
      let edges = false
      for (const part of parts) {
        if (crfUnigrams(part).length !== n)
          throw new OmaiError('shape', 'combineExtractors: the extractors disagree on the length of the input')
        for (const [t, item] of crfUnigrams(part).entries()) addInto(unigram[t] as Record<string, number>, item)
        const b = crfBigrams(part)
        if (b !== undefined) {
          edges = true
          for (const [t, edge] of b.entries()) addInto(bigram[t] as Record<string, number>, edge)
        }
      }
      return edges ? { unigram, bigram } : unigram
    },
    explain: (input, position) => extractors.flatMap((e) => e.explain(input, position)),
  }
}

/**
 * An extractor with every feature name prefixed, so that extractors with overlapping names can be combined.
 *
 * @param extractor - The extractor.
 * @param prefix - The prefix, for example `ctx:`.
 * @returns The extractor with prefixed features (and prefixed names in its explanations).
 * @concept crf-features
 * @example
 * ```ts
 * import { prefixedExtractor, wapitiExtractor } from 'omai-engine/pgm'
 * prefixedExtractor(wapitiExtractor('u:w=%x[0,0]'), 'x/').extract([['a']]) // => { unigram: [{ 'x/u:w=a': 1 }], bigram: [{}] }
 * ```
 */
export function prefixedExtractor<I>(extractor: FeatureExtractor<I>, prefix: string): FeatureExtractor<I> {
  const rename = (item: ItemFeatures): Record<string, number> =>
    Object.fromEntries(Object.entries(item).map(([f, v]) => [`${prefix}${f}`, v]))
  return {
    name: `${prefix}${extractor.name}`,
    labelTransitions: extractor.labelTransitions,
    extract: (input) => {
      const x = extractor.extract(input)
      const edges = crfBigrams(x)
      const unigram = crfUnigrams(x).map(rename)
      return edges === undefined ? unigram : { unigram, bigram: edges.map(rename) }
    },
    explain: (input, position) =>
      extractor.explain(input, position).map((o) => ({ ...o, feature: `${prefix}${o.feature}` })),
  }
}

/** Options of {@link hashedExtractor}. */
export interface HashedExtractorOptions {
  /** The number of buckets (default 2^18). */
  readonly nFeatures?: number
  /** Flip the sign of features whose hash is negative, so collisions cancel on average (default true). */
  readonly alternateSign?: boolean
}

/**
 * The hashing trick on an extractor, for template sets whose vocabulary is too large to index (conjunctions of
 * several words): each feature goes to the bucket `#k`, `k = |h| mod nFeatures` of the MurmurHash3 of its name with its
 * value signed by the hash (`text.hashedColumn`, as scikit-learn's FeatureHasher), collisions summed. The vocabulary
 * is then at most `nFeatures` buckets, and a feature never seen in training still lands in a bucket. Explanations keep
 * the unhashed feature names, so a fitted model can still be read back through them (`crfTemplateWeights` does not
 * apply to hashed models, whose weights belong to buckets).
 *
 * @param extractor - The extractor.
 * @param options - The number of buckets and sign alternation.
 * @returns The hashed extractor.
 * @throws OmaiError with code `value` for a non-positive width.
 * @concept crf-features
 * @concept feature-hashing
 * @index function Hash an extractor's features into a fixed number of buckets (the hashing trick).
 * @example
 * ```ts
 * import { crfppExtractor, hashedExtractor } from 'omai-engine/pgm'
 * hashedExtractor(crfppExtractor('U00:%x[0,0]'), { nFeatures: 4, alternateSign: false }).extract([['a'], ['b']]) // => { unigram: [{ '#3': 1 }, { '#1': 1 }], bigram: [{}, {}] }
 * ```
 */
export function hashedExtractor<I>(
  extractor: FeatureExtractor<I>,
  options: HashedExtractorOptions = {},
): FeatureExtractor<I> {
  const width = options.nFeatures ?? 2 ** 18
  if (!Number.isInteger(width) || width < 1)
    throw new OmaiError('value', 'hashedExtractor: nFeatures must be a positive integer')
  const hash = (item: ItemFeatures): Record<string, number> => {
    const out: Record<string, number> = {}
    for (const [f, v] of Object.entries(item)) {
      const { column, sign } = hashedColumn(f, width, options.alternateSign !== false)
      out[`#${column}`] = (out[`#${column}`] ?? 0) + sign * v
    }
    return out
  }
  return {
    name: `hashed(${extractor.name})`,
    labelTransitions: extractor.labelTransitions,
    extract: (input) => {
      const x = extractor.extract(input)
      const edges = crfBigrams(x)
      const unigram = crfUnigrams(x).map(hash)
      return edges === undefined ? unigram : { unigram, bigram: edges.map(hash) }
    },
    explain: extractor.explain,
  }
}

/** Options of {@link crfModelFor}. */
export interface CrfModelOptions {
  /** Keep the features seen at least this often in the training inputs (CRF++'s `-f`; default 1). */
  readonly minFrequency?: number
  /** Whether to have plain label-pair transition weights (default: what the extractor asks for). */
  readonly transitions?: boolean
}

/**
 * The vocabulary of item features and of edge features of extracted inputs, each feature kept when it fires at least
 * `minFrequency` times (once per item or edge it occurs in).
 *
 * @param inputs - Extracted inputs.
 * @param options - The minimum frequency.
 * @returns The item and edge feature indexes.
 * @concept crf-features
 * @example
 * ```ts
 * import { crfVocabulary } from 'omai-engine/pgm'
 * crfVocabulary([[{ a: 1, b: 1 }, { a: 1 }]], { minFrequency: 2 }).unigram.features // => ['a']
 * ```
 */
export function crfVocabulary(
  inputs: readonly CrfInput[],
  options: { readonly minFrequency?: number } = {},
): { readonly unigram: FeatureIndex; readonly bigram: FeatureIndex } {
  const min = options.minFrequency === undefined ? {} : { minFrequency: options.minFrequency }
  return {
    unigram: featureIndex(
      inputs.map((x) => crfUnigrams(x)),
      min,
    ),
    bigram: featureIndex(
      inputs.map((x) => (crfBigrams(x) ?? []).slice(1)),
      min,
    ),
  }
}

/**
 * A linear-chain CRF whose attribute vocabulary is that of already extracted inputs (see {@link crfModelFor}), with zero
 * weights.
 *
 * @param labels - The labels.
 * @param inputs - The extracted training inputs.
 * @param options - The minimum frequency, and whether to have label-pair transitions (default true).
 * @returns The model, ready for `fitCrf` or `trainCrf` on the same inputs.
 * @concept crf-features
 * @example
 * ```ts
 * import { crfModelFromInputs } from 'omai-engine/pgm'
 * crfModelFromInputs(['A', 'B'], [[{ f: 1 }, { f: 1, g: 1 }]]).attributes // => ['f', 'g']
 * ```
 */
export function crfModelFromInputs(
  labels: readonly string[],
  inputs: readonly CrfInput[],
  options: CrfModelOptions = {},
): LinearChainCrf {
  const vocabulary = crfVocabulary(inputs, options)
  return linearChainCrf(labels, vocabulary.unigram.features, {
    transitions: options.transitions ?? true,
    bigramAttributes: vocabulary.bigram.features,
  })
}

/**
 * The linear-chain CRF for an extractor and training inputs: a state weight per kept item feature and label, a weight
 * per kept edge feature and label pair, and label-pair transitions if the extractor's language has them (CRFsuite) or
 * leaves them to a bare bigram line (CRF++ and Wapiti), over the features seen at least `minFrequency` times. It
 * generalises `crfppCrf` to every template language, the hashing trick and combinations.
 *
 * @param extractor - The extractor.
 * @param labels - The labels.
 * @param inputs - The training inputs (token tables for the template languages).
 * @param options - The minimum feature frequency and whether to have label-pair transitions.
 * @returns The model with zero weights; fit it on `labelledSequences(extractor, inputs, labels)`.
 * @concept crf-features
 * @index function The linear-chain CRF with its feature vocabulary built from an extractor and training inputs.
 * @example
 * ```ts
 * import { attributeExtractor, crfModelFor } from 'omai-engine/pgm'
 * const e = attributeExtractor('w[0]', { columns: ['w'] })
 * const crf = crfModelFor(e, ['D', 'N'], [[['the'], ['dog']], [['the'], ['cat']]], { minFrequency: 2 })
 * [crf.attributes, crf.transitionWeights.length] // => [['w[0]=the'], 4]
 * ```
 */
export function crfModelFor<I>(
  extractor: FeatureExtractor<I>,
  labels: readonly string[],
  inputs: readonly I[],
  options: CrfModelOptions = {},
): LinearChainCrf {
  return crfModelFromInputs(labels, inputs.map(extractor.extract), {
    ...options,
    transitions: options.transitions ?? extractor.labelTransitions,
  })
}

/**
 * Labelled training sequences from raw inputs: the extractor applied to each input, paired with its labels.
 *
 * @param extractor - The extractor.
 * @param inputs - The inputs.
 * @param labels - The labels of each input, the same lengths as the inputs.
 * @returns The labelled sequences.
 * @throws OmaiError with code `shape` for a different number of inputs and label lists.
 * @concept crf-features
 * @example
 * ```ts
 * import { functionExtractor, labelledSequences } from 'omai-engine/pgm'
 * const e = functionExtractor<string[]>('w', { length: (w) => w.length, unigram: (w, t) => [`w=${w[t]}`] })
 * labelledSequences(e, [['a', 'b']], [['X', 'Y']]) // => [{ x: [{ 'w=a': 1 }, { 'w=b': 1 }], y: ['X', 'Y'] }]
 * ```
 */
export function labelledSequences<I>(
  extractor: FeatureExtractor<I>,
  inputs: readonly I[],
  labels: readonly (readonly string[])[],
): LabelledSequence[] {
  if (labels.length !== inputs.length)
    throw new OmaiError('shape', `labelledSequences: ${inputs.length} inputs and ${labels.length} label lists`)
  return inputs.map((input, i) => ({ x: extractor.extract(input), y: labels[i] ?? [] }))
}

/**
 * The observation of a sequence under a CRF++ template (`text.parseCrfppTemplate`): the expanded unigram features of
 * each item and bigram features of each edge, as attributes that count their firings.
 *
 * @param template - The parsed CRF++ template.
 * @param table - The token-by-attribute table of the sequence.
 * @returns The observation.
 * @concept crf-features
 * @example
 * ```ts
 * import { crfppObservation } from 'omai-engine/pgm'
 * import { parseCrfppTemplate } from 'omai-engine/text'
 * crfppObservation(parseCrfppTemplate('U00:%x[0,0]\nB'), [['a'], ['b']]) // => { unigram: [{ 'U00:a': 1 }, { 'U00:b': 1 }], bigram: [{}, { B: 1 }] }
 * ```
 */
export function crfppObservation(template: CrfppTemplate, table: TokenTable): CrfObservation {
  return crfppExtractor(template).extract(table) as CrfObservation
}

/**
 * A linear-chain CRF in CRF++'s form for a template and training tables: a state weight per unigram feature and label
 * and a weight per bigram feature and label pair (the bare `B` line is the label-pair transition, as a bigram feature
 * that fires on every edge), over the features seen at least `minFrequency` times (CRF++'s `-f`). The CRF++ case of
 * {@link crfModelFor}.
 *
 * @param template - The parsed CRF++ template.
 * @param labels - The labels.
 * @param tables - The token-by-attribute tables of the training sequences.
 * @param options - The minimum feature frequency (default 1).
 * @returns The model with zero weights, ready for `fitCrf` on {@link crfppObservation} inputs.
 * @concept crf-features
 * @example
 * ```ts
 * import { crfppCrf } from 'omai-engine/pgm'
 * import { parseCrfppTemplate } from 'omai-engine/text'
 * const crf = crfppCrf(parseCrfppTemplate('U00:%x[0,0]\nB'), ['D', 'N'], [[['the'], ['dog']]])
 * [crf.attributes, crf.bigramAttributes, crf.transitionWeights.length] // => [['U00:the', 'U00:dog'], ['B'], 0]
 * ```
 */
export function crfppCrf(
  template: CrfppTemplate,
  labels: readonly string[],
  tables: readonly TokenTable[],
  options: { readonly minFrequency?: number } = {},
): LinearChainCrf {
  return crfModelFor(crfppExtractor(template), labels, tables, options)
}

/** Options of {@link crfTopFeatures}. */
export interface TopFeaturesOptions {
  /** How many features per label (and transitions) to keep (default 10). */
  readonly top?: number
  /** Rank by the weight (`weight`, the strongest evidence for a label first; default) or by its size (`magnitude`). */
  readonly by?: 'weight' | 'magnitude'
}

/** The strongest features of a fitted CRF, per label and for transitions, as coefficient tables. */
export interface CrfFeatureSummary {
  /** For each label, its top item features with their weights. */
  readonly labels: readonly { readonly label: string; readonly features: CoefficientTable }[]
  /** The top label-pair transitions (`A->B`) and edge features (`B: A->B`) with their weights. */
  readonly transitions: CoefficientTable
}

function rank(
  entries: readonly (readonly [string, number])[],
  top: number,
  by: 'weight' | 'magnitude',
): CoefficientTable {
  const sorted = [...entries].sort((a, b) => (by === 'weight' ? b[1] - a[1] : Math.abs(b[1]) - Math.abs(a[1])))
  const kept = sorted.slice(0, top)
  return coefficientTable({ names: kept.map(([n]) => n), estimates: kept.map(([, w]) => w) })
}

/**
 * The top-weighted features of a fitted linear-chain CRF by label, and its strongest transitions: the teaching output of
 * a CRF (which suffix says "noun", which label follows which). Item features are the model's attributes (the strings
 * the extractor made); transitions include the weights of edge features, as `B: A->B` for the bare `B` of CRF++.
 *
 * @param crf - The fitted model.
 * @param options - How many to keep and how to rank.
 * @returns A coefficient table of features per label and one of transitions.
 * @concept crf-features
 * @index function The top-weighted features of a fitted CRF per label, and its strongest transitions.
 * @example
 * ```ts
 * import { crfTopFeatures, linearChainCrf } from 'omai-engine/pgm'
 * const crf = linearChainCrf(['A', 'B'], ['f', 'g'], { state: [2, -1, 0.5, 3] })
 * crfTopFeatures(crf, { top: 1 }).labels.map((l) => l.features.names) // => [['f'], ['g']]
 * ```
 */
export function crfTopFeatures(crf: LinearChainCrf, options: TopFeaturesOptions = {}): CrfFeatureSummary {
  const top = options.top ?? 10
  const by = options.by ?? 'weight'
  const K = crf.labels.length
  const labels = crf.labels.map((label, y) => ({
    label,
    features: rank(
      crf.attributes.map((a, i) => [a, crf.stateWeights[i * K + y] ?? 0] as const),
      top,
      by,
    ),
  }))
  const pairs: (readonly [string, number])[] = []
  for (let i = 0; i < K; i++)
    for (let j = 0; j < K; j++) {
      if (crf.transitionWeights.length > 0)
        pairs.push([`${crf.labels[i]}->${crf.labels[j]}`, crf.transitionWeights[i * K + j] ?? 0])
      for (const [a, b] of crf.bigramAttributes.entries())
        pairs.push([`${b}: ${crf.labels[i]}->${crf.labels[j]}`, crf.bigramWeights[(a * K + i) * K + j] ?? 0])
    }
  return { labels, transitions: rank(pairs, top, by) }
}

/** The weight a template carries in a fitted model. */
export interface TemplateWeight {
  readonly template: string
  /** How many of the model's features the template produced. */
  readonly features: number
  /** The sum over those features of the largest absolute weight over labels (and label pairs). */
  readonly weight: number
}

/**
 * Which templates carry a fitted model: each feature of the model is traced to the template that produced it on the
 * given inputs (through the extractor's `explain`, the first template to fire it), and the largest absolute weight of
 * each feature (over labels, or label pairs for an edge feature) is summed per template. Templates whose features the
 * minimum-frequency cut-off removed show with zero features. Not meaningful for hashed extractors.
 *
 * @param crf - The fitted model.
 * @param extractor - The extractor the model was built with.
 * @param inputs - Inputs to trace features on (usually the training inputs).
 * @returns The templates, heaviest first.
 * @concept crf-features
 * @index function The weight carried by each template of a fitted CRF, traced through explain.
 * @example
 * ```ts
 * import { crfTemplateWeights, crfppExtractor, linearChainCrf } from 'omai-engine/pgm'
 * const e = crfppExtractor('U00:%x[0,0]\nU01:%x[-1,0]')
 * const crf = linearChainCrf(['A', 'B'], ['U00:a', 'U01:_B-1'], { state: [3, -3, 1, 1] })
 * crfTemplateWeights(crf, e, [[['a']]]).map((t) => [t.template, t.weight]) // => [['U00:%x[0,0]', 3], ['U01:%x[-1,0]', 1]]
 * ```
 */
export function crfTemplateWeights<I>(
  crf: LinearChainCrf,
  extractor: FeatureExtractor<I>,
  inputs: readonly I[],
): TemplateWeight[] {
  const K = crf.labels.length
  const owner = new Map<string, string>()
  const templates: string[] = []
  const seen = new Set<string>()
  for (const input of inputs) {
    const n = crfUnigrams(extractor.extract(input)).length
    for (let t = 0; t < n; t++)
      for (const o of extractor.explain(input, t)) {
        const key = `${o.kind}\u0000${o.feature}`
        if (!owner.has(key)) owner.set(key, o.template)
        if (!seen.has(o.template)) {
          seen.add(o.template)
          templates.push(o.template)
        }
      }
  }
  const total = new Map(templates.map((t) => [t, { features: 0, weight: 0 }]))
  const credit = (kind: 'unigram' | 'bigram', feature: string, weight: number): void => {
    const template = owner.get(`${kind}\u0000${feature}`)
    const cell = template === undefined ? undefined : total.get(template)
    if (cell === undefined) return
    cell.features++
    cell.weight += weight
  }
  const biggest = (from: number, count: number, w: Float64Array): number => {
    let m = 0
    for (let k = from; k < from + count; k++) m = Math.max(m, Math.abs(w[k] ?? 0))
    return m
  }
  for (const [i, a] of crf.attributes.entries()) credit('unigram', a, biggest(i * K, K, crf.stateWeights))
  for (const [i, a] of crf.bigramAttributes.entries()) credit('bigram', a, biggest(i * K * K, K * K, crf.bigramWeights))
  return [...total].map(([template, v]) => ({ template, ...v })).sort((a, b) => b.weight - a.weight)
}
