import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { parseCrfppTemplate, type Token, type TokenTable } from '../../text/index.ts'
import {
  attributeExtractor,
  combineExtractors,
  crfBigrams,
  crfDecode,
  crfModelFor,
  crfModelFromInputs,
  crfObjective,
  crfppCrf,
  crfppExtractor,
  crfppObservation,
  crfTemplateWeights,
  crfTopFeatures,
  crfUnigrams,
  crfVocabulary,
  type FeatureExtractor,
  functionExtractor,
  hashedExtractor,
  labelledSequences,
  linearChainCrf,
  prefixedExtractor,
  trainCrf,
  wapitiExtractor,
  wapitiObservation,
} from '../index.ts'

const TABLES: TokenTable[] = [
  [
    ['the', 'DT'],
    ['cat', 'NN'],
    ['runs', 'VB'],
  ],
  [
    ['a', 'DT'],
    ['big', 'JJ'],
    ['dog', 'NN'],
    ['barks', 'VB'],
  ],
  [['dogs', 'NN']],
]
const LABELS = [['D', 'N', 'V'], ['D', 'J', 'N', 'V'], ['N']]

/** The same templates in the three languages, with the same literal separators, so that their values can be compared. */
const CRFPP = 'U00:%x[-1,0]\nU01:%x[0,0]|%x[1,1]\nU02:%x[0,1]\nU03:%x[-2,0]|%x[0,0]\nB10:%x[-1,1]|%x[0,1]'
const WAPITI = 'u:a=%x[-1,0]\nu:b=%x[0,0]|%x[1,1]\nu:c=%x[0,1]\nu:d=%x[-2,0]|%x[0,0]\nb:e=%x[-1,1]|%x[0,1]'
const ATTRIBUTES = 'w[-1]\nw[0]|p[1]\np[0]\nw[-2]|w[0]'

/** A feature's value alone, with the template's name and each language's boundary marker removed. */
const featureValue = (feature: string): string =>
  feature.slice(Math.max(feature.indexOf(':'), feature.indexOf('=')) + 1).replaceAll(/_[Bx]([-+])/g, '_$1')

const values = (items: readonly Record<string, number>[]): string[][] =>
  items.map((item) => Object.keys(item).map(featureValue).sort())

describe('the three template languages give the same features', () => {
  const crfpp = crfppExtractor(CRFPP)
  const wapiti = wapitiExtractor(WAPITI)
  const attributes = attributeExtractor(ATTRIBUTES, { columns: ['w', 'p'], boundary: 'pad' })

  it('item features agree value by value, with each language boundary markers', () => {
    for (const table of TABLES) {
      const a = values(crfUnigrams(crfpp.extract(table)) as Record<string, number>[])
      const b = values(crfUnigrams(wapiti.extract(table)) as Record<string, number>[])
      const c = values(crfUnigrams(attributes.extract(table)) as Record<string, number>[])
      expect(b).toEqual(a)
      expect(c).toEqual(a)
    }
  })

  it('edge features agree between CRF++ and Wapiti, and are absent at the first position', () => {
    for (const table of TABLES) {
      const a = crfBigrams(crfpp.extract(table)) as Record<string, number>[]
      const b = crfBigrams(wapiti.extract(table)) as Record<string, number>[]
      expect(a[0]).toEqual({})
      expect(b[0]).toEqual({})
      expect(values(b)).toEqual(values(a))
    }
  })

  it('records the boundary values of each language', () => {
    const table: TokenTable = [['a', 'X']]
    expect(crfUnigrams(crfpp.extract(table))[0]).toMatchObject({ 'U00:_B-1': 1, 'U03:_B-2|a': 1 })
    expect(crfUnigrams(wapiti.extract(table))[0]).toMatchObject({ 'u:a=_x-1': 1, 'u:d=_x-2|a': 1 })
    expect(crfUnigrams(attributes.extract(table))[0]).toMatchObject({ 'w[-1]=_B-1': 1, 'w[-2]|w[0]=_B-2|a': 1 })
    // CRFsuite's default skips a template whose cell is outside the sequence.
    const skipping = attributeExtractor(ATTRIBUTES, { columns: ['w', 'p'] })
    expect(Object.keys(crfUnigrams(skipping.extract(table))[0] ?? {})).toEqual(['p[0]=X'])
  })

  it('trains to the same optimum through the three routes', () => {
    // The same item templates, and one label-pair transition: CRF++'s and Wapiti's bare bigram line, CRFsuite's own.
    const items = CRFPP.split('\n').slice(0, 4)
    const routes = [
      crfppExtractor(`${items.join('\n')}\nB`),
      wapitiExtractor(`${WAPITI.split('\n').slice(0, 4).join('\n')}\nb`),
      attributeExtractor(ATTRIBUTES, { columns: ['w', 'p'], boundary: 'pad' }),
    ] as FeatureExtractor<TokenTable>[]
    const classes = ['D', 'J', 'N', 'V']
    const fitted = routes.map((e) => {
      const data = labelledSequences(e, TABLES, LABELS)
      return trainCrf(crfModelFor(e, classes, TABLES), data, { c2: 0.5 }, 300)
    })
    for (const f of fitted) expect(Math.abs(f.objective - (fitted[0]?.objective ?? 0))).toBeLessThan(1e-5)
    const probe = TABLES[1] as TokenTable
    const decoded = routes.map((e, i) => crfDecode((fitted[i] as (typeof fitted)[number]).crf, e.extract(probe)).labels)
    expect(decoded[1]).toEqual(decoded[0])
    expect(decoded[2]).toEqual(decoded[0])
  })
})

describe('the feature extractors', () => {
  it('wapitiObservation and the Wapiti extractor use regular-expression tests and matches', () => {
    const patterns = 'u:cap=%t[0,0,"^\\u"]\nU:suf=%m[0,0,"..$"]\n*:w=%x[0,0]'
    const x = wapitiObservation(patterns, [['The'], ['cat']])
    expect(x.unigram).toEqual([
      { 'u:cap=true': 1, 'U:suf=he': 1, '*:w=The': 1 },
      { 'u:cap=false': 1, 'U:suf=at': 1, '*:w=cat': 1 },
    ])
    expect(x.bigram).toEqual([{}, { '*:w=cat': 1 }])
    const e = wapitiExtractor(patterns)
    expect(e.explain([['The'], ['cat']], 1).map((o) => [o.kind, o.template, o.feature])).toEqual([
      ['unigram', 'u:cap=%t[0,0,"^\\u"]', 'u:cap=false'],
      ['unigram', 'U:suf=%m[0,0,"..$"]', 'U:suf=at'],
      ['unigram', '*:w=%x[0,0]', '*:w=cat'],
      ['bigram', '*:w=%x[0,0]', '*:w=cat'],
    ])
  })

  it('reproduces the feature dictionaries of the sklearn-crfsuite case from its templates', () => {
    const c = loadFixture('pgm/structured/crf').cases.find((k) => k.name === 'crfsuite/c2=0.1')
    const X = c?.params['x'] as Record<string, number>[][]
    // The fixture's templates: bias, w[0], w[-1]|w[0], w[1] and the 2-letter suffix.
    const suffix = functionExtractor<readonly Token[]>('suffix', {
      length: (tokens) => tokens.length,
      unigram: (tokens, t) => [`suf2=${(tokens[t]?.['w'] ?? '').slice(-2)}`],
    })
    const extractor = combineExtractors<readonly Token[]>([
      attributeExtractor('w[0]\nw[-1]|w[0]\nw[1]', { bias: true }) as FeatureExtractor<readonly Token[]>,
      suffix,
    ])
    for (const sentence of X) {
      const words = sentence.map(
        (item) =>
          Object.keys(item)
            .find((k) => k.startsWith('w[0]='))
            ?.slice(5) ?? '',
      )
      const tokens = words.map((w) => ({ w }))
      expect(extractor.extract(tokens)).toEqual(sentence)
    }
  })

  it('attribute templates take the bias and sentence markers, from tokens or table rows', () => {
    const e = attributeExtractor('w[0]', { columns: ['w'], bias: true, bosEos: true })
    expect(e.extract([['a'], ['b']])).toEqual([
      { bias: 1, 'w[0]=a': 1, __BOS__: 1 },
      { bias: 1, 'w[0]=b': 1, __EOS__: 1 },
    ])
    expect(e.extract([{ w: 'a' }, { w: 'b' }] as never)).toEqual(e.extract([['a'], ['b']]))
    expect(e.explain([['a'], ['b']], 1).map((o) => o.template)).toEqual(['w[0]', 'bias', 'bosEos'])
    expect(() => attributeExtractor('w[0]').extract([['a']])).toThrow(/names of their columns/)
  })

  it('a function extractor takes feature strings or records and optional edge features', () => {
    const e = functionExtractor<string>('chars', {
      length: (s) => s.length,
      unigram: (s, t) => ({ [`c=${s[t]}`]: 2 }),
      bigram: (s, t) => [`pair=${s[t - 1]}${s[t]}`],
      labelTransitions: false,
    })
    expect(e.extract('ab')).toEqual({ unigram: [{ 'c=a': 2 }, { 'c=b': 2 }], bigram: [{}, { 'pair=ab': 1 }] })
    expect(e.labelTransitions).toBe(false)
    expect(e.explain('ab', 1).map((o) => [o.kind, o.feature, o.value])).toEqual([
      ['unigram', 'c=b', 2],
      ['bigram', 'pair=ab', 1],
    ])
  })

  it('combines extractors by adding the values of shared features, and prefixes separate them', () => {
    const w = functionExtractor<string[]>('w', { length: (x) => x.length, unigram: (x, t) => [`f=${x[t]}`] })
    const v = functionExtractor<string[]>('v', { length: (x) => x.length, unigram: (x, t) => [`f=${x[t]}`, 'one'] })
    const both = combineExtractors([w, v])
    expect(both.name).toBe('w+v')
    expect(both.extract(['a'])).toEqual([{ 'f=a': 2, one: 1 }])
    const apart = combineExtractors([w, prefixedExtractor(v, 'v:')])
    expect(apart.extract(['a'])).toEqual([{ 'f=a': 1, 'v:f=a': 1, 'v:one': 1 }])
    expect(apart.explain(['a'], 0).map((o) => [o.source, o.feature])).toEqual([
      ['w', 'f=a'],
      ['v', 'v:f=a'],
      ['v', 'v:one'],
    ])
    const edged = combineExtractors<TokenTable>([crfppExtractor('U00:%x[0,0]'), crfppExtractor('B01:%x[0,0]')])
    expect(edged.extract([['a'], ['b']])).toEqual({
      unigram: [{ 'U00:a': 1 }, { 'U00:b': 1 }],
      bigram: [{}, { 'B01:b': 1 }],
    })
    expect(() => combineExtractors([])).toThrow(/no extractors/)
    expect(() =>
      combineExtractors<string[]>([w, functionExtractor('short', { length: () => 0, unigram: () => [] })]).extract([
        'a',
      ]),
    ).toThrow(/disagree/)
  })

  it('the hashing trick bounds the vocabulary and signs the values as FeatureHasher', () => {
    const base = crfppExtractor('U00:%x[0,0]\nU01:%x[-1,0]|%x[0,0]\nB')
    const hashed = hashedExtractor(base, { nFeatures: 8 })
    const table: TokenTable = [['a'], ['b'], ['a']]
    const x = hashed.extract(table)
    for (const item of crfUnigrams(x)) for (const k of Object.keys(item)) expect(Number(k.slice(1))).toBeLessThan(8)
    const crf = crfModelFor(hashed, ['P', 'Q'], [table])
    expect(crf.attributes.length).toBeLessThanOrEqual(8)
    // Each item holds the signed counts of its features' buckets: the total |value| is the number of features that fired.
    const total = crfUnigrams(x).map((item) => Object.values(item).reduce((s, v) => s + Math.abs(v), 0))
    expect(total.every((t) => t <= 2)).toBe(true)
    const unsigned = hashedExtractor(base, { nFeatures: 8, alternateSign: false }).extract(table)
    expect(crfUnigrams(unsigned).map((item) => Object.values(item).reduce((s, v) => s + v, 0))).toEqual([2, 2, 2])
    expect(hashed.explain(table, 0)).toEqual(base.explain(table, 0))
    expect(() => hashedExtractor(base, { nFeatures: 0 })).toThrow(/positive integer/)
  })
})

describe('crfModelFor and the explanation helpers', () => {
  const crfpp = crfppExtractor(CRFPP)

  it('builds the vocabulary with the frequency cut-off, generalising crfppCrf', () => {
    const template = parseCrfppTemplate(CRFPP)
    for (const minFrequency of [1, 2]) {
      const a = crfppCrf(template, ['D', 'N'], TABLES, { minFrequency })
      const b = crfModelFor(crfpp, ['D', 'N'], TABLES, { minFrequency })
      expect(b).toEqual(a)
    }
    expect(crfppCrf(template, ['D', 'N'], TABLES).transitionWeights.length).toBe(0)
    expect(crfppObservation(template, TABLES[0] as TokenTable)).toEqual(crfpp.extract(TABLES[0] as TokenTable))
    const cut = crfModelFor(crfpp, ['D', 'N'], TABLES, { minFrequency: 2 })
    expect(cut.attributes.length).toBeLessThan(crfModelFor(crfpp, ['D', 'N'], TABLES).attributes.length)
  })

  it('asks for label-pair transitions as the language does, unless told otherwise', () => {
    const attributes = attributeExtractor('w[0]', { columns: ['w', 'p'] })
    expect(crfModelFor(attributes, ['A', 'B'], TABLES).transitionWeights.length).toBe(4)
    expect(crfModelFor(attributes, ['A', 'B'], TABLES, { transitions: false }).transitionWeights.length).toBe(0)
    expect(crfModelFor(wapitiExtractor('u:w=%x[0,0]\nb'), ['A', 'B'], TABLES).transitionWeights.length).toBe(0)
    const v = crfVocabulary([crfpp.extract(TABLES[0] as TokenTable)], { minFrequency: 2 })
    expect(v.bigram.features).toEqual([])
    expect(crfModelFromInputs(['A'], [[{ f: 1 }]]).attributes).toEqual(['f'])
  })

  it('checks that inputs and labels match', () => {
    expect(() => labelledSequences(crfpp, TABLES, [['D']])).toThrow(/3 inputs and 1 label lists/)
  })

  it('explain accounts for every extracted feature, with its template', () => {
    const extractors: FeatureExtractor<TokenTable>[] = [
      crfpp,
      wapitiExtractor(WAPITI),
      attributeExtractor(ATTRIBUTES, { columns: ['w', 'p'], bias: true, bosEos: true, boundary: 'pad' }),
    ]
    for (const e of extractors)
      for (const table of TABLES) {
        const x = e.extract(table)
        for (let t = 0; t < table.length; t++) {
          const explained = e.explain(table, t)
          const fromUnigram = Object.fromEntries(
            explained.filter((o) => o.kind === 'unigram').map((o) => [o.feature, o.value]),
          )
          expect(fromUnigram).toEqual(crfUnigrams(x)[t])
          const edge = crfBigrams(x)?.[t] ?? {}
          expect(
            Object.fromEntries(explained.filter((o) => o.kind === 'bigram').map((o) => [o.feature, o.value])),
          ).toEqual(edge)
        }
      }
  })

  it('reads a fitted model back: top features by label, strongest transitions and the weight per template', () => {
    const data = labelledSequences(crfpp, TABLES, LABELS)
    const start = crfModelFor(crfpp, ['D', 'J', 'N', 'V'], TABLES)
    const { crf } = trainCrf(start, data, { c2: 0.2 }, 100)
    const summary = crfTopFeatures(crf, { top: 3 })
    expect(summary.labels.map((l) => l.label)).toEqual(['D', 'J', 'N', 'V'])
    const v = summary.labels[3]?.features
    expect(v?.names.length).toBe(3)
    // The part-of-speech column says the label outright: the verb's top feature carries it.
    expect(v?.names[0]).toMatch(/VB/)
    expect(v?.estimates.every((w, i, a) => i === 0 || (a[i - 1] ?? 0) >= w)).toBe(true)
    expect(summary.transitions.names.length).toBe(3)
    expect(summary.transitions.names.every((n) => n.includes('->'))).toBe(true)
    const byMagnitude = crfTopFeatures(crf, { top: 2, by: 'magnitude' }).labels[0]?.features.estimates ?? []
    expect(Math.abs(byMagnitude[0] ?? 0)).toBeGreaterThanOrEqual(Math.abs(byMagnitude[1] ?? 0))
    const weights = crfTemplateWeights(crf, crfpp, TABLES)
    expect(weights.map((w) => w.template).sort()).toEqual(
      parseCrfppTemplate(CRFPP)
        .unigram.concat(parseCrfppTemplate(CRFPP).bigram)
        .map((l) => l.source)
        .sort(),
    )
    expect(weights.reduce((s, w) => s + w.features, 0)).toBe(crf.attributes.length + crf.bigramAttributes.length)
    // Heaviest first; the column that names the label outright (%x[0,1]) carries the most.
    expect(weights[0]?.template).toMatch(/%x\[0,1\]/)
    expect(weights.every((w, i, a) => i === 0 || (a[i - 1]?.weight ?? 0) >= w.weight)).toBe(true)
  })

  it('a model with the frequency cut-off removing a template shows it with zero features', () => {
    const crf = linearChainCrf(['A', 'B'], ['U00:a'], { state: [1, 2] })
    const weights = crfTemplateWeights(crf, crfppExtractor('U00:%x[0,0]\nU01:%x[1,0]'), [[['a'], ['b']]])
    expect(weights).toEqual([
      { template: 'U00:%x[0,0]', features: 1, weight: 2 },
      { template: 'U01:%x[1,0]', features: 0, weight: 0 },
    ])
  })

  it('evaluating the objective does not depend on how the model was built', () => {
    const data = labelledSequences(crfpp, TABLES, LABELS)
    const crf = crfModelFor(crfpp, ['D', 'J', 'N', 'V'], TABLES)
    expect(crfObjective(crf, data, { c2: 1 }).value).toBeCloseTo(
      crfObjective(crfppCrf(parseCrfppTemplate(CRFPP), ['D', 'J', 'N', 'V'], TABLES), data, { c2: 1 }).value,
      12,
    )
  })
})
