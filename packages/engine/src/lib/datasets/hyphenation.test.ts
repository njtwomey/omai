import { describe, expect, it } from 'vitest'
import { lookup } from '../base/index.ts'
import { trainTestSplit } from '../data/index.ts'
import type { FeatureExtractor } from '../pgm/index.ts'
import {
  attributeExtractor,
  crfDecode,
  crfModelFor,
  crfppExtractor,
  labelledSequences,
  trainCrf,
} from '../pgm/index.ts'
import { key } from '../random/index.ts'
import { hyphenate, type TokenTable } from '../text/index.ts'
import { datasetRegistry } from './catalogue.ts'
import {
  hyphenatedWord,
  hyphenationLabels,
  hyphenationPatternsEnUs,
  hyphenationTables,
  hyphenationWords,
} from './hyphenation.ts'
import { loadDataset } from './registry.ts'

describe('the hyphenation dataset', () => {
  const data = hyphenationWords()

  it('is a sample of the Moby list, with one tag per letter and no break after the last letter', () => {
    expect(data.kind).toBe('sequence-labelling')
    expect(data.tokens.length).toBe(3984)
    expect(data.tagNames).toEqual(['O', 'B'])
    expect(data.meta.licence).toMatch(/Public domain/)
    for (const [i, letters] of data.tokens.entries()) {
      expect(letters.length).toBeGreaterThanOrEqual(4)
      expect(letters.length).toBeLessThanOrEqual(14)
      expect(letters.every((c) => c >= 'a' && c <= 'z')).toBe(true)
      expect(data.tags[i]?.length).toBe(letters.length)
      expect(data.tags[i]?.at(-1)).toBe(0)
    }
    expect(new Set(data.tokens.map((t) => t.join(''))).size).toBe(data.tokens.length)
    const words = data.tokens.map((letters, i) => hyphenatedWord(letters, data.tags[i] ?? []))
    expect(words.slice(0, 4)).toEqual(['ab-a-cus', 'ab-duce', 'ab-ir-ri-ta-tive', 'ab-lu-ent'])
    expect(words.filter((w) => !w.includes('-')).length).toBeLessThan(0.1 * words.length)
  })

  it('is registered with its tags, and splits like any dataset', () => {
    const entry = lookup(datasetRegistry, 'hyphenationWords')
    expect([entry.form, entry.kind, entry.modality, entry.real, entry.size]).toEqual([
      'vendored',
      'sequence-labelling',
      'sequence',
      true,
      3984,
    ])
    expect(loadDataset(entry).data.kind).toBe('sequence-labelling')
    const { train, test } = trainTestSplit(data, { testSize: 400, key: key(1) })
    expect([train.tokens.length, test.tokens.length]).toEqual([3584, 400])
    expect(train.tags[0]?.length).toBe(train.tokens[0]?.length)
  })

  it('makes letter tables with a vowel class column', () => {
    expect(hyphenationTables(['yes', [...'ab']])).toEqual([
      [
        ['y', 'V'],
        ['e', 'V'],
        ['s', 'C'],
      ],
      [
        ['a', 'V'],
        ['b', 'C'],
      ],
    ])
    expect(hyphenationLabels([[0, 1, 0]])).toEqual([['O', 'B', 'O']])
  })

  it('has the TeX patterns of American English as a baseline hyphenator', () => {
    const patterns = hyphenationPatternsEnUs()
    expect(patterns.patterns.size).toBeGreaterThan(4000)
    expect(hyphenate('hyphenation', patterns).parts).toEqual(['hy', 'phen', 'ation'])
    expect(hyphenate('table', patterns).parts).toEqual(['ta', 'ble'])
  })
})

/** A word is hyphenated right when every gap between letters is right; the points are per gap after each letter. */
function scores(
  predicted: readonly (readonly boolean[])[],
  gold: readonly (readonly boolean[])[],
): { word: number; precision: number; recall: number } {
  let right = 0
  let tp = 0
  let fp = 0
  let fn = 0
  for (const [i, g] of gold.entries()) {
    const p = predicted[i] ?? []
    if (g.every((b, k) => b === p[k])) right++
    for (const [k, b] of g.entries()) {
      if (b && p[k]) tp++
      else if (p[k]) fp++
      else if (b) fn++
    }
  }
  return { word: right / gold.length, precision: tp / (tp + fp), recall: tp / (tp + fn) }
}

describe('hyphenation with a CRF, against Liang patterns', () => {
  const data = hyphenationWords()
  const { train, test } = trainTestSplit(data, { testSize: 200, key: key(1) })
  const nTrain = 200
  const trainWords = train.tokens.slice(0, nTrain)
  const trainTables = hyphenationTables(trainWords)
  const trainLabels = hyphenationLabels(train.tags.slice(0, nTrain))
  const testTables = hyphenationTables(test.tokens)
  const gold = test.tags.map((t) => t.slice(0, -1).map((k) => k === 1))

  /** Letter windows of one to three letters and the vowel/consonant classes, in the CRF++ and the CRFsuite forms. */
  const crfpp = crfppExtractor(
    [
      ...[-2, -1, 0, 1, 2].map((o, i) => `U0${i}:%x[${o},0]`),
      'U05:%x[-1,0]/%x[0,0]',
      'U06:%x[0,0]/%x[1,0]',
      'U07:%x[-2,0]/%x[-1,0]/%x[0,0]',
      'U08:%x[-1,0]/%x[0,0]/%x[1,0]',
      'U09:%x[0,0]/%x[1,0]/%x[2,0]',
      'U10:%x[-1,1]/%x[0,1]/%x[1,1]',
      'B',
    ].join('\n'),
  )
  const attributes = attributeExtractor(
    'c[-2]\nc[-1]\nc[0]\nc[1]\nc[2]\nc[-1]|c[0]\nc[0]|c[1]\nc[-2]|c[-1]|c[0]\nc[-1]|c[0]|c[1]\nc[0]|c[1]|c[2]\nv[-1]|v[0]|v[1]',
    { columns: ['c', 'v'], boundary: 'pad' },
  )

  const evaluate = (extractor: FeatureExtractor<TokenTable>): { word: number; precision: number; recall: number } => {
    const crf = crfModelFor(extractor, ['O', 'B'], trainTables)
    const fitted = trainCrf(crf, labelledSequences(extractor, trainTables, trainLabels), { c2: 1 }, 40).crf
    const predicted = testTables.map((t) =>
      crfDecode(fitted, extractor.extract(t))
        .labels.slice(0, -1)
        .map((l) => l === 'B'),
    )
    return scores(predicted, gold)
  }

  it('learns where words break from letter windows, in both template forms', () => {
    const a = evaluate(crfpp)
    const b = evaluate(attributes as FeatureExtractor<TokenTable>)
    for (const s of [a, b]) {
      // Far better than never hyphenating (no hyphen recalled, almost no word right) after 200 training words.
      expect(s.recall).toBeGreaterThan(0.5)
      expect(s.precision).toBeGreaterThan(0.6)
      expect(s.word).toBeGreaterThan(0.2)
    }
    // The two forms are the same model up to the label-pair transitions' parametrisation: they land close together.
    expect(Math.abs(a.word - b.word)).toBeLessThan(0.1)
  }, 30_000)

  it('is compared with the TeX patterns on the same held-out words', () => {
    const patterns = hyphenationPatternsEnUs()
    const predicted = test.tokens.map((letters) => {
      const points = new Set(hyphenate(letters.join(''), patterns, { leftMin: 1, rightMin: 1 }).points)
      return letters.slice(0, -1).map((_, k) => points.has(k + 1))
    })
    const liang = scores(predicted, gold)
    // Liang patterns are conservative (high precision): they miss breaks that a dictionary has.
    expect(liang.precision).toBeGreaterThan(0.8)
    expect(liang.recall).toBeGreaterThan(0.5)
    expect(liang.word).toBeGreaterThan(0.3)
  })
})
