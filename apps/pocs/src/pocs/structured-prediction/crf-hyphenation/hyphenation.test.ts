import { crf, crfData } from 'omai-engine/pgm'
import type { TokenTable } from 'omai-engine/text'
import { describe, expect, it } from 'vitest'
import {
  extractorOf,
  firstWords,
  gapEvidence,
  patternLabels,
  readWord,
  scoreLabels,
  templateLines,
  templateSurvival,
  wordSplit,
} from './hyphenation.ts'

const window = { radius: 2, classes: true }

describe('the hyphenation poc compute', () => {
  const { train, test } = wordSplit()

  it('splits the word list once into 400 held-out words and the rest', () => {
    expect(test.words).toHaveLength(400)
    expect(train.words.length + test.words.length).toBe(3984)
    expect(new Set(test.words).size).toBe(400)
    expect(wordSplit()).toBe(wordSplit())
  })

  it('writes the templates of a window', () => {
    expect(templateLines({ radius: 1, classes: false })).toEqual(['c[-1]', 'c[0]', 'c[1]', 'c[-1]|c[0]', 'c[0]|c[1]'])
    expect(templateLines({ radius: 2, classes: true })).toContain('k[-2]|k[-1]|k[0]|k[1]|k[2]')
    expect(templateLines({ radius: 3, classes: true }).length).toBeGreaterThan(templateLines(window).length)
  })

  it('scores the TeX patterns well above never hyphenating, by the library metrics', () => {
    const baseline = scoreLabels(
      test.labels,
      test.words.map((w) => patternLabels(w)),
    )
    expect(baseline.precision).toBeGreaterThan(0.8)
    expect(baseline.recall).toBeGreaterThan(0.5)
  })

  it('fits a CRF in seconds, prunes templates with L1 and reads a word and a gap', () => {
    const few = firstWords(train, 200)
    const extractor = extractorOf(window)
    const lines = templateLines(window)
    const fit = (c1: number) =>
      crf<TokenTable>({ features: extractor, c1, c2: 0.1, maxIterations: 60 }).fit(crfData(few.tables, few.labels))
    const dense = fit(0)
    const sparse = fit(1)
    expect(sparse.params.nonzeros).toBeLessThan(dense.params.nonzeros)
    const survival = templateSurvival(sparse, lines)
    expect(survival.map((s) => s.template)).toEqual(['bias', ...lines])
    expect(survival.reduce((n, s) => n + s.nonzero, 0)).toBeLessThan(
      templateSurvival(dense, lines).reduce((n, s) => n + s.nonzero, 0),
    )
    const scores = scoreLabels(test.labels, dense.predict(test.tables))
    expect(scores.words).toBeGreaterThan(0.1)
    const reading = readWord(dense, 'hyphenation')
    expect(reading.letters).toHaveLength(11)
    expect(reading.probability).toHaveLength(10)
    expect(reading.tex).toBe('hy-phen-ation')
    expect(reading.crf.replaceAll('-', '')).toBe('hyphenation')
    const evidence = gapEvidence(dense, extractor, test.tables[0] as TokenTable, 1, 5)
    expect(evidence.length).toBeGreaterThan(0)
    expect(Math.abs(evidence[0]?.push ?? 0)).toBeGreaterThanOrEqual(Math.abs(evidence.at(-1)?.push ?? 0))
  })
})
