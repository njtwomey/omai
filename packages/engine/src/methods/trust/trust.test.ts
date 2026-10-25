import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { biasedCredit, heteroscedastic, makeClassification } from '../../lib/datasets/index.ts'
import { recordsOf, setSizes } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { gaussianNB } from '../naive-bayes/index.ts'
import { kNeighborsRegressor } from '../neighbours/index.ts'
import { calibrateClassifier } from './calibration.ts'
import { conformalClassification, conformalRegression, onlineConformal } from './conformal.ts'
import { modelCard } from './model-cards.ts'
import { robustnessStudy } from './robustness.ts'
import { shiftStudy } from './shift.ts'
import { ensembleUncertainty } from './uncertainty.ts'

const curve = generate(heteroscedastic, { key: key(1), knobs: { n: 600 } }).data
const classes = generate(makeClassification, {
  key: key(2),
  knobs: { n: 800, nFeatures: 10, nInformative: 3, nRedundant: 5, classSep: 0.8 },
}).data

describe('calibration', () => {
  it('recalibrates an overconfident naive Bayes model, lowering its calibration error', () => {
    const model = gaussianNB().fit(classes)
    const r = calibrateClassifier(model, classes, { key: key(3) })
    const ece = r.ece
    expect(ece['uncalibrated']).toBeGreaterThan(0.05)
    for (const m of ['Platt scaling', 'isotonic', 'temperature']) expect(ece[m]).toBeLessThan(ece['uncalibrated'] ?? 0)
    const rec = recordsOf(r.exposition, 'calibration')[0]
    expect(rec?.curves).toHaveLength(4)
    expect(rec?.curves[1]?.map?.x).toHaveLength(101)
  })

  it('draws each group’s curve when asked', () => {
    const credit = generate(biasedCredit, { key: key(4), knobs: { n: 800 } }).data
    const r = calibrateClassifier(logisticRegression().fit(credit), credit, { sensitive: 'group', methods: ['platt'] })
    const groups = recordsOf(r.exposition, 'calibration')[0]?.curves.filter((c) => c.group !== undefined)
    expect(groups).toHaveLength(4)
  })
})

describe('conformal prediction', () => {
  for (const method of ['split', 'normalised', 'cv-plus'] as const)
    it(`${method} intervals cover about 1 - alpha of new rows`, () => {
      const r = conformalRegression(kNeighborsRegressor({ nNeighbors: 15 }), curve, { alpha: 0.1, method, key: key(5) })
      expect(r.coverage).toBeGreaterThan(0.8)
      const rec = recordsOf(r.exposition, 'conformal')[0]
      expect(rec?.x).toHaveLength(180)
      expect([...(rec?.x ?? [])]).toEqual([...(rec?.x ?? [])].sort((a, b) => a - b))
    })

  it('normalised scores adapt the width to the noise', () => {
    const r = conformalRegression(kNeighborsRegressor({ nNeighbors: 15 }), curve, { method: 'normalised', key: key(5) })
    const rec = recordsOf(r.exposition, 'conformal')[0]
    const w = (rec?.upper ?? []).map((u, i) => u - (rec?.lower?.[i] ?? 0))
    const avg = (v: number[]): number => v.reduce((a, b) => a + b, 0) / v.length
    const q = Math.floor(w.length / 4)
    expect(avg(w.slice(-q))).toBeGreaterThan(1.3 * avg(w.slice(0, q)))
  })

  it('prediction sets hold the true class about 1 - alpha of the time', () => {
    const r = conformalClassification(logisticRegression(), classes, { alpha: 0.1, key: key(6) })
    expect(r.coverage).toBeGreaterThan(0.82)
    const rec = recordsOf(r.exposition, 'conformal')[0]
    if (rec === undefined) throw new Error('no record')
    expect(setSizes(rec).counts.reduce((a, b) => a + b, 0)).toBe(rec.sets?.length)
    const aps = conformalClassification(logisticRegression(), classes, { score: 'aps', key: key(6) })
    expect(aps.meanSize).toBeGreaterThanOrEqual(1)
  })

  it('adaptive conformal inference keeps its running coverage near the target as noise grows', () => {
    const r = onlineConformal(kNeighborsRegressor({ nNeighbors: 15 }), curve, { alpha: 0.1, key: key(7) })
    const rec = recordsOf(r.exposition, 'conformal')[0]
    const s = rec?.stream
    expect(s?.t).toHaveLength(300)
    expect(Math.abs(r.coverage - 0.9)).toBeLessThan(0.05)
    // The fixed interval over-covers the quiet start of the stream.
    expect(s?.baseline?.[60]).toBeGreaterThan((s?.coverage[60] ?? 1) - 0.02)
  })
})

describe('uncertainty, shift and robustness', () => {
  it('the ensemble disagrees more away from the data', () => {
    const r = ensembleUncertainty(kNeighborsRegressor({ nNeighbors: 10 }), curve, { members: 8, key: key(8) })
    expect(r.aleatoric).toBeGreaterThan(0)
    expect(recordsOf(r.exposition, 'curves')[0]?.series.length).toBeGreaterThan(2)
  })

  it('covariate shift is detected; concept shift is not, though accuracy falls', () => {
    const cov = shiftStudy(logisticRegression(), classes, { magnitudes: [0, 1.5], key: key(9) })
    expect(cov.points[1]?.domainAuc).toBeGreaterThan(0.8)
    expect(cov.points[1]?.ksPValue).toBeLessThan(0.01)
    const concept = shiftStudy(logisticRegression(), classes, { kind: 'concept', magnitudes: [0, 0.4], key: key(9) })
    expect(concept.points[1]?.domainAuc).toBeLessThan(0.65)
    expect(concept.points[1]?.accuracy).toBeLessThan((concept.points[0]?.accuracy ?? 0) - 0.15)
  })

  it('accuracy falls with noise and missing values', () => {
    const model = logisticRegression().fit(classes)
    const r = robustnessStudy(model, classes, { noise: [0, 2], missing: [0, 0.8], repeats: 1 })
    expect(r.noise[1]).toBeLessThan(r.noise[0] ?? 0)
    expect(r.missing[1]).toBeLessThan(r.missing[0] ?? 0)
    expect(r.perFeature).toHaveLength(10)
  })

  it('a model card gathers performance, groups, calibration and robustness', () => {
    const credit = generate(biasedCredit, { key: key(10), knobs: { n: 600, labelBias: 0.4 } }).data
    const r = modelCard(logisticRegression().fit(credit), credit, { intendedUse: 'a demo' })
    const headings = r.card.sections.map((s) => s.heading)
    expect(headings).toContain('Performance by group')
    expect(headings).toContain('Caveats')
    expect(r.exposition.map((e) => e.kind)).toEqual(['groupReport', 'calibration', 'curves', 'curves'])
  })
})
