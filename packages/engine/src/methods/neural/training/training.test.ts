import { describe, expect, it } from 'vitest'
import { datasetCollections, loadEntry } from '../../../lib/datasets/index.ts'
import { recordsOf } from '../../../lib/exposition/index.ts'
import type { LearnData } from '../../../lib/learn/index.ts'
import { key } from '../../../lib/random/index.ts'
import { initialisationStudy, normalisationStudy } from './initialisation.ts'
import { learningRateFinder, optimiserRace } from './optimisers.ts'
import { regularisationStudy } from './regularisation.ts'

const moons = (n: number): LearnData => {
  const entry = datasetCollections['classification2d']?.entries.find((e) => e.name === 'moons')
  if (entry === undefined) throw new Error('no moons')
  return loadEntry(entry, key(0), { n }).data as unknown as LearnData
}

describe('optimiserRace', () => {
  it('races the optimisers from the same start and records each', () => {
    const r = optimiserRace(moons(80), { epochs: 15, hiddenLayerSizes: [8] }, { key: key(0) })
    expect(r.results.map((x) => x.optimiser)).toEqual(['sgd', 'momentum', 'rmsprop', 'adam'])
    for (const x of r.results) expect(x.losses.length).toBe(15)
    // Adam leaves SGD behind on the same budget.
    const loss = (name: string): number => r.results.find((x) => x.optimiser === name)?.finalLoss ?? 0
    expect(loss('adam')).toBeLessThan(loss('sgd'))
    expect(recordsOf(r.exposition, 'decisionSurface').length).toBe(4)
    expect(recordsOf(r.exposition, 'decisionSurface')[3]?.steps?.at(-1)?.iteration).toBe(15)
    expect(recordsOf(r.exposition, 'learningCurve').length).toBe(4)
    expect(recordsOf(r.exposition, 'searchTrials')[0]?.trials.length).toBe(4)
  })
})

describe('learningRateFinder', () => {
  it('sweeps the rate geometrically and suggests one inside the range', () => {
    const r = learningRateFinder(moons(80), { steps: 60, hiddenLayerSizes: [8] }, { key: key(0) })
    expect(r.learningRates.length).toBe(r.losses.length)
    expect(r.learningRates[0]).toBeCloseTo(1e-5, 12)
    for (let i = 1; i < r.learningRates.length; i++)
      expect(r.learningRates[i] ?? 0).toBeGreaterThan(r.learningRates[i - 1] ?? 0)
    expect(r.steepest).toBeGreaterThan(1e-5)
    expect(r.steepest).toBeLessThan(10)
    expect(recordsOf(r.exposition, 'learningCurve')[0]?.resource).toBe('learning rate')
  })
})

describe('initialisationStudy', () => {
  it('shows activations vanishing under a small initialisation and not under Xavier', () => {
    const r = initialisationStudy(
      moons(60),
      { schemes: ['small', 'xavier'], depth: 4, width: 8, epochs: 3 },
      { key: key(0) },
    )
    const [small, xavier] = r.schemes
    expect(small?.activationStd.length).toBe(5)
    expect(small?.activationStd.at(-1) ?? 1).toBeLessThan(1e-4)
    expect(xavier?.activationStd.at(-1) ?? 0).toBeGreaterThan(0.05)
    expect(small?.gradientNorm.every((g) => g < (xavier?.gradientNorm[0] ?? 0))).toBe(true)
    expect(recordsOf(r.exposition, 'learningCurve').length).toBe(4)
  })
})

describe('normalisationStudy', () => {
  it('trains with batch and layer normalisation', () => {
    const r = normalisationStudy(moons(60), { depth: 2, width: 8, epochs: 3 }, { key: key(0) })
    expect(r.results.map((x) => x.norm)).toEqual(['none', 'batch', 'layer'])
    for (const x of r.results) expect(Number.isFinite(x.finalLoss)).toBe(true)
    expect(recordsOf(r.exposition, 'decisionSurface').length).toBe(3)
  })
})

describe('regularisationStudy', () => {
  it('records the validation curve and the train-validation gap for dropout and weight decay', () => {
    const data = moons(80)
    const r = regularisationStudy(data, { values: [0, 0.5], hiddenLayerSizes: [16], epochs: 20 }, { key: key(0) })
    expect(r.results.length).toBe(2)
    const [curve] = recordsOf(r.exposition, 'learningCurve')
    expect(curve?.sizes).toEqual([0, 0.5])
    expect(curve?.resource).toBe('dropout rate')
    const wd = regularisationStudy(data, { kind: 'weightDecay', values: [0, 1], hiddenLayerSizes: [16], epochs: 20 })
    // A strong penalty fits the training rows less closely.
    expect(wd.results[1]?.trainLoss ?? 0).toBeGreaterThan(wd.results[0]?.trainLoss ?? 1)
  })
})
