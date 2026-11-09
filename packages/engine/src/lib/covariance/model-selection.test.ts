import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blockCovarianceData, factorCovarianceData, spikedCovarianceData } from '../datasets/index.ts'
import { key, normal } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  type CovarianceCandidate,
  CovarianceSelectionEstimator,
  covarianceSelectionEstimator,
  defaultCovarianceCandidates,
  heldOutLogLikelihood,
  selectCovariance,
  selectCovarianceAlgorithm,
} from './model-selection.ts'

const NAMES = [
  'empirical',
  'ledoit-wolf',
  'oas',
  'diagonal',
  'constant-correlation',
  'single-factor',
  'nonlinear',
  'denoised',
]

describe('heldOutLogLikelihood', () => {
  it('is the mean Gaussian log density of the rows', () => {
    expect(
      heldOutLogLikelihood(
        [[0, 0]],
        [0, 0],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toBeCloseTo(-Math.log(2 * Math.PI), 14)
    const v = heldOutLogLikelihood(
      [
        [1, 0],
        [-1, 0],
      ],
      [0, 0],
      [
        [2, 0],
        [0, 1],
      ],
    )
    expect(v).toBeCloseTo(-0.5 * (0.5 + Math.log(2) + 2 * Math.log(2 * Math.PI)), 12)
  })

  it('is -Infinity for a singular covariance and checks the columns', () => {
    expect(
      heldOutLogLikelihood(
        [[1, 1]],
        [0, 0],
        [
          [1, 1],
          [1, 1],
        ],
      ),
    ).toBe(Number.NEGATIVE_INFINITY)
    expect(() =>
      heldOutLogLikelihood(
        [[1, 1, 1]],
        [0, 0],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toThrow(/expected 2 features/)
  })
})

describe('defaultCovarianceCandidates', () => {
  it('names the empirical covariance, the shrinkages, the nonlinear shrinkage and the denoised matrix', () => {
    expect(defaultCovarianceCandidates().map((c) => c.name)).toEqual(NAMES)
  })
})

describe('selectCovariance', () => {
  it('picks the candidate with the best mean score and refits it on all the data', () => {
    const { data } = generate(blockCovarianceData, {
      key: key(1),
      knobs: { n: 80, p: 12, blocks: 1, within: 0.5, between: 0 },
    })
    const r = selectCovariance(data.x, undefined, { folds: 4 })
    expect(r.scores.map((s) => s.name)).toEqual(NAMES)
    const finite = r.scores.filter((s) => Number.isFinite(s.mean))
    expect(r.best).toBe(finite.reduce((a, s) => (s.mean > a.mean ? s : a)).name)
    expect(r.covariance).toHaveLength(12)
    expect(r.location).toHaveLength(12)
    for (const s of r.scores) expect(s.folds).toHaveLength(4)
  })

  it('prefers a shrinkage to the empirical covariance when the sample is short and the truth is simple', () => {
    const x = normal(key(2), [40, 25])
    const r = selectCovariance(x, undefined, { folds: 5 })
    const empirical = r.scores.find((s) => s.name === 'empirical')?.mean ?? Number.NEGATIVE_INFINITY
    const best = r.scores.find((s) => s.name === r.best)?.mean ?? Number.NEGATIVE_INFINITY
    expect(best).toBeGreaterThan(empirical)
    expect(r.best).not.toBe('empirical')
  })

  it('scores the empirical covariance -Infinity and still selects a candidate when there are more variables than rows', () => {
    const { data } = generate(factorCovarianceData, { key: key(3), knobs: { n: 24, p: 40, factors: 2 } })
    const r = selectCovariance(data.x, undefined, { folds: 4 })
    expect(r.scores.find((s) => s.name === 'empirical')?.mean).toBe(Number.NEGATIVE_INFINITY)
    expect(r.best).not.toBe('')
    expect(r.best).not.toBe('empirical')
  })

  it('selects the nonlinear or denoised candidate for strongly spiked data with many variables', () => {
    const { data } = generate(spikedCovarianceData, { key: key(4), knobs: { n: 120, p: 60, spikes: 3, strength: 12 } })
    const r = selectCovariance(data.x, undefined, { folds: 5 })
    expect(['nonlinear', 'denoised']).toContain(r.best)
  })

  it('compares a given list, and a shuffling key changes the folds but not the rank structure', () => {
    const x = normal(key(5), [60, 6])
    const candidates: CovarianceCandidate[] = defaultCovarianceCandidates().filter(
      (c) => c.name === 'empirical' || c.name === 'ledoit-wolf',
    )
    const a = selectCovariance(x, candidates, { folds: 3 })
    const b = selectCovariance(x, candidates, { folds: 3, key: key(9) })
    expect(a.scores.map((s) => s.name)).toEqual(['empirical', 'ledoit-wolf'])
    expect(a.scores[0]?.folds).not.toEqual(b.scores[0]?.folds)
    expect(selectCovariance(x, candidates, { folds: 3, key: key(9) }).scores[0]?.folds).toEqual(b.scores[0]?.folds)
  })

  it('marks a candidate whose fit throws with NaN and does not select it', () => {
    const x = normal(key(6), [30, 4])
    const broken: CovarianceCandidate = {
      name: 'broken',
      fit: () => {
        throw new Error('no fit')
      },
    }
    const r = selectCovariance(x, [broken, ...defaultCovarianceCandidates().slice(0, 2)], { folds: 3 })
    expect(Number.isNaN(r.scores[0]?.mean ?? 0)).toBe(true)
    expect(r.best).not.toBe('broken')
  })

  it('rejects no candidates, a single fold and too few rows', () => {
    const x = normal(key(7), [20, 3])
    expect(() => selectCovariance(x, [])).toThrow(/candidate/)
    expect(() => selectCovariance(x, undefined, { folds: 1 })).toThrow(/folds/)
    expect(() =>
      selectCovariance([
        [1, 2],
        [3, 4],
      ]),
    ).toThrow(/at least 4/)
  })
})

describe('selectCovarianceAlgorithm', () => {
  it('scores one candidate per step and refits the winner in a last step, the best score never falling', () => {
    const x = normal(key(8), [50, 8])
    const out = runAlgorithm(selectCovarianceAlgorithm(x, undefined, { folds: 5 }), { maxSteps: 20 })
    expect(out.steps).toBe(NAMES.length + 1)
    expect(out.state.scores).toHaveLength(NAMES.length)
    expect(out.state.done).toBe(true)
    expect(out.state.covariance).toHaveLength(8)
    let best = Number.NEGATIVE_INFINITY
    for (let t = 1; t <= NAMES.length; t++) {
      const state = runAlgorithm(selectCovarianceAlgorithm(x, undefined, { folds: 5 }), { maxSteps: t }).state
      const top = state.scores.find((s) => s.name === state.best)?.mean ?? Number.NEGATIVE_INFINITY
      expect(top).toBeGreaterThanOrEqual(best)
      best = top
    }
  })
})

describe('covarianceSelectionEstimator', () => {
  const { data } = generate(spikedCovarianceData, { key: key(9), knobs: { n: 100, p: 20, spikes: 2, strength: 10 } })

  it('fits the winner with its scores and the verbs', () => {
    const fit = covarianceSelectionEstimator({ folds: 4 }).fit({ x: data.x })
    expect(fit.params.candidates).toEqual(NAMES)
    expect(NAMES).toContain(fit.params.selected)
    expect(fit.params.scores).toHaveLength(NAMES.length)
    expect(fit.params.covariance).toHaveLength(20)
    expect(Number.isFinite(fit.score({ x: data.x }))).toBe(true)
    expect(fit.exposition.map((r) => r.kind)).toEqual(['importances', 'anomalyScores'])
  })

  it('works as a class', () => {
    expect(new CovarianceSelectionEstimator({ folds: 3 }).fit(data.x).model.params.selected).toBeTypeOf('string')
  })
})
