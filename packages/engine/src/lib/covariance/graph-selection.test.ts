import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { gaussianGraph } from '../datasets/index.ts'
import { recordsOf } from '../exposition/index.ts'
import { kFold } from '../learn/model-selection/index.ts'
import { key } from '../random/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { adaptiveGraphicalLasso } from './adaptive-graphical-lasso.ts'
import { edgeRecovery, precisionGraph } from './evaluation.ts'
import {
  GraphicalLassoCV,
  gaussianLogLikelihood,
  graphicalLassoEbic,
  graphicalLassoSelectionAlgorithm,
  graphicalLassoSelectionEstimator,
  type PenaltySelector,
  selectGraphicalLassoPenalty,
  selectionRecords,
} from './graph-selection.ts'

const model = (n: number, p = 6, graphSeed = 0, seed = 1) => {
  const { data, truth } = generate(gaussianGraph, {
    key: key(seed),
    knobs: { n, p, structure: 'band', strength: 0.4, graphSeed },
  })
  return { x: toArray(data.x) as number[][], adjacency: truth.extra?.['adjacency'] as boolean[][] }
}

describe('the Gaussian log-likelihood and the extended BIC', () => {
  it('match the closed forms', () => {
    expect(gaussianLogLikelihood([[1]], [[1]])).toBeCloseTo(-0.5 * (1 + Math.log(2 * Math.PI)), 12)
    expect(
      gaussianLogLikelihood(
        [
          [2, 0],
          [0, 1],
        ],
        [
          [0.5, 0],
          [0, 1],
        ],
      ),
    ).toBeCloseTo((-2 + Math.log(0.5) - 2 * Math.log(2 * Math.PI)) / 2, 12)
    expect(
      gaussianLogLikelihood(
        [
          [1, 0],
          [0, 1],
        ],
        [
          [-1, 0],
          [0, 1],
        ],
      ),
    ).toBe(Number.NEGATIVE_INFINITY)
  })

  it('charges log n per edge, and 4 gamma log p more for each', () => {
    const s = [
      [1, 0.3, 0],
      [0.3, 1, 0],
      [0, 0, 1],
    ]
    const precision = [
      [1.2, -0.4, 0],
      [-0.4, 1.2, 0],
      [0, 0, 1],
    ]
    const base = -2 * 50 * gaussianLogLikelihood(s, precision)
    expect(graphicalLassoEbic(s, precision, 50, 0)).toBeCloseTo(base + Math.log(50), 12)
    expect(graphicalLassoEbic(s, precision, 50, 0.5)).toBeCloseTo(base + Math.log(50) + 2 * Math.log(3), 12)
  })
})

describe('selecting the penalty', () => {
  const { x, adjacency } = model(300)

  it('cross-validation picks the penalty with the best mean held-out score (the smaller on a tie) and refines the grid', () => {
    const out = selectGraphicalLassoPenalty(x, { method: 'cv', cv: kFold(4) })
    expect(out.alphas.length).toBe(16)
    expect([...out.alphas].sort((a, b) => b - a)).toEqual(out.alphas)
    const best = Math.max(...out.criterion)
    expect(out.criterion[out.bestIndex]).toBe(best)
    expect(out.alpha).toBe(out.alphas[out.bestIndex])
    expect(out.scores.length).toBe(16)
    expect(out.scores[0]?.length).toBe(4)
    for (const [i, row] of out.scores.entries())
      expect(row.reduce((a, v) => a + v, 0) / 4).toBeCloseTo(out.criterion[i] ?? 0, 12)
    expect(out.unpenalisedScores.length).toBe(4)
    expect(out.direction).toBe('max')
  })

  it('EBIC picks the penalty with the lowest criterion and recovers the chain better than cross-validation', () => {
    const ebic = selectGraphicalLassoPenalty(x, { method: 'ebic' })
    expect(ebic.criterion[ebic.bestIndex]).toBe(Math.min(...ebic.criterion))
    expect(ebic.direction).toBe('min')
    const cv = selectGraphicalLassoPenalty(x, { method: 'cv' })
    expect(edgeRecovery(ebic.graph, adjacency).f1).toBeGreaterThan(edgeRecovery(cv.graph, adjacency).f1)
    expect(edgeRecovery(ebic.graph, adjacency).recall).toBe(1)
  })

  it('StARS starts from an empty graph, has the sparsest graph under the instability bound and is reproducible', () => {
    const a = selectGraphicalLassoPenalty(x, { method: 'stars', nSubsamples: 10 }, key(4))
    const b = selectGraphicalLassoPenalty(x, { method: 'stars', nSubsamples: 10 }, key(4))
    expect(a.instability[0]).toBe(0)
    expect(a.edges[0]).toBe(0)
    expect(a.alpha).toBe(b.alpha)
    expect(a.instability).toEqual(b.instability)
    // The chosen penalty is the smallest whose monotone instability is within beta.
    expect(a.criterion[a.bestIndex]).toBeLessThanOrEqual(0.05)
    if (a.bestIndex + 1 < a.alphas.length) expect(a.criterion[a.bestIndex + 1]).toBeGreaterThan(0.05)
    expect(selectGraphicalLassoPenalty(x, { method: 'stars', nSubsamples: 10, beta: 1 }, key(4)).alpha).toBe(
      Math.min(...a.alphas),
    )
  })

  it('a stricter instability bound gives a sparser graph', () => {
    const loose = selectGraphicalLassoPenalty(x, { method: 'stars', nSubsamples: 10, beta: 0.2 }, key(2))
    const strict = selectGraphicalLassoPenalty(x, { method: 'stars', nSubsamples: 10, beta: 0.02 }, key(2))
    expect(strict.alpha).toBeGreaterThanOrEqual(loose.alpha)
  })

  it('an explicit grid is used as given (largest first) and not refined; the result carries its path of precisions', () => {
    const out = selectGraphicalLassoPenalty(x, { method: 'ebic', alphas: [0.01, 0.3, 0.1] })
    expect(out.alphas).toEqual([0.3, 0.1, 0.01])
    expect(out.precisions.length).toBe(3)
    expect(out.edges[0]).toBeLessThanOrEqual(out.edges[2] ?? 0)
    expect(selectGraphicalLassoPenalty(x, { method: 'cv', alphas: [0.3, 0.1], nRefinements: 5 }).alphas.length).toBe(2)
    expect(selectGraphicalLassoPenalty(x, { method: 'ebic', alphas: 5 }).alphas.length).toBe(5)
  })

  it('the refit at the chosen penalty is the graphical lasso there, and its graph is the support of its precision', () => {
    const out = selectGraphicalLassoPenalty(x, { method: 'ebic' })
    expect(out.precision.length).toBe(6)
    for (const [a, row] of out.graph.entries())
      for (const [b, v] of row.entries()) expect(v).toBe(a !== b && Math.abs(out.precision[a]?.[b] ?? 0) > 1e-10)
    expect(out.nIter).toBeGreaterThan(0)
  })

  it('recovers the true graph better as the sample grows: the chosen penalty finds every true edge, and the adaptive refit at it removes the extra ones', () => {
    const rows = [60, 600, 6000].map((n) => {
      const m = model(n, 8)
      const chosen = selectGraphicalLassoPenalty(m.x, { method: 'ebic' })
      const s = toArray(empiricalCovariance(m.x)) as number[][]
      const adaptive = adaptiveGraphicalLasso(s, chosen.alpha)
      return {
        plain: edgeRecovery(chosen.graph, m.adjacency),
        adaptive: edgeRecovery(precisionGraph(adaptive.precision), m.adjacency),
      }
    })
    expect(rows.every((r) => r.plain.recall === 1)).toBe(true)
    expect(rows[2]?.adaptive.f1).toBe(1)
    expect(rows[2]?.adaptive.f1).toBeGreaterThanOrEqual(rows[0]?.adaptive.f1 ?? 1)
    expect(rows[0]?.adaptive.f1).toBeGreaterThan(rows[0]?.plain.f1 ?? 1)
  })

  it('rejects too few rows, uncorrelated data and a bad grid', () => {
    expect(() =>
      selectGraphicalLassoPenalty([
        [1, 2],
        [2, 1],
      ]),
    ).toThrow(/at least 4/)
    expect(() =>
      selectGraphicalLassoPenalty(
        [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1],
        ],
        { method: 'ebic' },
      ),
    ).toThrow(/uncorrelated/)
    expect(() => selectGraphicalLassoPenalty(x, { alphas: [0.1, -1] })).toThrow(/non-negative/)
    expect(() => selectGraphicalLassoPenalty(x, { alphas: 0 })).toThrow(/count/)
  })
})

describe('the selection as a traced run', () => {
  const { x } = model(120)

  it('steps through the penalties and a final refit, recording the penalty, criterion, edges and gap', () => {
    const algorithm = graphicalLassoSelectionAlgorithm(x, { method: 'ebic', alphas: [0.4, 0.2, 0.1] })
    const run = runAlgorithm(algorithm, { maxSteps: 20 })
    expect(run.steps).toBe(4)
    expect(run.status).toBe('converged')
    expect(run.state.stage).toBe('done')
    expect(run.state.entries.map((e) => e.alpha)).toEqual([0.4, 0.2, 0.1])
    const restored = algorithm.deserialise?.(algorithm.serialise?.(run.state) ?? null)
    expect(restored?.chosen).toBe(run.state.chosen)
  })

  it('the estimator exposes the criterion curve and the partial-correlation path, with the choice marked', () => {
    for (const method of ['cv', 'ebic', 'stars'] as PenaltySelector[]) {
      const fitted = graphicalLassoSelectionEstimator({
        method,
        folds: 3,
        nSubsamples: 6,
        alphas: [0.4, 0.2, 0.1, 0.05],
      }).fit({
        x,
      })
      const [curve] = recordsOf(fitted.exposition, 'crossValidationCurve')
      const [path] = recordsOf(fitted.exposition, 'regularisationPath')
      expect(curve?.values).toEqual([0.4, 0.2, 0.1, 0.05])
      expect(curve?.best).toBe(fitted.params.alpha)
      expect(path?.chosen).toBe(fitted.params.alpha)
      expect(path?.features.length).toBe(15)
      expect(fitted.params.method).toBe(method)
      expect(fitted.run.events.length).toBeGreaterThan(0)
      expect(Number.isFinite(fitted.score({ x }))).toBe(true)
    }
  })

  it('GraphicalLassoCV is the estimator for the cross-validated rule, and records carry feature names', () => {
    const fitted = new GraphicalLassoCV({ folds: 3, alphas: [0.3, 0.1] }).fit(x).model
    expect(fitted.params.method).toBe('cv')
    const sel = selectGraphicalLassoPenalty(x, { method: 'ebic', alphas: [0.3, 0.1] })
    expect(selectionRecords(sel, ['a', 'b', 'c', 'd', 'e', 'f'])[1].features[0]).toBe('a~b')
    const session = graphicalLassoSelectionEstimator({ method: 'ebic', alphas: [0.3, 0.1] }).session(
      { x },
      { mode: 'eager' },
    )
    expect(session.fittedAt(session.t).params.alphas.length).toBeGreaterThan(0)
  })

  it('the covariance of the data sets the top of the default grid', () => {
    const cov = toArray(empiricalCovariance(x)) as number[][]
    let top = 0
    for (let a = 0; a < 6; a++) for (let b = 0; b < 6; b++) if (a !== b) top = Math.max(top, Math.abs(cov[a]?.[b] ?? 0))
    expect(selectGraphicalLassoPenalty(x, { method: 'ebic', alphas: 3 }).alphas[0]).toBeCloseTo(top, 12)
  })
})
