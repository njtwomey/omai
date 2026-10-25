import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { fromStorage, matmul, type Tensor, toFlat, type Value } from '../tensor/index.ts'
import { conceptActivationVector, directionalDerivatives, linearProbe, tcav } from './concepts.ts'
import { explainableBoosting, glassBoxContributions, glassBoxPredict, scorecard, termImportances } from './glassbox.ts'
import { linearityMeasure, ruleFit, ruleFitPredict, ruleMask, treeRules } from './rules.ts'
import { betaShapley, dataBanzhaf, representerValues, subsetUtility } from './valuation.ts'

const x = Array.from({ length: 120 }, (_, i) => [Math.sin(i * 1.3) * 2, Math.cos(i * 0.7) * 2, (i % 5) - 2])
const y = x.map(([a = 0, b = 0, c = 0]) => a * a + (b > 0 ? 2 : 0) + 0.5 * c + (a > 0 && c > 0 ? 1.5 : 0))

describe('glass-box models', () => {
  it('explainable boosting learns the shape functions and a pairwise term, its contributions summing to the score', () => {
    const m = explainableBoosting({ x, y }, { bins: 16, rounds: 150, learningRate: 0.2, interactions: 1, pairBins: 4 })
    const pred = glassBoxPredict(m, x)
    const r2 = 1 - pred.reduce((s, p, i) => s + (p - (y[i] ?? 0)) ** 2, 0) / y.reduce((s, v) => s + (v - 1.5) ** 2, 0)
    expect(r2).toBeGreaterThan(0.9)
    expect(m.terms.at(-1)?.features).toEqual([0, 2])
    const c = glassBoxContributions(m, x.slice(0, 3))
    c.forEach((row, i) => {
      expect(row.reduce((a, b) => a + b, m.intercept)).toBeCloseTo(pred[i] ?? 0, 10)
    })
    // Every term is centred over the data.
    const all = glassBoxContributions(m, x)
    for (let t = 0; t < m.terms.length; t++)
      expect(all.reduce((s, r) => s + (r[t] ?? 0), 0) / x.length).toBeCloseTo(0, 8)
    expect(termImportances(m, x)[2] ?? 1).toBeLessThan(termImportances(m, x)[0] ?? 0)
  })

  it('classification boosting gives probabilities and a scorecard tracks the log odds', () => {
    const yc = x.map(([a = 0]) => (a > 0 ? 1 : 0))
    const m = explainableBoosting({ x, y: yc }, { task: 'classification', bins: 8, rounds: 60 })
    const p = glassBoxPredict(m, x)
    expect(p.every((v) => v > 0 && v < 1)).toBe(true)
    expect(p.filter((v, i) => (v > 0.5 ? 1 : 0) === yc[i]).length / p.length).toBeGreaterThan(0.95)
    const card = scorecard([1, 0, 0], 0, x, { bins: 4, pdo: 20, baseScore: 600, baseOdds: 1 })
    const pts = card.features[0]?.points ?? []
    for (let b = 1; b < pts.length; b++) expect(pts[b] ?? 0).toBeGreaterThan(pts[b - 1] ?? 0)
    expect(card.basePoints).toBeCloseTo(600, 9)
  })
})

describe('rules', () => {
  it('tree rules cover their nodes and RuleFit recovers a step and a slope', () => {
    const stump = {
      childrenLeft: [1, -1, -1],
      childrenRight: [2, -1, -1],
      feature: [0, -2, -2],
      threshold: [0.5, 0, 0],
      value: [0.5, 0, 1],
      cover: [2, 1, 1],
    }
    const rules = treeRules(stump)
    expect(ruleMask(rules[0] ?? [], [[0], [1]])).toEqual([1, 0])
    const ys = x.map(([a = 0, , c = 0]) => (a > 1 ? 3 : 0) + 0.5 * c)
    const model = ruleFit({ x, y: ys }, { nTrees: 10, key: key(1) })
    const pred = ruleFitPredict(model, x)
    const mse = pred.reduce((s, p, i) => s + (p - (ys[i] ?? 0)) ** 2, 0) / x.length
    expect(mse).toBeLessThan(0.3)
    expect(model.terms.some((t) => t.feature === 2)).toBe(true)
  })

  it('the linearity measure is zero for linear models and grows with curvature', () => {
    const sq = (v: Tensor): Tensor => {
      const d = toFlat(v)
      return fromStorage(
        Float64Array.from({ length: v.shape[0] ?? 0 }, (_, i) => (d[i] ?? 0) ** 2),
        [v.shape[0] ?? 0],
        'float64',
      )
    }
    expect(linearityMeasure(sq, [1], { epsilon: 0.5, key: key(2) })).toBeGreaterThan(0.01)
    expect(linearityMeasure(sq, [1], { epsilon: 0.05, key: key(2) })).toBeLessThan(
      linearityMeasure(sq, [1], { epsilon: 0.5, key: key(2) }),
    )
  })
})

describe('concepts', () => {
  it('probes, CAVs and TCAV separate a concept the output depends on from one it ignores', () => {
    const probe = linearProbe(
      [
        [0, 1],
        [0, 2],
        [1, 0],
        [2, 0],
      ],
      [1, 1, 0, 0],
    )
    expect(probe.weights[1] ?? 0).toBeGreaterThan(0)
    const tail = (h: Value): Value => matmul(h, [[1], [0]])
    const cls = Array.from({ length: 10 }, (_, i) => [i / 10, -i / 10])
    const randoms = Array.from({ length: 12 }, (_, r) =>
      Array.from({ length: 8 }, (_, i) => [Math.sin(r * 7.1 + i * 1.7), Math.cos(r * 3.3 + i * 2.9)]),
    )
    const along = tcav(
      tail,
      cls,
      Array.from({ length: 8 }, (_, i) => [3 + i / 8, Math.sin(i)]),
      randoms,
    )
    expect(along.mean).toBe(1)
    const against = tcav(
      tail,
      cls,
      Array.from({ length: 8 }, (_, i) => [-3 - i / 8, Math.sin(i)]),
      randoms,
    )
    expect(against.mean).toBe(0)
    expect(against.pValue).toBeLessThan(0.05)
    const cav = conceptActivationVector(
      [
        [0, 3],
        [0.2, 4],
      ],
      [
        [0, 0],
        [0.1, -0.2],
      ],
    )
    expect(directionalDerivatives(tail, [[1, 1]], cav.vector)[0] ?? 1).toBeCloseTo(cav.vector[0] ?? 0, 12)
  })
})

describe('data attribution additions', () => {
  it('Beta Shapley and data Banzhaf rank a harmful point last, and representer values decompose a prediction', () => {
    const worth = [1, 0.5, -2, 0.2]
    const u = (rows: readonly number[]): number =>
      rows.reduce((s, i) => s + (worth[i] ?? 0), 0) + (rows.length > 2 ? 0.3 : 0)
    const beta = betaShapley(u, 4, { nPermutations: 200, key: key(4) })
    expect(beta.values.indexOf(Math.min(...beta.values))).toBe(2)
    const banzhaf = dataBanzhaf(u, 4, { nSamples: 600, key: key(5) })
    expect(banzhaf.values.indexOf(Math.min(...banzhaf.values))).toBe(2)
    const r = representerValues(
      [
        [1, 2],
        [3, -1],
      ],
      [0.5, -1],
      [[1, 1]],
      0.1,
    )
    expect(r.values[0]?.reduce((a, b) => a + b, 0)).toBeCloseTo((-0.5 / 0.4) * 3 + (1 / 0.4) * 2, 10)
    expect(() =>
      subsetUtility(
        {
          name: 'x',
          kind: 'regressor',
          hyper: {},
          fit: () => {
            throw new Error('no')
          },
          revive: () => {
            throw new Error('no')
          },
          with: () => {
            throw new Error('no')
          },
        } as never,
        { x: [[0]], y: [0] },
        { x: [[0]], y: [0] },
        { scoring: 'r2', emptyUtility: -1 },
      )([0]),
    ).not.toThrow()
  })
})
