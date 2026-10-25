import { describe, expect, it } from 'vitest'
import { diabetes, iris } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { gradientBoostingClassifier, randomForestClassifier, randomForestRegressor } from '../ensemble/index.ts'
import { decisionTreeRegressor } from '../trees/index.ts'
import { explainDataset, explainPrediction } from './attributions.ts'
import { featureImportance } from './inspection.ts'

const reg = diabetes()
const forest = randomForestRegressor({ nEstimators: 8, maxDepth: 4 }).fit(reg, { key: key(1) })
const sum = (v: readonly number[]): number => v.reduce((a, b) => a + b, 0)

describe('explainPrediction', () => {
  it('tree SHAP of a forest adds up to its prediction, with named features', () => {
    const r = explainPrediction(forest, reg, 3)
    expect(r.method).toBe('tree')
    expect(r.baseValue + sum(r.values)).toBeCloseTo(toFlat(forest.predict(reg.x))[3] ?? 0, 8)
    const [rec] = recordsOf(r.exposition, 'featureAttributions')
    expect(rec?.features[2]).toBe('bmi')
    expect(rec?.indices).toEqual([3])
    expect(rec?.output).toBe('prediction')
  })

  it('kernel SHAP (exhaustive for four features) matches the exact explainer', () => {
    const data = iris()
    const model = randomForestClassifier({ nEstimators: 5, maxDepth: 3 }).fit(data, { key: key(2) })
    const exact = explainPrediction(model, data, 120, { method: 'exact', background: 20 })
    const kernel = explainPrediction(model, data, 120, { method: 'kernel', background: 20 })
    for (const [j, v] of kernel.values.entries()) expect(v).toBeCloseTo(exact.values[j] ?? 0, 8)
    expect(exact.baseValue + sum(exact.values)).toBeCloseTo(exact.prediction, 8)
    expect(recordsOf(exact.exposition, 'featureAttributions')[0]?.output).toBe('P(virginica)')
  })

  it('sampling SHAP records its running estimate, ending at the values', () => {
    const tree = decisionTreeRegressor({ maxDepth: 3 }).fit(reg)
    const r = explainPrediction(tree, reg, 0, { method: 'sampling', nPermutations: 30, background: 10 })
    const c = recordsOf(r.exposition, 'featureAttributions')[0]?.convergence
    expect(c?.samples.at(-1)).toBe(30)
    expect(c?.estimates.at(-1)).toEqual(r.values)
    expect(c?.standardErrors?.[0]).toHaveLength(10)
  })

  it('LIME reports its local fit and starts from its intercept', () => {
    const r = explainPrediction(forest, reg, 5, { method: 'lime', nSamples: 300 })
    const rec = recordsOf(r.exposition, 'featureAttributions')[0]
    expect(rec?.localFit?.[0]).toBeGreaterThan(0)
    expect(rec?.baseValues?.[0]).toBe(r.baseValue)
  })

  it('a booster is explained in log odds by its trees', () => {
    const data = { ...iris(), y: toFlat(iris().y).map((c) => (c === 2 ? 1 : 0)) }
    const gb = gradientBoostingClassifier({ nEstimators: 10, maxDepth: 2 }).fit(data)
    const r = explainPrediction(gb, data, 140)
    const raw = toFlat(gb.decisionFunction(data.x))[140] ?? 0
    expect(r.baseValue + sum(r.values)).toBeCloseTo(raw, 8)
    expect(() => explainPrediction(gb, data, 0, { method: 'tree', output: 'probability' })).toThrow(/tree explainer/)
  })
})

describe('explainDataset and featureImportance', () => {
  it('explains a sample of rows for the summary views', () => {
    const r = explainDataset(forest, reg, { maxRows: 25 })
    const rec = recordsOf(r.exposition, 'featureAttributions')[0]
    expect(rec?.values).toHaveLength(25)
    expect(rec?.data[0]).toHaveLength(10)
    expect(rec?.indices).toEqual(r.rows)
    expect(rec?.partners).toHaveLength(10)
  })

  it('measures permutation and impurity importances', () => {
    const p = featureImportance(forest, reg, { nRepeats: 2 })
    const rec = recordsOf(p.exposition, 'importances')[0]
    expect(rec?.repeats?.[0]).toHaveLength(2)
    expect(rec?.metric).toBe('R²')
    // bmi and s5 carry the diabetes signal.
    const top = rec?.mean.indexOf(Math.max(...(rec?.mean ?? [])))
    expect([2, 8]).toContain(top)
    const imp = featureImportance(forest, reg, { method: 'impurity' })
    expect(sum(imp.mean)).toBeCloseTo(1, 6)
  })
})
