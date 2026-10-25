import { describe, expect, it } from 'vitest'
import { diabetes } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { matrixView, takeColumns, takeData } from '../../lib/learn/index.ts'
import { key } from '../../lib/random/index.ts'
import { fromStorage } from '../../lib/tensor/index.ts'
import { randomForestRegressor } from '../ensemble/index.ts'
import { decisionTreeRegressor } from '../trees/index.ts'
import { aleEffects, dependencePlot, dropColumnImportances, featureImportance } from './inspection.ts'

const full = diabetes()
const data = takeData(
  full,
  Array.from({ length: 120 }, (_, i) => i),
)
const forest = randomForestRegressor({ nEstimators: 6, maxDepth: 3 }).fit(data, { key: key(1) })

describe('global effects', () => {
  it('partial dependence with centred ICE curves and a surface', () => {
    const r = dependencePlot(forest, data, {
      features: [2, 8],
      pairs: [[2, 8]],
      transform: 'centred',
      gridResolution: 8,
      maxRows: 20,
    })
    const [rec] = recordsOf(r.exposition, 'dependence')
    expect(rec?.curves.length).toBe(2)
    expect(rec?.curves[0]?.individual?.length).toBe(20)
    for (const c of rec?.curves[0]?.individual ?? []) expect(c[0]).toBe(0)
    expect(rec?.surfaces?.[0]?.values.length).toBe(rec?.surfaces?.[0]?.gridX.length)
    expect(rec?.features[2]).toBe('bmi')
  })

  it('ALE of first and second order', () => {
    const r = aleEffects(forest, data, { features: [2], pairs: [[2, 3]], bins: 5 })
    const [rec] = recordsOf(r.exposition, 'dependence')
    expect(rec?.method).toBe('ale')
    expect(rec?.curves[0]?.counts?.length).toBe((rec?.curves[0]?.grid.length ?? 0) - 1)
    expect(rec?.surfaces?.[0]?.counts).toBeDefined()
  })

  it('drop-column and conditional importances as importance records', () => {
    const four = takeColumns(matrixView(data.x, 'test'), [0, 1, 2, 3])
    const d = {
      ...data,
      x: fromStorage(four.data, [four.n, 4], 'float64'),
      featureNames: data.featureNames.slice(0, 4),
    }
    const drop = dropColumnImportances(decisionTreeRegressor({ maxDepth: 3 }), d)
    expect(recordsOf(drop.exposition, 'importances')[0]?.method).toBe('drop-column')
    const cond = featureImportance(forest, data, { method: 'conditional', nRepeats: 2, key: key(3) })
    expect(recordsOf(cond.exposition, 'importances')[0]?.method).toBe('conditional permutation')
    expect(cond.mean.length).toBe(10)
  })
})
