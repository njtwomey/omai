import { describe, expect, it } from 'vitest'
import { diabetes, digits } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { takeData } from '../../lib/learn/index.ts'
import { key } from '../../lib/random/index.ts'
import { randomForestClassifier, randomForestRegressor } from '../ensemble/index.ts'
import { explainImage, explainSeries, explainText, morrisScreening } from './perturbation.ts'

const images = takeData(
  digits(),
  Array.from({ length: 200 }, (_, i) => i),
)
const forest = randomForestClassifier({ nEstimators: 8, maxDepth: 6 }).fit(images, { key: key(1) })

describe('perturbation methods', () => {
  it('explains an image by occlusion, superpixel SHAP and RISE, as image-laid records', () => {
    for (const method of ['occlusion', 'shap', 'rise', 'lime', 'ablation'] as const) {
      const r = explainImage(forest, images, 3, { method, nMasks: 40, nSamples: 60, key: key(2) })
      const [rec] = recordsOf(r.exposition, 'featureAttributions')
      expect(rec?.layout).toEqual({ kind: 'image', shape: [8, 8] })
      expect(rec?.values[0]?.length).toBe(64)
      expect(rec?.output.startsWith('P(')).toBe(true)
    }
  })

  it('explains a text word by word', () => {
    const score = (texts: readonly string[]) => texts.map((t) => (t.includes('great') ? 0.9 : 0.2))
    const r = explainText(score, 'A great little film', { method: 'shap' })
    expect(r.tokens).toEqual(['A', 'great', 'little', 'film'])
    expect(r.values[1]).toBeCloseTo(0.7, 9)
    expect(recordsOf(r.exposition, 'featureAttributions')[0]?.layout?.kind).toBe('tokens')
  })

  it('explains a series by windows and screens features with Morris', () => {
    const reg = takeData(
      diabetes(),
      Array.from({ length: 80 }, (_, i) => i),
    )
    const model = randomForestRegressor({ nEstimators: 4, maxDepth: 3 }).fit(reg, { key: key(3) })
    const s = explainSeries(model, reg, 0, { method: 'ablation', window: 3 })
    expect(recordsOf(s.exposition, 'featureAttributions')[0]?.segments?.length).toBe(10)
    const m = morrisScreening(model, reg, { nTrajectories: 5 })
    expect(recordsOf(m.exposition, 'importances')[0]?.repeats?.[0]?.length).toBe(5)
  })
})
