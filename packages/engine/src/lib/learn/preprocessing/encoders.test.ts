import { describe, expect, it } from 'vitest'
import { rows } from '../../../testing/learn.ts'
import { revive, serialiseFitted } from '../estimator.ts'
import { labelEncoder, oneHotEncoder, ordinalEncoder, targetEncoder } from './encoders.ts'

describe('encoders', () => {
  it('raise on unknown categories by default', () => {
    expect(() =>
      oneHotEncoder()
        .fit({ x: [[1], [2]] })
        .transform([[3]]),
    ).toThrow(/unknown category/)
    expect(() =>
      ordinalEncoder()
        .fit({ x: [[1], [2]] })
        .transform([[3]]),
    ).toThrow(/unknown category/)
    expect(() =>
      labelEncoder()
        .fit({ x: [[0]], y: [1, 2] })
        .transform([5]),
    ).toThrow(/unseen/)
  })

  it('keep NaN as a category through JSON and revive', () => {
    const est = oneHotEncoder()
    const fitted = est.fit({ x: [[1], [Number.NaN], [2]] })
    const again = revive(est, JSON.parse(JSON.stringify(serialiseFitted(fitted))))
    expect(rows(again.transform([[Number.NaN], [2]]))).toEqual([
      [0, 0, 1],
      [0, 1, 0],
    ])
    expect(again.featureNamesOut(['colour'])).toEqual(['colour_1', 'colour_2', 'colour_nan'])
  })

  it('target encoding cross-fits the training rows', () => {
    const x = [[0], [0], [0], [1], [1], [1]]
    const y = [1, 2, 3, 10, 11, 12]
    const est = targetEncoder({ smooth: 0, targetType: 'continuous', nSplits: 3 })
    const fitTransform = est.fitTransform
    if (fitTransform === undefined) throw new Error('missing fitTransform')
    const { fitted, transformed } = fitTransform({ x, y })
    expect(rows(fitted.transform([[0], [1]]))).toEqual([[2], [11]])
    // Each training row is encoded without its own target: never the full-data category mean.
    const cross = rows(transformed).map((r) => r[0] ?? 0)
    expect(cross.some((v, i) => v !== (i < 3 ? 2 : 11))).toBe(true)
    expect(() => fitted.inverseTransform([[1]])).toThrow()
  })
})
