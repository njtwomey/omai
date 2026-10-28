import { describe, expect, it } from 'vitest'
import { defaults, grid } from '../space/index.ts'
import { item, toArray } from '../tensor/index.ts'
import { classificationLosses, type Penalty, penalties, regressionLosses } from './tables.ts'

describe('objectives', () => {
  it('every regression loss is zero at a perfect prediction and positive away from it', () => {
    for (const config of grid(regressionLosses.space, 1)) {
      const loss = regressionLosses.make(config)
      expect(item(loss([1, 2, 3], [1, 2, 3])), JSON.stringify(config)).toBe(0)
      expect(item(loss([1, 2, 3], [1.5, 2, 3])), JSON.stringify(config)).toBeGreaterThan(0)
    }
    expect(defaults(regressionLosses.space)).toEqual({ kind: 'squared' })
    expect(regressionLosses.where((p) => p.likelihood)).toEqual(['squared', 'absolute'])
    expect(regressionLosses.where((p) => !p.smooth)).toEqual(['absolute', 'pinball'])
  })

  it('every classification loss falls as the margin grows', () => {
    for (const config of grid(classificationLosses.space, 1)) {
      const loss = classificationLosses.make(config)
      // Just on the wrong side against well on the right side: every margin loss falls.
      const wrong = item(loss([0.1, -0.1], [-1, 1]))
      const right = item(loss([3, -3], [1, -1]))
      expect(right, JSON.stringify(config)).toBeLessThan(wrong)
    }
    expect(classificationLosses.where((p) => p.likelihood)).toEqual(['logistic'])
    expect(classificationLosses.properties({ kind: 'hinge' }).smooth).toBe(false)
  })

  it('penalties carry their strength, value and proximal operator', () => {
    expect(defaults(penalties.space)).toEqual({ kind: 'none' })
    expect(penalties.where((p) => p.sparse)).toEqual(['l1', 'elasticNet'])
    const none = penalties.make({ kind: 'none' })
    expect(item(none.value([1, 2]))).toBe(0)
    const l1 = penalties.make({ kind: 'l1', alpha: 0.5 })
    expect(item(l1.value([1, -2]))).toBeCloseTo(1.5, 12)
    const proxOf = (p: Penalty): NonNullable<Penalty['prox']> => p.prox as NonNullable<Penalty['prox']>
    expect(toArray(proxOf(l1)([1, -0.2, 0.1], 1))).toEqual([0.5, 0, 0])
    const l2 = penalties.make({ kind: 'l2', alpha: 1 })
    expect(item(l2.value([3, 4]))).toBeCloseTo(12.5, 12)
    expect(toArray(proxOf(l2)([2], 1))).toEqual([1])
    const net = penalties.make({ kind: 'elasticNet', alpha: 1, l1Ratio: 0.5 })
    expect(item(net.value([2]))).toBeCloseTo(0.5 * 2 + 0.25 * 4, 12)
    expect(net.strength).toBe(1)
  })
})
