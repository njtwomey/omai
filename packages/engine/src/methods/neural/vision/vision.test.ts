import { describe, expect, it } from 'vitest'
import { take } from '../../../lib/data/index.ts'
import { digits } from '../../../lib/datasets/index.ts'
import { recordsOf } from '../../../lib/exposition/index.ts'
import type { LearnData } from '../../../lib/learn/index.ts'
import { key } from '../../../lib/random/index.ts'
import { toFlat } from '../../../lib/tensor/index.ts'
import { lenet } from './lenet.ts'

describe('lenet', () => {
  it('learns the 8 by 8 digits within a short budget and taps its feature maps at steps', () => {
    const data = take(
      digits(),
      Array.from({ length: 300 }, (_, i) => i),
    ) as unknown as LearnData
    const fitted = lenet({ epochs: 3, snapshots: 10 }).fit(data, { key: key(0) })
    expect(fitted.score?.(data)).toBeGreaterThan(0.6)
    const [loss, accuracy] = recordsOf(fitted.exposition, 'learningCurve')
    expect(loss?.sizes).toEqual([1, 2, 3])
    expect(accuracy?.labels?.validation).toBe('validation accuracy')
    const [image, conv1, conv2, filters] = recordsOf(fitted.exposition, 'decisionSurface')
    expect(image?.values?.length).toBe(8)
    // Six 8 by 8 maps tiled 3 across with gaps: 2 rows of 8 plus a gap, 3 columns of 8 plus two gaps.
    expect([conv1?.values?.length, conv1?.values?.[0]?.length]).toEqual([17, 26])
    expect(conv1?.steps?.length).toBeLessThanOrEqual(10)
    expect(conv1?.steps?.[0]?.iteration).toBe(0)
    expect([conv2?.values?.length, conv2?.values?.[0]?.length]).toEqual([14, 19])
    expect(filters?.steps?.length).toBe(conv1?.steps?.length)
    expect(recordsOf(fitted.exposition, 'coefficientTable')[0]?.names.length).toBe(10)
    const maps = fitted.featureMaps([toFlat((data as { x: never }).x).slice(0, 64)])
    expect(maps.map((l) => [l[0]?.length, l[0]?.[0]?.length])).toEqual([
      [6, 8],
      [12, 4],
    ])
    expect(fitted.describe().nodes.length).toBeGreaterThan(5)
  })
})
