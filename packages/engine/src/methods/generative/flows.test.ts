import { describe, expect, it } from 'vitest'
import { datasetCollections, loadEntry } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import type { LearnData } from '../../lib/learn/index.ts'
import { key } from '../../lib/random/index.ts'
import { realNvp } from './flows.ts'

const moons = (n: number): LearnData => {
  const entry = datasetCollections['density2d']?.entries.find((e) => e.name === 'moons')
  if (entry === undefined) throw new Error('no moons')
  return loadEntry(entry, key(0), { n }).data as unknown as LearnData
}

describe('realNvp', () => {
  it('raises the likelihood of the data and records density, samples, the warped grid and layers per step', () => {
    const r = realNvp(
      moons(120),
      { steps: 40, layers: 2, hidden: 8, resolution: 6, samples: 20, frames: 5 },
      { key: key(0) },
    )
    expect(r.negativeLogLikelihood).toBeLessThan(r.initialNegativeLogLikelihood)
    const [surface] = recordsOf(r.exposition, 'decisionSurface')
    expect(surface?.steps?.length).toBe(5)
    expect(surface?.steps?.[0]?.values?.length).toBe(6)
    const [samples, warp, layers] = recordsOf(r.exposition, 'embeddingSteps')
    expect(samples?.positions[0]?.length).toBe(120 + 20)
    expect(warp?.graph?.length).toBe(2 * 17 * 16)
    expect(layers?.positions.length).toBe(3)
    expect(recordsOf(r.exposition, 'learningCurve').length).toBe(1)
    expect(r.logProb([[0, 0]]).length).toBe(1)
  })

  it('reports progress at every recorded step', () => {
    const seen: number[] = []
    realNvp(
      moons(40),
      { steps: 6, layers: 2, hidden: 4, resolution: 4, samples: 5, onProgress: (p) => seen.push(p.fraction) },
      { key: key(1) },
    )
    expect(seen.at(-1)).toBe(1)
  })
})
