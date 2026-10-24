import { describe, expect, it } from 'vitest'
import { datasetCollections, digits, loadEntry } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import type { LearnData } from '../../lib/learn/index.ts'
import { key } from '../../lib/random/index.ts'
import { div, take } from '../../lib/tensor/index.ts'
import { vae } from './autoencoders.ts'

const moons = (n: number): LearnData => {
  const entry = datasetCollections['density2d']?.entries.find((e) => e.name === 'moons')
  if (entry === undefined) throw new Error('no moons')
  return loadEntry(entry, key(0), { n }).data as unknown as LearnData
}

describe('vae', () => {
  it('lowers the negative ELBO and records the latent space, reconstructions, samples and both terms', () => {
    const r = vae(moons(100), { steps: 60, hidden: 8, samples: 15, frames: 5 }, { key: key(0) })
    expect(r.negativeElbo).toBeLessThan(r.initialNegativeElbo)
    const [latent, reconstructions, samples] = recordsOf(r.exposition, 'embeddingSteps')
    expect(latent?.positions.length).toBe(5)
    expect(latent?.colourBy?.kind).toBe('classes')
    expect(reconstructions?.positions[0]?.length).toBe(200)
    expect(samples?.positions[0]?.length).toBe(115)
    const [terms, elbo] = recordsOf(r.exposition, 'learningCurve')
    expect(terms?.labels?.validation).toMatch(/KL/)
    expect(elbo?.train.length).toBeGreaterThan(0)
  })

  it('learns pixel images with a Bernoulli decoder and records them as images', () => {
    const d = digits()
    const rows = Array.from({ length: 60 }, (_, i) => i)
    const data = { x: div(take(d.x, rows), 16), y: take(d.y, rows) }
    const r = vae(
      data,
      { likelihood: 'bernoulli', steps: 10, hidden: 8, imageShape: [8, 8], images: 4, samples: 4, frames: 3 },
      { key: key(1) },
    )
    const [images] = recordsOf(r.exposition, 'imageSteps')
    expect(images?.groups).toEqual(['data', 'reconstruction', 'samples'])
    expect(images?.steps[0]?.images[1]?.length).toBe(4)
    expect(images?.steps[0]?.images[0]?.[0]?.length).toBe(64)
    expect(() =>
      vae(
        {
          x: [
            [2, 0],
            [0, 1],
          ],
        },
        { likelihood: 'bernoulli' },
      ),
    ).toThrow(/\[0, 1\]/)
  })
})
