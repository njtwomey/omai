import { describe, expect, it } from 'vitest'
import { lookup } from '../base/index.ts'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { spearman } from '../stats/index.ts'
import { toFlat } from '../tensor/index.ts'
import { datasetRegistry, irtResponses, pottsSamples, shapeletPatterns, shapeletSeries } from './index.ts'

describe('planted generators of the graphical-model and time-series domains', () => {
  it('irtResponses: 0/1 responses whose scores follow the true abilities, with the true items in the truth', () => {
    const { data, truth } = generate(irtResponses, {
      key: key(0),
      knobs: { respondents: 400, items: 15, model: '3pl' },
    })
    expect(data.x.shape).toEqual([400, 15])
    const x = toFlat(data.x)
    expect(x.every((v) => v === 0 || v === 1)).toBe(true)
    const score = Array.from({ length: 400 }, (_, i) => x.slice(i * 15, (i + 1) * 15).reduce((a, v) => a + v, 0))
    expect(spearman(score, toFlat(data.y)).statistic).toBeGreaterThan(0.7)
    expect((truth.latents?.['theta'] as number[]).length).toBe(400)
    const items = truth.extra?.['items'] as { a: number; b: number; c: number }[]
    expect(items.map((i) => i.c)).toEqual(new Array(15).fill(0.2))
    expect(items[0]?.b).toBe(-2)
    expect(items[14]?.b).toBe(2)
    // a harder item is answered correctly less often
    const rate = (j: number): number =>
      Array.from({ length: 400 }, (_, i) => x[i * 15 + j] ?? 0).reduce((a, v) => a + v, 0) / 400
    expect(rate(0)).toBeGreaterThan(rate(14) + 0.3)
  })

  it('pottsSamples: ordered above the critical coupling (near Onsager), disordered below it', () => {
    const ordered = generate(pottsSamples, {
      key: key(1),
      knobs: { samples: 30, side: 12, relativeCoupling: 2, burnIn: 50, thin: 3 },
    })
    expect(ordered.data.x.shape).toEqual([30, 144])
    const m = toFlat(ordered.data.y)
    const onsager = ordered.truth.extra?.['onsagerMagnetisation'] as number
    expect(onsager).toBeGreaterThan(0.99)
    expect(m.reduce((a, v) => a + v, 0) / m.length).toBeGreaterThan(0.9)
    const disordered = generate(pottsSamples, {
      key: key(1),
      knobs: { samples: 30, side: 12, relativeCoupling: 0.4, burnIn: 50, thin: 3 },
    })
    const d = toFlat(disordered.data.y)
    expect(d.reduce((a, v) => a + v, 0) / d.length).toBeLessThan(0.3)
    expect(disordered.truth.extra?.['onsagerMagnetisation']).toBe(0)
    expect(Math.abs((ordered.truth.extra?.['criticalCoupling'] as number) - Math.log(1 + Math.SQRT2))).toBeLessThan(
      1e-12,
    )
    // the energy per site is lower (more aligned bonds) in the ordered phase
    const energy = (t: typeof ordered.truth): number =>
      (t.latents?.['energy'] as number[]).reduce((a, v) => a + v, 0) / 30
    expect(energy(ordered.truth)).toBeLessThan(energy(disordered.truth) - 0.5)
  })

  it('shapeletSeries: the pattern sits where the truth says, in every series of its class', () => {
    const { data, truth } = generate(shapeletSeries, {
      key: key(2),
      knobs: { n: 30, length: 50, classes: 3, noise: 0.05 },
    })
    const x = toFlat(data.x)
    const y = toFlat(data.y)
    const at = truth.latents?.['positions'] as number[]
    for (let i = 0; i < 30; i++) {
      expect(y[i]).toBe(i % 3)
      if (y[i] === 0) {
        expect(at[i]).toBe(-1)
        continue
      }
      const pattern = shapeletPatterns[(y[i] ?? 1) - 1] ?? []
      const start = at[i] as number
      expect(start).toBeGreaterThanOrEqual(0)
      expect(start + pattern.length).toBeLessThanOrEqual(50)
      pattern.forEach((v, j) => expect(x[i * 50 + start + j]).toBeCloseTo(v, 0))
    }
    expect((truth.extra?.['patterns'] as number[][]).length).toBe(2)
  })

  it('are registered with tags', () => {
    for (const name of ['irtResponses', 'pottsSamples', 'shapeletSeries']) {
      const entry = lookup(datasetRegistry, name)
      expect(entry.name).toBe(name)
    }
    expect(lookup(datasetRegistry, 'shapeletSeries').modality).toBe('timeSeries')
  })
})
