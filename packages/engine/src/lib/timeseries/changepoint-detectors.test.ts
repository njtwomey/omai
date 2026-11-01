import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { changepointSeries, conjugateModelOf, type SegmentModel } from '../datasets/index.ts'
import { hazardValue, segmentModelValueOf } from '../dynamics/filtering/index.ts'
import { key, normal } from '../random/index.ts'
import { defaults } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import { changepointDetectors } from './changepoint-detectors.ts'
import { changepointPrecisionRecall } from './changepoint-scoring.ts'

describe('the changepoint detectors', () => {
  const noise = toFlat(normal(key(4), [120]))
  // Means 0, 4, -2 with noise 0.5: changes at 40 and 80.
  const y = noise.map((v, i) => 0.5 * v + (i < 40 ? 0 : i < 80 ? 4 : -2))
  const truth = [40, 80]

  it('lists the five detectors with their own knobs', () => {
    expect(changepointDetectors.names).toEqual(['bocpd', 'pelt', 'binseg', 'cusum', 'page-hinkley'])
    expect(changepointDetectors.where((p) => p.online)).toEqual(['bocpd', 'cusum', 'page-hinkley'])
    expect(changepointDetectors.where((p) => p.locatesChange)).toEqual(['bocpd', 'pelt', 'binseg'])
    expect(changepointDetectors.space.options['pelt']?.spaces).toHaveProperty('penalty')
  })

  it('every detector, from its default knobs, finds the changes of an easy series within its own delay', () => {
    const picks: Record<string, Record<string, unknown>> = {
      bocpd: {
        model: { kind: 'mean', sigma: 0.5, priorMean: 0, priorStd: 4 },
        hazard: hazardValue('constant', { rate: 0.02 }),
      },
      pelt: { cost: 'l2', penalty: 10, minSize: 2 },
      binseg: { cost: 'l2', changepoints: 2 },
      cusum: { warmup: 20, threshold: 8 },
      'page-hinkley': { threshold: 30, minInstances: 20 },
    }
    for (const name of changepointDetectors.names) {
      const detect = changepointDetectors.make({
        ...(defaults(changepointDetectors.space) as object),
        kind: name,
        ...picks[name],
      })
      const found = detect(y)
      expect(found, name).toEqual([...found].sort((a, b) => a - b))
      expect(new Set(found).size, name).toBe(found.length)
      const margin = changepointDetectors.properties({ kind: name }).locatesChange ? 3 : 15
      const scores = changepointPrecisionRecall(truth, found, margin)
      expect(scores.recall, `${name}: ${found.join(',')}`).toBe(1)
    }
  })

  it('the offline detectors agree with the library functions and drop the series end', () => {
    expect(
      changepointDetectors.make({ kind: 'pelt', cost: 'l2', penalty: 5, minSize: 2 })([0, 0, 0, 0, 5, 5, 5, 5]),
    ).toEqual([4])
    expect(
      changepointDetectors.make({ kind: 'binseg', cost: 'l2', changepoints: 1, minSize: 2 })([0, 0, 0, 5, 5, 5]),
    ).toEqual([3])
    expect(changepointDetectors.make({ kind: 'pelt' })([1, 1, 1, 1])).toEqual([])
  })

  it('BOCPD with the generator’s own model finds the changes of generated series of its kind', () => {
    const { data, truth: t } = generate(changepointSeries, {
      key: key(21),
      knobs: { n: 300, kind: 'mean', hazard: 0.02, noise: 0.5, shift: 4 },
    })
    const detect = changepointDetectors.make({
      kind: 'bocpd',
      model: segmentModelValueOf(conjugateModelOf(t.extra?.['model'] as SegmentModel)),
      hazard: hazardValue('constant', { rate: 0.02 }),
      changeThreshold: 0.5,
    })
    const s = changepointPrecisionRecall(t.changepoints ?? [], detect(Array.from(toFlat(data.values))), 3)
    expect(s.recall).toBeGreaterThanOrEqual(0.7)
    expect(s.precision).toBeGreaterThanOrEqual(0.7)
  })
})
