import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { blobs } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key, normal } from '../../lib/random/index.ts'
import { toArray, toFlat } from '../../lib/tensor/index.ts'
import { HistogramDensity, histogramDensity, kernelDensity } from './density.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 80, centers: 2, clusterStd: 0.7 } })
const line = (toFlat(normal(key(1), [60])) as number[]).map((v) => [v])

describe('density estimation', () => {
  it('KDE picks a bandwidth by cross-validation and draws its log density', () => {
    const f = kernelDensity({ bandwidth: 'cv', folds: 4 }).fit(data, { key: key(0) })
    const [cv] = recordsOf(f.exposition, 'crossValidationCurve')
    expect(cv?.values.length).toBe(12)
    expect(cv?.best).toBe(f.params.factor)
    const [surface] = recordsOf(f.exposition, 'decisionSurface')
    expect(surface?.values?.length).toBe(40)
    expect(Number.isFinite(f.score(data))).toBe(true)
  })

  it('a standard normal sample: KDE and histogram log densities near the truth at 0', () => {
    const kde = kernelDensity().fit({ x: line })
    expect(toFlat(kde.scoreSamples([[0]]))[0]).toBeCloseTo(-0.5 * Math.log(2 * Math.PI), 0)
    const hist = histogramDensity({ bins: 6 }).fit({ x: line })
    const p = hist.params
    const widths = (p.edges[0] ?? []).slice(1).map((e, i) => e - (p.edges[0]?.[i] ?? 0))
    expect(p.density.reduce((a, v, i) => a + v * (widths[i] ?? 0), 0)).toBeCloseTo(1, 10)
    expect(toFlat(hist.scoreSamples([[1e6]]))[0]).toBe(Number.NEGATIVE_INFINITY)
    const h2 = new HistogramDensity({ bins: 5 }).fit(toArray(data.x as never) as number[][]).model
    expect(recordsOf(h2.exposition, 'decisionSurface').length).toBe(1)
  })
})
