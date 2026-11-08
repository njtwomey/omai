import { featureAttributions } from 'omai-engine/exposition'
import { describe, expect, it } from 'vitest'
import { figure } from '../figure/figure.ts'
import {
  attributionImageView,
  attributionSeriesView,
  tokenLayout,
  tokenStripHeight,
  tokenStripView,
} from './modalities.ts'
import { stillOf } from './registry.ts'

type Opt = { grid: unknown[]; series: { name?: string }[]; graphic?: { type: string; style?: { text?: string } }[] }

const base = { output: 'y', baseValue: 0, predictions: [1] }

describe('modality views', () => {
  it('draws an image beside its attribution map', () => {
    const r = featureAttributions({
      ...base,
      method: 'occlusion',
      features: ['a', 'b', 'c', 'd', 'e', 'f'],
      values: [[1, 0, 0, -1, 0, 0]],
      data: [[1, 0, 0, 1, 1, 1]],
      layout: { kind: 'image', shape: [2, 3] },
    })
    expect((attributionImageView(r) as unknown as Opt).grid).toHaveLength(2)
    expect((stillOf(r) as unknown as Opt).grid).toHaveLength(2)
    expect(() => featureAttributions({ ...r, layout: { kind: 'image', shape: [2, 2] } })).toThrow(/image/)
  })

  it('wraps tokens into lines and writes each with its attribution', () => {
    const words = Array.from({ length: 40 }, (_, i) => `word${i}`)
    const boxes = tokenLayout(words, 300, 11)
    expect(boxes.at(-1)?.line ?? 0).toBeGreaterThan(2)
    for (const b of boxes) expect(b.x + b.width).toBeLessThanOrEqual(300 + 1e-9)
    const r = featureAttributions({
      ...base,
      method: 'ablation',
      features: ['not', 'bad'],
      values: [[0.3, -0.2]],
      data: [[1, 1]],
      layout: { kind: 'tokens' },
    })
    const texts = ((tokenStripView(r) as unknown as Opt).graphic ?? []).map((g) => g.style?.text)
    expect(texts).toContain('not')
    expect(texts.some((t) => t?.endsWith('0.2'))).toBe(true)
    expect(tokenStripHeight(words, figure())).toBeGreaterThan(tokenStripHeight(['a'], figure()))
  })

  it('draws a series over its attributions', () => {
    const r = featureAttributions({
      ...base,
      method: 'occlusion',
      features: ['t0', 't1', 't2'],
      values: [[0, 1, -1]],
      data: [[1, 2, 3]],
      layout: { kind: 'series', time: [10, 20, 30] },
    })
    expect((attributionSeriesView(r) as unknown as Opt).grid).toHaveLength(2)
  })
})
