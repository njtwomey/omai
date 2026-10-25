import { explanationQuality } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { explanationQualityView, qualityLayout } from './explanation-quality.ts'
import { stillOf, viewFor } from './registry.ts'

type Opt = { grid: unknown[]; title?: { text?: string }[] }

describe('explanation quality view', () => {
  it('draws a panel per metric and the agreement matrix', () => {
    const r = explanationQuality({
      explainers: ['tree', 'lime'],
      metrics: [
        { name: 'deletion AUC', category: 'faithfulness', values: [0.2, 0.4], better: 'lower' },
        { name: 'sparseness', category: 'complexity', values: [0.5, 0.3], spread: [0.1, 0.1], better: 'higher' },
      ],
      agreement: {
        metric: 'spearman',
        matrix: [
          [1, 0.4],
          [0.4, 1],
        ],
      },
    })
    expect(qualityLayout(r)).toEqual({ rows: 1, cols: 3 })
    expect((explanationQualityView(r) as unknown as Opt).grid).toHaveLength(3)
    expect(viewFor('explanationQuality').component).toBe('ExplanationQualityView')
    expect(stillOf(r)).toBeDefined()
  })
})
