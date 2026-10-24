import { describe, expect, it } from 'vitest'
import { crossValidationCurve, learningCurve, pruningPath, regularisationPath } from 'whitebox-engine/exposition'
import { coefficientsAt, crossValidationView, regularisationPathView } from './paths.ts'
import { stillOf, viewFor } from './registry.ts'

type Opt = { series: { name?: string }[]; grid: unknown[]; xAxis: { type: string; inverse?: boolean }[] }
const o = (x: unknown): Opt => x as Opt

const path = regularisationPath({
  penalties: [10, 1, 0.1, 0.01],
  features: ['a', 'b'],
  coefficients: [
    [0, 0],
    [0.5, 0],
    [0.9, -0.2],
    [1, -0.3],
  ],
  chosen: 0.1,
})

describe('path views', () => {
  it('draws a regularisation path as coefficients over the active-set size', () => {
    const opt = o(regularisationPathView(path, { selected: 1 }))
    expect(opt.grid).toHaveLength(2)
    expect(opt.xAxis[0]?.type).toBe('log')
    expect(opt.xAxis[0]?.inverse).toBe(true)
    expect(opt.series.filter((s) => s.name === 'a' || s.name === 'b')).toHaveLength(4)
  })

  it('interpolates the coefficients at a penalty in log penalty', () => {
    expect(coefficientsAt(path, 1)).toEqual([0.5, 0])
    expect(coefficientsAt(path, Math.sqrt(10))[0]).toBeCloseTo(0.25, 12)
  })

  it('draws a cross-validation curve with its band, mean and marks', () => {
    const curve = crossValidationCurve({
      parameter: 'alpha',
      values: [1, 2, 3],
      meanScores: [0.7, 0.8, 0.75],
      standardErrors: [0.05, 0.04, 0.05],
      foldScores: [
        [0.6, 0.8],
        [0.8, 0.8],
        [0.7, 0.8],
      ],
      higherIsBetter: true,
      best: 2,
    })
    const names = o(crossValidationView(curve)).series.map((s) => s.name)
    expect(names.filter((n) => n?.startsWith('__fold'))).toHaveLength(2)
    expect(names).toContain('mean')
    expect(names).toContain('best')
  })
})

describe('the view registry', () => {
  it('names a component and draws a still for the kinds it knows', () => {
    expect(viewFor('regularisationPath')?.component).toBe('PathView')
    const lc = learningCurve({ sizes: [10, 20], train: [1, 0.9], validation: [0.6, 0.8] })
    expect(stillOf(lc)).toBeDefined()
    expect(stillOf(pruningPath({ alphas: [0, 0.1], impurities: [0, 0.2], leaves: [4, 1] }))).toBeDefined()
  })
})
