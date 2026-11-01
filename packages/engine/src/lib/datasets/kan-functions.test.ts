import { describe, expect, it } from 'vitest'
import { type Generated, generate, type RegressionData } from '../data/index.ts'
import { key } from '../random/index.ts'
import { toFlat, type Value } from '../tensor/index.ts'
import { datasetRegistry } from './catalogue.ts'
import { kanFunctions } from './regression-curves.ts'

describe('kanFunctions', () => {
  const at = (name: string, n = 50): Generated<RegressionData> =>
    generate(kanFunctions, { key: key(1), knobs: { function: name, n } })

  it('has the paper’s functions as the regression function, exactly, on uniform inputs in [-1, 1]', () => {
    const a = at('exp-sin-square')
    const x = toFlat(a.data.x as Value)
    const y = toFlat(a.data.y as Value)
    expect(Math.min(...x)).toBeGreaterThanOrEqual(-1)
    expect(Math.max(...x)).toBeLessThanOrEqual(1)
    for (let r = 0; r < 50; r++)
      expect(y[r]).toBeCloseTo(Math.exp(Math.sin(Math.PI * (x[2 * r] as number)) + (x[2 * r + 1] as number) ** 2), 12)
    const m = toFlat(at('multiplication').data.y as Value)
    const mx = toFlat(at('multiplication').data.x as Value)
    expect(m[3]).toBeCloseTo((mx[6] as number) * (mx[7] as number), 12)
    const ring = at('ring-sums')
    expect(ring.data.x.shape).toEqual([50, 4])
    expect(toFlat(ring.truth.regressionFunction?.(ring.data.x as Value) as Value)).toEqual(toFlat(ring.data.y as Value))
  })

  it('carries the structure as truth and is registered', () => {
    expect(at('multiplication').truth.extra?.['minimalWidths']).toEqual([2, 2, 1])
    const edges = (at('exp-sin-square').truth.extra?.['edges'] ?? []) as { name: string }[]
    expect(edges.map((e) => e.name)).toEqual(['sin', 'x^2', 'exp'])
    expect(datasetRegistry.entries.some((e) => e.name === 'kanFunctions')).toBe(true)
  })
})
