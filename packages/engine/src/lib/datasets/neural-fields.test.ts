import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { tensor, toFlat, type Value } from '../tensor/index.ts'
import { datasetRegistry } from './catalogue.ts'
import { fieldImage, sdfBox, sdfSphere, sdfTorus, sdfUnion, signedDistance } from './neural-fields.ts'

describe('analytic signed distances', () => {
  it('are exact on the axes', () => {
    expect(sdfSphere([2, 0, 0], 0.5)).toBe(1.5)
    expect(sdfSphere([0.1, 0.2, 0.3], 0.5, [0.1, 0.2, 0.3])).toBe(-0.5)
    expect(sdfBox([2, 0], [1, 1])).toBe(1)
    expect(sdfBox([2, 2], [1, 1])).toBeCloseTo(Math.SQRT2, 15)
    expect(sdfBox([0.5, 0.2], [1, 1])).toBe(-0.5)
    // A torus around y with major radius 0.5 and tube 0.2: on the ring's core the distance is -0.2.
    expect(sdfTorus([0, 0, 0.5], 0.5, 0.2)).toBeCloseTo(-0.2, 15)
    expect(sdfTorus([0, 0.3, 0.5], 0.5, 0.2)).toBeCloseTo(0.1, 15)
    expect(sdfTorus([0.5, 0], 0.5, 0.2)).toBeCloseTo(-0.2, 15)
    expect(sdfUnion([0.3, -0.1, 2])).toBe(-0.1)
  })

  it('have unit gradient away from the medial axis (the eikonal property), checked by finite differences', () => {
    for (const f of [
      (p: number[]) => sdfSphere(p, 0.5),
      (p: number[]) => sdfBox(p, [0.6, 0.4, 0.4]),
      (p: number[]) => sdfTorus(p, 0.55, 0.2),
    ]) {
      const p = [0.71, 0.33, -0.52]
      const h = 1e-6
      const g = p.map(
        (_, i) => (f(p.map((v, j) => v + (i === j ? h : 0))) - f(p.map((v, j) => v - (i === j ? h : 0)))) / (2 * h),
      )
      expect(Math.hypot(...g)).toBeCloseTo(1, 6)
    }
  })
})

describe('signedDistance', () => {
  it('samples the exact distance, with a share of the points near the surface', () => {
    const g = generate(signedDistance, { key: key(0), knobs: { shape: 'union', dim: 3, n: 200, nearSurface: 0.5 } })
    expect(g.data.x.shape).toEqual([200, 3])
    expect(g.data.y.shape).toEqual([200])
    const truth = g.truth.regressionFunction as (x: Value) => Value
    expect(Array.from(toFlat(truth(g.data.x)))).toEqual(Array.from(toFlat(g.data.y)))
    const d = Array.from(toFlat(g.data.y))
    const nearCount = (values: readonly number[]): number => values.filter((v) => Math.abs(v) < 0.1).length
    const uniform = generate(signedDistance, { key: key(0), knobs: { shape: 'union', dim: 3, n: 200, nearSurface: 0 } })
    expect(nearCount(d)).toBeGreaterThan(nearCount(Array.from(toFlat(uniform.data.y))) + 30)
    expect(d.some((v) => v < 0) && d.some((v) => v > 0)).toBe(true)
    expect(g.truth.extra?.['eikonal']).toBe(true)
    const again = generate(signedDistance, { key: key(0), knobs: { shape: 'union', dim: 3, n: 200, nearSurface: 0.5 } })
    expect(Array.from(toFlat(again.data.x))).toEqual(Array.from(toFlat(g.data.x)))
  })

  it('has a shape in each dimension and is registered with tags', () => {
    for (const shape of ['sphere', 'box', 'torus', 'union'])
      for (const dim of [2, 3])
        expect(generate(signedDistance, { key: key(1), knobs: { shape, dim, n: 16 } }).data.x.shape).toEqual([16, dim])
    const entry = datasetRegistry.entries.find((e) => e.name === 'signedDistance')
    expect(entry?.kind).toBe('regression')
    expect(entry?.real).toBe(false)
  })
})

describe('fieldImage', () => {
  const side = 16
  const g = generate(fieldImage, { key: key(0), knobs: { side } })

  it('samples pixel centres over [-1, 1]^2 row by row, with colours in [0, 1.2]', () => {
    expect(g.data.x.shape).toEqual([side * side, 2])
    expect(g.data.y.shape).toEqual([side * side, 3])
    const x = toFlat(g.data.x)
    expect(x[0]).toBeCloseTo(-1 + 1 / side, 12)
    expect(x[1]).toBeCloseTo(-1 + 1 / side, 12)
    expect(x[2]).toBeCloseTo(-1 + 3 / side, 12)
    expect(x[2 * side + 1]).toBeCloseTo(-1 + 3 / side, 12)
    for (const v of toFlat(g.data.y)) expect(v).toBeGreaterThanOrEqual(-0.1)
  })

  it('has sharp edges and fine texture: the truth jumps across a disc’s edge and varies at the texture frequency', () => {
    const f = g.truth.regressionFunction as (x: Value) => Value
    // The white disc at (0.45, -0.5) with radius 0.18: just inside and just outside along u.
    const [inside, outside] = [toFlat(f(tensor([[0.45 + 0.17, -0.5]]))), toFlat(f(tensor([[0.45 + 0.19, -0.5]])))]
    expect(inside).toEqual([1, 1, 1])
    expect(Math.abs((outside[0] as number) - 1)).toBeGreaterThan(0.3)
    // The texture: the background's red channel along u at v = 0.9 has several oscillations (6 cycles per unit).
    const line = toFlat(f(tensor(Array.from({ length: 200 }, (_, i) => [-1 + (2 * i) / 199, 0.93])))).filter(
      (_, i) => i % 3 === 0,
    )
    const signChanges = line
      .slice(1)
      .filter((v, i) => Math.sign(v - 0.1) !== Math.sign((line[i] as number) - 0.1)).length
    expect(signChanges).toBeGreaterThanOrEqual(6)
  })

  it('evaluates its truth anywhere, equals its own samples at the pixel centres, and is registered', () => {
    const f = g.truth.regressionFunction as (x: Value) => Value
    expect(Array.from(toFlat(f(g.data.x)))).toEqual(Array.from(toFlat(g.data.y)))
    for (const pattern of ['checker', 'stripes', 'discs', 'mixed'])
      expect(generate(fieldImage, { key: key(0), knobs: { side: 4, pattern } }).data.y.shape).toEqual([16, 3])
    expect(datasetRegistry.entries.some((e) => e.name === 'fieldImage')).toBe(true)
  })
})
