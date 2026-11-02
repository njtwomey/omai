import { describe, expect, it } from 'vitest'
import { gradCheck } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { fromStorage, square, sum, tensor, toArray, toFlat, type Value } from '../../tensor/index.ts'
import {
  fourierFeatures,
  gaussianFrequencies,
  hashGridEncode,
  hashGridIndex,
  hashGridResolutions,
  hashGridTableShape,
  positionalEncoding,
  positionalEncodingWidth,
  resolveHashGrid,
} from './coordinate-encodings.ts'

const close = (a: ArrayLike<number>, b: readonly number[], tol = 1e-12): void => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < b.length; i++) expect(Math.abs((a[i] as number) - (b[i] as number))).toBeLessThan(tol)
}

describe('positionalEncoding', () => {
  it('is the NeRF encoding, octave by octave: all sines of an octave, then all its cosines', () => {
    const x = [[0.25, -0.5]]
    const y = toFlat(positionalEncoding(x, { octaves: 2 }))
    const expected = [
      Math.sin(Math.PI * 0.25),
      Math.sin(-Math.PI * 0.5),
      Math.cos(Math.PI * 0.25),
      Math.cos(-Math.PI * 0.5),
      Math.sin(2 * Math.PI * 0.25),
      Math.sin(-2 * Math.PI * 0.5),
      Math.cos(2 * Math.PI * 0.25),
      Math.cos(-2 * Math.PI * 0.5),
    ]
    close(y, expected)
  })

  it('prepends the input when asked and has the stated width, with any leading shape', () => {
    const o = { octaves: 3, includeInput: true }
    expect(positionalEncodingWidth(3, o)).toBe(21)
    expect(
      positionalEncoding(
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        o,
      ).shape,
    ).toEqual([2, 21])
    expect(positionalEncoding(tensor([[[1, 2, 3]], [[4, 5, 6]]]), o).shape).toEqual([2, 1, 21])
    close(toFlat(positionalEncoding([[0.3, 0.1, 0.2]], o)).slice(0, 3), [0.3, 0.1, 0.2])
  })

  it('is differentiable in the input', () => {
    const f = (v: Value): Value => sum(square(positionalEncoding(v, { octaves: 3, includeInput: true })))
    expect(
      gradCheck(
        f,
        tensor([
          [0.3, -0.2],
          [0.7, 0.1],
        ]),
      ).passed,
    ).toBe(true)
  })

  it('rejects fewer than one octave', () => {
    expect(() => positionalEncoding([[1]], { octaves: 0 })).toThrow(/octaves/)
  })
})

describe('fourierFeatures', () => {
  it('has the sines then the cosines of 2 pi v B', () => {
    const B = [
      [0.5, 1],
      [0.25, -2],
    ]
    const y = toFlat(fourierFeatures([[1, 2]], B))
    const angles = [2 * Math.PI * (0.5 + 0.5), 2 * Math.PI * (1 - 4)]
    close(y, [
      Math.sin(angles[0] as number),
      Math.sin(angles[1] as number),
      Math.cos(angles[0] as number),
      Math.cos(angles[1] as number),
    ])
  })

  it('has a stationary kernel: the mean product of two encodings is exp(-2 pi^2 sigma^2 |x - y|^2)', () => {
    const sigma = 0.3
    const m = 4000
    const B = gaussianFrequencies(key(7), 2, m, sigma)
    const a = toFlat(fourierFeatures([[0.1, 0.2]], B))
    const b = toFlat(fourierFeatures([[0.3, 0.2]], B))
    let dot = 0
    for (let i = 0; i < a.length; i++) dot += (a[i] as number) * (b[i] as number)
    const expected = Math.exp(-2 * Math.PI ** 2 * sigma ** 2 * 0.2 ** 2)
    expect(Math.abs(dot / m - expected)).toBeLessThan(0.03)
  })

  it('draws the same frequencies from the same key, with the scale knob setting their spread', () => {
    expect(toFlat(gaussianFrequencies(key(3), 2, 8, 2))).toEqual(toFlat(gaussianFrequencies(key(3), 2, 8, 2)))
    const spread = (s: number): number => {
      const b = toFlat(gaussianFrequencies(key(1), 2, 2000, s))
      return Math.sqrt(b.reduce((a, v) => a + v * v, 0) / b.length)
    }
    expect(spread(10) / spread(1)).toBeCloseTo(10, 6)
    expect(spread(4)).toBeGreaterThan(3.7)
    expect(spread(4)).toBeLessThan(4.3)
  })

  it('checks the coordinate dimension against B', () => {
    expect(() => fourierFeatures([[1, 2, 3]], [[1], [2]])).toThrow(/B must be/)
  })

  it('is differentiable in the input', () => {
    const B = gaussianFrequencies(key(0), 2, 5, 1)
    expect(gradCheck((v: Value) => sum(square(fourierFeatures(v, B))), tensor([[0.3, -0.2]])).passed).toBe(true)
  })
})

describe('hash grid', () => {
  const config = {
    dim: 2,
    levels: 2,
    log2TableSize: 8,
    featuresPerLevel: 2,
    baseResolution: 4,
    finestResolution: 8,
  } as const

  it('has geometric resolutions from the base to the finest and a table of the stated shape', () => {
    expect(hashGridResolutions({ dim: 2, levels: 3, baseResolution: 4, finestResolution: 16 })).toEqual([4, 8, 16])
    expect(hashGridResolutions({ dim: 3, levels: 1, baseResolution: 5 })).toEqual([5])
    expect(hashGridTableShape(config)).toEqual([512, 2])
    expect(() => resolveHashGrid({ dim: 4 })).toThrow(/dimension/)
    expect(() => resolveHashGrid({ dim: 2, baseResolution: 8, finestResolution: 4 })).toThrow(/baseResolution/)
  })

  it('indexes vertices densely when the level fits in the table, and by a deterministic hash when it does not', () => {
    // 5 x 5 vertices fit in 256 slots: row-major, x fastest.
    const seen = new Set<number>()
    for (let j = 0; j <= 4; j++) for (let i = 0; i <= 4; i++) seen.add(hashGridIndex([i, j], 4, 8))
    expect(seen.size).toBe(25)
    expect(hashGridIndex([2, 1], 4, 8)).toBe(7)
    // 1001 x 1001 vertices do not fit in 16 slots: hashed into range, the same every time, and not all one slot.
    const slots = new Set<number>()
    for (let i = 0; i < 40; i++) slots.add(hashGridIndex([i * 7, i * 13], 1000, 4))
    expect([...slots].every((s) => s >= 0 && s < 16)).toBe(true)
    expect(slots.size).toBeGreaterThan(4)
    expect(hashGridIndex([123, 456, 789], 1000, 12)).toBe(hashGridIndex([123, 456, 789], 1000, 12))
    expect(hashGridIndex([123, 456, 789], 1000, 12)).toBeLessThan(4096)
  })

  it('reproduces a table value at a lattice point and interpolates multilinearly between', () => {
    const [rows] = hashGridTableShape(config)
    const table = normal(key(2), [rows, 2])
    const t = toFlat(table)
    // Level 0: N = 4 on [-1, 1]: vertex (i, j) is at (-1 + i / 2, -1 + j / 2), stored at row j * 5 + i.
    const at = (u: number, v: number): number[] => Array.from(toFlat(hashGridEncode([[u, v]], table, config)))
    const level0 = (i: number, j: number): number[] => [t[(j * 5 + i) * 2] as number, t[(j * 5 + i) * 2 + 1] as number]
    close(at(0, -0.5).slice(0, 2), level0(2, 1))
    close(at(-1, 1).slice(0, 2), level0(0, 4))
    // The centre of the cell with corners (2,1), (3,1), (2,2), (3,2): the mean of the four.
    const cu = (2 + 3) / 2
    const cv = (1 + 2) / 2
    const corners = [level0(2, 1), level0(3, 1), level0(2, 2), level0(3, 2)]
    const mean = [0, 1].map((k) => corners.reduce((a, v) => a + (v[k] as number), 0) / 4)
    close(at(-1 + cu / 2, -1 + cv / 2).slice(0, 2), mean)
    // A quarter of the way along one edge: 3/4 of the first corner and 1/4 of the second (bilinear in x).
    const quarter = [0, 1].map((k) => 0.75 * (level0(2, 1)[k] as number) + 0.25 * (level0(3, 1)[k] as number))
    close(at(0 + 0.125, -0.5).slice(0, 2), quarter)
  })

  it('has the 1-D example in its documentation: vertices 0, 1, 4 and the point half way along the last cell', () => {
    const cfg = {
      dim: 1,
      levels: 1,
      log2TableSize: 2,
      featuresPerLevel: 1,
      baseResolution: 2,
      finestResolution: 2,
      bounds: [0, 1],
    } as const
    const table = fromStorage(Float64Array.of(0, 1, 4, 0), [4, 1], 'float64')
    expect(toArray(hashGridEncode([[0.75]], table, cfg))).toEqual([[2.5]])
  })

  it('clamps coordinates outside the bounds to the border and concatenates levels', () => {
    const [rows] = hashGridTableShape(config)
    const table = normal(key(4), [rows, 2])
    const inside = toFlat(hashGridEncode([[1, 1]], table, config))
    const outside = toFlat(hashGridEncode([[5, 9]], table, config))
    close(outside, Array.from(inside))
    expect(
      hashGridEncode(
        tensor([
          [0.1, 0.2],
          [0.3, 0.4],
          [0.5, 0.6],
        ]),
        table,
        config,
      ).shape,
    ).toEqual([3, 4])
    expect(hashGridEncode(tensor([[[0.1, 0.2]], [[0.3, 0.4]]]), table, config).shape).toEqual([2, 1, 4])
  })

  it('is differentiable in the table and, away from lattice points, in the coordinates', () => {
    const [rows] = hashGridTableShape(config)
    const table = normal(key(5), [rows, 2])
    const x = tensor([
      [0.13, -0.31],
      [0.52, 0.27],
    ])
    expect(gradCheck((t: Value) => sum(square(hashGridEncode(x, t, config))), table).passed).toBe(true)
    expect(gradCheck((v: Value) => sum(square(hashGridEncode(v, table, config))), x).passed).toBe(true)
  })

  it('works in three dimensions with hashed levels', () => {
    const cfg = {
      dim: 3,
      levels: 2,
      log2TableSize: 6,
      featuresPerLevel: 2,
      baseResolution: 8,
      finestResolution: 32,
    } as const
    const [rows] = hashGridTableShape(cfg)
    const table = normal(key(6), [rows, 2])
    const y = hashGridEncode(
      tensor([
        [0.1, -0.2, 0.3],
        [0.9, 0.9, -0.9],
      ]),
      table,
      cfg,
    )
    expect(y.shape).toEqual([2, 4])
    expect(toFlat(y).every(Number.isFinite)).toBe(true)
    expect(
      gradCheck((t: Value) => sum(square(hashGridEncode(tensor([[0.13, -0.21, 0.33]]), t, cfg))), table).passed,
    ).toBe(true)
  })
})
