import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { decodeField, fieldBackbones, fieldEncodings, gridCoordinates, neuralField, psnrFromMse } from './index.ts'

describe('neuralField', () => {
  it('fills the encoding and backbone slots and sizes the backbone from the encoding', () => {
    const encodings = [
      [{ kind: 'none' }, 2],
      [{ kind: 'frequency', octaves: 3, includeInput: true }, 14],
      [{ kind: 'fourier', features: 8, sigma: 4 }, 16],
      [{ kind: 'hash', levels: 3, log2TableSize: 6, featuresPerLevel: 2, finestResolution: 16 }, 6],
    ] as const
    for (const [encoding, width] of encodings)
      for (const backbone of [{ kind: 'mlp' }, { kind: 'siren', omega0: 20 }]) {
        const f = neuralField({ inputDim: 2, outputDim: 3, encoding, backbone, width: 8, depth: 2, key: key(0) })
        expect(f.encodedDim).toBe(width)
        expect(
          f.forward([
            [0.1, 0.2],
            [0.3, 0.4],
          ]).shape,
        ).toEqual([2, 3])
        expect(f.outputShape([5, 2])).toEqual([5, 3])
      }
  })

  it('is a plain ReLU MLP by default, a sigmoid head squashes to (0, 1), and the key fixes the weights', () => {
    const f = neuralField({ inputDim: 1, outputDim: 1, width: 4, depth: 2, key: key(0) })
    // 1 -> 4 -> 4 -> 1: (1 + 1) * 4 + (4 + 1) * 4 + (4 + 1) = 33 parameters.
    expect(f.parameterCount()).toBe(33)
    const s = neuralField({ inputDim: 1, outputDim: 2, width: 4, depth: 1, output: 'sigmoid', key: key(0) })
    for (const v of toFlat(s.forward([[-50], [0], [50]]))) {
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThanOrEqual(1)
    }
    const a = neuralField({ inputDim: 2, outputDim: 1, encoding: { kind: 'fourier' }, width: 4, depth: 1, key: key(9) })
    const b = neuralField({ inputDim: 2, outputDim: 1, encoding: { kind: 'fourier' }, width: 4, depth: 1, key: key(9) })
    expect(toFlat(a.forward([[0.2, 0.3]]))).toEqual(toFlat(b.forward([[0.2, 0.3]])))
  })

  it('has the tables’ properties and defaults', () => {
    expect(fieldEncodings.names).toEqual(['none', 'frequency', 'fourier', 'hash'])
    expect(fieldEncodings.where((p) => p.periodic)).toEqual(['frequency', 'fourier'])
    expect(fieldEncodings.where((p) => p.trainable)).toEqual(['hash'])
    expect(fieldBackbones.where((p) => p.periodic)).toEqual(['siren'])
    expect(fieldEncodings.optionOf(undefined).name).toBe('none')
  })
})

describe('decoding and quality', () => {
  it('lays a grid of cell centres, u across and v down, with the plane of a 3-D field', () => {
    expect(Array.from(toFlat(gridCoordinates({ dim: 2, resolution: 2 })))).toEqual([
      -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5,
    ])
    expect(Array.from(toFlat(gridCoordinates({ dim: 1, resolution: 2, bounds: [0, 4] })))).toEqual([1, 3])
    const g = gridCoordinates({ dim: 3, resolution: 2, slice: 0.25 })
    expect(g.shape).toEqual([4, 3])
    expect(Array.from(toFlat(g)).filter((_, i) => i % 3 === 2)).toEqual([0.25, 0.25, 0.25, 0.25])
  })

  it('decodes a field to plain data', () => {
    const f = neuralField({ inputDim: 3, outputDim: 1, width: 4, depth: 1, key: key(0) })
    const grid = decodeField(f, f.params, { dim: 3, resolution: 5 })
    expect(grid).toMatchObject({ kind: 'fieldGrid', dim: 3, resolution: 5, channels: 1, slice: 0 })
    expect(grid.values.length).toBe(25)
    expect(JSON.parse(JSON.stringify(grid))).toEqual(grid)
  })

  it('converts a mean squared error to decibels', () => {
    expect(psnrFromMse(0.01)).toBeCloseTo(20, 12)
    expect(psnrFromMse(0.01, 2)).toBeCloseTo(26.0206, 4)
  })
})
