import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal, uniform } from '../../random/index.ts'
import { fromStorage, toArray, toFlat, type Value } from '../../tensor/index.ts'
import * as NN from '../index.ts'
import { affineGrid, gridSample, identityTheta, roiAlign, roiPool } from './sampling.ts'
import { interpolate, pad, patchEmbedding, pixelShuffle, pixelUnshuffle } from './vision.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const ok = (f: (...xs: Value[]) => Value, xs: Value[]): void => {
  expect(checkPrimitive(f, xs, { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
}

describe('vision gradients', () => {
  it('interpolation, shuffles and padding', () => {
    ok((x) => interpolate(x, { size: [5, 3], mode: 'bilinear' }), [n(0, [1, 1, 3, 4])])
    ok((x) => interpolate(x, { scaleFactor: 1.5, mode: 'bicubic', alignCorners: true }), [n(1, [1, 1, 3, 3])])
    ok((x) => pixelShuffle(x, 2), [n(2, [1, 4, 2, 2])])
    ok((x) => pixelUnshuffle(x, 2), [n(3, [1, 1, 4, 2])])
    ok((x) => pad(x, [1, 2, 1, 0], 'reflect'), [n(4, [1, 1, 3, 3])])
    ok((x, w, b) => patchEmbedding(x, w, b), [n(5, [1, 2, 4, 4]), n(6, [3, 2, 2, 2]), n(7, [3])])
  })
  it('grid sampling in the input and the grid', () => {
    const grid = uniform(key(8), [1, 2, 3, 2], { low: -0.9, high: 0.9 })
    for (const mode of ['bilinear', 'bicubic'] as const) {
      ok((x, g) => gridSample(x, g, { mode, paddingMode: 'border' }), [n(9, [1, 2, 3, 4]), grid])
    }
    ok((t) => affineGrid(t, [1, 1, 2, 3]), [n(10, [1, 2, 3])])
    ok((x) => roiAlign(x, [[0, 0.3, 0.2, 2.6, 1.9]], 2, { samplingRatio: 2 }), [n(11, [1, 2, 3, 4])])
  })
})

describe('sampling behaviour', () => {
  it('the identity transform resamples an image to itself', () => {
    const x = n(0, [2, 1, 3, 4])
    const g = affineGrid(identityTheta(2), x.shape)
    const y = toFlat(gridSample(x, g))
    for (const [i, v] of toFlat(x).entries()) expect(y[i]).toBeCloseTo(v, 12)
  })

  it('ROI Align samples a linear ramp at bin centres', () => {
    // f(y, x) = x on a 6 x 8 grid: the average over a bin is its centre's x.
    const data = Float64Array.from({ length: 48 }, (_, i) => i % 8)
    const ramp = fromStorage(data, [1, 1, 6, 8], 'float64')
    const y = toArray(roiAlign(ramp, [[0, 1, 1, 5, 3]], [1, 2], { aligned: true, samplingRatio: 2 })) as number[][][][]
    expect(y[0]?.[0]?.[0]?.map((v) => Math.round(v * 1e9) / 1e9)).toEqual([1.5, 3.5])
    // A constant map pools to the constant everywhere inside.
    const ones = fromStorage(new Float64Array(48).fill(2), [1, 1, 6, 8], 'float64')
    expect(toFlat(roiAlign(ones, [[0, 0.5, 0.5, 4, 3]], 3)).every((v) => Math.abs(v - 2) < 1e-12)).toBe(true)
    expect(roiAlign(ones, [] as unknown as number[][], 2).shape).toEqual([0, 1, 2, 2])
  })

  it('ROI pooling takes the maximum of each quantised bin', () => {
    const data = Float64Array.from({ length: 16 }, (_, i) => i)
    const x = fromStorage(data, [1, 1, 4, 4], 'float64')
    expect(toArray(roiPool(x, [[0, 0, 0, 3, 3]], 2))).toEqual([
      [
        [
          [5, 7],
          [13, 15],
        ],
      ],
    ])
  })

  it('modules', () => {
    const x = n(1, [1, 4, 4, 6])
    for (const m of [
      new NN.Upsample({ size: [5, 7], mode: 'bicubic' }),
      new NN.UpsamplingNearest2d(2),
      new NN.UpsamplingBilinear2d(2),
      new NN.PixelShuffle(2),
      new NN.PixelUnshuffle(2),
      new NN.ZeroPad2d([1, 2, 0, 1]),
      new NN.ReflectionPad2d(1),
      new NN.ReplicationPad1d([2, 0]),
      new NN.CircularPad3d(1),
      new NN.ConstantPad2d(1, 0.5),
    ]) {
      const input = m.kind === 'CircularPad3d' ? n(2, [1, 1, 2, 3, 3]) : x
      expect(m.forward(input).shape).toEqual(m.outputShape(input.shape))
    }
    const pe = new NN.PatchEmbedding(2, 4, 8, { key: key(3) })
    expect(pe.forward(x).shape).toEqual(pe.outputShape(x.shape))
    const loc = new NN.Sequential(new NN.Flatten(), new NN.Linear(96, 6, { key: key(4) }))
    const st = new NN.SpatialTransformer(loc)
    expect(st.forward(x).shape).toEqual([1, 4, 4, 6])
    expect(Object.keys(st.stateDict())).toEqual(['localisation.1.bias', 'localisation.1.weight'])
  })
})
