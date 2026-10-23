import { describe, expect, it } from 'vitest'
import { checkPrimitive, jacfwd } from '../../autodiff/index.ts'
import { det } from '../../numerics/linalg/index.ts'
import { key, normal } from '../../random/index.ts'
import { item, reshape, toFlat, type Value } from '../../tensor/index.ts'
import { Linear } from '../module/layers.ts'
import {
  AdditiveCoupling,
  AffineCoupling,
  DiffusionDownStage,
  DiffusionResBlock,
  DiffusionUpStage,
  fourierFrequencies,
  noiseLevelEmbedding,
  SpatialSelfAttention,
  TimeEmbedding,
  timestepEmbedding,
} from './generative.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

describe('generative blocks', () => {
  it('time and noise-level embeddings', () => {
    const e = toFlat(timestepEmbedding([3], 6, 100))
    expect(e[1]).toBeCloseTo(Math.cos(3 * 100 ** (-1 / 3)), 14)
    expect(e[4]).toBeCloseTo(Math.sin(3 * 100 ** (-1 / 3)), 14)
    expect(timestepEmbedding([1, 2], 5).shape).toEqual([2, 5])
    expect(noiseLevelEmbedding([1], fourierFrequencies(key(0), 6)).shape).toEqual([1, 6])
    expect(new TimeEmbedding(8, 12, { key: key(1) }).forward([10]).shape).toEqual([1, 12])
    expect(checkPrimitive((t) => timestepEmbedding(t, 4), [n(2, [3])]).failures).toEqual([])
  })

  it('a U-Net down and up stage round-trip the resolution with skips', () => {
    const t = n(3, [2, 8])
    const down = new DiffusionDownStage(2, 4, 8, { groups: 2, attention: true, heads: 2, key: key(4) })
    const x = n(5, [2, 2, 8, 8])
    const d = down.forward({ x, t })
    expect(d.output.shape).toEqual([2, 4, 4, 4])
    expect(d.skips.map((s) => s.shape)).toEqual([
      [2, 4, 8, 8],
      [2, 4, 8, 8],
    ])
    const up = new DiffusionUpStage(4, [4, 4], 2, 8, { groups: 2, upsample: false, key: key(6) })
    expect(up.forward({ x: d.skips[1] as Value, t, skips: [...d.skips].reverse() }).shape).toEqual([2, 2, 8, 8])
    expect(() => up.forward({ x: d.output, t, skips: [] })).toThrow()
    const res = new DiffusionResBlock(2, 2, 8, { groups: 1, key: key(7) })
    expect(Object.keys(res.stateDict())).not.toContain('conv_shortcut.weight')
    expect(new SpatialSelfAttention(4, { groups: 2, key: key(8) }).forward(n(9, [1, 4, 2, 3])).shape).toEqual([
      1, 4, 2, 3,
    ])
  })

  it('coupling layers invert exactly with the log-determinant of their Jacobian', () => {
    const affine = new AffineCoupling(new Linear(4, 8, { key: key(10) }), 4, { parity: 1 })
    const x = n(11, [1, 4])
    const { output, logDet } = affine.forward(x)
    const back = affine.inverse(affine.params, output).output
    for (const [i, v] of toFlat(back).entries()) expect(v).toBeCloseTo(toFlat(x)[i] ?? 0, 12)
    // log |det J| from the Jacobian of the map at one example.
    const f = (v: Value): Value => reshape(affine.apply(affine.params, reshape(v, [1, 4])).output, [4])
    const jac = jacfwd(f)(reshape(x, [4]))
    expect(Math.log(Math.abs(item(det(jac))))).toBeCloseTo(item(logDet), 10)
    const additive = new AdditiveCoupling(new Linear(2, 2, { key: key(12) }), 2)
    const y = additive.forward(n(13, [3, 2]))
    expect(toFlat(y.logDet)).toEqual([0, 0, 0])
    for (const [i, v] of toFlat(additive.inverse(additive.params, y.output).output).entries())
      expect(v).toBeCloseTo(toFlat(n(13, [3, 2]))[i] ?? 0, 12)
  })
})
