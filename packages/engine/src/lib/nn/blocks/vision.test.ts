import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { toFlat, type Value } from '../../tensor/index.ts'
import {
  BasicBlock,
  Bottleneck,
  ConvNeXtBlock,
  DoubleConv,
  FeaturePyramid,
  MBConv,
  SqueezeExcitation,
  toTokens,
  UNetDown,
  UNetUp,
} from './vision.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

describe('vision blocks', () => {
  it('residual blocks match their shapes and torchvision names, and update batch statistics', () => {
    const b = new BasicBlock(4, 6, { stride: 2, key: key(0) })
    const x = n(1, [2, 4, 6, 6])
    expect(b.forward(x).shape).toEqual(b.outputShape(x.shape))
    const names = Object.keys(b.stateDict())
    for (const k of ['conv1.weight', 'bn1.running_mean', 'downsample.0.weight', 'downsample.1.weight'])
      expect(names).toContain(k)
    expect(toFlat(b.stateDict()['bn1.num_batches_tracked'] ?? n(0, [1]))).toEqual([1])
    const same = new BasicBlock(4, 4, { key: key(2) })
    expect(Object.keys(same.stateDict()).some((k) => k.startsWith('downsample'))).toBe(false)
    const bott = new Bottleneck(8, 2, { stride: 2, key: key(3) })
    expect(bott.forward(n(4, [1, 8, 4, 4])).shape).toEqual([1, 8, 2, 2])
  })

  it('squeeze-excitation, ConvNeXt and MBConv', () => {
    const x = n(5, [1, 4, 5, 5])
    const se = new SqueezeExcitation(4, 2, { key: key(6) })
    expect(se.forward(x).shape).toEqual([1, 4, 5, 5])
    const cn = new ConvNeXtBlock(4, { layerScale: 0.5, key: key(7) })
    expect(cn.eval().forward(x).shape).toEqual([1, 4, 5, 5])
    expect(toFlat(cn.params['gamma'] as Value)).toEqual([0.5, 0.5, 0.5, 0.5])
    const mb = new MBConv(4, 4, { key: key(8) })
    expect(mb.residual).toBe(true)
    expect(mb.eval().forward(x).shape).toEqual([1, 4, 5, 5])
    expect(new MBConv(4, 6, { stride: 2, expandRatio: 1, key: key(9) }).eval().forward(x).shape).toEqual([1, 6, 3, 3])
  })

  it('U-Net stages and the feature pyramid', () => {
    const down = new UNetDown(2, 4, { key: key(10) })
    const skip = n(11, [1, 2, 8, 8])
    const h = down.forward(skip)
    expect(h.shape).toEqual([1, 4, 4, 4])
    const up = new UNetUp(4, 2, { key: key(12) })
    expect(up.forward([h, skip]).shape).toEqual([1, 2, 8, 8])
    expect(new UNetUp(4, 2, { bilinear: true, key: key(13) }).forward([h, skip]).shape).toEqual([1, 2, 8, 8])
    expect(() => up.forward([h, n(14, [1, 2, 6, 6])])).toThrow()
    const fpn = new FeaturePyramid([2, 4], 3, { key: key(15) })
    expect(fpn.forward([skip, h]).map((f) => f.shape)).toEqual([
      [1, 3, 8, 8],
      [1, 3, 4, 4],
    ])
    expect(toTokens(h).shape).toEqual([1, 16, 4])
  })

  it('blocks pass the gradient conformance check (evaluation mode)', () => {
    const b = new DoubleConv(1, 2, { key: key(16) }).eval()
    const f = (x: Value): Value => b.apply(b.params, x, { train: false })
    expect(checkPrimitive(f, [n(17, [1, 1, 3, 3])], { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
    const se = new SqueezeExcitation(2, 1, { key: key(18) })
    expect(checkPrimitive((x) => se.apply(se.params, x), [n(19, [1, 2, 2, 2])], { rtol: 1e-5 }).failures).toEqual([])
  })
})
