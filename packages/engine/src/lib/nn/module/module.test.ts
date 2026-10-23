import { describe, expect, it } from 'vitest'
import { checkPrimitive, valueAndGrad } from '../../autodiff/index.ts'
import { svd } from '../../numerics/linalg/index.ts'
import { key, normal } from '../../random/index.ts'
import { item, mean, mul, type Tensor, toArray, toFlat, type Value } from '../../tensor/index.ts'
import { tapContext } from '../../trace/index.ts'
import * as NN from '../index.ts'

const n = (seed: number, shape: number[]): Tensor => normal(key(seed), shape)
const flat = (v: Value): number[] => toFlat(v)

describe('module infrastructure', () => {
  it('initParams(key) gives the parameters construction with that key gives', () => {
    for (const make of [
      (k?: ReturnType<typeof key>) => new NN.Linear(3, 2, k === undefined ? {} : { key: k }),
      (k?: ReturnType<typeof key>) => new NN.Conv2d(2, 3, 3, k === undefined ? {} : { key: k }),
      (k?: ReturnType<typeof key>) => new NN.Embedding(5, 2, k === undefined ? {} : { key: k }),
    ]) {
      const a = make(key(7))
      const b = make().resetParameters(key(7))
      expect(b.stateDict()).toEqual(a.stateDict())
    }
  })

  it('containers route buffers: a forward pass in training updates a nested BatchNorm', () => {
    const net = new NN.Sequential(new NN.Linear(2, 3, { key: key(0) }), new NN.BatchNorm1d(3), new NN.ReLU())
    expect(Object.keys(net.stateDict())).toContain('1.running_mean')
    const before = flat(net.stateDict()['1.running_mean'] as Tensor)
    net.forward(n(1, [8, 2]))
    const after = flat(net.stateDict()['1.running_mean'] as Tensor)
    expect(after).not.toEqual(before)
    expect(item(net.stateDict()['1.num_batches_tracked'] as Tensor)).toBe(1)
    // The child holds the same buffers as its parent.
    expect(flat((net.layers[1] as NN.Module).buffers['running_mean'] as Value)).toEqual(after)
    // Evaluation does not change them.
    net.eval().forward(n(2, [4, 2]))
    expect(flat(net.stateDict()['1.running_mean'] as Tensor)).toEqual(after)
  })

  it('the functional form threads buffers as data and is differentiable', () => {
    const net = new NN.Sequential(new NN.Linear(2, 3, { key: key(0) }), new NN.BatchNorm1d(3))
    const f = NN.functional(net)
    const params = f.init(key(4))
    const buffers = f.initBuffers()
    const x = n(5, [6, 2])
    const { output, buffers: next } = f.applyWithState(params, buffers, x)
    expect(output.shape).toEqual([6, 3])
    expect(flat((next['1'] as NN.Params)['running_mean'] as Value)).not.toEqual([0, 0, 0])
    // Buffers held by the module are unchanged by the functional form.
    expect(flat((net.buffers['1'] as NN.Params)['running_mean'] as Value)).toEqual([0, 0, 0])
    const { grad } = valueAndGrad((p: NN.Params) => mean(mul(f.apply(p, x), f.apply(p, x))))(params)
    expect(Object.keys(NN.functional(net).init(key(4)))).toEqual(Object.keys(grad as NN.Params))
  })

  it('loads a state dict without buffers when not strict', () => {
    const bn = new NN.BatchNorm1d(2)
    bn.loadStateDict({ weight: [2, 3], bias: [0, 1] }, { strict: false })
    expect(flat(bn.params['weight'] as Value)).toEqual([2, 3])
    expect(() => bn.loadStateDict({ weight: [2, 3], bias: [0, 1] })).toThrow()
  })

  it('freezing marks parameters in the trainable mask', () => {
    const net = new NN.Sequential(new NN.Linear(2, 2, { key: key(0) }), new NN.Linear(2, 1, { key: key(1) }))
    ;(net.layers[0] as NN.Module).requiresGrad(false)
    const mask = NN.functional(net) && net.trainableMask()
    expect(flat((mask['0'] as NN.Params)['weight'] as Value)).toEqual([0])
    expect(flat((mask['1'] as NN.Params)['weight'] as Value)).toEqual([1])
  })

  it('taps record activations under module paths', () => {
    const taps = tapContext()
    const probe = new NN.Lambda((v, ctx) => NN.tap(ctx, 'h', v))
    const net = new NN.Sequential(new NN.Linear(2, 2, { key: key(0) }), probe)
    net.forward(n(0, [1, 2]), { taps, path: 'net' })
    expect([...taps.entries.keys()]).toEqual(['net.1.h'])
  })
})

describe('containers', () => {
  it('ModuleList and ModuleDict hold and name modules', () => {
    const list = new NN.ModuleList([
      new NN.Linear(2, 2, { key: key(0) }),
      new NN.ReLU(),
      new NN.Linear(2, 1, { key: key(1) }),
    ])
    expect(Object.keys(list.stateDict())).toEqual(['0.bias', '0.weight', '2.bias', '2.weight'])
    expect(list.at(2).kind).toBe('Linear')
    expect(list.length).toBe(3)
    expect(() => list.at(5)).toThrow()
    const dict = new NN.ModuleDict({
      enc: new NN.Linear(2, 2, { key: key(0) }),
      dec: new NN.Linear(2, 2, { key: key(1) }),
    })
    expect(dict.keys()).toEqual(['enc', 'dec'])
    expect(dict.get('dec').kind).toBe('Linear')
    expect(() => dict.get('x')).toThrow()
  })

  it('Residual adds the shortcut and Parallel combines branches', () => {
    const body = new NN.Linear(3, 3, { key: key(0) })
    const r = new NN.Residual(body)
    const x = n(1, [2, 3])
    expect(flat(r.forward(x))).toEqual(flat(body.forward(x)).map((v, i) => v + (flat(x)[i] ?? 0)))
    const proj = new NN.Residual(new NN.Linear(3, 4, { key: key(2) }), new NN.Linear(3, 4, { key: key(3) }))
    expect(proj.forward(x).shape).toEqual([2, 4])
    expect(Object.keys(proj.stateDict())).toEqual(['body.bias', 'body.weight', 'shortcut.bias', 'shortcut.weight'])
    const branches = [new NN.Linear(3, 2, { key: key(4) }), new NN.Linear(3, 2, { key: key(5) })]
    expect(new NN.Parallel(branches, 'concat').forward(x).shape).toEqual([2, 4])
    expect(new NN.Parallel(branches, 'stack', 1).forward(x).shape).toEqual([2, 2, 2])
    expect(new NN.Parallel(branches, 'stack', 1).outputShape([2, 3])).toEqual([2, 2, 2])
    expect(new NN.Parallel(branches).forward(x).shape).toEqual([2, 2])
    expect(new NN.Lambda((v) => mul(v, 2)).forward([1, 2]).shape).toEqual([2])
  })
})

describe('layer modules', () => {
  it('convolutions and pooling match their functions and shapes', () => {
    const conv = new NN.Conv2d(2, 3, 3, { padding: 1, stride: 2, key: key(0) })
    const x = n(1, [1, 2, 5, 5])
    const y = conv.forward(x)
    expect(y.shape).toEqual(conv.outputShape(x.shape))
    expect(flat(y)).toEqual(
      flat(NN.conv2d(x, conv.params['weight'] as Value, { bias: conv.params['bias'] as Value, padding: 1, stride: 2 })),
    )
    const bound = 1 / Math.sqrt(2 * 9)
    expect(flat(conv.params['weight'] as Value).every((v) => Math.abs(v) <= bound)).toBe(true)
    expect(conv.extraRepr()).toBe('2, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)')
    const t = new NN.ConvTranspose2d(3, 2, 4, { stride: 2, padding: 1, key: key(2) })
    expect(t.forward(y).shape).toEqual(t.outputShape(y.shape))
    for (const pool of [
      new NN.MaxPool2d(2),
      new NN.AvgPool2d(3, { stride: 1, padding: 1 }),
      new NN.AdaptiveAvgPool2d([2, 3]),
      new NN.AdaptiveMaxPool2d(1),
      new NN.LPPool2d(2, 2),
      new NN.GlobalPool('max'),
    ]) {
      expect(pool.forward(x).shape).toEqual(pool.outputShape(x.shape))
    }
    const c1 = new NN.Conv1d(2, 4, 3, { groups: 2, paddingMode: 'circular', padding: 1, key: key(3) })
    expect(c1.forward(n(4, [2, 2, 6])).shape).toEqual([2, 4, 6])
    expect(new NN.Conv3d(1, 2, 2, { key: key(5) }).forward(n(6, [1, 1, 3, 3, 3])).shape).toEqual([1, 2, 2, 2, 2])
    expect(new NN.Unfold([2, 2]).forward(x).shape).toEqual(new NN.Unfold([2, 2]).outputShape(x.shape))
  })

  it('modules pass the gradient conformance check', () => {
    const conv = new NN.Conv1d(2, 2, 3, { padding: 1, key: key(0) })
    const w0 = conv.params['weight'] as Value
    const b0 = conv.params['bias'] as Value
    expect(
      checkPrimitive((x, w, b) => conv.apply({ weight: w, bias: b }, x), [n(1, [1, 2, 4]), w0, b0], { rtol: 1e-5 })
        .failures,
    ).toEqual([])
    const bn = new NN.BatchNorm2d(2)
    expect(
      checkPrimitive((x, w, b) => bn.apply({ weight: w, bias: b }, x), [n(2, [2, 2, 2, 2]), n(3, [2]), n(4, [2])], {
        rtol: 1e-5,
      }).failures,
    ).toEqual([])
    const ln = new NN.LayerNorm(3)
    expect(checkPrimitive((x) => ln.apply(ln.params, x), [n(5, [2, 3])], { rtol: 1e-5 }).failures).toEqual([])
  })

  it('embedding, bilinear, flatten and unflatten modules', () => {
    const e = new NN.Embedding(6, 3, { paddingIdx: 0, key: key(0) })
    expect(flat(e.forward([0])).every((v) => v === 0)).toBe(true)
    const pre = NN.Embedding.fromPretrained([
      [1, 2],
      [3, 4],
    ])
    expect(toArray(pre.forward([1]))).toEqual([[3, 4]])
    expect(pre.trainable).toBe(false)
    const bag = new NN.EmbeddingBag(6, 2, { mode: 'sum', key: key(1) })
    expect(bag.forward({ input: [1, 2, 3], offsets: [0, 1] }).shape).toEqual([2, 2])
    expect(new NN.Bilinear(2, 3, 4, { key: key(2) }).parameterCount()).toBe(4 * 2 * 3 + 4)
    expect(new NN.Flatten(0).outputShape([2, 3, 4])).toEqual([24])
    expect(new NN.Flatten(1, 2).outputShape([2, 3, 4, 5])).toEqual([2, 12, 5])
    expect(new NN.Unflatten(1, [2, -1]).outputShape([3, 8])).toEqual([3, 2, 4])
    expect(() => new NN.Unflatten(1, [3, 3]).outputShape([3, 8])).toThrow()
  })

  it('activation modules', () => {
    const x = n(0, [2, 4])
    for (const m of [
      new NN.ReLU6(),
      new NN.ELU(),
      new NN.CELU(),
      new NN.SELU(),
      new NN.Mish(),
      new NN.Softplus(),
      new NN.Softsign(),
      new NN.LogSigmoid(),
      new NN.Hardtanh(),
      new NN.Hardswish(),
      new NN.Tanhshrink(),
      new NN.Softshrink(),
      new NN.Hardshrink(),
      new NN.Threshold(0, 1),
      new NN.Softmax(),
      new NN.LogSoftmax(),
      new NN.Softmin(),
      new NN.PReLU(),
    ]) {
      expect(m.forward(x).shape).toEqual([2, 4])
    }
    expect(new NN.GLU().forward(x).shape).toEqual([2, 2])
    expect(new NN.PReLU(4).forward(n(1, [2, 4, 3])).shape).toEqual([2, 4, 3])
  })

  it('regularisation modules need a key in training and are the identity in evaluation', () => {
    const x = n(0, [4, 3, 2])
    for (const m of [
      new NN.Dropout1d(0.5),
      new NN.Dropout2d(0.5),
      new NN.AlphaDropout(0.3),
      new NN.FeatureAlphaDropout(0.3),
      new NN.DropPath(0.5),
    ]) {
      expect(() => m.forward(x)).toThrow()
      expect(m.forward(x, { key: key(1) }).shape).toEqual([4, 3, 2])
      expect(flat(m.eval().forward(x))).toEqual(flat(x))
    }
    expect(new NN.GaussianNoise(0.1).forward(x, { key: key(2) }).shape).toEqual([4, 3, 2])
    expect(new NN.Cutout(1).forward(n(3, [1, 1, 3, 3]), { key: key(3) }).shape).toEqual([1, 1, 3, 3])
  })
})

describe('parametrisations', () => {
  it('weight normalisation starts at the original weight and trains g and v', () => {
    const lin = new NN.Linear(3, 2, { key: key(0) })
    const x = n(1, [4, 3])
    const before = flat(lin.forward(x))
    const wn = new NN.WeightNorm(lin)
    flat(wn.forward(x)).forEach((v, i) => {
      expect(v).toBeCloseTo(before[i] ?? 0, 12)
    })
    expect((wn.params['weight_g'] as Value).shape).toEqual([2, 1])
    expect(new NN.WeightNorm(new NN.Linear(3, 2, { key: key(0) }), { dim: null }).params['weight_g']).toBeDefined()
  })

  it('spectral normalisation updates its vectors in training only', () => {
    const sn = new NN.SpectralNorm(new NN.Linear(4, 3, { key: key(0) }), { key: key(1) })
    const u0 = flat(sn.buffers['weight_u'] as Value)
    const x = n(2, [2, 4])
    sn.forward(x)
    const u1 = flat(sn.buffers['weight_u'] as Value)
    expect(u1).not.toEqual(u0)
    sn.eval().forward(x)
    expect(flat(sn.buffers['weight_u'] as Value)).toEqual(u1)
    for (let i = 0; i < 40; i++) sn.train().forward(x)
    const top = toFlat(svd(sn.params['weight_orig'] as Value).s)[0] ?? 0
    expect(NN.spectralNormEstimate(sn)).toBeCloseTo(top, 4)
    expect(new NN.SpectralNorm(new NN.ConvTranspose1d(2, 3, 2, { key: key(3) }), { key: key(4) }).dim).toBe(1)
  })
})
