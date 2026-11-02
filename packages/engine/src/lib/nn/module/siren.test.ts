import { describe, expect, it } from 'vitest'
import { grad, gradCheck } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import { square, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import {
  FourierFeatureEncoding,
  FrequencyEncoding,
  HashGridEncoding,
  IdentityEncoding,
} from './coordinate-encodings.ts'
import { SineLayer, siren } from './siren.ts'

const maxAbs = (v: Value): number => Math.max(...Array.from(toFlat(v), Math.abs))

describe('SineLayer and siren', () => {
  it('initialises the first layer on +-1/n and later layers on +-sqrt(6/n)/omega0', () => {
    const first = new SineLayer(2, 512, { first: true, key: key(0) })
    const hidden = new SineLayer(64, 512, { omega0: 30, key: key(1) })
    expect(maxAbs(first.params['weight'] as Value)).toBeLessThanOrEqual(1 / 2)
    expect(maxAbs(first.params['weight'] as Value)).toBeGreaterThan(0.45)
    const bound = Math.sqrt(6 / 64) / 30
    expect(maxAbs(hidden.params['weight'] as Value)).toBeLessThanOrEqual(bound)
    expect(maxAbs(hidden.params['weight'] as Value)).toBeGreaterThan(0.95 * bound)
    expect(first.weightBound).toBe(0.5)
    expect(hidden.weightBound).toBeCloseTo(bound, 15)
    // Biases are PyTorch's default, +-1/sqrt(n).
    expect(maxAbs(hidden.params['bias'] as Value)).toBeLessThanOrEqual(1 / 8)
  })

  it('computes sin(omega0 (W x + b))', () => {
    const layer = new SineLayer(2, 1, { omega0: 10, key: key(0) })
    const w = toFlat(layer.params['weight'] as Value)
    const b = toFlat(layer.params['bias'] as Value)
    const y = toFlat(layer.forward([[0.3, -0.2]]))
    expect(y[0]).toBeCloseTo(Math.sin(10 * ((w[0] as number) * 0.3 + (w[1] as number) * -0.2 + (b[0] as number))), 12)
  })

  it('keeps the activations of hidden layers spread over (-1, 1) through depth (the point of the initialisation)', () => {
    const net = siren([2, 256, 256, 256, 1], { key: key(3) })
    const x = tensor(Array.from({ length: 64 }, (_, i) => [Math.sin(i), Math.cos(1.7 * i)]))
    // Run the first three sine layers by hand and look at the last hidden layer's spread.
    let h: Value = x
    for (const layer of net.layers.slice(0, 3))
      h = layer.apply(((net.params as Record<string, unknown>)[String(net.layers.indexOf(layer))] ?? {}) as never, h)
    const v = Array.from(toFlat(h))
    const rms = Math.sqrt(v.reduce((a, t) => a + t * t, 0) / v.length)
    // The arcsine law has variance 1/2.
    expect(rms).toBeGreaterThan(0.55)
    expect(rms).toBeLessThan(0.85)
  })

  it('ends in an affine layer (no sine) and is differentiable in its input twice over (smooth derivatives)', () => {
    const net = siren([1, 16, 16, 1], { key: key(2) })
    expect((net.layers.at(-1) as SineLayer).affine).toBe(true)
    expect(net.layers.slice(0, -1).every((l) => !(l as SineLayer).affine)).toBe(true)
    const f = (x: Value): Value => sum(net.apply(net.params, x))
    expect(gradCheck((x: Value) => square(f(x)), tensor([[0.2], [-0.4]]), { eps: 1e-6 }).passed).toBe(true)
    // The second derivative is available through nested autodiff.
    const second = grad((x: Value) => sum(grad(f)(x)))(tensor([[0.2], [-0.4]]))
    expect(second.shape).toEqual([2, 1])
  })

  it('needs at least two widths', () => {
    expect(() => siren([3])).toThrow(/at least/)
  })
})

describe('encoding modules', () => {
  it('report their output widths and shapes', () => {
    const enc = [
      new IdentityEncoding(2),
      new FrequencyEncoding(2, { octaves: 3, includeInput: true }),
      new FourierFeatureEncoding(2, { features: 7, sigma: 2, key: key(0) }),
      new HashGridEncoding({ dim: 2, levels: 3, log2TableSize: 6, featuresPerLevel: 2 }, { key: key(0) }),
    ]
    expect(enc.map((e) => e.outputDim)).toEqual([2, 14, 14, 6])
    for (const e of enc) {
      expect(
        e.forward([
          [0.1, 0.2],
          [0.3, 0.4],
        ]).shape,
      ).toEqual([2, e.outputDim])
      expect(e.outputShape([5, 2])).toEqual([5, e.outputDim])
    }
    expect(enc.map((e) => e.parameterCount())).toEqual([0, 0, 0, 3 * 64 * 2])
  })

  it('fixes the Fourier frequencies by the key, and initialises the hash table near zero', () => {
    const a = new FourierFeatureEncoding(2, { features: 4, key: key(5) })
    const b = new FourierFeatureEncoding(2, { features: 4, key: key(5) })
    expect(toFlat(a.matrix)).toEqual(toFlat(b.matrix))
    const h = new HashGridEncoding({ dim: 2, levels: 2, log2TableSize: 5 }, { key: key(1) })
    expect(maxAbs(h.params['table'] as Value)).toBeLessThanOrEqual(1e-4)
    expect(maxAbs(h.params['table'] as Value)).toBeGreaterThan(0)
  })
})
