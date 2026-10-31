import { describe, expect, it } from 'vitest'
import { expectClose } from '../../../testing/close.ts'
import { minimise } from '../../../testing/optimise.ts'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { add, mean, square, sub, tensor, toFlat, type Value } from '../../tensor/index.ts'
import type { Params } from '../module/module.ts'
import { DeepONet, FNO, FourierLayer, SpectralConv1d, SpectralConv2d } from './operators.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

/** A `[a, b, c]` tensor as nested arrays. */
function nested(t: Value): number[][][] {
  const [a, b, c] = t.shape as [number, number, number]
  const flat = toFlat(t)
  return Array.from({ length: a }, (_, i) =>
    Array.from({ length: b }, (_, j) => flat.slice((i * b + j) * c, (i * b + j + 1) * c)),
  )
}

/** The parameter tree with one leaf replaced, for differentiating in a single parameter. */
function replaced(params: Params, path: readonly string[], value: Value): Params {
  const [head, ...rest] = path
  if (head === undefined) return params
  return { ...params, [head]: rest.length === 0 ? value : replaced(params[head] as Params, rest, value) }
}

function leaf(params: Params, path: readonly string[]): Value {
  return path.reduce<unknown>((t, k) => (t as Params)[k], params) as Value
}

describe('DeepONet', () => {
  const sensors = 6
  const queries = [-1, -0.3, 0.4, 1]
  const operator = (u: Value): Value => tensor(toFlat(mean(u, 1)).map((m) => queries.map((v) => m * v)))

  it('has the branch-trunk dot product shape for shared and per-function query points', () => {
    const net = new DeepONet(sensors, 2, 8, { branchHidden: [10], trunkHidden: [10], key: key(0) })
    expect(net.forward({ u: n(1, [3, sensors]), y: n(2, [5, 2]) }).shape).toEqual([3, 5])
    expect(net.forward({ u: n(1, [3, sensors]), y: n(3, [3, 5, 2]) }).shape).toEqual([3, 5])
    expect(Object.keys(net.stateDict())).toContain('bias')
  })

  it('represents a linear operator exactly with a linear branch: G(u)(y) = y mean(u)', () => {
    const net = new DeepONet(sensors, 1, 1, {
      branchHidden: [],
      trunkHidden: [],
      trunkOutputActivation: false,
      key: key(4),
    })
    net.loadStateDict({
      'branch.0.weight': tensor([Array.from({ length: sensors }, () => 1 / sensors)]),
      'branch.0.bias': tensor([0]),
      'trunk.0.weight': tensor([[1]]),
      'trunk.0.bias': tensor([0]),
      bias: tensor(0),
    })
    const u = n(5, [4, sensors])
    expectClose(net.forward({ u, y: tensor(queries.map((v) => [v])) }), operator(u))
  })

  it('learns that operator from examples and generalises to new functions', () => {
    const net = new DeepONet(sensors, 1, 2, {
      branchHidden: [],
      trunkHidden: [],
      trunkOutputActivation: false,
      key: key(6),
    })
    const y = tensor(queries.map((v) => [v]))
    const u = n(7, [40, sensors])
    const fit = minimise(net.params, (p) => mean(square(sub(net.apply(p, { u, y }), operator(u)))), 600, 0.03)
    expect(fit.losses[0] as number).toBeGreaterThan(0.05)
    expect(fit.losses.at(-1) as number).toBeLessThan(1e-4)
    const test = n(8, [10, sensors])
    const err = toFlat(sub(net.apply(fit.params, { u: test, y }), operator(test))).map(Math.abs)
    expect(Math.max(...err)).toBeLessThan(0.02)
  })

  it('passes gradients to the branch and the trunk', () => {
    const small = new DeepONet(3, 1, 2, { branchHidden: [4], trunkHidden: [4], key: key(9) })
    const input = { u: n(10, [2, 3]), y: n(11, [3, 1]) }
    for (const path of [['branch', '0', 'weight'], ['trunk', '2', 'bias'], ['bias']]) {
      const f = (p: Value): Value => small.apply(replaced(small.params, path, p), input)
      expect(checkPrimitive(f, [leaf(small.params, path)], { rtol: 1e-5 }).failures, path.join('.')).toEqual([])
    }
  })
})

/** The spectral convolution with plain sums: Y_k = sum_i W_ik X_ik and y_n = (1/N) sum_k c_k Re(Y_k e^{2 pi i k n / N}). */
function reference(x: number[][][], real: number[][][], imag: number[][][], modes: number): number[][][] {
  const N = ((x[0] as number[][])[0] as number[]).length
  return x.map((batch) =>
    Array.from({ length: (real[0] as number[][]).length }, (_, o) =>
      Array.from({ length: N }, (_, t) => {
        let total = 0
        for (let k = 0; k < modes; k++) {
          let yr = 0
          let yi = 0
          for (const [i, row] of batch.entries()) {
            let xr = 0
            let xi = 0
            for (let m = 0; m < N; m++) {
              xr += (row[m] as number) * Math.cos((2 * Math.PI * k * m) / N)
              xi -= (row[m] as number) * Math.sin((2 * Math.PI * k * m) / N)
            }
            const wr = ((real[i] as number[][])[o] as number[])[k] as number
            const wi = ((imag[i] as number[][])[o] as number[])[k] as number
            yr += xr * wr - xi * wi
            yi += xr * wi + xi * wr
          }
          const c = k === 0 || (N % 2 === 0 && k === N / 2) ? 1 : 2
          total += (c * (yr * Math.cos((2 * Math.PI * k * t) / N) - yi * Math.sin((2 * Math.PI * k * t) / N))) / N
        }
        return total
      }),
    ),
  )
}

describe('spectral convolution', () => {
  it('multiplies the lowest modes by complex weights: it matches a direct evaluation of the sums', () => {
    const layer = new SpectralConv1d(2, 3, 3, { key: key(20) })
    const x = n(21, [2, 2, 8])
    const want = reference(nested(x), nested(layer.params['real'] as Value), nested(layer.params['imag'] as Value), 3)
    expectClose(layer.forward(x), tensor(want), 1e-12)
  })

  it('with one mode (k = 1) it is a circular convolution with a cosine kernel', () => {
    const layer = new SpectralConv1d(1, 1, 2, { key: key(22) })
    const wr = toFlat(layer.params['real'] as Value)[1] as number
    const wi = toFlat(layer.params['imag'] as Value)[1] as number
    // Keep only mode 1 by zeroing the mode-0 weight.
    layer.loadStateDict({ real: tensor([[[0, wr]]]), imag: tensor([[[0, wi]]]) })
    const N = 12
    const x = toFlat(n(23, [1, 1, N]))
    const kernel = Array.from(
      { length: N },
      (_, d) => (2 / N) * (wr * Math.cos((2 * Math.PI * d) / N) - wi * Math.sin((2 * Math.PI * d) / N)),
    )
    const want = Array.from({ length: N }, (_, t) =>
      x.reduce((a, v, m) => a + v * (kernel[(((t - m) % N) + N) % N] as number), 0),
    )
    expectClose(layer.forward(tensor([[x]])), want, 1e-12)
  })

  it('has no dependence on the resolution: the same function sampled finer gives the same output', () => {
    const layer = new SpectralConv1d(1, 2, 3, { key: key(24) })
    const f = (s: number): number => Math.cos(2 * Math.PI * s) + 0.5 * Math.sin(4 * Math.PI * s) + 0.25
    const sample = (N: number): number[] => Array.from({ length: N }, (_, i) => f(i / N))
    const coarse = toFlat(layer.forward(tensor([[sample(16)]])))
    const fine = toFlat(layer.forward(tensor([[sample(32)]])))
    for (let o = 0; o < 2; o++)
      for (let i = 0; i < 16; i++) expect(coarse[o * 16 + i]).toBeCloseTo(fine[o * 32 + 2 * i] as number, 12)
  })

  it('in two dimensions it passes the retained plane waves and removes the others', () => {
    const H = 8
    const W = 8
    const layer = new SpectralConv2d(1, 1, 2, 3, { key: key(25) })
    const ones = tensor([
      [
        [
          [1, 1, 1],
          [1, 1, 1],
        ],
      ],
    ])
    const none = tensor([
      [
        [
          [0, 0, 0],
          [0, 0, 0],
        ],
      ],
    ])
    layer.loadStateDict({ real1: ones, imag1: none, real2: ones, imag2: none })
    type Wave = (x: number, y: number) => number
    const wave =
      (a: number, b: number, amp: number): Wave =>
      (x, y) =>
        amp * Math.cos(2 * Math.PI * ((a * x) / H + (b * y) / W))
    const kept: Wave[] = [wave(1, 0, 1), wave(1, 2, 0.7)]
    // (3, 0) lies outside the rows [0, 2) and [6, 8); (0, 3) outside the columns [0, 3).
    const dropped: Wave[] = [wave(3, 0, 0.3), wave(0, 3, 0.4)]
    const field = (fs: Wave[]): Value =>
      tensor([
        [Array.from({ length: H }, (_, x) => Array.from({ length: W }, (_, y) => fs.reduce((a, f) => a + f(x, y), 0)))],
      ])
    expectClose(layer.forward(field([...kept, ...dropped])), field(kept), 1e-12)
  })

  it('is translation equivariant and differentiable in the weights and the field', () => {
    const layer = new SpectralConv1d(2, 2, 3, { key: key(26) })
    const x = n(27, [1, 2, 10])
    const roll = (a: number[][][]): number[][][] => a.map((b) => b.map((row) => [...row.slice(3), ...row.slice(0, 3)]))
    expectClose(layer.forward(tensor(roll(nested(x)))), tensor(roll(nested(layer.forward(x)))), 1e-12)
    for (const name of ['real', 'imag']) {
      const f = (p: Value): Value => layer.apply({ ...layer.params, [name]: p }, x)
      expect(checkPrimitive(f, [layer.params[name] as Value], { rtol: 1e-5 }).failures, name).toEqual([])
    }
    expect(checkPrimitive((v: Value) => layer.apply(layer.params, v), [x], { rtol: 1e-5 }).failures).toEqual([])
    const two = new SpectralConv2d(1, 1, 2, 2, { key: key(28) })
    const field = n(29, [1, 1, 6, 6])
    for (const name of ['real1', 'imag2']) {
      const f = (p: Value): Value => two.apply({ ...two.params, [name]: p }, field)
      expect(checkPrimitive(f, [two.params[name] as Value], { rtol: 1e-5 }).failures, name).toEqual([])
    }
  })
})

describe('Fourier neural operator', () => {
  it('assembles the lift, Fourier layers and projection, in one and two dimensions, at any resolution', () => {
    const one = new FNO(1, 2, 1, { width: 6, modes: [3], layers: 2, key: key(30) })
    expect(one.forward(n(31, [2, 2, 16])).shape).toEqual([2, 1, 16])
    expect(one.forward(n(32, [2, 2, 40])).shape).toEqual([2, 1, 40])
    const two = new FNO(2, 1, 2, { width: 4, modes: [2, 2], layers: 2, key: key(33) })
    expect(two.forward(n(34, [1, 1, 8, 6])).shape).toEqual([1, 2, 8, 6])
    expect(one.outputShape([2, 2, 16])).toEqual([2, 1, 16])
    expect(Object.keys(one.stateDict())).toContain('fourier1.spectral.real')
  })

  it('the Fourier layer adds the pointwise path to the spectral one', () => {
    const layer = new FourierLayer(3, [2], { activation: false, key: key(35) })
    const x = n(36, [1, 3, 8])
    expectClose(layer.forward(x), add(layer.part('spectral').forward(x), layer.part('pointwise').forward(x)), 1e-12)
  })

  it('a spectral layer learns a low-pass operator that doubles the lowest frequencies', () => {
    const N = 16
    const ts = Array.from({ length: N }, (_, i) => (2 * Math.PI * i) / N)
    const inputs = Array.from({ length: 24 }, (_, i) => {
      const c = toFlat(n(100 + i, [5]))
      return ts.map(
        (t) =>
          (c[0] as number) +
          (c[1] as number) * Math.cos(t) +
          (c[2] as number) * Math.sin(t) +
          (c[3] as number) * Math.cos(5 * t) +
          0.5 * (c[4] as number) * Math.sin(6 * t),
      )
    })
    const lowpass = (row: number[]): number[] => {
      const c0 = row.reduce((a, v) => a + v, 0) / N
      const c1 = (2 / N) * row.reduce((a, v, i) => a + v * Math.cos(ts[i] as number), 0)
      const s1 = (2 / N) * row.reduce((a, v, i) => a + v * Math.sin(ts[i] as number), 0)
      return ts.map((t) => 2 * (c0 + c1 * Math.cos(t) + s1 * Math.sin(t)))
    }
    const x = tensor(inputs.map((r) => [r]))
    const y = tensor(inputs.map((r) => [lowpass(r)]))
    const layer = new SpectralConv1d(1, 1, 2, { key: key(40) })
    const fit = minimise(layer.params, (p) => mean(square(sub(layer.apply(p, x), y))), 500, 0.05)
    expect(fit.losses[0] as number).toBeGreaterThan(0.1)
    expect(fit.losses.at(-1) as number).toBeLessThan(1e-6)
  })
})
