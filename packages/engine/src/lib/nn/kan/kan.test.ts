import { describe, expect, it } from 'vitest'
import { grad, numericalGradient } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import { asTensor, fromStorage, mul, sum, toFlat, type Value, zeros } from '../../tensor/index.ts'
import type { Params } from '../module/module.ts'
import { ExpansionLayer, kanBases } from './edge-bases.ts'
import { KAN } from './kan.ts'
import { KanLayer } from './kan-layer.ts'

const grid = (n: number, lo = -1, hi = 1): number[] =>
  Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>): number =>
  Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - (b[i] ?? 0))), 0)
const silu = (x: number): number => x / (1 + Math.exp(-x))

describe('KanLayer', () => {
  const layer = new KanLayer({ inFeatures: 2, outFeatures: 3, grid: 5, degree: 3, key: key(1) })

  it('has a spline per edge plus the two scales: shapes and the parameter count', () => {
    const x = asTensor([
      [0.2, -0.4],
      [0.9, 0.1],
    ])
    expect(layer.forward(x).shape).toEqual([2, 3])
    expect(layer.activations(layer.params, x as Value).shape).toEqual([2, 2, 3])
    // coefficients out x in x (grid + degree), then w_b and w_s per edge.
    expect(layer.parameterCount()).toBe(3 * 2 * 8 + 3 * 2 + 3 * 2)
    const acts = toFlat(layer.activations(layer.params, x as Value))
    const y = toFlat(layer.forward(x))
    for (let n = 0; n < 2; n++)
      for (let o = 0; o < 3; o++)
        expect(y[n * 3 + o]).toBeCloseTo((acts[n * 2 * 3 + o] as number) + (acts[(n * 2 + 1) * 3 + o] as number), 12)
  })

  it('initialises as the paper: w_s = 1/sqrt(in), w_b within +-1/sqrt(in), small spline noise', () => {
    for (const w of toFlat(layer.params['wSpline'] as Value)) expect(w).toBeCloseTo(1 / Math.sqrt(2), 14)
    for (const w of toFlat(layer.params['wBase'] as Value)) expect(Math.abs(w)).toBeLessThanOrEqual(1 / Math.sqrt(2))
    const coef = toFlat((layer.params['edge'] as Params)['coef'] as Value)
    expect(Math.max(...coef.map(Math.abs))).toBeLessThan(0.3)
  })

  it('is phi(x) = w_b silu(x) + w_s spline(x) on each edge, with the spline the edge function’s own curve', () => {
    const l = new KanLayer({ inFeatures: 2, outFeatures: 2, grid: 4, key: key(2) })
    const g = grid(11)
    const wb = toFlat(l.params['wBase'] as Value)
    const ws = toFlat(l.params['wSpline'] as Value)
    for (const [o, i] of [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ] as const) {
      const spline = toFlat(l.edge.curve(l.params['edge'] as Params, g, i, o))
      const expected = g.map(
        (v, k) => (wb[o * 2 + i] as number) * silu(v) + (ws[o * 2 + i] as number) * (spline[k] as number),
      )
      expect(maxAbs(toFlat(l.edgeCurve(l.params, g, i, o)), expected)).toBeLessThan(1e-12)
    }
  })

  it('reproduces a constant for constant coefficients, and the base function alone without the spline', () => {
    const l = new KanLayer({ inFeatures: 1, outFeatures: 1, grid: 5, key: key(3) })
    const ones = fromStorage(new Float64Array(8).fill(1), [1, 1, 8], 'float64')
    const params = {
      ...l.params,
      edge: { coef: ones },
      wSpline: asTensor([[2]]),
      wBase: asTensor([[0]]),
    }
    for (const v of toFlat(l.apply(params, asTensor(grid(9)) as Value, {}))) expect(v).toBeCloseTo(2, 12)
    const base = { ...params, wSpline: asTensor([[0]]), wBase: asTensor([[1]]) }
    const g = grid(9, -3, 3)
    expect(maxAbs(toFlat(l.edgeCurve(base, g)), g.map(silu))).toBeLessThan(1e-12)
  })

  it('stays finite and differentiable outside the grid, continuing each spline with its tangent', () => {
    const l = new KanLayer({ inFeatures: 1, outFeatures: 1, grid: 5, base: { kind: 'none' }, key: key(4), noise: 3 })
    const far = [10, 100, 1e4, -50]
    for (const v of toFlat(l.forward(asTensor(far.map((v) => [v]))))) expect(Number.isFinite(v)).toBe(true)
    const g = toFlat(grad((xs: Value) => sum(l.apply(l.params, xs)))(asTensor(far.map((v) => [v])) as Value) as Value)
    for (const v of g) expect(Number.isFinite(v)).toBe(true)
    // A straight line beyond the upper end: equal second differences.
    const [a, b, c] = [2, 3, 4].map((v) => toFlat(l.forward(asTensor([[v]])))[0] as number) as [number, number, number]
    expect(a - 2 * b + c).toBeCloseTo(0, 10)
    const held = new KanLayer({ inFeatures: 1, outFeatures: 1, outside: 'clamp', base: { kind: 'none' }, key: key(4) })
    expect(toFlat(held.forward([[5]]))[0]).toBeCloseTo(toFlat(held.forward([[1]]))[0] as number, 12)
  })

  it('has parameter gradients that match finite differences', () => {
    const l = new KanLayer({ inFeatures: 2, outFeatures: 2, grid: 4, key: key(5), noise: 2 })
    const x = asTensor([
      [0.3, -0.6],
      [-0.2, 0.8],
      [0.55, 0.1],
    ]) as Value
    for (const name of ['wSpline', 'wBase'] as const) {
      const f = (w: Value): Value =>
        sum(mul(l.apply({ ...l.params, [name]: w }, x), l.apply({ ...l.params, [name]: w }, x)))
      const at = l.params[name] as Value
      expect(maxAbs(toFlat(grad(f)(at) as Value), toFlat(numericalGradient(f, at, 1e-6)))).toBeLessThan(1e-6)
    }
    const coef = (l.params['edge'] as Params)['coef'] as Value
    const f = (c: Value): Value => sum(l.apply({ ...l.params, edge: { coef: c } }, x))
    expect(maxAbs(toFlat(grad(f)(coef) as Value), toFlat(numericalGradient(f, coef, 1e-6)))).toBeLessThan(1e-6)
  })

  it('switches an edge off with the mask buffer', () => {
    const l = new KanLayer({ inFeatures: 2, outFeatures: 1, grid: 4, key: key(6) })
    const x = asTensor([[0.4, 0.7]])
    const mask = fromStorage(Float64Array.from([1, 0]), [1, 2], 'float64')
    const buffers = { ...l.buffers, mask }
    const full = toFlat(l.activations(l.params, x as Value))
    const masked = toFlat(l.activations(l.params, x as Value, buffers))
    expect(masked[0]).toBeCloseTo(full[0] as number, 14)
    expect(masked[1]).toBe(0)
    expect(toFlat(l.edgeCurve(l.params, [0.7], 1, 0, buffers))[0]).toBe(0)
  })

  it('can add a bias per output node and keeps free knots as parameters', () => {
    const b = new KanLayer({ inFeatures: 1, outFeatures: 2, bias: true, key: key(7) })
    expect(Object.keys(b.params).sort()).toEqual(['bias', 'edge', 'wBase', 'wSpline'])
    const free = new KanLayer({
      inFeatures: 2,
      outFeatures: 1,
      grid: 4,
      knots: { kind: 'learnable' },
      key: key(8),
      noise: 3,
    })
    const edge = free.params['edge'] as Params
    expect(Object.keys(edge).sort()).toEqual(['coef', 'knotLogits'])
    const x = asTensor([
      [0.3, -0.5],
      [-0.7, 0.4],
      [0.1, 0.9],
    ]) as Value
    const f = (logits: Value): Value =>
      sum(mul(free.apply({ ...free.params, edge: { ...edge, knotLogits: logits } }, x), 1))
    const at = edge['knotLogits'] as Value
    const analytic = toFlat(grad(f)(at) as Value)
    expect(analytic.some((v) => Math.abs(v) > 1e-6)).toBe(true)
    expect(maxAbs(analytic, toFlat(numericalGradient(f, at, 1e-6)))).toBeLessThan(1e-6)
  })

  it('takes the input range from a sample', () => {
    const sample = asTensor(Array.from({ length: 20 }, (_, i) => [i / 19 - 0.5, 2 + i / 19]))
    const l = new KanLayer({ inFeatures: 2, outFeatures: 1, sample, key: key(9) })
    const r = l.inputRange()
    expect(r[0]?.[0]).toBeCloseTo(-0.5, 12)
    expect(r[1]?.[1]).toBeCloseTo(3, 12)
  })
})

describe('edge function families', () => {
  const g = grid(21, -1, 1)
  for (const kind of ['bspline', 'rbf', 'chebyshev', 'fourier'] as const) {
    it(`${kind}: its curve is the edge of terms, and its size is what the table says`, () => {
      const basis = kanBases.make({ kind })
      const edge = basis.create({
        inFeatures: 2,
        outFeatures: 2,
        grid: 4,
        degree: 3,
        knots: { kind: 'uniform' },
        outside: 'linear',
        range: [-1, 1],
        initScale: 1,
        key: key(11),
      })
      expect(edge.basisSize).toBe(basis.size(4, 3))
      const p = edge.params
      const x = asTensor(g.map((v) => [v, -v])) as Value
      const terms = toFlat(edge.terms(p, x))
      for (const [i, o] of [
        [0, 0],
        [1, 1],
        [1, 0],
      ] as const) {
        const curve = toFlat(edge.curve(p, i === 0 ? g : g.map((v) => -v), i, o))
        const column = g.map((_, n) => terms[(n * 2 + i) * 2 + o] as number)
        expect(maxAbs(curve, column)).toBeLessThan(1e-12)
      }
    })
  }

  it('the constant term of a Chebyshev or Fourier edge, and the bump of a radial one, are what they should be', () => {
    const constant = (kind: 'chebyshev' | 'fourier'): number[] => {
      const l = new ExpansionLayer({
        inFeatures: 1,
        outFeatures: 1,
        expansion: kind,
        grid: 3,
        degree: 3,
        range: [-2, 2],
      })
      const coef = new Float64Array(l.basisSize)
      coef[0] = 1.5
      return toFlat(l.curve({ coef: fromStorage(coef, [1, 1, l.basisSize], 'float64') }, grid(7, -9, 9)))
    }
    for (const kind of ['chebyshev', 'fourier'] as const) for (const v of constant(kind)) expect(v).toBeCloseTo(1.5, 12)
    const rbf = new ExpansionLayer({ inFeatures: 1, outFeatures: 1, expansion: 'rbf', grid: 4, range: [0, 4] })
    // Centres at 0, 1, 2, 3, 4: the third bump is 1 at 2 and exp(-1) one spacing away.
    const b = toFlat(rbf.basis(rbf.params, asTensor([[2], [3]]) as Value))
    expect(b[2]).toBeCloseTo(1, 14)
    expect(b[5 + 2]).toBeCloseTo(Math.exp(-1), 14)
  })

  it('Chebyshev terms are the polynomials of the squashed input', () => {
    const l = new ExpansionLayer({ inFeatures: 1, outFeatures: 1, expansion: 'chebyshev', degree: 4, range: [-1, 1] })
    const xs = [-0.8, 0.1, 0.6]
    const b = toFlat(l.basis(l.params, asTensor(xs.map((v) => [v])) as Value))
    xs.forEach((v, n) => {
      const u = Math.tanh(v)
      for (let k = 0; k < 5; k++) expect(b[n * 5 + k]).toBeCloseTo(Math.cos(k * Math.acos(u)), 12)
    })
  })

  it('Fourier terms are cosines and sines of the standardised input, and the range is a buffer', () => {
    const l = new ExpansionLayer({ inFeatures: 1, outFeatures: 1, expansion: 'fourier', grid: 2, range: [0, 2] })
    const b = toFlat(l.basis(l.params, asTensor([[0.5]]) as Value))
    const theta = (Math.PI * 0.5) / 2
    expect(b.map((v) => Math.round(v * 1e12) / 1e12)).toEqual(
      [1, Math.cos(theta), Math.sin(theta), Math.cos(2 * theta), Math.sin(2 * theta)].map(
        (v) => Math.round(v * 1e12) / 1e12,
      ),
    )
    expect(toFlat(l.buffers['range'] as Value)).toEqual([0, 2])
  })

  it('the cheaper families refuse knots they cannot place', () => {
    expect(() =>
      kanBases.make({ kind: 'rbf' }).create({
        inFeatures: 1,
        outFeatures: 1,
        grid: 4,
        degree: 3,
        knots: { kind: 'learnable' },
        outside: 'linear',
        initScale: 0.1,
      }),
    ).toThrow(/only the spline family/)
  })

  it('every family trains through a network: the gradients are finite and match finite differences', () => {
    for (const kind of ['rbf', 'chebyshev', 'fourier'] as const) {
      const net = new KAN([2, 3, 1], { basis: { kind }, grid: 3, degree: 3, noise: 2, key: key(12) })
      const x = asTensor([
        [0.2, -0.3],
        [0.7, 0.5],
        [-0.4, 0.1],
      ]) as Value
      const f = (params: Params): Value => sum(mul(net.apply(params, x), net.apply(params, x)))
      const first = (net.params['0'] as Params)['edge'] as Params
      const g = (c: Value): Value => f({ ...net.params, '0': { ...(net.params['0'] as Params), edge: { coef: c } } })
      const at = first['coef'] as Value
      const analytic = toFlat(grad(g)(at) as Value)
      expect(analytic.every(Number.isFinite)).toBe(true)
      expect(maxAbs(analytic, toFlat(numericalGradient(g, at, 1e-6)))).toBeLessThan(1e-5)
    }
  })
})

describe('KAN', () => {
  const net = new KAN([3, 4, 2], { grid: 4, key: key(21), noise: 1 })
  const x = asTensor([
    [0.1, 0.5, -0.3],
    [-0.7, 0.2, 0.9],
    [0.4, -0.4, 0.0],
    [0.8, 0.3, -0.6],
  ]) as Value

  it('stacks the layers by the widths and counts its parameters', () => {
    expect(net.widths).toEqual([3, 4, 2])
    expect(net.layers.map((l) => [l.inFeatures, l.outFeatures])).toEqual([
      [3, 4],
      [4, 2],
    ])
    expect(net.forward(x).shape).toEqual([4, 2])
    expect(net.parameterCount()).toBe((3 * 4 + 4 * 2) * (4 + 3 + 2))
    expect(Object.keys(net.params)).toEqual(['0', '1'])
  })

  it('traces the layers: inputs, edge activations and the output agree with the forward pass', () => {
    const t = net.trace(net.params, x)
    expect(t.inputs.map((v) => v.shape)).toEqual([
      [4, 3],
      [4, 4],
    ])
    expect(t.activations.map((v) => v.shape)).toEqual([
      [4, 3, 4],
      [4, 4, 2],
    ])
    expect(maxAbs(toFlat(t.output), toFlat(net.forward(x)))).toBeLessThan(1e-14)
    // The second layer's input is the sum of the first layer's edges.
    const a = toFlat(t.activations[0] as Value)
    const h = toFlat(t.inputs[1] as Value)
    for (let n = 0; n < 4; n++)
      for (let o = 0; o < 4; o++)
        expect(h[n * 4 + o]).toBeCloseTo(
          [0, 1, 2].reduce((s, i) => s + (a[(n * 3 + i) * 4 + o] as number), 0),
          12,
        )
  })

  it('gives the function on any edge, equal to the edge’s activations, and its mean magnitude as the edge scale', () => {
    const t = net.trace(net.params, x)
    const acts = toFlat(t.activations[0] as Value)
    const xin = toFlat(x)
    const curve = toFlat(
      net.edgeCurve(
        0,
        2,
        3,
        Array.from({ length: 4 }, (_, n) => xin[n * 3 + 2] as number),
      ),
    )
    expect(
      maxAbs(
        curve,
        Array.from({ length: 4 }, (_, n) => acts[(n * 3 + 2) * 4 + 3] as number),
      ),
    ).toBeLessThan(1e-12)
    const scales = net.edgeScales(net.params, x)
    expect(scales.map((s) => s.shape)).toEqual([
      [4, 3],
      [2, 4],
    ])
    const mean = [0, 1, 2, 3].reduce((s, n) => s + Math.abs(acts[(n * 3 + 2) * 4 + 3] as number), 0) / 4
    expect(toFlat(scales[0] as Value)[3 * 3 + 2]).toBeCloseTo(mean, 12)
    expect(() => net.edgeCurve(0, 3, 0, [0])).toThrow(/no edge/)
  })

  it('restores from a state dict, with its ranges and masks', () => {
    const other = new KAN([3, 4, 2], { grid: 4, key: key(99), sample: x })
    other.loadStateDict(net.stateDict(), { strict: true })
    expect(maxAbs(toFlat(other.forward(x)), toFlat(net.forward(x)))).toBeLessThan(1e-14)
  })

  it('takes the first layer’s range from a sample and the others from the default', () => {
    const sample = asTensor(Array.from({ length: 10 }, (_, i) => [i, 2 * i, -i]))
    const n = new KAN([3, 2, 1], { sample, key: key(31) })
    expect(n.layers[0]?.inputRange()[1]).toEqual([0, 18])
    expect(n.layers[1]?.inputRange()[0]).toEqual([-1, 1])
    expect(zeros([1]).shape).toEqual([1])
  })

  it('runs with free knots, other bases and without the base function', () => {
    for (const options of [
      { knots: { kind: 'learnable' } },
      { base: { kind: 'none' } },
      { basis: { kind: 'rbf' } },
    ] as const) {
      const n = new KAN([2, 3, 1], { ...options, key: key(41) })
      for (const v of toFlat(n.forward([[0.2, 0.3]]))) expect(Number.isFinite(v)).toBe(true)
    }
  })
})
