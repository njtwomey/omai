import { describe, expect, it } from 'vitest'
import { grad, numericalGradient } from '../../autodiff/index.ts'
import { xy } from '../../data/index.ts'
import { mse } from '../../losses/index.ts'
import { adam } from '../../optim/index.ts'
import { key } from '../../random/index.ts'
import { compile, evaluate } from '../../symbolic/index.ts'
import { asTensor, fromStorage, sum, toFlat, type Value } from '../../tensor/index.ts'
import { grevilleAbscissae } from '../additive/bspline.ts'
import type { SplineLayer } from '../additive/spline-layer.ts'
import type { Params } from '../module/module.ts'
import { fit as fitModule } from '../training/fit.ts'
import { refineGrid, updateGrid } from './grid.ts'
import { KAN, type KanState } from './kan.ts'
import { pruneKan } from './prune.ts'
import { edgeSparsity, kanPenalties } from './sparsity.ts'
import { evaluateSymbolic, fixSymbolic, rankSymbolic, suggestSymbolic, symbolicFormula } from './symbolic.ts'

const grid = (n: number, lo = -1, hi = 1): number[] =>
  Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>): number =>
  Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - (b[i] ?? 0))), 0)

/**
 * A state of `kan` whose edges are the given functions (keyed `layer:in:out`), by setting the spline coefficients to the
 * function at the Greville abscissae (a quasi-interpolant, within 1e-3 of a smooth function on a grid of 20 intervals);
 * every other edge is zero, `w_s = 1` and `w_b = 0`.
 */
function plant(kan: KAN, edges: Readonly<Record<string, (x: number) => number>>): KanState {
  const params: Record<string, unknown> = {}
  kan.layers.forEach((layer, l) => {
    const p = (kan.params[String(l)] ?? {}) as Params
    const b = (kan.buffers[String(l)] ?? {}) as Params
    const edge = layer.edge as SplineLayer
    const abscissae = toFlat(grevilleAbscissae(edge.knotVectors(p['edge'] as Params, b['edge'] as Params), edge.degree))
    const k = edge.basisSize
    const coef = new Float64Array(layer.outFeatures * layer.inFeatures * k)
    for (let o = 0; o < layer.outFeatures; o++)
      for (let i = 0; i < layer.inFeatures; i++) {
        const f = edges[`${l}:${i}:${o}`]
        if (f !== undefined)
          for (let j = 0; j < k; j++) coef[(o * layer.inFeatures + i) * k + j] = f(abscissae[i * k + j] as number)
      }
    params[String(l)] = {
      ...p,
      edge: { coef: fromStorage(coef, [layer.outFeatures, layer.inFeatures, k], 'float64') },
      wSpline: fromStorage(
        new Float64Array(layer.outFeatures * layer.inFeatures).fill(1),
        [layer.outFeatures, layer.inFeatures],
        'float64',
      ),
      wBase: fromStorage(
        new Float64Array(layer.outFeatures * layer.inFeatures),
        [layer.outFeatures, layer.inFeatures],
        'float64',
      ),
    }
  })
  return { params: params as Params, buffers: kan.buffers }
}

const rows = (n: number, d: number, lo = -1, hi = 1): Value =>
  asTensor(
    Array.from({ length: n }, (_, r) =>
      Array.from({ length: d }, (__, i) => lo + ((hi - lo) * ((r * (2 * i + 3) * 7 + i * 5) % n)) / (n - 1)),
    ),
  ) as Value

describe('refineGrid and updateGrid', () => {
  const net = new KAN([2, 3, 1], { grid: 5, noise: 3, key: key(1) })
  const x = rows(60, 2)

  it('refines a cubic spline network to a grid twice as fine without changing its function', () => {
    const r = refineGrid(net, net.state, 10, x)
    expect(r.kan.spec.grid).toBe(10)
    expect(r.kan.layers[0]?.edge.basisSize).toBe(13)
    expect(r.drift.max).toBeLessThan(1e-7)
    // Off the sample too: the function agrees everywhere, not only where the data was.
    const probe = rows(37, 2)
    expect(
      maxAbs(
        toFlat(r.kan.apply(r.state.params, probe, { buffers: r.state.buffers })),
        toFlat(net.apply(net.params, probe, { buffers: net.buffers })),
      ),
    ).toBeLessThan(1e-7)
  })

  it('keeps the scales and the masks, and a higher degree refits closely', () => {
    // Cubic splines are not inside quartic splines on the same knots, so this one is a fit, not an identity.
    const r = refineGrid(net, net.state, 5, x, { degree: 4 })
    expect(r.kan.spec.degree).toBe(4)
    expect(r.drift.rms).toBeLessThan(0.05)
    expect(toFlat((r.state.params['0'] as Params)['wSpline'] as Value)).toEqual(
      toFlat((net.params['0'] as Params)['wSpline'] as Value),
    )
  })

  it('refits to the best function in a space that does not contain the old one (a grid that is not a multiple)', () => {
    const r = refineGrid(net, net.state, 7, x)
    expect(r.drift.rms).toBeLessThan(0.05)
    expect(r.drift.rms).toBeGreaterThan(1e-9)
  })

  it('is exact for nested polynomial and Fourier families when the size grows, and close for radial ones', () => {
    for (const [basis, drift] of [
      [{ kind: 'chebyshev' }, 1e-7],
      [{ kind: 'fourier' }, 1e-7],
    ] as const) {
      const n = new KAN([2, 2, 1], { basis, grid: 3, degree: 3, noise: 3, key: key(2) })
      const r = refineGrid(n, n.state, 6, x, basis.kind === 'chebyshev' ? { degree: 6 } : {})
      expect(r.drift.max).toBeLessThan(drift)
    }
    const rbf = new KAN([2, 2, 1], { basis: { kind: 'rbf' }, grid: 4, noise: 3, key: key(3) })
    expect(refineGrid(rbf, rbf.state, 8, x).drift.rms).toBeLessThan(0.05)
  })

  it('moves the range of every layer to the data, keeping the function where the old range had it', () => {
    const wide = asTensor(Array.from({ length: 50 }, (_, r) => [-3 + (6 * r) / 49, 2 + (r % 7) / 3]))
    const u = updateGrid(net, net.state, wide)
    const r0 = u.kan.layers[0]?.inputRange() ?? []
    expect(r0[0]?.[0]).toBeCloseTo(-3.3, 10)
    expect(r0[0]?.[1]).toBeCloseTo(3.3, 10)
    expect(r0[1]?.[0]).toBeLessThan(2)
    // The hidden layer's range now covers what the first layer sends it.
    const hidden = toFlat(net.trace(net.params, wide).inputs[1] as Value)
    const r1 = u.kan.layers[1]?.inputRange() ?? []
    for (let i = 0; i < 3; i++) {
      const col = Array.from({ length: 50 }, (_, r) => hidden[r * 3 + i] as number)
      expect(r1[i]?.[0]).toBeLessThan(Math.min(...col))
      expect(r1[i]?.[1]).toBeGreaterThan(Math.max(...col))
    }
    expect(u.drift.rms).toBeLessThan(0.1)
  })

  it('lets training go on from the refined network and lower the loss further', () => {
    const f = (v: number): number => Math.sin(3 * Math.PI * v)
    const xs = grid(80)
    const data = xy(
      xs.map((v) => [v]),
      xs.map((v) => [f(v)]),
    )
    const coarse = new KAN([1, 1], { grid: 3, key: key(5) })
    const train = (model: KAN, state: KanState, epochs: number): { loss: number; state: KanState } => {
      model.setParams(state.params)
      model.setBuffers(state.buffers)
      const r = fitModule(model, data, {
        loss: (p, t) => mse(p, t),
        optimizer: adam({ learningRate: 0.05 }),
        epochs,
        key: key(6),
      })
      return { loss: r.losses.at(-1) ?? Number.NaN, state: { params: r.params, buffers: r.buffers } }
    }
    const a = train(coarse, coarse.state, 400)
    const refined = refineGrid(coarse, a.state, 12, data.x)
    // The loss is where it was at the moment of refinement...
    const at = toFlat(
      mse(
        refined.kan.apply(refined.state.params, data.x, { buffers: refined.state.buffers }),
        asTensor(xs.map(f).map((v) => [v])),
      ),
    )[0] as number
    expect(at).toBeCloseTo(
      toFlat(
        mse(coarse.apply(a.state.params, data.x, { buffers: a.state.buffers }), asTensor(xs.map(f).map((v) => [v]))),
      )[0] as number,
      6,
    )
    // ...and the extra capacity then lowers it.
    const b = train(refined.kan, refined.state, 600)
    expect(b.loss).toBeLessThan(a.loss * 0.5)
  })
})

describe('sparsification penalties', () => {
  const net = new KAN([2, 3, 1], { grid: 4, noise: 1, key: key(7) })
  const x = rows(30, 2)

  it('are the sum of the edge magnitudes and the per-layer entropy of their shares', () => {
    const scales = net.edgeScales(net.params, x).map((s) => toFlat(s))
    const p = kanPenalties(net, net.params, x, { l1: 1, entropy: 2 })
    expect(toFlat(p.l1)[0]).toBeCloseTo(
      scales.flat().reduce((a, b) => a + b, 0),
      12,
    )
    const entropy = scales.reduce((acc, s) => {
      const total = s.reduce((a, b) => a + b, 0)
      return acc - s.reduce((a, v) => a + (v / total) * Math.log(v / total), 0)
    }, 0)
    expect(toFlat(p.entropy)[0]).toBeCloseTo(entropy, 9)
    expect(toFlat(p.total)[0]).toBeCloseTo((toFlat(p.l1)[0] as number) + 2 * entropy, 9)
    expect(toFlat(kanPenalties(net, net.params, x, {}).total)[0]).toBe(0)
  })

  it('have the entropy of one edge near 0 and of equal edges at log(count)', () => {
    const one = plant(new KAN([2, 1], { grid: 5, range: [-1, 1], key: key(8) }), { '0:0:0': (v) => v })
    const equal = plant(new KAN([2, 1], { grid: 5, range: [-1, 1], key: key(8) }), {
      '0:0:0': (v) => v,
      '0:1:0': (v) => -v,
    })
    const two = new KAN([2, 1], { grid: 5, range: [-1, 1], key: key(8) })
    const sym = asTensor(grid(41).map((v) => [v, v]))
    expect(toFlat(kanPenalties(two, one.params, sym, {}).entropy)[0]).toBeLessThan(1e-9)
    expect(toFlat(kanPenalties(two, equal.params, sym, {}).entropy)[0]).toBeCloseTo(Math.log(2), 6)
  })

  it('differentiate in the parameters: the gradient of the total matches finite differences', () => {
    const f = (w: Value): Value =>
      kanPenalties(net, { ...net.params, '0': { ...(net.params['0'] as Params), wSpline: w } }, x, {
        l1: 1,
        entropy: 1,
        coef: 0.5,
      }).total
    const at = (net.params['0'] as Params)['wSpline'] as Value
    const analytic = toFlat(grad(f)(at) as Value)
    expect(analytic.every(Number.isFinite)).toBe(true)
    expect(maxAbs(analytic, toFlat(numericalGradient(f, at, 1e-6)))).toBeLessThan(1e-5)
    const g = (c: Value): Value =>
      kanPenalties(net, { ...net.params, '1': { ...(net.params['1'] as Params), edge: { coef: c } } }, x, { coef: 1 })
        .total
    const c0 = ((net.params['1'] as Params)['edge'] as Params)['coef'] as Value
    expect(maxAbs(toFlat(grad(g)(c0) as Value), toFlat(numericalGradient(g, c0, 1e-6)))).toBeLessThan(1e-5)
  })

  it('count the negligible edges among those switched on', () => {
    const planted = new KAN([2, 2, 1], { grid: 5, range: [-1, 1], key: key(9) })
    const s = plant(planted, { '0:0:0': Math.sin, '1:0:0': (v) => v })
    const r = edgeSparsity(planted, s, rows(40, 2))
    expect(r.edges).toBe(6)
    expect(r.active).toBe(6)
    expect(r.negligible).toBe(4)
    expect(r.fraction).toBeCloseTo(4 / 6, 12)
  })
})

describe('pruneKan', () => {
  const net = new KAN([3, 3, 1], { grid: 6, range: [-1, 1], key: key(10) })
  const planted = plant(net, {
    '0:0:0': (v) => Math.sin(Math.PI * v),
    '0:1:1': (v) => v * v,
    '1:0:0': (v) => v,
    '1:1:0': (v) => 2 * v,
  })
  const x = rows(60, 3)

  it('removes the nodes that cannot reach the output and keeps the function', () => {
    const p = pruneKan(net, planted, x)
    expect(p.kan.widths).toEqual([3, 2, 1])
    expect(p.kept).toEqual([[0, 1, 2], [0, 1], [0]])
    expect(p.parameters.after).toBeLessThan(p.parameters.before)
    const before = toFlat(net.apply(planted.params, x, { buffers: planted.buffers }))
    const after = toFlat(p.kan.apply(p.state.params, x, { buffers: p.state.buffers }))
    expect(maxAbs(after, before)).toBeLessThan(1e-12)
    // The scores that decided it: the dead node has none going in or out.
    expect(p.scores[1]?.incoming[2]).toBeLessThan(1e-6)
    expect(p.scores[1]?.outgoing[0]).toBeGreaterThan(0.1)
  })

  it('switches off the weak edges among the survivors, with their own threshold', () => {
    const weak = plant(net, {
      '0:0:0': (v) => Math.sin(Math.PI * v),
      '0:2:0': (v) => 0.001 * v,
      '1:0:0': (v) => v,
    })
    const p = pruneKan(net, weak, x, { edgeThreshold: 0.01 })
    expect(p.kan.widths).toEqual([3, 1, 1])
    const mask = toFlat((p.state.buffers['0'] as Params)['mask'] as Value)
    expect(mask).toEqual([1, 0, 0])
    expect(p.edgesOff).toBe(2)
    // With a lower threshold the weak edge stays.
    expect(
      toFlat((pruneKan(net, weak, x, { edgeThreshold: 1e-6 }).state.buffers['0'] as Params)['mask'] as Value),
    ).toEqual([1, 0, 1])
  })

  it('can drop inputs that feed nothing, and keeps one node per layer at least', () => {
    const p = pruneKan(net, planted, x, { pruneInputs: true })
    expect(p.kept[0]).toEqual([0, 1])
    expect(p.kan.widths).toEqual([2, 2, 1])
    const xr = asTensor(
      Array.from({ length: 60 }, (_, r) => [toFlat(x)[r * 3] as number, toFlat(x)[r * 3 + 1] as number]),
    )
    expect(
      maxAbs(
        toFlat(p.kan.apply(p.state.params, xr, { buffers: p.state.buffers })),
        toFlat(net.apply(planted.params, x, { buffers: planted.buffers })),
      ),
    ).toBeLessThan(1e-12)
    const all = pruneKan(net, planted, x, { nodeThreshold: 1e9 })
    expect(all.kan.widths).toEqual([3, 1, 1])
  })

  it('works for the cheaper families, free knots and a bias', () => {
    for (const options of [
      { basis: { kind: 'chebyshev' } },
      { knots: { kind: 'learnable' } },
      { bias: true },
    ] as const) {
      const n = new KAN([2, 3, 1], { ...options, key: key(11), noise: 2 })
      const p = pruneKan(n, n.state, rows(20, 2), { nodeThreshold: 0.05 })
      for (const v of toFlat(p.kan.apply(p.state.params, rows(5, 2), { buffers: p.state.buffers })))
        expect(Number.isFinite(v)).toBe(true)
    }
  })
})

describe('symbolic regression on the edges', () => {
  const xs = grid(80, 0.5, 3)

  it('names the planted function of a table first, up to the affine fit a f(b x + c) + d', () => {
    const cases: [string, (x: number) => number][] = [
      ['sin', (x) => 1.5 * Math.sin(2 * x + 0.3) - 0.2],
      ['x^2', (x) => 2 * (x - 1) ** 2 + 0.5],
      ['exp', (x) => 0.5 * Math.exp(0.8 * x)],
      ['log', (x) => 2 * Math.log(1.5 * x) + 1],
      ['tanh', (x) => Math.tanh(2 * (x - 1.7)) * 3],
      ['1/x', (x) => 1 / x + 0.2],
      ['sqrt', (x) => 3 * Math.sqrt(x) - 1],
      ['x^3', (x) => 0.5 * (x - 1.5) ** 3],
    ]
    for (const [name, f] of cases) {
      const ranked = rankSymbolic(xs, xs.map(f))
      expect(ranked[0]?.name, name).toBe(name)
      expect(ranked[0]?.r2).toBeGreaterThan(0.999999)
    }
  })

  it('recovers the parameters of the affine form', () => {
    const f = (x: number): number => 1.5 * Math.sin(2 * x + 0.3) - 0.2
    const r = rankSymbolic(xs, xs.map(f), { library: ['sin'] })[0]
    // Equal functions: a f(b x + c) + d is the same curve for (a, b, c, d) and (-a, -b, -c, 0 + ...) up to a half period.
    expect(
      maxAbs(
        xs.map((x) => (r?.a ?? 0) * Math.sin((r?.b ?? 0) * x + (r?.c ?? 0)) + (r?.d ?? 0)),
        xs.map(f),
      ),
    ).toBeLessThan(1e-6)
    expect(r?.d).toBeCloseTo(-0.2, 5)
  })

  it('reads a constant edge as zero and finds nothing in noise', () => {
    expect(
      rankSymbolic(
        xs,
        xs.map(() => 0),
      )[0]?.name,
    ).toBe('0')
    expect(
      rankSymbolic(
        xs,
        xs.map(() => 2.5),
      )[0]?.r2,
    ).toBeCloseTo(1, 12)
  })

  it('suggests, from a network, the functions planted on its edges', () => {
    const net = new KAN([3, 1], { grid: 20, range: [-1, 1], key: key(12) })
    const planted = plant(net, {
      '0:0:0': (v) => Math.sin(Math.PI * v),
      '0:1:0': (v) => v * v,
      '0:2:0': (v) => Math.exp(v),
    })
    const x = rows(120, 3)
    const names = [0, 1, 2].map((i) => suggestSymbolic(net, planted, x, 0, i, 0)[0]?.name)
    expect(names).toEqual(['sin', 'x^2', 'exp'])
    expect(suggestSymbolic(net, planted, x, 0, 0, 0)[0]?.r2).toBeGreaterThan(0.999)
  })

  it('fixes a network and gives its closed form, which computes what the fixed edges compute', () => {
    const net = new KAN([3, 1], { grid: 20, range: [-1, 1], key: key(12) })
    const planted = plant(net, {
      '0:0:0': (v) => Math.sin(Math.PI * v),
      '0:1:0': (v) => v * v,
      '0:2:0': (v) => Math.exp(v),
    })
    const x = rows(120, 3)
    const fixed = fixSymbolic(net, planted, x)
    expect(fixed.layers[0]?.edges[0]?.map((e) => e?.name)).toEqual(['sin', 'x^2', 'exp'])
    const target = toFlat(net.apply(planted.params, x, { buffers: planted.buffers }))
    const numeric = evaluateSymbolic(fixed, x).map((r) => r[0] as number)
    expect(maxAbs(numeric, target)).toBeLessThan(0.02)
    const formula = symbolicFormula(fixed, ['u', 'v', 'w'], 0)
    expect(formula.text[0]).toContain('sin')
    expect(formula.text[0]).toContain('exp')
    const f = compile(formula.expressions[0] as never, ['u', 'v', 'w'])
    const cols = [0, 1, 2].map((i) => Array.from({ length: 120 }, (_, r) => toFlat(x)[r * 3 + i] as number))
    expect(maxAbs(toFlat(f(...cols)), numeric)).toBeLessThan(1e-10)
    expect(evaluate(formula.expressions[0] as never, { u: 0.2, v: -0.5, w: 0.7 })).toBeCloseTo(
      evaluateSymbolic(fixed, [[0.2, -0.5, 0.7]])[0]?.[0] as number,
      10,
    )
  })

  it('composes layers: a hidden node is the sum of its edges and the next layer applies its function', () => {
    const net = new KAN([2, 1, 1], { grid: 20, range: [-2, 2], key: key(13) })
    const planted = plant(net, {
      '0:0:0': (v) => Math.sin(Math.PI * v),
      '0:1:0': (v) => v * v,
      '1:0:0': (v) => 0.4 * Math.exp(v),
    })
    const x = rows(150, 2, -1, 1)
    const fixed = fixSymbolic(net, planted, x)
    expect(fixed.layers.map((l) => l.edges.flat().map((e) => e?.name))).toEqual([['sin', 'x^2'], ['exp']])
    const target = toFlat(net.apply(planted.params, x, { buffers: planted.buffers }))
    expect(
      maxAbs(
        evaluateSymbolic(fixed, x).map((r) => r[0] as number),
        target,
      ),
    ).toBeLessThan(0.05)
    expect(symbolicFormula(fixed).text[0]).toContain('exp')
  })

  it('leaves an unfixed edge open and refuses to print a formula with one', () => {
    const net = new KAN([1, 1], { grid: 30, range: [-1, 1], key: key(14) })
    const wiggly = plant(net, { '0:0:0': (v) => Math.sin(6 * v) + Math.sin(14 * v) })
    const fixed = fixSymbolic(net, wiggly, rows(100, 1), { r2Threshold: 0.99 })
    expect(fixed.layers[0]?.edges[0]?.[0]).toBeUndefined()
    expect(() => symbolicFormula(fixed)).toThrow(/not fixed/)
    expect(() => rankSymbolic([1, 2], [1, 2])).toThrow()
    expect(sum(asTensor([1])).shape).toEqual([])
  })
})
