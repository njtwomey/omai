import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import { add, exp, item, mul, neg, sum, swapaxes, tensor, toArray, type Value } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import {
  barbellGraph,
  chebyshevFilter,
  chebyshevFilterCoefficients,
  cycleGraph,
  dirichletEnergy,
  fiedlerVector,
  graph,
  graphWavelets,
  heatDiffusion,
  heatKernel,
  iterativeSpectralEmbedding,
  laplacian,
  normalisedLaplacian,
  pathGraph,
  propagate,
  spectralEmbedding,
  spectralEmbeddingAlgorithm,
  spectralFilter,
} from './index.ts'

const L = laplacian(
  graph([
    [0, 1, 1.2],
    [1, 2, 0.7],
    [2, 3, 1.9],
    [0, 2, 0.3],
    [1, 3, 0.4],
  ]),
)
const X = tensor([0.3, -1.1, 0.8, 0.5])
const W = tensor([
  [0, 1.2, 0.3, 0],
  [1.2, 0, 0.7, 0.4],
  [0.3, 0.7, 0, 1.9],
  [0, 0.4, 1.9, 0],
])
const round = (v: number): number => Math.round(v * 1e8) / 1e8 + 0

describe('graph signal processing', () => {
  it('computes the Dirichlet energy as the sum over edges of w (x_i - x_j)^2', () => {
    const g = graph([
      [0, 1, 2],
      [1, 2, 3],
    ])
    expect(item(dirichletEnergy(g, [1, 0, 2]))).toBe(2 * 1 + 3 * 4)
    expect(
      item(
        dirichletEnergy(g, [
          [1, 1],
          [0, 0],
          [2, 2],
        ]),
      ),
    ).toBe(28)
  })

  it('diffuses heat conservatively and towards the mean', () => {
    const g = pathGraph(5)
    const x = [5, 0, 0, 0, 0]
    expect(round(item(sum(heatDiffusion(g, x, 3))))).toBe(5)
    const far = toArray(heatDiffusion(g, x, 200)) as number[]
    expect(far.map(round)).toEqual([1, 1, 1, 1, 1])
    // The heat kernel of a normalised Laplacian is symmetric with unit diagonal at t = 0.
    expect((toArray(heatKernel(g, 0, { normalised: true })) as number[][]).map((r) => r.map(round))[0]).toEqual([
      1, 0, 0, 0, 0,
    ])
  })

  it('approximates the heat kernel with a Chebyshev filter', () => {
    const l = normalisedLaplacian(cycleGraph(7))
    const theta = chebyshevFilterCoefficients((lambda) => Math.exp(-0.5 * lambda), 12)
    const approx = toArray(chebyshevFilter(l, [1, 0, 0, 0, 0, 0, 0], theta)) as number[]
    const exact = toArray(heatDiffusion(l, [1, 0, 0, 0, 0, 0, 0], 0.5)) as number[]
    approx.forEach((v, i) => {
      expect(Math.abs(v - (exact[i] ?? 0))).toBeLessThan(1e-10)
    })
    expect(chebyshevFilterCoefficients(() => 2, 0)).toEqual([2])
    expect(() => chebyshevFilter(l, [1, 0, 0, 0, 0, 0, 0], [])).toThrow()
  })

  it('keeps a Chebyshev filter of order K within K hops', () => {
    const g = pathGraph(8)
    const y = toArray(chebyshevFilter(normalisedLaplacian(g), [1, 0, 0, 0, 0, 0, 0, 0], [0.2, 0.5, -0.3])) as number[]
    expect(y.slice(3).every((v) => v === 0)).toBe(true)
    expect(y[2]).not.toBe(0)
  })

  it('filters with an identity response and builds wavelets with zero DC response', () => {
    const x = [1, 2, 3, 4]
    expect((toArray(spectralFilter(L, x, (w) => exp(mul(w, 0)))) as number[]).map(round)).toEqual([1, 2, 3, 4])
    // The mexican-hat kernel vanishes at frequency 0, so the constant signal has zero wavelet coefficients.
    const c = toArray(graphWavelets(cycleGraph(6), [1, 3], { signal: [1, 1, 1, 1, 1, 1] })) as number[][]
    expect(c.flat().every((v) => Math.abs(v) < 1e-12)).toBe(true)
    expect(graphWavelets(cycleGraph(6), [1], { kernel: 'heat-band' }).shape).toEqual([1, 6, 6])
  })

  it('embeds a barbell so the Fiedler coordinate separates the bells', () => {
    const g = barbellGraph(5, 0)
    const f = fiedlerVector(g)
    const left = Array.from(f.subarray(0, 5))
    const right = Array.from(f.subarray(5))
    expect(left.every((v) => Math.sign(v) === Math.sign(left[0] ?? 0))).toBe(true)
    expect(right.every((v) => Math.sign(v) === -Math.sign(left[0] ?? 0))).toBe(true)
    const y = toArray(spectralEmbedding(g, 2, { dropFirst: true, rowNormalise: true })) as number[][]
    for (const row of y) expect(Math.abs(dnrm2(row) - 1)).toBeLessThan(1e-12)
    expect(() => spectralEmbedding(g, 11)).toThrow()
    expect(() => fiedlerVector(graph([], { nodes: 1 }))).toThrow()
  })

  it('runs the subspace iteration as a traced algorithm that matches the dense eigenvalues', () => {
    const g = barbellGraph(4, 1)
    const rec = recorder()
    const r = iterativeSpectralEmbedding(g, 3, { tracer: rec, seed: 2 })
    const dense = toArray(spectralEmbedding(g, 3)) as number[][]
    const v = toArray(r.vectors) as number[][]
    for (let i = 0; i < g.n; i++) {
      for (let j = 1; j < 3; j++) expect(Math.abs((v[i]?.[j] ?? 0) - (dense[i]?.[j] ?? 0))).toBeLessThan(1e-6)
    }
    expect(rec.events.some((e) => e.type === 'metric')).toBe(true)
    expect(spectralEmbeddingAlgorithm(g, 2).name).toBe('spectral-subspace-iteration')
    expect(() => iterativeSpectralEmbedding(g, 3, { maxIter: 1 })).toThrow()
  })

  it('propagates messages with normalised adjacencies, SGC and APPNP style (checkPrimitive)', () => {
    const g = pathGraph(4)
    // Mean aggregation keeps constant signals constant; symmetric GCN normalisation has eigenvalues in (-1, 1].
    expect((toArray(propagate(g, [2, 2, 2, 2], { kind: 'rw' })) as number[]).map(round)).toEqual([2, 2, 2, 2])
    expect(
      propagate(
        g,
        [
          [1, 0],
          [0, 1],
          [1, 1],
          [0, 0],
        ],
        { steps: 3 },
      ).shape,
    ).toEqual([4, 2])
    expect(toArray(propagate(g, [1, 2, 3, 4], { steps: 0 })) as number[]).toEqual([1, 2, 3, 4])
    expect(() => propagate(g, [1, 2, 3, 4], { steps: -1 })).toThrow()
    // With teleport 1 the features never move.
    expect((toArray(propagate(g, [1, 2, 3, 4], { steps: 5, teleport: 1 })) as number[]).map(round)).toEqual([
      1, 2, 3, 4,
    ])
    expect(checkPrimitive((x: Value) => propagate(W, x, { steps: 2 }), [X]).failures).toEqual([])
    expect(
      checkPrimitive((alpha: Value) => propagate(W, X, { steps: 3, teleport: alpha }), [tensor(0.2)]).failures,
    ).toEqual([])
    expect(checkPrimitive((a: Value) => propagate(add(a, swapaxes(a, -1, -2)), X, { steps: 2 }), [W]).failures).toEqual(
      [],
    )
  })

  it('differentiates filters and energies in the signal, the operator and the parameters (checkPrimitive)', () => {
    expect(checkPrimitive((x: Value) => dirichletEnergy(L, x), [X]).failures).toEqual([])
    expect(checkPrimitive((l: Value) => dirichletEnergy(l, X), [L]).failures).toEqual([])
    expect(checkPrimitive((t: Value) => heatDiffusion(L, X, t), [tensor(0.6)]).failures).toEqual([])
    expect(checkPrimitive((x: Value) => heatDiffusion(L, x, 0.6), [X]).failures).toEqual([])
    expect(
      checkPrimitive((theta: Value) => chebyshevFilter(L, X, theta, { lambdaMax: 6 }), [tensor([0.4, -0.2, 0.1])])
        .failures,
    ).toEqual([])
    expect(
      checkPrimitive((l: Value) => chebyshevFilter(l, X, [0.4, -0.2, 0.1], { lambdaMax: 6 }), [L]).failures,
    ).toEqual([])
    expect(checkPrimitive((s: Value) => graphWavelets(L, s, { signal: X }), [tensor([0.5, 1.5])]).failures).toEqual([])
    const response = (beta: Value) => (w: Value) => exp(neg(mul(w, beta)))
    expect(checkPrimitive((b: Value) => spectralFilter(L, X, response(b)), [tensor(0.8)]).failures).toEqual([])
  })
})
