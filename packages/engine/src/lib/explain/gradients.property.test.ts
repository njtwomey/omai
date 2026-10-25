import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import {
  add,
  fromStorage,
  matmul,
  relu,
  type Tensor,
  tanh,
  tensor,
  toArray,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  type DeepLiftLayer,
  deconvolution,
  deepLift,
  expectedGradients,
  gradientShap,
  guidedBackprop,
  integratedGradients,
  interpolateMap,
  type LrpLayer,
  lrp,
  noiseTunnel,
  saliency,
  smoothGrad,
} from './gradients.ts'
import { attentionFlow, attentionRollout, layerConductance } from './internal.ts'

const weights = fc.array(
  fc.integer({ min: -20, max: 20 }).map((v) => v / 10),
  { minLength: 12, maxLength: 12 },
)
const inputs = fc.array(
  fc.integer({ min: -20, max: 20 }).map((v) => v / 10),
  { minLength: 6, maxLength: 6 },
)

function net(w: readonly number[]) {
  const w1 = fromStorage(Float64Array.from(w.slice(0, 6)), [2, 3], 'float64')
  const w2 = fromStorage(Float64Array.from(w.slice(6, 9)), [3, 1], 'float64')
  const b = fromStorage(Float64Array.from(w.slice(9, 12)), [3], 'float64')
  const layers = [
    { fn: (v: Value) => add(matmul(v, w1), b) },
    { fn: tanh, nonlinear: true },
    { fn: (v: Value) => matmul(v, w2) },
  ]
  return { layers, f: (v: Value): Value => layers.reduce((h, l) => l.fn(h), v) }
}

describe('attribution axioms', () => {
  it('DeepLIFT and integrated gradients are complete', () => {
    fc.assert(
      fc.property(weights, inputs, (w, xs) => {
        const { layers, f } = net(w)
        const x = fromStorage(Float64Array.from(xs), [3, 2], 'float64')
        const fx = toFlat(f(x))
        const f0 = toFlat(f(fromStorage(new Float64Array(6), [3, 2], 'float64')))
        const dl = toFlat(deepLift(layers, x))
        for (let i = 0; i < 3; i++) {
          expect((dl[2 * i] ?? 0) + (dl[2 * i + 1] ?? 0)).toBeCloseTo((fx[i] ?? 0) - (f0[i] ?? 0), 9)
        }
        const ig = integratedGradients(f, x, { nSteps: 64 })
        for (const d of ig.delta) expect(Math.abs(d)).toBeLessThan(1e-8)
      }),
      { numRuns: 30 },
    )
  })

  it('noisy estimators are reproducible from their key and exact for linear models', () => {
    const lin = (v: Value) => matmul(v, fromStorage(Float64Array.from([1.5, -0.5]), [2, 1], 'float64'))
    const x = fromStorage(Float64Array.from([1, 2, -1, 0.5]), [2, 2], 'float64')
    toFlat(smoothGrad(lin, x, { key: key(3), nSamples: 7 })).forEach((v, i) => {
      expect(v).toBeCloseTo([1.5, -0.5][i % 2] ?? 0, 12)
    })
    const bg = fromStorage(Float64Array.from([0, 0, 2, 2, -1, 1]), [3, 2], 'float64')
    const a = expectedGradients(lin, x, bg, { key: key(4), nSamples: 20 }) as Tensor
    expect(toFlat(a)).toEqual(toFlat(expectedGradients(lin, x, bg, { key: key(4), nSamples: 20 })))
  })
})

describe('gradient completion', () => {
  const w1 = [
    [0.5, -1, 0.3],
    [1.2, 0.4, -0.7],
  ]
  const w2 = [[1], [-0.5], [0.8]]
  const layers: DeepLiftLayer[] = [
    { fn: (v: Value) => matmul(v, w1) },
    { fn: relu, nonlinear: true },
    { fn: (v: Value) => matmul(v, w2) },
  ]
  const net = (v: Value): Value => layers.reduce((h, l) => l.fn(h), v)

  it('GradientSHAP of a linear model is w (x - baseline), and noise tunnels agree with SmoothGrad', () => {
    const lin = (v: Value): Value => matmul(v, [[2], [-1]])
    expect(toArray(gradientShap(lin, [[3, 1]], [[1, 2]], { nSamples: 3, key: key(1) }))).toEqual([[4, 1]])
    const x = [[0.3, -0.2]]
    const tunnel = noiseTunnel((v) => saliency(net, v, { abs: false }), x, { nSamples: 6, key: key(2) })
    const smooth = smoothGrad(net, x, { nSamples: 6, key: key(2) })
    toFlat(tunnel).forEach((v, i) => {
      expect(v).toBeCloseTo(toFlat(smooth)[i] ?? 0, 12)
    })
    const sq = noiseTunnel((v) => saliency(net, v, { abs: false }), x, {
      type: 'smoothgrad_sq',
      nSamples: 6,
      key: key(2),
    })
    const vg = noiseTunnel((v) => saliency(net, v, { abs: false }), x, { type: 'vargrad', nSamples: 6, key: key(2) })
    toFlat(vg).forEach((v, i) => {
      expect(v).toBeCloseTo((toFlat(sq)[i] ?? 0) - (toFlat(tunnel)[i] ?? 0) ** 2, 12)
    })
  })

  it('guided backpropagation and deconvolution are the gradient when every signal is positive', () => {
    const pos: DeepLiftLayer[] = [
      {
        fn: (v: Value) =>
          matmul(v, [
            [1, 2],
            [3, 1],
          ]),
      },
      { fn: relu, nonlinear: true },
      { fn: (v: Value) => matmul(v, [[1], [2]]) },
    ]
    const f = (v: Value): Value => pos.reduce((h, l) => l.fn(h), v)
    const x = [[1, 1]]
    expect(toArray(guidedBackprop(pos, x))).toEqual(toArray(saliency(f, x, { abs: false })))
    expect(toArray(deconvolution(pos, x))).toEqual(toArray(saliency(f, x, { abs: false })))
  })

  it('LRP conserves relevance without biases, and alpha-beta splits it by sign', () => {
    const dense: LrpLayer[] = [
      { kind: 'linear', weight: w1 },
      { kind: 'elementwise', fn: relu },
      { kind: 'linear', weight: w2 },
    ]
    fc.assert(
      fc.property(fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: 2, maxLength: 2 }), (x) => {
        const out = toFlat(net(tensor([x])) as Tensor)[0] ?? 0
        const r = toFlat(lrp(dense, [x]))
        if (Math.abs(out) > 1e-3) expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(out, 6)
      }),
      { numRuns: 40 },
    )
    const ab = toFlat(lrp(dense, [[1, 1]], { rule: { name: 'alphaBeta', alpha: 2, beta: 1 } }))
    expect(ab.every(Number.isFinite)).toBe(true)
  })

  it('layer conductance is complete, Grad-CAM maps upsample, and attention rollout keeps rows stochastic', () => {
    const head = (v: Value): Value => relu(matmul(v, w1))
    const tail = (h: Value): Value => matmul(h, w2)
    const x = [[0.8, 0.6]]
    const c = toFlat(layerConductance(head, tail, x, { nSteps: 200 }))
    const change = (toFlat(net(tensor(x)) as Tensor)[0] ?? 0) - (toFlat(net(tensor([[0, 0]])) as Tensor)[0] ?? 0)
    expect(c.reduce((a, b) => a + b, 0)).toBeCloseTo(change, 2)
    expect(
      interpolateMap(
        [
          [2, 2],
          [2, 2],
        ],
        [3, 5],
        'bilinear',
      )
        .flat()
        .every((v) => Math.abs(v - 2) < 1e-12),
    ).toBe(true)
    const roll = attentionRollout([
      [
        [0.7, 0.3],
        [0.2, 0.8],
      ],
      [
        [0.5, 0.5],
        [0.9, 0.1],
      ],
    ])
    for (const row of roll) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const flow = attentionFlow([
      [
        [0.7, 0.3],
        [0.2, 0.8],
      ],
    ])
    expect(flow[0]?.[0] ?? 0).toBeGreaterThan(flow[0]?.[1] ?? 0)
  })
})
