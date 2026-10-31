import { describe, expect, it } from 'vitest'
import { crossEntropy } from '../losses/index.ts'
import { adam } from '../optim/index.ts'
import { key, normal } from '../random/index.ts'
import { argmax, matmul, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as NN from './index.ts'

const flat = (v: unknown): number[] => toFlat(v as Value)

describe('randomised smoothing', () => {
  // A binary linear classifier with logits (0, w.x + b).
  const w = [3, 4]
  const b = 0
  const model: NN.LogitsFunction = (x) => {
    const f = matmul(
      x,
      tensor([
        [0, w[0] as number],
        [0, w[1] as number],
      ]),
    )
    return f
  }
  const sigma = 0.5
  // w . x = 2.5, so margin / (sigma |w|) = 1 and the exact radius is the distance to the hyperplane, 0.5.
  const x = [0.5, 0.25]

  it('the closed form gives the distance to the hyperplane for any sigma', () => {
    const r = NN.linearSmoothedRadius(w, b, x, sigma)
    expect(r.radius).toBeCloseTo(0.5, 14)
    expect(r.probability).toBeCloseTo(0.8413447460685429, 12)
    expect(NN.linearSmoothedRadius(w, b, x, 3).radius).toBeCloseTo(0.5, 14)
  })

  it('the certified radius is a valid lower bound close to the closed form', () => {
    const exact = NN.linearSmoothedRadius(w, b, x, sigma)
    const c = NN.randomisedSmoothing(model, x, { sigma, nSelect: 200, nEstimate: 20000, alpha: 0.001, key: key(0) })
    expect(c.prediction).toBe(1)
    expect(c.pLower).toBeLessThanOrEqual(exact.probability)
    expect(c.pLower).toBeGreaterThan(exact.probability - 0.02)
    expect(c.radius).toBeLessThanOrEqual(exact.radius)
    expect(c.radius).toBeGreaterThan(0.9 * exact.radius)
    expect(c.counts.reduce((a, v) => a + v, 0)).toBe(20000)
  })

  it('abstains on the decision boundary and certifies a larger radius for a larger margin', () => {
    const on = NN.randomisedSmoothing(model, [0.4, -0.3], { sigma, nEstimate: 2000, key: key(1) })
    expect(on.prediction).toBe(-1)
    expect(on.radius).toBe(0)
    const near = NN.randomisedSmoothing(model, [0.2, 0.1], { sigma, nEstimate: 2000, key: key(2) })
    const far = NN.randomisedSmoothing(model, [2, 2], { sigma, nEstimate: 2000, key: key(2) })
    expect(far.radius).toBeGreaterThan(near.radius)
  })

  it('certified radius is sigma Phi^-1(p) and zero at or below one half', () => {
    expect(NN.certifiedRadius(0.8413447460685429, 2)).toBeCloseTo(2, 9)
    expect(NN.certifiedRadius(0.5, 1)).toBe(0)
    expect(NN.certifiedRadius(0.3, 1)).toBe(0)
  })

  it('certified accuracy falls with the radius and abstentions count as errors', () => {
    const xs = tensor([
      [2, 2],
      [-2, -2],
      [0.4, -0.3],
    ])
    const r = NN.certifiedAccuracy(model, xs, [1, 0, 1], [0, 0.5, 1, 4], { sigma, nEstimate: 2000, key: key(3) })
    const acc = r.curve.map((c) => c.accuracy)
    expect(acc[0]).toBeCloseTo(2 / 3, 12)
    for (let i = 1; i < acc.length; i++) expect(acc[i] as number).toBeLessThanOrEqual(acc[i - 1] as number)
    expect(acc.at(-1)).toBe(0)
  })

  it('the stepped run tightens the bound as samples arrive and serialises', () => {
    const alg = NN.smoothingAlgorithm(model, x, { sigma, perStep: 500, steps: 8 })
    const bounds: number[] = []
    const r = runAlgorithm(
      {
        ...alg,
        step: (s, ctx) => {
          const next = alg.step(s, ctx)
          bounds.push(next.pLower)
          return next
        },
      },
      { maxSteps: 20, seed: 4 },
    )
    expect(r.state.n).toBe(4000)
    expect(bounds.at(-1) as number).toBeGreaterThan(bounds[0] as number)
    expect(r.state.radius).toBeLessThanOrEqual(0.5)
    expect(alg.deserialise?.(alg.serialise?.(r.state) ?? null)?.n).toBe(4000)
  })
})

describe('interval bound propagation', () => {
  it('is exact for a single linear layer: the bound is attained at a corner of the box', () => {
    const net = new NN.Sequential(new NN.Linear(3, 2, { key: key(0) }))
    const lo = [-0.3, 0.1, -1]
    const hi = [0.5, 0.9, 0.2]
    const r = NN.intervalBoundPropagation(net, [lo], [hi])
    const wMat = flat((net.params['0'] as NN.Params)['weight'])
    const bias = flat((net.params['0'] as NN.Params)['bias'])
    for (let o = 0; o < 2; o++) {
      let max = bias[o] as number
      let min = bias[o] as number
      for (let k = 0; k < 3; k++) {
        const wk = wMat[o * 3 + k] as number
        max += wk >= 0 ? wk * (hi[k] as number) : wk * (lo[k] as number)
        min += wk >= 0 ? wk * (lo[k] as number) : wk * (hi[k] as number)
      }
      expect(flat(r.output.upper)[o]).toBeCloseTo(max, 14)
      expect(flat(r.output.lower)[o]).toBeCloseTo(min, 14)
    }
  })

  it('widens with depth and collapses to the point for an empty box', () => {
    const net = NN.mlp([2, 8, 8, 2], { key: key(1) })
    const wide = NN.intervalBoundPropagation(net, [[-0.2, -0.2]], [[0.2, 0.2]])
    const widths = wide.layers.map(
      (b) => flat(b.upper).reduce((a, v, i) => a + v - (flat(b.lower)[i] as number), 0) / flat(b.upper).length,
    )
    expect(widths[0]).toBeCloseTo(0.4, 14)
    expect(widths.at(-1) as number).toBeGreaterThan(0)
    const point = NN.intervalBoundPropagation(net, [[0.1, 0.2]], [[0.1, 0.2]])
    const out = flat(net.forward([[0.1, 0.2]]))
    flat(point.output.lower).forEach((v, i) => {
      expect(v).toBeCloseTo(out[i] as number, 12)
    })
    flat(point.output.upper).forEach((v, i) => {
      expect(v).toBeCloseTo(out[i] as number, 12)
    })
  })

  it('refuses a layer it cannot bound', () => {
    expect(() => NN.intervalBoundPropagation(new NN.Sequential(new NN.Softmax()), [[0]], [[1]])).toThrow(/cannot bound/)
  })

  it('certifies exactly the points whose margin survives the box, and certified points resist PGD', () => {
    const x = normal(key(2), [60, 2]) as Tensor
    const labels = tensor(
      Array.from({ length: 60 }, (_, i) => ((flat(x)[2 * i] as number) + (flat(x)[2 * i + 1] as number) > 0 ? 1 : 0)),
      { dtype: 'int32' },
    ) as Tensor
    const net = NN.mlp([2, 10, 2], { key: key(3) })
    const opt = adam({ learningRate: 0.05 })
    let state = NN.trainState(net, opt, key(0))
    for (let i = 0; i < 150; i++) state = NN.trainStep(net, state, x, (z) => crossEntropy(z, labels), opt)
    net.setParams(state.params)
    const clean = NN.ibpCertify(net, x, labels, 0)
    const pred = flat(argmax(net.forward(x), -1))
    clean.certified.forEach((c, i) => {
      expect(c).toBe(pred[i] === flat(labels)[i])
    })
    const eps = 0.05
    const cert = NN.ibpCertify(net, x, labels, eps)
    expect(cert.certified.filter(Boolean).length).toBeGreaterThan(20)
    expect(cert.certified.filter(Boolean).length).toBeLessThan(clean.certified.filter(Boolean).length + 1)
    const f: NN.LogitsFunction = (v) => net.apply(net.params, v, { train: false })
    const adv = NN.pgd(f, x, labels, { epsilon: eps, steps: 20, key: key(5) })
    const advPred = flat(argmax(f(adv), -1))
    cert.certified.forEach((c, i) => {
      if (c) expect(advPred[i]).toBe(flat(labels)[i])
    })
    // The bound is conservative: the bounds are looser than the attack's real margin.
    cert.margin.forEach((m, i) => {
      if (clean.certified[i] === true) expect(m).toBeLessThanOrEqual((clean.margin[i] as number) + 1e-12)
    })
  })
})

describe('Lipschitz certificates', () => {
  const net = NN.mlp([3, 10, 10, 2], { key: key(4) })

  it('the bound is the product of spectral norms and dominates every observed ratio', () => {
    const bound = NN.lipschitzBound(net)
    // The largest singular value by power iteration on W^T W, in plain arithmetic.
    const sv = (layer: string): number => {
      const w = (net.params[layer] as NN.Params)['weight'] as Value
      const [r, c] = [w.shape[0] ?? 0, w.shape[1] ?? 0]
      const m = flat(w)
      let v = Array.from({ length: c }, (_, k) => 1 + k)
      for (let it = 0; it < 2000; it++) {
        const u = Array.from({ length: r }, (_, i) => v.reduce((a, q, k) => a + (m[i * c + k] as number) * q, 0))
        const t = Array.from({ length: c }, (_, k) => u.reduce((a, q, i) => a + (m[i * c + k] as number) * q, 0))
        const n = Math.hypot(...t)
        v = t.map((q) => q / n)
      }
      return Math.hypot(
        ...Array.from({ length: r }, (_, i) => v.reduce((a, q, k) => a + (m[i * c + k] as number) * q, 0)),
      )
    }
    expect(bound).toBeCloseTo(sv('0') * sv('2') * sv('4'), 6)
    const a = normal(key(1), [200, 3]) as Tensor
    const b = normal(key(2), [200, 3]) as Tensor
    const fa = flat(net.forward(a))
    const fb = flat(net.forward(b))
    let worst = 0
    for (let i = 0; i < 200; i++) {
      const num = Math.hypot(
        (fa[2 * i] as number) - (fb[2 * i] as number),
        (fa[2 * i + 1] as number) - (fb[2 * i + 1] as number),
      )
      const den = Math.hypot(...[0, 1, 2].map((k) => (flat(a)[3 * i + k] as number) - (flat(b)[3 * i + k] as number)))
      worst = Math.max(worst, num / den)
    }
    expect(worst).toBeLessThanOrEqual(bound + 1e-12)
  })

  it('the certified radius is at most the distance DeepFool finds to the boundary', () => {
    const x = normal(key(7), [30, 3]) as Tensor
    const f: NN.LogitsFunction = (v) => net.apply(net.params, v, { train: false })
    const radius = NN.lipschitzCertifiedRadius(f(x), NN.lipschitzBound(net))
    const df = NN.deepFool(f, x)
    let compared = 0
    df.success.forEach((ok, i) => {
      if (!ok) return
      compared++
      expect(radius[i] as number).toBeLessThanOrEqual((df.distance[i] as number) + 1e-9)
    })
    expect(compared).toBeGreaterThan(20)
  })

  it('the radius formula is margin over sqrt(2) L', () => {
    const r = NN.lipschitzCertifiedRadius(
      [
        [3, 1],
        [0, 0],
      ],
      Math.SQRT2,
    )
    expect(r[0]).toBeCloseTo(1, 14)
    expect(r[1]).toBe(0)
  })
})
