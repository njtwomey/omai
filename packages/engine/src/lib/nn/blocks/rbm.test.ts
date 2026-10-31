import { describe, expect, it } from 'vitest'
import { expectClose } from '../../../testing/close.ts'
import { grad } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import {
  item,
  matmul,
  mean,
  mul,
  neg,
  sub,
  tensor,
  toArray,
  toFlat,
  transpose,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../../trace/index.ts'
import type { Params } from '../module/module.ts'
import {
  contrastiveDivergenceAlgorithm,
  RBM,
  type RbmVisible,
  rbmContrastiveLoss,
  rbmFreeEnergy,
  rbmGibbs,
  rbmHiddenProbabilities,
  rbmLogLikelihood,
  rbmLogPartition,
  rbmModelStatistics,
  rbmParamsOf,
  sampleBinary,
} from './rbm.ts'

const small = (visible: RbmVisible = 'binary'): Params => ({
  weight: normal(key(1), [3, 2], { scale: visible === 'binary' ? 1 : 0.5 }),
  vbias: normal(key(2), [3], { scale: 0.5 }),
  hbias: normal(key(3), [2], { scale: 0.5 }),
})

const configs = (n: number): number[][] =>
  Array.from({ length: 2 ** n }, (_, r) => Array.from({ length: n }, (_, j) => (r >> j) & 1))

/** E(v, h) of the binary-visible model, computed entry by entry. */
const energy = (p: Params, v: readonly number[], h: readonly number[], gaussian = false): number => {
  const w = toArray(p['weight'] as Value) as number[][]
  const b = toFlat(p['vbias'] as Value)
  const c = toFlat(p['hbias'] as Value)
  let e = 0
  for (const [i, vi] of v.entries()) {
    e += gaussian ? 0.5 * (vi - (b[i] as number)) ** 2 : -(b[i] as number) * vi
    for (const [j, hj] of h.entries()) e -= vi * ((w[i] as number[])[j] as number) * hj
  }
  for (const [j, hj] of h.entries()) e -= (c[j] as number) * hj
  return e
}

describe('RBM energy, free energy and partition function', () => {
  it('the free energy is minus the log of the sum over hidden units, for binary and Gaussian visible units', () => {
    for (const visible of ['binary', 'gaussian'] as const) {
      const p = small(visible)
      const v = visible === 'binary' ? [1, 0, 1] : [0.3, -1.2, 0.8]
      const brute = -Math.log(configs(2).reduce((a, h) => a + Math.exp(-energy(p, v, h, visible === 'gaussian')), 0))
      expect(item(rbmFreeEnergy(p, [v], visible))).toBeCloseTo(brute, 12)
    }
  })

  it('log Z matches the brute-force sum over every visible and hidden configuration', () => {
    const p = small()
    const z = configs(3).reduce((a, v) => a + configs(2).reduce((b, h) => b + Math.exp(-energy(p, v, h)), 0), 0)
    expect(item(rbmLogPartition(p))).toBeCloseTo(Math.log(z), 12)
    // The probabilities of all visible rows sum to one.
    const total = configs(3).reduce((a, v) => a + Math.exp(item(rbmLogLikelihood(p, [v]))), 0)
    expect(total).toBeCloseTo(1, 12)
  })

  it('for Gaussian visible units exp(-F) / Z integrates to one', () => {
    const p: Params = { weight: tensor([[0.8, -0.6]]), vbias: tensor([0.3]), hbias: tensor([0.2, -0.1]) }
    const lz = item(rbmLogPartition(p, 'gaussian'))
    let total = 0
    const h = 0.01
    for (let x = -15; x <= 15; x += h) total += Math.exp(-item(rbmFreeEnergy(p, [[x]], 'gaussian')) - lz) * h
    expect(total).toBeCloseTo(1, 6)
  })
})

describe('contrastive divergence statistics', () => {
  const data = [
    [1, 1, 0],
    [1, 1, 1],
    [0, 0, 1],
    [1, 0, 0],
  ]

  it('the exact log-likelihood gradient is the data statistics minus the model statistics', () => {
    const p = small()
    const g = grad((q: Params) => rbmLogLikelihood(q, data))(p) as unknown as {
      weight: Value
      vbias: Value
      hbias: Value
    }
    const n = data.length
    const h0 = rbmHiddenProbabilities(p, data)
    const stats = rbmModelStatistics(p)
    const positive = mul(matmul(transpose(tensor(data)), h0), 1 / n)
    expectClose(g.weight, sub(positive, stats.coupling))
    expectClose(g.vbias, sub(mean(tensor(data), 0), stats.visible))
    expectClose(g.hbias, sub(mean(h0, 0), stats.hidden))
  })

  it('the surrogate loss has the negative of the CD update as its gradient', () => {
    const p = small()
    const model = [
      [0, 1, 1],
      [1, 0, 0],
    ]
    const g = grad((q: Params) => rbmContrastiveLoss(q, data, model))(p) as unknown as { weight: Value }
    const pos = matmul(transpose(tensor(data)), rbmHiddenProbabilities(p, data))
    const negative = matmul(transpose(tensor(model)), rbmHiddenProbabilities(p, model))
    const cd = toFlat(sub(mul(negative, 1 / 2), mul(pos, 1 / data.length)))
    expectClose(g.weight, cd, 1e-12)
  })

  it('a long Gibbs chain reproduces the exact model statistics and the CD gradient approaches the exact one', () => {
    const p = small()
    const chains = 6000
    const start = sampleBinary(key(5), tensor(Array.from({ length: chains }, () => [0.5, 0.5, 0.5])))
    const end = rbmGibbs(p, start, 25, key(6))
    const stats = rbmModelStatistics(p)
    expectClose(mean(end.visible, 0), stats.visible, 0.04)
    expectClose(mul(matmul(transpose(end.visible), end.hiddenProbabilities), 1 / chains), stats.coupling, 0.04)
    // The CD-k update with long chains against the exact log-likelihood gradient (3 standard errors is about 0.04).
    const g = grad((q: Params) => rbmContrastiveLoss(q, data, end.visible))(p) as unknown as { weight: Value }
    const exact = grad((q: Params) => rbmLogLikelihood(q, data))(p) as unknown as { weight: Value }
    expectClose(g.weight, neg(exact.weight), 0.05)
  })
})

describe('training by contrastive divergence', () => {
  // Two prototypes with a little noise: 6 visible units.
  const rows = Array.from({ length: 40 }, (_, i) => {
    const base = i % 2 === 0 ? [1, 1, 1, 0, 0, 0] : [0, 0, 0, 1, 1, 1]
    return base.map((v, j) => (i % 7 === j ? 1 - v : v))
  })

  it('CD-1 and persistent CD lower the reconstruction error and raise the exact log-likelihood', () => {
    for (const mode of ['cd', 'pcd'] as const) {
      const a = contrastiveDivergenceAlgorithm(rows, {
        hidden: 3,
        mode,
        k: 1,
        learningRate: 0.2,
        momentum: 0.5,
        batchSize: 20,
        exactLogPartition: true,
      })
      const rec = recorder()
      const r = runAlgorithm(a, { maxSteps: 400, seed: 3, tracer: rec })
      const err = metricSeries(rec.run(), 'train/reconstructionError').values.filter(Number.isFinite)
      const early = err.slice(0, 20).reduce((s, v) => s + v, 0) / 20
      const late = err.slice(-20).reduce((s, v) => s + v, 0) / 20
      expect(late).toBeLessThan(early * 0.7)
      expect(metricSeries(rec.run(), 'train/logPartition').values.length).toBeGreaterThan(0)
      const p = rbmParamsOf(r.state)
      const trained = item(rbmLogLikelihood(p, rows))
      const fresh = item(rbmLogLikelihood({ weight: zeros([6, 3]), vbias: zeros([6]), hbias: zeros([3]) }, rows))
      expect(trained).toBeGreaterThan(fresh + 0.5)
    }
  })

  it('records the free-energy gap to held-out rows, serialises, and resumes identically', () => {
    const validation = [
      [1, 0, 1, 0, 1, 0],
      [0, 1, 0, 1, 0, 1],
    ]
    const a = contrastiveDivergenceAlgorithm(rows, { hidden: 2, batchSize: 10, validation, learningRate: 0.2 })
    const rec = recorder()
    const full = runAlgorithm(a, { maxSteps: 150, seed: 1, tracer: rec })
    const gap = metricSeries(rec.run(), 'train/freeEnergyGap').values
    expect(gap.at(-1) as number).toBeGreaterThan(0)
    const half = runAlgorithm(a, { maxSteps: 75, seed: 1 })
    const json = JSON.parse(JSON.stringify(a.serialise?.(half.state)))
    const resumed = runAlgorithm(a, { maxSteps: 75, seed: 1, start: { state: json, step: 75, serialised: true } })
    expect(toFlat(resumed.state.weight)).toEqual(toFlat(full.state.weight))
  })

  it('the RBM module encodes to hidden probabilities', () => {
    const m = new RBM(6, 3, { key: key(0) })
    const out = toFlat(m.forward(rows.slice(0, 2)))
    expect(out.length).toBe(6)
    for (const v of out) expect(v).toBeGreaterThan(0.3)
    expect(m.parameterCount()).toBe(6 * 3 + 6 + 3)
  })
})
