import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { grad } from '../autodiff/index.ts'
import { key } from '../random/index.ts'
import { add, item, log, mean, mul, sliceAxis, sub, sum, tensor, toArray, type Value } from '../tensor/index.ts'
import * as D from './index.ts'
import { probabilityFlow, reverseSde } from './sde.ts'

/** Element (i, c) of a matrix value. */
const cell = (m: Value | number[][], i: number, c: number): number =>
  (Array.isArray(m) ? m : (toArray(m) as number[][]))[i]?.[c] ?? Number.NaN

const schedules: readonly D.NoiseSchedule[] = [
  D.linearSchedule(),
  D.cosineSchedule(),
  D.sigmoidSchedule(),
  D.discreteSchedule(D.linearBetas(200)),
  D.veSchedule(),
  D.edmSchedule(),
]

describe('noise schedules', () => {
  it('are consistent: alpha and sigma, variance preservation, a falling log-SNR, and the SDE coefficients by differences', () => {
    for (const s of schedules) {
      const [lo, hi] = [s.tMin * 2, s.tMax * 0.9]
      for (const t of [lo, 0.5 * (lo + hi), hi]) {
        if (s.kind === 'vp') expect(s.alpha(t) ** 2 + s.sigma(t) ** 2, `${s.name} ${t}`).toBeCloseTo(1, 12)
        else expect(s.alpha(t), s.name).toBe(1)
        expect(s.logSnr(t + 0.01 * (hi - lo)), `${s.name} falls`).toBeLessThan(s.logSnr(t))
        // f = d log alpha / dt and g^2 = d sigma^2 / dt - 2 f sigma^2 by central differences (away from kinks).
        if (s.name === 'discrete') continue
        const h = 1e-5 * (hi - lo)
        const f = (Math.log(s.alpha(t + h)) - Math.log(s.alpha(t - h))) / (2 * h)
        const dSigma2 = (s.sigma(t + h) ** 2 - s.sigma(t - h) ** 2) / (2 * h)
        expect(s.drift(t), `${s.name} drift at ${t}`).toBeCloseTo(f, 5)
        expect(s.diffusionSquared(t), `${s.name} g2 at ${t}`).toBeCloseTo(dSigma2 - 2 * f * s.sigma(t) ** 2, 4)
        expect(s.diffusionSquared(t)).toBeGreaterThan(0)
      }
    }
  })

  it('has the linear schedule the VP-SDE: f = -beta / 2 and g^2 = beta', () => {
    const s = D.linearSchedule({ betaMin: 0.1, betaMax: 20 })
    for (const t of [0.1, 0.5, 0.9]) {
      expect(s.drift(t)).toBeCloseTo(-0.5 * (0.1 + t * 19.9), 14)
      expect(s.diffusionSquared(t)).toBeCloseTo(0.1 + t * 19.9, 12)
    }
    expect(s.alpha(1)).toBeCloseTo(Math.exp(-(19.9 / 4 + 0.05)), 14)
  })

  it('has the discrete schedule exact on its grid: the cumulative products of 1 - beta', () => {
    const betas = D.linearBetas(1000)
    const s = D.discreteSchedule(betas)
    let bar = 1
    for (const [i, b] of betas.entries()) {
      bar *= 1 - b
      if ([0, 9, 99, 499, 999].includes(i)) expect(s.alpha((i + 1) / 1000) ** 2).toBeCloseTo(bar, 12)
    }
    // DDPM's final signal level is about 4e-5.
    expect(bar).toBeGreaterThan(3e-5)
    expect(bar).toBeLessThan(5e-5)
  })

  it('has the discrete cosine betas reproduce the closed-form cosine alpha-bar', () => {
    const n = 500
    const s = D.discreteSchedule(D.cosineBetas(n))
    const f = (t: number): number => Math.cos(((t + 0.008) / 1.008) * (Math.PI / 2)) ** 2
    for (const i of [50, 200, 400]) expect(s.alpha(i / n) ** 2).toBeCloseTo(f(i / n) / f(0), 10)
  })

  it('spaces sampler times from tMax down to tMin and then zero', () => {
    const s = D.linearSchedule()
    for (const spacing of ['uniform', 'quadratic', 'logSnr', 'karras'] as const) {
      const times = D.samplerTimes(s, 10, { spacing })
      expect(times.length, spacing).toBe(11)
      expect(times[0]).toBeCloseTo(s.tMax, 6)
      expect(times[9]).toBeCloseTo(s.tMin, 6)
      expect(times[10]).toBe(0)
      for (let i = 1; i < times.length; i++) expect(times[i] ?? 0, spacing).toBeLessThan(times[i - 1] ?? 0)
    }
    // Uniform in log-SNR: equal gaps.
    const t = D.samplerTimes(s, 6, { spacing: 'logSnr' })
    const gaps = t.slice(0, 5).map((u, i) => s.logSnr(t[i + 1] ?? 0) - s.logSnr(u))
    for (const g of gaps) expect(g).toBeCloseTo(gaps[0] ?? 0, 6)
  })

  it('inverts the log-SNR', () => {
    for (const s of schedules) {
      const t = 0.4 * (s.tMax - s.tMin) + s.tMin
      expect(D.timeAtLogSnr(s, s.logSnr(t))).toBeCloseTo(t, 8)
    }
  })

  it('gives EDM its preconditioning: unit variance in and out', () => {
    const { cSkip, cOut, cIn } = D.edmPreconditioning(2, 0.5)
    // Var[c_in x] = c_in^2 (sigma^2 + sigma_data^2) = 1 and c_skip^2 (sigma^2 + sd^2) + c_out^2 / (the effective target) closes.
    expect(cIn ** 2 * (4 + 0.25)).toBeCloseTo(1, 14)
    expect(cSkip + 0 * cOut).toBeCloseTo(0.25 / 4.25, 14)
    expect(D.edmPreconditioning(0.5, 0.5).cSkip).toBe(0.5)
    expect(D.edmNoiseLevels(key(1), 2000).every((v) => v > 0)).toBe(true)
  })
})

describe('the forward process', () => {
  const s = D.linearSchedule()

  it('forms x_t = alpha x0 + sigma noise exactly, with one time or a time per sample', () => {
    const x0 = tensor([
      [1, -1],
      [2, 0.5],
      [0, 3],
    ])
    const f = D.forwardSample(s, x0, 0.3, key(5))
    const want = add(mul(x0, s.alpha(0.3)), mul(f.noise, s.sigma(0.3)))
    expect(toArray(f.xt)).toEqual(toArray(want))
    const g = D.forwardSample(s, x0, [0.1, 0.5, 0.9], key(5))
    const rows = toArray(g.xt) as number[][]
    const z = toArray(g.noise) as number[][]
    for (const [i, t] of [0.1, 0.5, 0.9].entries())
      for (const c of [0, 1])
        expect(rows[i]?.[c]).toBeCloseTo(s.alpha(t) * cell(x0, i, c) + s.sigma(t) * (z[i]?.[c] ?? 0), 14)
  })

  it('has the marginal mean alpha x0 and standard deviation sigma over many draws', () => {
    const x0 = tensor(Array.from({ length: 4000 }, () => [1.5]))
    const f = D.forwardSample(s, x0, 0.4, key(9))
    const values = toArray(f.xt) as number[][]
    const m = values.reduce((a, r) => a + (r[0] ?? 0), 0) / values.length
    const v = values.reduce((a, r) => a + ((r[0] ?? 0) - m) ** 2, 0) / values.length
    expect(m).toBeCloseTo(s.alpha(0.4) * 1.5, 1)
    expect(Math.sqrt(v)).toBeCloseTo(s.sigma(0.4), 1)
  })

  it('gives the VP-SDE the drift and diffusion of its schedule, and works with the reverse-time and flow forms of sde.ts', () => {
    const sde = D.vpSde()
    const score = (_t: number, x: Value): Value => mul(x, -2)
    const x = tensor([[1, 2]])
    const t = 0.3
    const beta = 0.1 + t * 19.9
    expect(toArray(sde.drift(t, x))).toEqual([[-0.5 * beta, -beta]])
    expect((toArray(sde.diffusion(t, x)) as number[][])[0]).toEqual([Math.sqrt(beta), Math.sqrt(beta)])
    // Reverse time s = 1 - t: drift = -f x + g^2 score.
    const rev = toArray(reverseSde(sde, score, 1).drift(1 - t, x)) as number[][]
    expect(rev[0]?.[0]).toBeCloseTo(0.5 * beta * 1 + beta * -2 * 1, 12)
    const flow = toArray(probabilityFlow(sde, score)(t, x)) as number[][]
    expect(flow[0]?.[1]).toBeCloseTo(-0.5 * beta * 2 - 0.5 * beta * -4, 12)
    expect(D.veSde().schedule.kind).toBe('ve')
  })
})

describe('losses and weightings', () => {
  const s = D.linearSchedule()
  const x0 = tensor([
    [1, -0.5],
    [0.3, 2],
    [-1.2, 0.4],
  ])
  const noise = tensor([
    [0.2, -1],
    [1.1, 0.5],
    [-0.3, 0.7],
  ])
  const times = [0.2, 0.55, 0.9]
  const alpha = times.map((t) => s.alpha(t))
  const sigma = times.map((t) => s.sigma(t))
  const xt = tensor(
    (toArray(x0) as number[][]).map((row, i) =>
      row.map((v, c) => (alpha[i] ?? 0) * v + (sigma[i] ?? 0) * cell(noise, i, c)),
    ),
  )
  const delta = tensor([
    [0.1, -0.2],
    [0.05, 0.3],
    [-0.15, 0.1],
  ])

  /** The epsilon-MSE of an epsilon estimate noise + delta, and the same error seen through other parameterisations. */
  const mseOf = (d: Value): number => item(mean(mul(d, d)))
  const viaOtherPredictions = (prediction: 'x0' | 'v' | 'score'): Value => {
    const eps = add(noise, delta)
    const rows = (toArray(eps) as number[][]).map((r, i) =>
      r.map((e, c) => {
        const [a, sg] = [alpha[i] ?? 1, sigma[i] ?? 1]
        const x0hat = (cell(xt, i, c) - sg * e) / a
        if (prediction === 'x0') return x0hat
        if (prediction === 'v') return a * e - sg * x0hat
        return -e / sg
      }),
    )
    return tensor(rows)
  }

  it('makes the simple weighting the epsilon-MSE whichever prediction the network outputs', () => {
    const want = mseOf(delta)
    // The per-sample MSE is averaged over samples, each with weight one: the mean of the per-sample epsilon MSEs.
    const eps = item(D.weightedDiffusionLoss(add(noise, delta), x0, noise, times, s, { prediction: 'epsilon' }))
    expect(eps).toBeCloseTo(want, 14)
    for (const prediction of ['x0', 'v', 'score'] as const)
      expect(
        item(D.weightedDiffusionLoss(viaOtherPredictions(prediction), x0, noise, times, s, { prediction })),
        prediction,
      ).toBeCloseTo(want, 12)
  })

  it('makes the SNR-plus-one weighting the plain velocity-MSE for a variance-preserving schedule', () => {
    const v = viaOtherPredictions('v')
    const target = tensor(
      (toArray(x0) as number[][]).map((r, i) =>
        r.map((value, c) => (alpha[i] ?? 0) * cell(noise, i, c) - (sigma[i] ?? 0) * value),
      ),
    )
    const plain = item(mean(mul(sub(v, target), sub(v, target))))
    expect(
      item(D.weightedDiffusionLoss(v, x0, noise, times, s, { prediction: 'v', weighting: 'snr-plus-one' })),
    ).toBeCloseTo(plain, 12)
  })

  it('orders the weightings and reads the ELBO weight off the SNR derivative', () => {
    for (const t of [0.1, 0.5, 0.9]) {
      expect(D.x0ErrorWeight(s, t, 'min-snr', { gamma: 5 })).toBeLessThanOrEqual(D.x0ErrorWeight(s, t, 'simple'))
      expect(D.x0ErrorWeight(s, t, 'truncated-snr')).toBeGreaterThanOrEqual(1)
      const h = 1e-6
      const dSnr = (D.snr(s, t + h) - D.snr(s, t - h)) / (2 * h)
      expect(D.x0ErrorWeight(s, t, 'elbo')).toBeCloseTo(0.5 * Math.abs(dSnr), 3)
    }
    expect(D.x0ErrorWeight(D.edmSchedule(), 0.5, 'edm', { sigmaData: 0.5 })).toBeCloseTo(2 / 0.25, 12)
  })

  it('passes the conformance check and builds batches', () => {
    expectConformant(
      (o) => D.weightedDiffusionLoss(o, x0, noise, times, s, { prediction: 'v', weighting: 'min-snr' }),
      [delta],
    )
    const b = D.diffusionBatch(s, x0, key(3))
    expect(b.times.every((t) => t >= s.tMin && t <= s.tMax)).toBe(true)
    expect(b.xt.shape).toEqual([3, 2])
    const e = D.diffusionBatch(D.edmSchedule(), x0, key(3), { logNormal: true })
    expect(e.times.every((t) => t > 0)).toBe(true)
  })
})

describe('guidance', () => {
  it('extrapolates from the unconditional to the conditional prediction', () => {
    expect(toArray(D.classifierFreeGuidance([2], [1], 1))).toEqual([2])
    expect(toArray(D.classifierFreeGuidance([2], [1], 0))).toEqual([1])
    expect(toArray(D.classifierFreeGuidance([2], [1], 3))).toEqual([4])
    expect(toArray(D.guidedScore([1], [2], 3))).toEqual([7])
    expect(toArray(D.guidedNoise([1], [2], 0.5, 3))).toEqual([-2])
    expect(
      toArray(
        D.guidedModel(
          () => tensor([2]),
          () => tensor([1]),
          3,
        )(tensor([0]), 0.5),
      ),
    ).toEqual([4])
  })
})

describe('the Gaussian-mixture truth', () => {
  const s = D.linearSchedule()
  const truth = D.gaussianMixtureTruth({
    means: [
      [-2, 0],
      [2, 1],
      [0, -2],
    ],
    stds: [0.3, 0.5, 0.4],
    weights: [2, 1, 1],
  })
  const x = tensor([
    [0.3, 0.2],
    [-1.5, 0.1],
    [1.1, 0.7],
    [0.2, -1.4],
  ])

  it('has a score equal to the gradient of the log density at every noise level', () => {
    for (const t of [0.05, 0.4, 0.9]) {
      const g = toArray(grad((v: Value) => sum(truth.logDensity(v, s, t)))(x)) as number[][]
      const exact = toArray(truth.score(x, s, t)) as number[][]
      for (const [i, row] of exact.entries())
        for (const [c, v] of row.entries()) expect(v, `t=${t}`).toBeCloseTo(g[i]?.[c] ?? Number.NaN, 10)
    }
  })

  it('integrates the noised density to one and starts at the data density', () => {
    const one = D.gaussianMixtureTruth({ means: [[0.5]], stds: 0.4 })
    for (const t of [0.01, 0.3, 0.8]) {
      let total = 0
      const step = 0.01
      for (let u = -12; u <= 12; u += step) total += Math.exp(item(one.logDensity([[u]], s, t))) * step
      expect(total, `t=${t}`).toBeCloseTo(1, 3)
    }
    // At t = 0 the marginal is the data: N(0.5, 0.4^2) at 0.5 has density 1 / (0.4 sqrt(2 pi)).
    expect(
      Math.exp(
        item(
          one.logDensity(
            [[0.5]],
            D.defineSchedule({ name: 'clean', kind: 'vp', tMin: 0, tMax: 1, alpha: () => 1, sigma: () => 1e-9 }),
            0,
          ),
        ),
      ),
    ).toBeCloseTo(1 / (0.4 * Math.sqrt(2 * Math.PI)), 6)
  })

  it('matches the closed-form Tweedie posterior mean for one Gaussian component', () => {
    const [m, sd] = [1.5, 0.6]
    const one = D.gaussianMixtureTruth({ means: [[m]], stds: sd })
    const t = 0.5
    const [a, sg] = [s.alpha(t), s.sigma(t)]
    const xt = 0.7
    const x0Hat = item(D.toX0AndNoise('score', one.score([[xt]], s, t), [[xt]], a, sg).x0)
    expect(x0Hat).toBeCloseTo(m + ((a * sd * sd) / (a * a * sd * sd + sg * sg)) * (xt - a * m), 12)
  })

  it('gives the same clean-sample and noise estimates through every parameterisation', () => {
    const t = 0.35
    const [a, sg] = [s.alpha(t), s.sigma(t)]
    const reference = D.toX0AndNoise('score', truth.score(x, s, t), x, a, sg)
    for (const prediction of ['epsilon', 'x0', 'v'] as const) {
      const e = D.toX0AndNoise(prediction, truth.model(s, prediction)(x, t), x, a, sg)
      const [got, want] = [toArray(e.x0) as number[][], toArray(reference.x0) as number[][]]
      const [gotNoise, wantNoise] = [toArray(e.noise) as number[][], toArray(reference.noise) as number[][]]
      for (const [i, row] of want.entries())
        for (const [c, v] of row.entries()) {
          expect(got[i]?.[c], `${prediction} x0`).toBeCloseTo(v, 9)
          expect(gotNoise[i]?.[c], `${prediction} noise`).toBeCloseTo(wantNoise[i]?.[c] ?? Number.NaN, 9)
        }
    }
  })

  it('has classifier-free guidance equal classifier guidance with the exact classifier gradient (Bayes)', () => {
    const t = 0.4
    const scale = 2.5
    const target = 1
    const unconditional = truth.score(x, s, t)
    const conditional = truth.componentScore(target, x, s, t)
    const cfg = D.classifierFreeGuidance(conditional, unconditional, scale)
    const classifierGradient = grad((v: Value) =>
      sum(log(sliceAxis(truth.responsibilities(v, s, t), 1, target, target + 1))),
    )(x)
    const guided = D.guidedScore(unconditional, classifierGradient, scale)
    const a = toArray(cfg) as number[][]
    const b = toArray(guided) as number[][]
    for (const [i, row] of a.entries())
      for (const [c, v] of row.entries()) expect(v).toBeCloseTo(b[i]?.[c] ?? Number.NaN, 9)
  })

  it('draws samples near its components with the right proportions', () => {
    const { x: draws, component } = truth.sample(key(11), 4000)
    const counts = [0, 1, 2].map((k) => component.filter((c) => c === k).length / 4000)
    expect(counts[0]).toBeCloseTo(0.5, 1)
    expect(counts[1]).toBeCloseTo(0.25, 1)
    const rows = toArray(draws) as number[][]
    const [mx, my, sd] = [
      [-2, 2, 0],
      [0, 1, -2],
      [0.3, 0.5, 0.4],
    ]
    const near = rows.filter((r, i) => {
      const k = component[i] ?? 0
      return Math.hypot((r[0] ?? 0) - (mx?.[k] ?? 0), (r[1] ?? 0) - (my?.[k] ?? 0)) < 3 * (sd?.[k] ?? 1) * Math.SQRT2
    })
    expect(near.length / 4000).toBeGreaterThan(0.95)
  })
})
