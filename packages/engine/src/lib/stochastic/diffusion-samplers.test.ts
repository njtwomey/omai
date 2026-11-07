import { describe, expect, it } from 'vitest'
import { foldIn, key } from '../random/index.ts'
import { full, item, tensor, toArray } from '../tensor/index.ts'
import { nullTracer, runAlgorithm, startRun, stepContext } from '../trace/index.ts'
import * as D from './index.ts'

const schedule = D.linearSchedule()
const [m, s] = [2, 0.5]
const gaussian = D.gaussianMixtureTruth({ means: [[m]], stds: s })

/** The probability-flow ODE's flow map of N(m, s^2) data, which keeps the standardised coordinate: x_T to x_0. */
const flowMap = (xT: number, tMax: number): number => {
  const a = schedule.alpha(tMax)
  return m + (s * (xT - a * m)) / Math.sqrt(a * a * s * s + schedule.sigma(tMax) ** 2)
}

/** Run a sampler from the start of `seed`, returning the samples and the flow-map prediction for the same starts. */
function runFrom(
  method: D.SamplerMethod,
  steps: number,
  seed: number,
  prediction: D.DiffusionPrediction = 'epsilon',
  spacing: D.TimeSpacing = 'uniform',
) {
  const options: D.DiffusionSamplerOptions = {
    schedule,
    model: gaussian.model(schedule, prediction),
    prediction,
    shape: [200, 1],
    steps,
    method,
    spacing,
  }
  const algorithm = D.diffusionSampler(options)
  const start = algorithm.init(foldIn(key(seed), 0)).x
  const end = runAlgorithm(algorithm, { maxSteps: steps + 1, seed }).state.x
  const want = (toArray(start) as number[][]).map((r) => flowMap(r[0] ?? 0, schedule.tMax))
  const got = (toArray(end) as number[][]).map((r) => r[0] ?? 0)
  const error = Math.sqrt(got.reduce((a, g, i) => a + (g - (want[i] ?? 0)) ** 2, 0) / got.length)
  return { error, got }
}

describe('deterministic reverse samplers against the exact flow map', () => {
  it('converge to the probability-flow map of a Gaussian as the steps grow, DDIM and Euler at first order', () => {
    for (const method of ['ddim', 'euler'] as const) {
      const [e10, e20, e40, e80] = [10, 20, 40, 80].map((n) => runFrom(method, n, 1).error)
      expect(e80, method).toBeLessThan((e40 ?? 0) * 0.7)
      expect(e40, method).toBeLessThan((e20 ?? 0) * 0.7)
      expect(e20, method).toBeLessThan((e10 ?? 0) * 0.7)
      expect(e80, method).toBeLessThan(0.05)
    }
  })

  it('are second order for Heun and DPM-Solver-2: much smaller error at the same number of steps', () => {
    const first = runFrom('ddim', 20, 2).error
    for (const method of ['heun', 'dpm2'] as const) {
      const [e10, e20, e40] = [10, 20, 40].map((n) => runFrom(method, n, 2).error)
      expect(e20, method).toBeLessThan(first / 2)
      // Halving the step cuts a second-order error by about four.
      expect((e20 ?? 0) / (e40 ?? 1), `${method} ratio`).toBeGreaterThan(2.8)
      expect((e10 ?? 0) / (e20 ?? 1), `${method} ratio`).toBeGreaterThan(2.8)
    }
  })

  it('give the same samples whichever parameterisation the exact model is written in', () => {
    const reference = runFrom('ddim', 30, 3, 'epsilon').got
    for (const prediction of ['x0', 'v', 'score'] as const) {
      const g = runFrom('ddim', 30, 3, prediction).got
      for (const [i, v] of g.entries()) expect(v, prediction).toBeCloseTo(reference[i] ?? Number.NaN, 8)
    }
  })

  it('reach the data distribution: the mean and spread of the samples', () => {
    for (const [method, spacing] of [
      ['ddim', 'logSnr'],
      ['heun', 'karras'],
      ['dpm2', 'logSnr'],
    ] as const) {
      const { got } = runFrom(method, 60, 4, 'epsilon', spacing)
      const mu = got.reduce((a, v) => a + v, 0) / got.length
      const sd = Math.sqrt(got.reduce((a, v) => a + (v - mu) ** 2, 0) / got.length)
      expect(mu, method).toBeCloseTo(m, 0)
      expect(sd, method).toBeCloseTo(s, 1)
    }
  })
})

describe('the stochastic samplers', () => {
  const posterior = (t: number, sNext: number): { mean: (x: number, x0: number) => number; variance: number } => {
    const [at, st, as, ss] = [schedule.alpha(t), schedule.sigma(t), schedule.alpha(sNext), schedule.sigma(sNext)]
    const transition = at / as
    const t2s = st * st - transition * transition * ss * ss
    return {
      mean: (x, x0) => ((transition * ss * ss) / (st * st)) * x + ((as * t2s) / (st * st)) * x0,
      variance: (t2s * ss * ss) / (st * st),
    }
  }

  it('takes DDPM steps with the posterior mean and variance of the forward process', () => {
    // One step from t = 0.6 to 0.5 on a known point: the mean is the DDPM posterior mean and the spread its variance.
    const [t, sNext] = [0.6, 0.5]
    const x = tensor(Array.from({ length: 4000 }, () => [1.3]))
    const model = gaussian.model(schedule, 'epsilon')
    const algorithm = D.ddpmSampler({ schedule, model, shape: [4000, 1], steps: 2, times: [t, sNext, 0] })
    const state = { x, x0Hat: x, step: 0, t } as D.SamplerState
    const next = algorithm.step(state, stepContext(0, key(3), startRun(nullTracer, { name: 'x', seed: 0, config: {} })))
    const x0Hat = item(
      D.toX0AndNoise('epsilon', model(tensor([[1.3]]), t), [[1.3]], schedule.alpha(t), schedule.sigma(t)).x0,
    )
    const want = posterior(t, sNext)
    const values = (toArray(next.x) as number[][]).map((r) => r[0] ?? 0)
    const mu = values.reduce((a, v) => a + v, 0) / values.length
    const variance = values.reduce((a, v) => a + (v - mu) ** 2, 0) / values.length
    expect(Math.abs(mu - want.mean(1.3, x0Hat))).toBeLessThan(4 * Math.sqrt(want.variance / values.length))
    expect(variance).toBeGreaterThan(want.variance * 0.9)
    expect(variance).toBeLessThan(want.variance * 1.1)
  })

  it('is DDPM when DDIM takes eta = 1 (same noise), and deterministic when eta = 0', () => {
    const base = { schedule, model: gaussian.model(schedule, 'epsilon'), shape: [20, 1], steps: 15 } as const
    const ddpm = D.sampleDiffusion({ ...base, method: 'ddpm' }, key(8)).x
    const eta1 = D.sampleDiffusion({ ...base, method: 'ddim', eta: 1 }, key(8)).x
    expect(toArray(eta1)).toEqual(toArray(ddpm))
    const a = D.sampleDiffusion({ ...base, method: 'ddim' }, key(8)).x
    const b = D.sampleDiffusion({ ...base, method: 'ddim' }, key(8)).x
    expect(toArray(a)).toEqual(toArray(b))
  })

  it('reach the data distribution too as the steps grow: ancestral sampling is biased at few steps and converges', () => {
    const spread = (steps: number, eta: number): { mu: number; sd: number } => {
      const { x } = D.sampleDiffusion(
        { schedule, model: gaussian.model(schedule, 'epsilon'), shape: [2000, 1], steps, eta },
        key(6),
      )
      const values = (toArray(x) as number[][]).map((r) => r[0] ?? 0)
      const mu = values.reduce((a, v) => a + v, 0) / values.length
      return { mu, sd: Math.sqrt(values.reduce((a, v) => a + (v - mu) ** 2, 0) / values.length) }
    }
    for (const eta of [0.5, 1]) {
      const [coarse, fine] = [spread(25, eta), spread(400, eta)]
      expect(fine.mu, `eta ${eta}`).toBeCloseTo(m, 1)
      expect(Math.abs(fine.sd - s), `eta ${eta} fine`).toBeLessThan(0.03)
      expect(Math.abs(fine.sd - s), `eta ${eta} closer`).toBeLessThan(Math.abs(coarse.sd - s) + 0.01)
    }
  })

  it('samples a two-dimensional mixture: every point near a mode, all modes visited', () => {
    const truth = D.gaussianMixtureTruth({
      means: [
        [-2, -2],
        [2, -2],
        [-2, 2],
        [2, 2],
      ],
      stds: 0.3,
    })
    const { x } = D.sampleDiffusion(
      { schedule, model: truth.model(schedule, 'epsilon'), shape: [800, 2], steps: 60, method: 'heun' },
      key(2),
    )
    const rows = toArray(x) as number[][]
    const modes = [
      [-2, -2],
      [2, -2],
      [-2, 2],
      [2, 2],
    ]
    const counts = [0, 0, 0, 0]
    let near = 0
    for (const r of rows) {
      const d = modes.map((c) => Math.hypot((r[0] ?? 0) - (c[0] ?? 0), (r[1] ?? 0) - (c[1] ?? 0)))
      const k = d.indexOf(Math.min(...d))
      counts[k] = (counts[k] ?? 0) + 1
      if ((d[k] ?? 9) < 1.2) near += 1
    }
    expect(near / rows.length).toBeGreaterThan(0.97)
    for (const c of counts) expect(c / rows.length).toBeGreaterThan(0.18)
  })

  it('concentrates on a component under classifier-free guidance, more as the scale grows', () => {
    const truth = D.gaussianMixtureTruth({
      means: [
        [-2, 0],
        [2, 0],
        [0, 3],
      ],
      stds: 0.3,
    })
    const share = (scale: number): number => {
      const model = D.guidedModel(
        truth.conditionalModel(1, schedule, 'epsilon'),
        truth.model(schedule, 'epsilon'),
        scale,
      )
      const { x } = D.sampleDiffusion({ schedule, model, shape: [300, 2], steps: 40, method: 'ddim' }, key(10))
      const rows = toArray(x) as number[][]
      return rows.filter((r) => (r[0] ?? 0) > 1).length / rows.length
    }
    const [unguided, mild, strong] = [share(0), share(1), share(3)]
    expect(unguided).toBeLessThan(0.5)
    expect(mild).toBeGreaterThan(0.95)
    expect(strong).toBeGreaterThan(0.95)
  })
})

describe('the sampler as an algorithm', () => {
  it('serialises its state and resumes to the same samples', () => {
    const options = {
      schedule,
      model: gaussian.model(schedule, 'epsilon'),
      shape: [5, 1],
      steps: 12,
      method: 'ddpm',
    } as const
    const algorithm = D.diffusionSampler(options)
    const full = runAlgorithm(algorithm, { maxSteps: 13, seed: 4 }).state
    const half = runAlgorithm(algorithm, { maxSteps: 5, seed: 4 })
    const stored = algorithm.serialise?.(half.state)
    expect(stored).toBeDefined()
    const resumed = runAlgorithm(algorithm, {
      maxSteps: 13,
      seed: 4,
      start: { state: stored, step: 5, serialised: true },
    }).state
    expect(toArray(resumed.x)).toEqual(toArray(full.x))
    expect(full.step).toBe(12)
    expect(full.t).toBe(0)
  })

  it('has the contract paths and logs the cloud, the clean estimate and the noise level', () => {
    expect(Object.keys(D.diffusionSamplerContract).sort()).toEqual([
      'sampler/cloud',
      'sampler/noiseLevel',
      'sampler/step',
      'sampler/time',
      'sampler/x0Hat',
      'sampler/xt',
    ])
    const algorithm = D.diffusionSampler({
      schedule,
      model: gaussian.model(schedule, 'epsilon'),
      shape: [3, 1],
      steps: 4,
    })
    const logged: string[] = []
    const ctx = {
      ...stepContext(0, key(1), startRun(nullTracer, { name: 'x', seed: 0, config: {} })),
      log: (path: string) => void logged.push(path),
    }
    algorithm.step(algorithm.init(key(0)), ctx)
    expect(logged).toEqual(['sampler/xt', 'sampler/x0Hat', 'sampler/noiseLevel', 'sampler/time', 'sampler/step'])
    const cloud = D.diffusionSamplerContract['sampler/cloud']
    expect(cloud.kind).toBe('derived')
  })

  it('refuses times that do not end at zero', () => {
    expect(() =>
      D.diffusionSampler({
        schedule,
        model: gaussian.model(schedule, 'epsilon'),
        shape: [1, 1],
        steps: 3,
        times: [1, 0.5, 0.2],
      }),
    ).toThrow()
  })

  it('lists the samplers with their properties', () => {
    expect(D.samplerProperties.heun.order).toBe(2)
    expect(D.samplerProperties.ddpm.stochastic).toBe(true)
  })

  it('clips an infinite clean-sample estimate to the bound instead of producing NaN', () => {
    const algorithm = D.diffusionSampler({
      schedule,
      model: () => full([4, 1], Number.POSITIVE_INFINITY),
      prediction: 'x0',
      shape: [4, 1],
      steps: 2,
      clipX0: [-1, 1],
    })
    const { state } = runAlgorithm(algorithm, { maxSteps: 3, seed: 0 })
    expect((toArray(state.x) as number[][]).every(([v]) => Number.isFinite(v))).toBe(true)
  })
})
