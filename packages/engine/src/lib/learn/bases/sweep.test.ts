import { describe, expect, it } from 'vitest'
import { loggedSolver, solverContract } from '../../optim/index.ts'
import { normal, split } from '../../random/index.ts'
import { choice, dict, int, real } from '../../space/index.ts'
import { add, matmul, mean, mul, square, sub, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import {
  defineMethod,
  Log,
  metricSeries,
  recorder,
  runAlgorithm,
  type StepContext,
  scalar,
  session,
} from '../../trace/index.ts'
import { knobsOf, race, sweep, sweepBest, sweepConfigurations, sweepMethod, sweepTrials } from './sweep.ts'

/**
 * Ridge regression with more coefficients than the data can pin down (30 coefficients, 40 rows), fitted by L-BFGS and
 * scored on a held-out set: the data come from the key, so every repeat is a fresh draw.
 */
const ridge = defineMethod<unknown, { alpha: number }>({
  name: 'ridge',
  config: dict({ alpha: real(1e-3, 1e3, { log: true, default: 1, label: 'penalty' }) }),
  make: (config, k) => {
    const [kw, kx, ke, kt, kf] = split(k, 5) as [typeof k, typeof k, typeof k, typeof k, typeof k]
    const p = 30
    const truth = mul(normal(kw, [p]), 0.5)
    const xs = normal(kx, [40, p])
    const y = add(matmul(xs, truth), normal(ke, [40]))
    const testX = normal(kt, [500, p])
    const testY = add(matmul(testX, truth), normal(kf, [500]))
    const objective = (w: Value): Value => add(mean(square(sub(matmul(xs, w), y))), mul(sum(square(w)), config.alpha))
    const test = (x: Float64Array): number =>
      toFlat(mean(square(sub(matmul(testX, tensor(Array.from(x))), testY))))[0] ?? 0
    return loggedSolver(
      { objective, x0: new Array<number>(p).fill(0) },
      { kind: 'lbfgs', memory: 10, gradTol: 1e-9 },
      k,
      (x, ctx: StepContext) => ctx.log('eval/test', test(x)),
    )
  },
  contract: {
    ...solverContract,
    'eval/test': scalar({ description: 'mean squared error on held-out rows', direction: 'lower' }),
  },
})

/** The mean of n standard normal draws: an estimator whose variance is 1/n, drawn from the key. */
const sampleMean = defineMethod<number, { n: number; shift: number }>({
  name: 'sample-mean',
  config: dict({ n: int(1, 100_000, { default: 10 }), shift: real(-10, 10, { default: 0 }) }),
  make: (config, k) => ({
    name: 'sample-mean',
    config: {},
    init: () => 0,
    step: (s, ctx) => {
      ctx.log('estimate/mean', (toFlat(mean(normal(k, [config.n])))[0] ?? 0) + config.shift)
      return s + 1
    },
    done: (s) => (s >= 1 ? 'completed' : undefined),
    serialise: (s) => s,
    deserialise: (j) => j as number,
  }),
  contract: { 'estimate/mean': scalar() },
})

const std = (xs: readonly number[]): number => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1))
}

describe('the sweep base: a regularisation sweep', () => {
  const study = sweep({
    method: ridge,
    vary: knobsOf(ridge, ['alpha']),
    points: 7,
    repeats: 3,
    steps: 80,
    metrics: ['eval/test'],
    final: [],
  })
  const { state, steps, status } = runAlgorithm(study, { seed: 1, maxSteps: 21 })

  it('runs one trial per step, configurations times repeats, and ends there', () => {
    expect(sweepTrials({ method: ridge, vary: knobsOf(ridge, ['alpha']), points: 7, repeats: 3, metrics: [] })).toBe(21)
    expect(steps).toBe(21)
    expect(status).toBe('completed')
    expect(state.trials).toHaveLength(21)
    expect(state.trials.map((t) => t.config).slice(0, 4)).toEqual([0, 0, 0, 1])
    expect(state.trials.slice(0, 3).map((t) => t.repeat)).toEqual([0, 1, 2])
  })

  it('finds a penalty in the interior that beats both ends, by the mean over fresh data draws', () => {
    const configs = sweepConfigurations(state)
    expect(configs).toHaveLength(7)
    const alphas = configs.map((c) => (c.values as { alpha: number }).alpha)
    expect(alphas[0]).toBeCloseTo(1e-3, 9)
    expect(alphas[6]).toBeCloseTo(1e3, 6)
    const errors = configs.map((c) => c.mean['eval/test'] as number)
    const best = sweepBest(state)
    expect(best?.index).toBeGreaterThan(0)
    expect(best?.index).toBeLessThan(6)
    expect(best?.mean['eval/test']).toBeLessThan(Math.min(errors[0] as number, errors[6] as number))
    // Too little penalty overfits (the error far above the noise floor of 1), too much underfits.
    expect(errors[0]).toBeGreaterThan(1.5)
    expect(state.best).toBeLessThanOrEqual(
      Math.min(...state.trials.map((t) => t.metrics['eval/test'] as number)) + 1e-12,
    )
  })

  it('summarises the inner tapes in the trial records and keeps none of them by default', () => {
    expect(state.tapes).toHaveLength(0)
    const first = state.trials[0]
    expect(first?.series['eval/test']?.n).toBeGreaterThan(5)
    expect(first?.series['eval/test']?.last).toBe(first?.metrics['eval/test'])
    expect(first?.status).toBe('converged')
  })

  it('keeps the last tape, or every tape, on request', () => {
    const small = { method: ridge, vary: knobsOf(ridge, ['alpha']), points: 3, steps: 30, metrics: ['eval/test'] }
    const run = (keep: 'none' | 'last' | 'all'): number =>
      runAlgorithm(sweep({ ...small, keep, logs: { 'solver/objective': Log() } }), { seed: 0, maxSteps: 3 }).state.tapes
        .length
    expect([run('none'), run('last'), run('all')]).toEqual([0, 1, 3])
    const kept = runAlgorithm(sweep({ ...small, keep: 'last', logs: { 'solver/objective': Log() } }), {
      seed: 0,
      maxSteps: 3,
    })
    expect(metricSeries(kept.state.tapes[0] as never, 'solver/objective').values.length).toBeGreaterThan(1)
  })
})

describe('the sweep base: a repeated-fit study of an estimator’s variance', () => {
  const configs = [{ n: 10 }, { n: 40 }, { n: 160 }]
  const study = sweep({ method: sampleMean, configs, repeats: 300, metrics: ['estimate/mean'], steps: 5 })
  const { state } = runAlgorithm(study, { seed: 7, maxSteps: 900 })

  it('has the spread of the estimate fall as one over the root of the sample size', () => {
    const summaries = sweepConfigurations(state)
    for (const [i, c] of summaries.entries()) {
      const n = (configs[i] as { n: number }).n
      expect(c.n).toBe(300)
      const spread = c.std['estimate/mean'] as number
      expect(spread / (1 / Math.sqrt(n))).toBeGreaterThan(0.8)
      expect(spread / (1 / Math.sqrt(n))).toBeLessThan(1.2)
      expect(Math.abs(c.mean['estimate/mean'] as number)).toBeLessThan(4 / Math.sqrt(n * 300))
    }
  })

  it('draws fresh keys per repeat, deterministically', () => {
    const xs = state.trials.filter((t) => t.config === 0).map((t) => t.metrics['estimate/mean'] as number)
    expect(new Set(xs).size).toBe(300)
    expect(std(xs)).toBeCloseTo(sweepConfigurations(state)[0]?.std['estimate/mean'] as number, 12)
    const again = runAlgorithm(study, { seed: 7, maxSteps: 900 }).state
    expect(again.trials.map((t) => t.metrics['estimate/mean'])).toEqual(
      state.trials.map((t) => t.metrics['estimate/mean']),
    )
    const other = runAlgorithm(study, { seed: 8, maxSteps: 900 }).state
    expect(other.trials[0]?.metrics['estimate/mean']).not.toBe(state.trials[0]?.metrics['estimate/mean'])
  })
})

describe('the sweep base: designs, pairing, serialisation', () => {
  const base = { method: sampleMean, metrics: ['estimate/mean'], steps: 5 } as const

  it('samples the design from the study’s key, in the range, the same every time', () => {
    const study = sweep({
      ...base,
      vary: knobsOf(sampleMean, ['shift'], { shift: real(-1, 1) }),
      design: 'sample',
      trials: 6,
    })
    const a = runAlgorithm(study, { seed: 2, maxSteps: 6 }).state
    const b = runAlgorithm(study, { seed: 2, maxSteps: 6 }).state
    const shifts = a.configs.map((c) => (c as { shift: number }).shift)
    expect(shifts).toEqual(b.configs.map((c) => (c as { shift: number }).shift))
    expect(new Set(shifts).size).toBe(6)
    expect(shifts.every((x) => x >= -1 && x <= 1)).toBe(true)
  })

  it('pairs the repeats across configurations by default, so only the configuration differs', () => {
    const run = (paired: boolean): number[] => {
      const { state } = runAlgorithm(sweep({ ...base, configs: [{ shift: 0 }, { shift: 2 }], repeats: 2, paired }), {
        seed: 3,
        maxSteps: 4,
      })
      return [0, 1].map((r) => {
        const at = (c: number): number =>
          state.trials.find((t) => t.config === c && t.repeat === r)?.metrics['estimate/mean'] as number
        return at(1) - at(0)
      })
    }
    expect(run(true)[0]).toBeCloseTo(2, 12)
    expect(run(true)[1]).toBeCloseTo(2, 12)
    expect(Math.abs((run(false)[0] as number) - 2)).toBeGreaterThan(1e-3)
  })

  it('logs the trial, the configuration’s running spread and the best so far through ctx.log', () => {
    const rec = recorder()
    const study = sweep({ ...base, configs: [{ shift: 3 }, { shift: 1 }, { shift: 2 }], repeats: 2 })
    runAlgorithm(study, { seed: 4, maxSteps: 6, tracer: rec })
    const run = rec.run()
    expect(metricSeries(run, 'trial/shift').values).toEqual([3, 3, 1, 1, 2, 2])
    expect(metricSeries(run, 'trial/config').values).toEqual([0, 0, 1, 1, 2, 2])
    expect(metricSeries(run, 'trial/repeat').values).toEqual([0, 1, 0, 1, 0, 1])
    const best = metricSeries(run, 'trial/best').values
    expect(best).toHaveLength(6)
    best.forEach((b, i) => {
      if (i > 0) expect(b).toBeLessThanOrEqual(best[i - 1] as number)
    })
    const spread = metricSeries(run, 'config/estimate/mean/std').values
    expect(spread[0]).toBe(0)
    expect(spread[1]).toBeGreaterThan(0)
    expect(spread[2]).toBe(0)
  })

  it('serialises, so a study resumes from a checkpoint exactly', () => {
    const study = sweep({ ...base, configs: [{ n: 5 }, { n: 20 }], repeats: 4 })
    const straight = runAlgorithm(study, { seed: 9, maxSteps: 8 })
    const half = runAlgorithm(study, { seed: 9, maxSteps: 5 })
    const json = JSON.parse(JSON.stringify(study.serialise?.(half.state)))
    const resumed = runAlgorithm(study, { seed: 9, maxSteps: 3, start: { state: json, step: 5, serialised: true } })
    expect(resumed.state.trials).toEqual(straight.state.trials)
    expect(resumed.status).toBe('completed')
  })

  it('refuses a study that reads nothing, or judges by something it does not read', () => {
    expect(() => sweep({ ...base, metrics: [] })).toThrow(/at least one metric/)
    expect(() => sweep({ ...base, objective: { metric: 'other' } })).toThrow(/not one of the metrics/)
    expect(() => runAlgorithm(sweep({ ...base, metrics: ['no/such'] }), { seed: 0, maxSteps: 1 })).toThrow(
      /no path 'no\/such'/,
    )
    expect(() => knobsOf(sampleMean, ['nope'])).toThrow(/no knob/)
  })
})

describe('the sweep as a method', () => {
  const method = sweepMethod({
    method: sampleMean,
    vary: knobsOf(sampleMean, ['n'], { n: int(5, 50) }),
    points: 3,
    metrics: ['estimate/mean'],
    repeats: 5,
    steps: 5,
  })

  it('runs as a session with the study’s knobs, a contract and derived probes', () => {
    const s = session(method, {
      mode: 'eager',
      steps: 100,
      config: { repeats: 4 },
      logs: { 'trial/n': Log(), 'trial/best': Log(), 'config/estimate/mean/mean': Log() },
    })
    expect(s.state.trials).toHaveLength(12)
    expect(metricSeries(s.run(), 'trial/n').values.slice(0, 5)).toEqual([5, 5, 5, 5, 28])
    const configurations = s.probe(s.t, 'sweep/configurations') as unknown as { n: number }[]
    expect(configurations.map((c) => c.n)).toEqual([4, 4, 4])
    expect(s.probe(s.t, 'sweep/best')).not.toBeNull()
    expect(s.stateAt(6).trials).toHaveLength(6)
  })

  it('offers the design as knobs: a grid’s points or a sample’s size', () => {
    const s = session(method, { mode: 'eager', steps: 100, config: { design: 'sample', trials: 2, repeats: 1 } })
    expect(s.state.trials).toHaveLength(2)
    expect(Object.keys(method.config.spaces)).toEqual(['repeats', 'steps', 'keep', 'design', 'points', 'trials'])
  })
})

describe('the race', () => {
  /** A least-squares problem from the key, minimised by the gradient mode with the optimiser the configuration names. */
  const fit = defineMethod<unknown, { optimiser: string; learningRate: number }>({
    name: 'fit',
    config: dict({
      optimiser: choice(['sgd', 'adam']),
      learningRate: real(1e-3, 1, { log: true, default: 0.05 }),
    }),
    make: (config, k) => {
      const [kx, ke] = split(k, 2) as [typeof k, typeof k]
      const xs = normal(kx, [60, 3])
      const y = add(matmul(xs, tensor([1, -2, 0.5])), mul(normal(ke, [60]), 0.1))
      return loggedSolver(
        { objective: (w: Value) => mean(square(sub(matmul(xs, w), y))), x0: [0, 0, 0] },
        { kind: 'gradient', learningRate: config.learningRate, optimiser: { kind: config.optimiser } },
        k,
      )
    },
    contract: { ...solverContract, 'solver/lr': scalar() },
  })
  const study = race(fit, {
    slow: { optimiser: 'sgd', learningRate: 0.01 },
    quick: { optimiser: 'adam', learningRate: 0.2 },
  })

  it('drives the variants in one run, from the same key, logging each under its own prefix', () => {
    const s = session(study, {
      mode: 'eager',
      steps: 40,
      seed: 5,
      logs: {
        'race/slow/solver/objective': Log(),
        'race/quick/solver/objective': Log(),
        'race/quick/solver/lr': Log(),
      },
    })
    const slow = metricSeries(s.run(), 'race/slow/solver/objective').values
    const quick = metricSeries(s.run(), 'race/quick/solver/objective').values
    expect(slow).toHaveLength(40)
    // The same data and start: the objective at the first step is the same function of the same point, then they part.
    expect(slow[0]).toBeGreaterThan(0)
    expect(quick[39]).toBeLessThan(slow[39] as number)
    expect(metricSeries(s.run(), 'race/quick/solver/lr').values[0]).toBeCloseTo(0.2, 12)
    expect(Object.keys(s.contract).filter((p) => p.startsWith('race/slow/')).length).toBeGreaterThan(2)
  })

  it('gives every variant the data of the key, and another key other data', () => {
    const start = (seed: number): number[] => {
      const s = session(study, { mode: 'eager', steps: 5, seed, logs: 'off' })
      const at0 = s.stateAt(0).states as Record<string, { value: number; x: Float64Array }>
      return [at0['slow']?.value as number, at0['quick']?.value as number]
    }
    // Both variants start at zero on the same data, so the objective at the start is the same number.
    expect(start(5)[0]).toBe(start(5)[1])
    expect(start(6)[0]).not.toBeCloseTo(start(5)[0] as number, 3)
  })

  it('seeks: the state at any step is every variant’s state there', () => {
    const s = session(study, { mode: 'eager', steps: 30, seed: 5, logs: 'off', checkpointEvery: 7 })
    const at10 = s.stateAt(10)
    const live = session(study, { mode: 'eager', steps: 10, seed: 5, logs: 'off' }).state
    expect(JSON.stringify(at10.states)).toBe(JSON.stringify(live.states))
    expect(() => race(fit, {})).toThrow(/at least one variant/)
  })
})
