import { describe, expect, it } from 'vitest'
import { key, normal, uniform } from '../random/index.ts'
import { dict, type SpaceValue } from '../space/index.ts'
import {
  add,
  matmul,
  mean,
  mul,
  relu,
  softplus,
  square,
  sub,
  sum,
  take,
  tensor,
  toArray,
  type Value,
} from '../tensor/index.ts'
import { defineMethod, Log, metricSeries, runAlgorithm, session } from '../trace/index.ts'
import { optimisers } from './optimiser-table.ts'
import { schedules } from './schedule-table.ts'
import { gradientContract, loggedSolver, type SmoothProblem, smoothSolvers, solverContract } from './solvers.ts'
import { gradientAlgorithm } from './stochastic.ts'

// An ill-conditioned bowl centred at (1, -2): f(x) = (x0 - 1)^2 + 10 (x1 + 2)^2.
const bowl: SmoothProblem = {
  objective: (x: Value): Value => sum(mul(square(sub(x, [1, -2])), [1, 10])),
  x0: [3, 3],
}

/** A logistic regression on 300 planted points: the weights are (2, -1) and the intercept 0.5. */
function logistic(): SmoothProblem & { readonly batches: { readonly n: number } } {
  const n = 300
  const xs = normal(key(11), [n, 2])
  const noise = uniform(key(12), [n])
  const z = toArray(add(matmul(xs, tensor([2, -1])), 0.5) as never) as number[]
  const y = tensor(z.map((v, i) => ((noise.data[i] ?? 0) < 1 / (1 + Math.exp(-v)) ? 1 : 0)))
  return {
    x0: [0, 0, 0],
    batches: { n },
    objective: (w: Value, batch?: readonly number[]): Value => {
      const x = batch === undefined ? xs : take(xs, batch)
      const t = batch === undefined ? y : take(y, batch)
      const logit = add(matmul(x, take(w, [0, 1])), take(w, [2]))
      return mean(sub(softplus(logit), mul(t, logit)))
    },
  }
}

const distance = (x: ArrayLike<number>): number => Math.hypot((x[0] ?? 0) - 1, (x[1] ?? 0) + 2)
const cosine: SpaceValue = { kind: 'cosine', steps: 400, endRatio: 0 }
const constant: SpaceValue = { kind: 'constant' }

const reach = (config: SpaceValue, steps = 400): Float64Array => {
  const { state } = runAlgorithm(smoothSolvers.make(bowl, config, key(0)), { seed: 0, maxSteps: steps })
  return (state as { x: Float64Array }).x
}

describe('the gradient mode', () => {
  it('reaches the minimum of the bowl with every optimiser', () => {
    const cases: readonly (readonly [string, number, SpaceValue, object?])[] = [
      ['sgd', 0.03, constant],
      ['sgd', 0.01, constant, { momentum: 0.9, nesterov: true }],
      ['rmsprop', 0.1, cosine],
      ['adagrad', 1, cosine],
      ['adadelta', 10, constant],
      ['adam', 0.1, cosine],
      ['adamw', 0.1, cosine, { weightDecay: 0 }],
      ['amsgrad', 0.1, cosine],
      ['adamax', 0.3, cosine],
      ['nadam', 0.1, cosine],
      ['radam', 0.05, constant],
      ['lion', 0.1, cosine],
    ]
    expect(new Set(cases.map(([name]) => name))).toEqual(new Set(optimisers.names))
    for (const [name, learningRate, schedule, knobs] of cases) {
      const config = { kind: 'gradient', learningRate, schedule, optimiser: { kind: name, ...knobs } }
      expect(distance(reach(config)), `${name} ${JSON.stringify(knobs ?? {})}`).toBeLessThan(1e-2)
    }
  })

  it('is plain gradient descent for SGD at a constant rate, step for step', () => {
    const alg = gradientAlgorithm(bowl.objective, bowl.x0, { learningRate: 0.02, gradTol: 0 })
    const { state } = runAlgorithm(alg, { seed: 0, maxSteps: 7 })
    let x = [3, 3]
    for (let i = 0; i < 7; i++)
      x = [(x[0] ?? 0) - 0.02 * 2 * ((x[0] ?? 0) - 1), (x[1] ?? 0) - 0.02 * 20 * ((x[1] ?? 0) + 2)]
    expect(Array.from(state.x)[0]).toBeCloseTo(x[0] ?? 0, 12)
    expect(Array.from(state.x)[1]).toBeCloseTo(x[1] ?? 0, 12)
    expect(state.t).toBe(7)
  })

  it('fills the knobs a configuration leaves out', () => {
    expect(distance(reach({ kind: 'gradient', learningRate: 0.03 }))).toBeLessThan(1e-2)
    // A schedule named without its knobs takes their defaults (a cosine over a thousand steps), the optimiser too.
    const partial = { kind: 'gradient', learningRate: 0.1, optimiser: { kind: 'adam' }, schedule: { kind: 'cosine' } }
    expect(distance(reach(partial))).toBeLessThan(0.5)
  })

  it('trains a batched logistic regression to the full-data optimum', () => {
    const problem = logistic()
    const full = (x: ArrayLike<number>): number =>
      toArray(problem.objective(tensor(Array.from(x))) as never) as unknown as number
    const bfgs = runAlgorithm(smoothSolvers.make(problem, { kind: 'bfgs', damped: false, gradTol: 1e-8 }, key(0)), {
      seed: 0,
      maxSteps: 100,
    }).state as { x: Float64Array }
    const config = {
      kind: 'gradient',
      learningRate: 0.3,
      batch: 32,
      optimiser: { kind: 'adam' },
      schedule: { kind: 'cosine', steps: 400, endRatio: 0 },
    }
    const { state } = runAlgorithm(smoothSolvers.make(problem, config, key(0)), { seed: 0, maxSteps: 400 })
    const x = (state as { x: Float64Array }).x
    expect(full(x) - full(bfgs.x)).toBeLessThan(0.02)
    expect(full(x) - full(bfgs.x)).toBeGreaterThanOrEqual(-1e-9)
    expect(x[0]).toBeGreaterThan(1.3)
    expect(x[1]).toBeLessThan(-0.7)
  })

  it('draws the same batches for the same key, other batches for another, and the whole set when it fits', () => {
    const problem = logistic()
    const run = (seed: number, batch: number): number[] => {
      const alg = gradientAlgorithm(problem.objective, problem.x0, {
        learningRate: 0.1,
        batch,
        batches: problem.batches,
      })
      return Array.from(runAlgorithm(alg, { seed, maxSteps: 15 }).state.x)
    }
    expect(run(3, 16)).toEqual(run(3, 16))
    expect(run(3, 16)).not.toEqual(run(4, 16))
    expect(run(3, 300)).toEqual(run(4, 300))
    const wholeSet = gradientAlgorithm(problem.objective, problem.x0, { learningRate: 0.1 })
    expect(run(3, 300)).toEqual(Array.from(runAlgorithm(wholeSet, { seed: 3, maxSteps: 15 }).state.x))
  })

  it('serialises: resuming from a checkpoint continues exactly where the run would have gone', () => {
    const problem = logistic()
    const alg = gradientAlgorithm(problem.objective, problem.x0, {
      learningRate: 0.1,
      batch: 20,
      batches: problem.batches,
      optimiser: (lr) => optimisers.make({ kind: 'adam' })(lr),
    })
    const straight = runAlgorithm(alg, { seed: 5, maxSteps: 30 })
    const half = runAlgorithm(alg, { seed: 5, maxSteps: 12 })
    const json = alg.serialise?.(half.state)
    expect(JSON.parse(JSON.stringify(json))).toEqual(json)
    const resumed = runAlgorithm(alg, {
      seed: 5,
      maxSteps: 18,
      start: { state: JSON.parse(JSON.stringify(json)), step: 12, serialised: true },
    })
    expect(Array.from(resumed.state.x)).toEqual(Array.from(straight.state.x))
    expect(resumed.state.t).toBe(30)
  })

  // A batch whose examples all have zero loss has an exactly zero gradient far from the minimum: the tolerance is a
  // statement about the whole gradient, so a lucky batch must not end the run as `converged`.
  it('declares convergence from the whole gradient, never from the gradient of a batch', () => {
    const xs = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    const labels = tensor([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    const squaredHinge = (w: Value, batch?: readonly number[]): Value => {
      const s = batch === undefined ? xs : take(xs, batch)
      const y = batch === undefined ? labels : take(labels, batch)
      return mean(square(relu(sub(1, mul(y, mul(s, w))))))
    }
    const run = (batch: number | null, seed: number): { status: string; steps: number; w: number } => {
      const alg = gradientAlgorithm(squaredHinge, tensor([0.5]), {
        learningRate: 0.01,
        batch,
        batches: { n: 10 },
        gradTol: 1e-5,
      })
      const r = runAlgorithm(alg, { seed, maxSteps: 60 })
      return { status: r.status, steps: r.steps, w: r.state.x[0] ?? 0 }
    }
    // Start 0.5, optimum -0.065: the examples with y s w >= 1 have no loss, so batches of one or two often see no gradient.
    for (const seed of [0, 1, 2, 3, 4, 5]) {
      for (const batch of [1, 2]) {
        const r = run(batch, seed)
        expect(r.status, `batch ${batch} seed ${seed}`).toBe('completed')
        expect(r.steps, `batch ${batch} seed ${seed}`).toBe(60)
      }
    }
    // The whole set, with the whole gradient in hand, still stops at the minimum.
    const whole = run(null, 0)
    expect(whole.status).toBe('converged')
    expect(whole.w).toBeCloseTo(-0.065, 2)
  })

  it('refuses a batch for a problem that cannot be batched', () => {
    expect(() => gradientAlgorithm(bowl.objective, bowl.x0, { batch: 4 })).toThrow(/how many examples/)
    expect(() => gradientAlgorithm(logistic().objective, [0, 0, 0], { batch: 0, batches: { n: 5 } })).toThrow(
      /positive integer/,
    )
  })

  it('logs the objective, the gradient norm, the step and the scheduled learning rate through loggedSolver', () => {
    const config = {
      kind: 'gradient',
      learningRate: 0.1,
      optimiser: { kind: 'sgd' },
      schedule: { kind: 'linear', steps: 8, endRatio: 0 },
    }
    const method = defineMethod({
      name: 'bowl-gradient',
      config: dict({}),
      make: (_c, k) => loggedSolver(bowl, config, k),
      contract: { ...solverContract, ...gradientContract },
    })
    const s = session(method, {
      mode: 'eager',
      steps: 6,
      logs: { 'solver/lr': Log(), 'solver/objective': Log(), 'solver/gradNorm': Log(), 'solver/step': Log() },
    })
    const lr = metricSeries(s.run(), 'solver/lr').values
    const expected = schedules.make(config.schedule)(0.1)
    expect(lr).toHaveLength(6)
    lr.forEach((v, i) => {
      expect(v).toBeCloseTo(expected(i), 12)
    })
    const objective = metricSeries(s.run(), 'solver/objective').values
    expect(objective[5]).toBeLessThan(objective[0] ?? 0)
    expect(metricSeries(s.run(), 'solver/step').values).toHaveLength(6)
  })
})
