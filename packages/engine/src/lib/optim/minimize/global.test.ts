import { describe, expect, it } from 'vitest'
import { ackley, himmelblau, hop2, quadratic, rastrigin, rosen, sixHumpCamel } from '../../../testing/optim-problems.ts'
import { add, sliceAxis, square, sub, sum, toFlat, type Value } from '../../tensor/index.ts'
import { messages, metricSeries, policies, recorder, runTree, tensorSeries } from '../../trace/index.ts'
import { cmaes, cmaesAlgorithm } from './cmaes.ts'
import { nelderMead } from './derivative-free.ts'
import { differentialEvolution } from './differential-evolution.ts'
import { dualAnnealing } from './dual-annealing.ts'
import { basinHopping, simulatedAnnealing } from './global.ts'
import { minimize } from './minimizer.ts'
import { particleSwarm } from './particle-swarm.ts'
import { powell } from './powell.ts'

const v = (x: unknown): Value => x as Value
const flat = (x: unknown): number[] => toFlat(x as Value)
const close = (a: readonly number[], b: readonly number[], tol: number): void => {
  expect(a).toHaveLength(b.length)
  for (const [i, x] of a.entries()) expect(Math.abs(x - (b[i] ?? 0))).toBeLessThan(tol)
}
const box2: [number, number][] = [
  [-5, 5],
  [-5, 5],
]
const sixHumpMin = -1.0316284534898774

describe('differential evolution', () => {
  it('is reproducible from its seed and differs across seeds', () => {
    const a = differentialEvolution((x) => rastrigin(v(x)), [1, 1], {
      bounds: box2,
      seed: 3,
      polish: false,
      maxIterations: 20,
    })
    const b = differentialEvolution((x) => rastrigin(v(x)), [1, 1], {
      bounds: box2,
      seed: 3,
      polish: false,
      maxIterations: 20,
    })
    const c = differentialEvolution((x) => rastrigin(v(x)), [1, 1], {
      bounds: box2,
      seed: 4,
      polish: false,
      maxIterations: 20,
    })
    expect(flat(a.x)).toEqual(flat(b.x))
    expect(flat(a.x)).not.toEqual(flat(c.x))
  })

  it('supports every strategy, initialisation and updating mode', () => {
    const strategies = [
      'best1bin',
      'best1exp',
      'rand1bin',
      'rand1exp',
      'randtobest1bin',
      'randtobest1exp',
      'currenttobest1bin',
      'currenttobest1exp',
      'best2bin',
      'best2exp',
      'rand2bin',
      'rand2exp',
    ] as const
    for (const strategy of strategies) {
      const r = differentialEvolution((x) => quadratic(v(x)), [0, 0, 0], {
        bounds: [
          [-3, 3],
          [-3, 3],
          [-3, 3],
        ],
        strategy,
        seed: 0,
        updating: strategy.startsWith('rand') ? 'deferred' : 'immediate',
      })
      close(flat(r.x), [2 / 9, 1 / 9, 13 / 9], 1e-6)
    }
    for (const init of ['random', 'halton'] as const) {
      const r = differentialEvolution((x) => himmelblau(v(x)), [0, 0], { bounds: box2, init, seed: 2 })
      expect(r.fun).toBeLessThan(1e-12)
    }
    const points = Array.from({ length: 10 }, (_, i) => [i - 4.5, 4.5 - i])
    const r = differentialEvolution((x) => sixHumpCamel(v(x)), [0, 0], {
      bounds: [
        [-3, 3],
        [-2, 2],
      ],
      init: points,
      seed: 0,
    })
    expect(Math.abs(r.fun - sixHumpMin)).toBeLessThan(1e-9)
    expect(() => differentialEvolution((x) => sum(v(x)), [0, 0], { bounds: box2, init: [[0, 0]] })).toThrow()
  })

  it('polishes as a child run, with a custom method, or not at all', () => {
    const rec = recorder()
    const polished = differentialEvolution((x) => rosen(v(x)), [0, 0], { bounds: box2, seed: 0, tracer: rec })
    const [root] = runTree(rec.run())
    expect(root?.run.name).toBe('differentialEvolution')
    expect(root?.children.map((c) => c.run.name)).toEqual(['nelderMead'])
    close(flat(polished.x), [1, 1], 1e-7)
    const custom = differentialEvolution((x) => rosen(v(x)), [0, 0], { bounds: box2, seed: 0, polish: powell })
    close(flat(custom.x), [1, 1], 1e-4)
    const raw = differentialEvolution((x) => rosen(v(x)), [0, 0], { bounds: box2, seed: 0, polish: false })
    expect(raw.fun).toBeGreaterThanOrEqual(polished.fun)
  })

  it('handles constraints by feasibility, includes x0, and honours the budget', () => {
    // minimise x + y on the box subject to x^2 + y^2 <= 1: (-1/sqrt2, -1/sqrt2)
    const r = differentialEvolution((x) => sum(v(x)), [0, 0], {
      bounds: box2,
      constraints: [(x) => sub(1, sum(square(v(x))))],
      seed: 0,
    })
    expect(r.converged).toBe(true)
    close(flat(r.x), [-Math.SQRT1_2, -Math.SQRT1_2], 1e-6)
    const infeasible = differentialEvolution((x) => sum(v(x)), [0, 0], {
      bounds: box2,
      constraints: [(x) => sub(-100, sum(square(v(x))))],
      seed: 0,
      maxIterations: 3,
    })
    expect(infeasible.status).toBe('failed')
    const withX0 = differentialEvolution((x) => sum(square(sub(v(x), [1, 1]))), [1, 1], {
      bounds: box2,
      includeX0: true,
      seed: 0,
      polish: false,
      maxIterations: 1,
    })
    close(flat(withX0.x), [1, 1], 1e-12)
    const capped = differentialEvolution((x) => rosen(v(x)), [0, 0], {
      bounds: box2,
      seed: 0,
      maxEvaluations: 100,
      polish: false,
    })
    expect(capped.status).toBe('failed')
    expect(capped.nfev).toBeLessThanOrEqual(100)
  })

  it('records convergence and the population', () => {
    const rec = recorder()
    differentialEvolution((x) => ackley(v(x)), [1, 1], {
      bounds: box2,
      seed: 0,
      tracer: rec,
      policy: { ...policies.debug, detail: 'full' },
      polish: false,
    })
    const run = rec.run()
    const conv = metricSeries(run, 'convergence').values
    expect((conv.at(-1) ?? 1) < (conv[0] ?? 0)).toBe(true)
    expect(tensorSeries(run, 'population')[0]?.shape).toEqual([30, 2])
  })
})

describe('CMA-ES', () => {
  it('converges on Rosenbrock and adapts sigma and the covariance', () => {
    const rec = recorder()
    const r = cmaes((x) => rosen(v(x)), [0, 0, 0], { seed: 0, tracer: rec })
    expect(r.converged).toBe(true)
    close(flat(r.x), [1, 1, 1], 1e-6)
    const run = rec.run()
    const sigma = metricSeries(run, 'sigma').values
    expect((sigma.at(-1) ?? 1) < 1e-5).toBe(true)
    expect(Math.max(...metricSeries(run, 'condition').values.slice(0, 50))).toBeGreaterThan(10)
  })

  it('keeps offspring in bounds and works through minimize', () => {
    const r = cmaes((x) => sum(square(sub(v(x), [3, 3]))), [0, 0], {
      bounds: [
        [-1, 1],
        [-1, 2],
      ],
      seed: 0,
    })
    close(flat(r.x), [1, 2], 1e-8)
    const m = minimize((x) => quadratic(v(x)), [1, 1, 1], { method: cmaes, seed: 1, popSize: 12 })
    close(flat(m.x), [2 / 9, 1 / 9, 13 / 9], 1e-6)
    expect(() => cmaesAlgorithm((x) => sum(v(x)), [0], { popSize: 1 })).toThrow()
  })
})

describe('particle swarm', () => {
  it('finds global minima with global and ring topologies and a decreasing inertia', () => {
    const g = particleSwarm((x) => ackley(v(x)), [3, 3], { bounds: box2, seed: 0 })
    expect(g.fun).toBeLessThan(1e-8)
    const ring = particleSwarm((x) => himmelblau(v(x)), [0, 0], {
      bounds: box2,
      seed: 0,
      topology: 'ring',
      maxIterations: 400,
    })
    expect(ring.fun).toBeLessThan(1e-8)
    const linear = particleSwarm((x) => sixHumpCamel(v(x)), [0, 0], {
      bounds: [
        [-3, 3],
        [-2, 2],
      ],
      seed: 0,
      inertia: [0.9, 0.4],
      maxIterations: 300,
    })
    expect(Math.abs(linear.fun - sixHumpMin)).toBeLessThan(1e-8)
  })

  it('records diversity and positions and stays in the box', () => {
    const rec = recorder()
    const r = particleSwarm((x) => sum(square(sub(v(x), [9, 9]))), [0, 0], {
      bounds: box2,
      seed: 0,
      tracer: rec,
      policy: { ...policies.debug, detail: 'full' },
      swarmSize: 10,
      maxIterations: 50,
    })
    close(flat(r.x), [5, 5], 1e-9)
    const run = rec.run()
    const d = metricSeries(run, 'diversity').values
    expect((d.at(-1) ?? 1) < (d[0] ?? 0)).toBe(true)
    for (const e of tensorSeries(run, 'positions'))
      for (const p of e.values ?? []) expect(Math.abs(p)).toBeLessThanOrEqual(5)
  })
})

describe('annealing and basin hopping', () => {
  it('simulated annealing cools on each schedule and adapts its steps', () => {
    for (const schedule of ['exponential', 'logarithmic', 'fast', (k: number, t0: number) => t0 * 0.9 ** k] as const) {
      const rec = recorder()
      const r = simulatedAnnealing((x) => himmelblau(v(x)), [0, 0], {
        bounds: box2,
        schedule,
        seed: 0,
        maxIterations: 200,
        tracer: rec,
      })
      expect(r.fun).toBeLessThan(schedule === 'logarithmic' || schedule === 'fast' ? 1e-2 : 1e-4)
      const t = metricSeries(rec.run(), 'temperature').values
      expect((t.at(-1) ?? 1) < (t[0] ?? 0)).toBe(true)
    }
    const stop = simulatedAnnealing((x) => sum(square(v(x))), [1], { seed: 0, minTemperature: 0.5 })
    expect(stop.converged).toBe(true)
    expect(stop.steps).toBe(14)
  })

  it('basin hopping nests its local searches and stops on niterSuccess', () => {
    const rec = recorder()
    const r = basinHopping((x) => hop2(v(x)), [1, 1], {
      seed: 0,
      maxIterations: 40,
      tracer: rec,
      policy: policies.debug,
    })
    close(flat(r.x), [-0.19506755, -0.1], 1e-6)
    const run = rec.run()
    const [root] = runTree(run)
    expect(root?.children).toHaveLength(41)
    expect(messages(run, 'hop').length).toBeGreaterThan(0)
    const quick = basinHopping((x) => sum(square(v(x))), [1, 1], { seed: 0, niterSuccess: 3, localMethod: nelderMead })
    expect(quick.converged).toBe(true)
    expect(quick.steps).toBeLessThan(30)
    const bounded = basinHopping((x) => sum(square(sub(v(x), [9, 9]))), [0, 0], {
      seed: 0,
      maxIterations: 5,
      bounds: box2,
    })
    close(flat(bounded.x), [5, 5], 1e-6)
  })

  it('dual annealing reanneals and can run without local search', () => {
    const rec = recorder()
    const r = dualAnnealing((x) => rastrigin(v(x)), [3, -2], {
      bounds: [
        [-5.12, 5.12],
        [-5.12, 5.12],
      ],
      seed: 0,
      maxIterations: 300,
      restartTemperatureRatio: 0.05,
      tracer: rec,
    })
    expect(r.fun).toBeLessThan(1e-10)
    expect(metricSeries(rec.run(), 'restarts').values.at(-1)).toBeGreaterThan(0)
    const plain = dualAnnealing(
      (x) => add(sum(square(sliceAxis(v(x), 0, 0, 1))), sum(square(sliceAxis(v(x), 0, 1, 2)))),
      [1, 1],
      {
        bounds: box2,
        seed: 0,
        noLocalSearch: true,
        maxIterations: 300,
      },
    )
    expect(plain.fun).toBeLessThan(1e-2)
    const capped = dualAnnealing((x) => rosen(v(x)), [0, 0], { bounds: box2, seed: 0, maxEvaluations: 50 })
    expect(capped.status).toBe('failed')
    expect(() => dualAnnealing((x) => sum(v(x)), [0], { bounds: [[0, Infinity]] })).toThrow()
  })
})
