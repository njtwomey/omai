import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { type Algorithm, policies, recorder, runAlgorithm } from '../trace/index.ts'
import {
  coalitionalGame,
  gloveGame,
  orderSamplers,
  sampledShapley,
  sampledShapleyAlgorithm,
  shapleyContract,
  shapleyValue,
} from './cooperative.ts'
import {
  asymmetricReplicatorDynamics,
  asymmetricReplicatorDynamicsAlgorithm,
  replicatorContract,
  replicatorDynamics,
  replicatorDynamicsAlgorithm,
  replicatorIntegrators,
} from './evolutionary.ts'
import {
  dominanceRules,
  eliminationContract,
  iteratedElimination,
  iteratedEliminationAlgorithm,
  prisonersDilemma,
  rockPaperScissors,
  stagHunt,
} from './normal-form/index.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

/** Run to the end, and again from a JSON checkpoint cut at `cut` steps; both must end the same. */
function expectResumes<S>(alg: Algorithm<S>, cut: number, project: (s: S) => unknown): S {
  const whole = runAlgorithm(alg, { maxSteps: 10_000 })
  const half = runAlgorithm(alg, { maxSteps: cut })
  const stored = viaJson(alg.serialise?.(half.state) ?? null)
  const rest = runAlgorithm(alg, { maxSteps: 10_000, start: { state: stored, step: half.steps, serialised: true } })
  expect(project(rest.state)).toEqual(project(whole.state))
  expect(rest.status).toBe(whole.status)
  expect(rest.steps).toBe(whole.steps)
  return whole.state
}

/** The values logged to scalar and tensor paths of a run. */
function logged<S>(alg: Algorithm<S>, maxSteps = 10_000): Map<string, unknown[]> {
  const rec = recorder()
  runAlgorithm(alg, {
    maxSteps,
    tracer: rec,
    policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
  })
  const out = new Map<string, unknown[]>()
  for (const e of rec.events) {
    if (e.type === 'metric') out.set(e.name, [...(out.get(e.name) ?? []), e.value])
    if (e.type === 'tensor') out.set(e.name, [...(out.get(e.name) ?? []), Array.from(e.values as ArrayLike<number>)])
  }
  return out
}

describe('replicator dynamics algorithms', () => {
  const hawkDove = [
    [0, 3],
    [1, 2],
  ]
  const timepoints = [0, 1, 2, 4, 8]

  it('step through the time points and give the trajectory the function returns', () => {
    const alg = replicatorDynamicsAlgorithm(hawkDove, { x0: [0.9, 0.1], timepoints })
    const run = runAlgorithm(alg, { maxSteps: 10 })
    expect(run.steps).toBe(4)
    expect(run.status).toBe('completed')
    const want = replicatorDynamics(hawkDove, { x0: [0.9, 0.1], timepoints })
    for (const [i, x] of run.state.path.entries()) {
      for (const [j, v] of x.entries()) expect(v).toBeCloseTo(want[i]?.[j] ?? Number.NaN, 12)
    }
    // The mixed equilibrium of hawk-dove is at x = 1/2 in this payoff matrix.
    expect(Math.abs((run.state.x[0] ?? 0) - 0.5)).toBeLessThan(0.05)
  })

  it('logs state, time, fitness and mean fitness, with the state on the simplex', () => {
    const l = logged(replicatorDynamicsAlgorithm(hawkDove, { x0: [0.9, 0.1], timepoints }))
    expect(l.get('replicator/time')).toEqual([1, 2, 4, 8])
    for (const x of l.get('replicator/x') as number[][]) expect((x[0] ?? 0) + (x[1] ?? 0)).toBeCloseTo(1, 9)
    const x1 = (l.get('replicator/x') as number[][])[0] as number[]
    const f1 = (l.get('replicator/fitness') as number[][])[0] as number[]
    expect(f1[0]).toBeCloseTo(3 * (x1[1] ?? 0), 9)
    expect((l.get('replicator/meanFitness') as number[][])[0]).toHaveLength(1)
    expect(Object.keys(replicatorContract)).toHaveLength(4)
  })

  it('integrators of higher order are closer to rk4 at the same step', () => {
    const run = (integrator: 'rk4' | 'midpoint' | 'euler', maxStep: number): number =>
      replicatorDynamics(hawkDove, { x0: [0.9, 0.1], timepoints: [0, 2], maxStep, integrator })[1]?.[0] ?? 0
    const reference = run('rk4', 0.001)
    const errors = (['euler', 'midpoint', 'rk4'] as const).map((k) => Math.abs(run(k, 0.1) - reference))
    expect(errors[0]).toBeGreaterThan(errors[1] ?? 0)
    expect(errors[1]).toBeGreaterThan(errors[2] ?? 0)
    expect(replicatorIntegrators.names).toEqual(['rk4', 'midpoint', 'euler'])
  })

  it('the asymmetric form logs both populations and resumes from JSON', () => {
    const game = prisonersDilemma()
    const alg = asymmetricReplicatorDynamicsAlgorithm(game, { x0: [0.9, 0.1], y0: [0.5, 0.5], timepoints })
    const l = logged(alg)
    expect((l.get('replicator/meanFitness') as number[][])[0]).toHaveLength(2)
    expect((l.get('replicator/x') as number[][])[0]).toHaveLength(4)
    const s = expectResumes(alg, 2, (x) => ({ i: x.index, p: x.path.map((q) => q.map((v) => Math.round(v * 1e9))) }))
    const want = asymmetricReplicatorDynamics(game, { x0: [0.9, 0.1], y0: [0.5, 0.5], timepoints })
    expect(s.x[0]).toBeCloseTo(want.row.at(-1)?.[0] ?? Number.NaN, 12)
    expect(() => asymmetricReplicatorDynamicsAlgorithm({ ...game, shape: [2, 2, 2] }, {})).toThrow(/two-player/)
  })

  it('rejects a decreasing time grid and a bad initial state', () => {
    expect(() => replicatorDynamicsAlgorithm(hawkDove, { timepoints: [1, 0] })).toThrow(/increase/)
    expect(() => replicatorDynamicsAlgorithm(hawkDove, { x0: [1] })).toThrow(/one share/)
  })
})

describe('iteratedEliminationAlgorithm', () => {
  // A game solvable by two rounds: column 2 is dominated by column 1, then row 1 by row 0.
  const game = {
    name: 'two rounds',
    shape: [2, 3],
    actionNames: [
      ['a', 'b'],
      ['x', 'y', 'z'],
    ],
    payoffs: [Float64Array.of(3, 2, 0, 1, 2, 1), Float64Array.of(2, 1, 0, 3, 1, 2)],
  }

  it('removes dominated actions round by round and matches iteratedElimination', () => {
    const run = runAlgorithm(iteratedEliminationAlgorithm(prisonersDilemma()), { maxSteps: 10 })
    expect(run.status).toBe('converged')
    expect(run.state.actions).toEqual([[1], [1]])
    expect(run.steps).toBe(2)
    const r = iteratedElimination(prisonersDilemma())
    expect(r.actions).toEqual([[1], [1]])
    expect(r.eliminated).toEqual(run.state.eliminated)
    expect(r.game.shape).toEqual([1, 1])
  })

  it('leaves a game with no dominated action untouched after one empty round', () => {
    const rps = rockPaperScissors()
    const run = runAlgorithm(iteratedEliminationAlgorithm(rps), { maxSteps: 10 })
    expect(run.steps).toBe(1)
    expect(run.state.eliminated).toEqual([])
    expect(iteratedElimination(rps).game).toBe(rps)
  })

  it('logs the round, the removals and what remains, shrinking monotonically', () => {
    const l = logged(iteratedEliminationAlgorithm(stagHunt(), { strict: false }))
    const rounds = l.get('elim/round') as number[]
    expect(rounds).toEqual(rounds.map((_, i) => i + 1))
    const remaining = l.get('elim/remaining') as number[][]
    for (let i = 1; i < remaining.length; i++) {
      expect((remaining[i] ?? []).every((k, p) => k <= (remaining[i - 1]?.[p] ?? 0))).toBe(true)
    }
    expect((l.get('elim/removed') as number[][]).at(-1)).toEqual([])
    expect(Object.keys(eliminationContract)).toHaveLength(3)
  })

  it('resumes from a JSON checkpoint, and the rules table gives the options', () => {
    const s = expectResumes(iteratedEliminationAlgorithm(game, {}), 1, (x) => ({
      a: x.actions,
      e: x.eliminated,
      r: x.round,
    }))
    expect(s.actions[1]).not.toContain(2)
    expect(dominanceRules.make({ kind: 'weak-mixed' })).toEqual({ strict: false, mixed: true })
    expect(dominanceRules.where((p) => p.linearProgram)).toEqual(['strict-mixed', 'weak-mixed'])
  })
})

describe('sampledShapleyAlgorithm', () => {
  const glove = gloveGame(1, 2)

  it('one order per step, giving the estimates sampledShapley returns', () => {
    const k = key(4)
    const run = runAlgorithm(sampledShapleyAlgorithm(3, glove, { samples: 40, key: k }), { maxSteps: 100 })
    expect(run.steps).toBe(40)
    expect(run.state.draws.every((d) => d.length === 40)).toBe(true)
    const r = sampledShapley(3, glove, { samples: 40, key: k })
    const means = run.state.draws.map((d) => d.reduce((a, b) => a + b, 0) / d.length)
    for (const [i, v] of r.values.entries()) expect(means[i]).toBeCloseTo(v, 12)
  })

  it('every sample distributes the whole worth (efficiency), and the estimate nears the exact value', () => {
    const l = logged(sampledShapleyAlgorithm(3, glove, { samples: 600, key: key(1) }))
    const phi = (l.get('shapley/phi') as number[][]).at(-1) as number[]
    expect(phi.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9)
    const exact = shapleyValue(glove)
    const se = (l.get('shapley/stderr') as number[][]).at(-1) as number[]
    for (const [i, v] of exact.entries()) expect(Math.abs((phi[i] ?? 0) - v)).toBeLessThan(5 * (se[i] ?? 1) + 1e-9)
    const orders = l.get('shapley/permutation') as number[][]
    expect(orders.every((o) => [...o].sort().join() === '0,1,2')).toBe(true)
    expect(Object.keys(shapleyContract)).toHaveLength(3)
  })

  it('antithetic orders alternate a permutation and its reverse and are exact for a symmetric game', () => {
    const l = logged(sampledShapleyAlgorithm(3, glove, { samples: 4, key: key(2), antithetic: true }))
    const orders = l.get('shapley/permutation') as number[][]
    expect(orders[1]).toEqual([...(orders[0] ?? [])].reverse())
    expect(orders[3]).toEqual([...(orders[2] ?? [])].reverse())
    // Two players that are interchangeable get the same value under antithetic pairs.
    const sym = coalitionalGame(2, (s) => (s.length === 2 ? 1 : 0))
    const r = sampledShapley(2, sym, { samples: 2, antithetic: true, key: key(3) })
    expect(r.values[0]).toBeCloseTo(0.5, 12)
    expect(r.values[1]).toBeCloseTo(0.5, 12)
    expect(orderSamplers.names).toEqual(['independent', 'antithetic'])
  })

  it('resumes from a JSON checkpoint and rejects zero samples', () => {
    const s = expectResumes(sampledShapleyAlgorithm(3, glove, { samples: 30, key: key(5) }), 12, (x) => x.draws)
    expect(s.t).toBe(30)
    expect(() => sampledShapleyAlgorithm(3, glove, { samples: 0 })).toThrow(/at least one/)
  })
})
