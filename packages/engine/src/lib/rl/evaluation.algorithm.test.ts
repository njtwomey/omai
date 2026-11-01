import { describe, expect, it } from 'vitest'
import { cartPole } from '../environments/index.ts'
import { key } from '../random/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import {
  policyEvaluation,
  policyEvaluationAlgorithm,
  policyEvaluationContract,
  sweepOrders,
  tabularMdp,
  valueIteration,
} from './mdp.ts'
import { rolloutContract, rolloutPlanner, rolloutPlannerAlgorithm } from './planning.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

// A three-state chain: action 0 stays, action 1 moves right; state 2 is terminal, reached with reward 1.
const chain = tabularMdp({
  transitions: [
    [
      [1, 0, 0],
      [0, 1, 0],
    ],
    [
      [0, 1, 0],
      [0, 0, 1],
    ],
    [
      [0, 0, 1],
      [0, 0, 1],
    ],
  ],
  rewards: [
    [0, 0],
    [0, 1],
    [0, 0],
  ],
  terminalStates: [2],
  gamma: 0.9,
})
const stochastic = tabularMdp({
  transitions: [
    [
      [0.5, 0.5, 0],
      [0, 0.5, 0.5],
    ],
    [
      [0.2, 0.3, 0.5],
      [0, 0.1, 0.9],
    ],
    [
      [0, 0, 1],
      [0, 0, 1],
    ],
  ],
  rewards: [
    [0, 0.5],
    [1, 2],
    [0, 0],
  ],
  terminalStates: [2],
  gamma: 0.9,
})

describe('policyEvaluationAlgorithm', () => {
  it('sweeps converge to the exact values, and the function gives the same numbers', () => {
    const policy = [1, 1, 0]
    const run = runAlgorithm(policyEvaluationAlgorithm(stochastic, policy, { tol: 1e-13 }), { maxSteps: 1000 })
    expect(run.status).toBe('converged')
    const exact = policyEvaluation(stochastic, policy)
    for (const [i, v] of exact.entries()) expect(run.state.V[i]).toBeCloseTo(v, 9)
    const iterative = policyEvaluation(stochastic, policy, { method: 'iterative', tol: 1e-13 })
    for (const [i, v] of iterative.entries()) expect(run.state.V[i]).toBeCloseTo(v, 12)
    expect(run.state.iteration).toBe(run.steps)
  })

  it('the largest change shrinks by at least gamma each synchronous sweep', () => {
    const rec = recorder()
    runAlgorithm(policyEvaluationAlgorithm(stochastic, [1, 1, 0], { tol: 1e-10 }), {
      maxSteps: 500,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const delta = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'eval/delta' ? [e.value] : []))
    expect(delta.length).toBeGreaterThan(5)
    for (let i = 1; i < delta.length; i++) expect(delta[i] ?? 0).toBeLessThanOrEqual((delta[i - 1] ?? 0) * 0.9 + 1e-15)
    const sweeps = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'eval/sweep' ? [e.value] : []))
    expect(sweeps).toEqual(delta.map((_, i) => i + 1))
    expect(rec.events.filter((e) => e.type === 'tensor' && e.name === 'eval/V')).toHaveLength(delta.length)
    expect(Object.keys(policyEvaluationContract)).toHaveLength(3)
  })

  it('in-place sweeps need no more sweeps than synchronous ones and reach the same values', () => {
    const run = (sweep: 'synchronous' | 'in-place') =>
      runAlgorithm(policyEvaluationAlgorithm(chain, [1, 1, 0], { sweep, tol: 1e-12 }), { maxSteps: 1000 })
    const a = run('synchronous')
    const b = run('in-place')
    expect(b.steps).toBeLessThanOrEqual(a.steps)
    for (const [i, v] of a.state.V.entries()) expect(b.state.V[i]).toBeCloseTo(v, 9)
    expect(sweepOrders.names).toEqual(['synchronous', 'in-place'])
  })

  it('agrees with value iteration for the greedy policy and resumes from a JSON checkpoint', () => {
    const best = valueIteration(stochastic).policy
    const alg = policyEvaluationAlgorithm(stochastic, best)
    const whole = runAlgorithm(alg, { maxSteps: 1000 })
    const vi = valueIteration(stochastic)
    for (const [i, v] of vi.V.entries()) expect(whole.state.V[i]).toBeCloseTo(v, 8)
    const half = runAlgorithm(alg, { maxSteps: 3 })
    const stored = viaJson(alg.serialise?.(half.state) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 1000, start: { state: stored, step: 3, serialised: true } })
    expect(Array.from(rest.state.V)).toEqual(Array.from(whole.state.V))
    expect(rest.steps).toBe(whole.steps)
    // The first sweep has an infinite residual before it: JSON must carry it.
    expect(viaJson(alg.serialise?.(alg.init(key(0))) ?? null)).toMatchObject({ delta: 'Infinity' })
  })
})

describe('rolloutPlannerAlgorithm', () => {
  const env = cartPole.create()
  const state: [number, number, number, number] = [0, 0, 0.15, 0.3]
  const options = { rollouts: 6, horizon: 15, key: key(3) }

  it('one batch per step, and rolloutPlanner is it run to the end', () => {
    const run = runAlgorithm(rolloutPlannerAlgorithm(env, state, options), { maxSteps: 20 })
    expect(run.steps).toBe(6)
    expect(run.status).toBe('converged')
    const planned = rolloutPlanner(env, state, options)
    const values = run.state.totals.map((t) => t / 6)
    for (const [i, v] of planned.values.entries()) expect(values[i]).toBeCloseTo(v, 12)
    expect(planned.action).toBe(values.indexOf(Math.max(...values)))
  })

  it('logs running means that settle on the final values, and per-batch returns', () => {
    const rec = recorder()
    runAlgorithm(rolloutPlannerAlgorithm(env, state, options), {
      maxSteps: 20,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const values = rec.events.filter((e) => e.type === 'tensor' && e.name === 'plan/value')
    expect(values).toHaveLength(6)
    const batches = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'plan/rollout' ? [e.value] : []))
    expect(batches).toEqual([1, 2, 3, 4, 5, 6])
    // A cart-pole return over a horizon of 15 is at most 15 per rollout.
    const returns = rec.events.filter((e) => e.type === 'tensor' && e.name === 'plan/return')
    for (const e of returns) if (e.type === 'tensor') for (const r of e.values ?? []) expect(r).toBeLessThanOrEqual(15)
    expect(Object.keys(rolloutContract)).toHaveLength(3)
  })

  it('resumes from a JSON checkpoint to the same totals', () => {
    const alg = rolloutPlannerAlgorithm(env, state, options)
    const whole = runAlgorithm(alg, { maxSteps: 20 })
    const half = runAlgorithm(alg, { maxSteps: 2 })
    const stored = viaJson(alg.serialise?.(half.state) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 20, start: { state: stored, step: 2, serialised: true } })
    expect(rest.state.totals).toEqual(whole.state.totals)
    expect(rest.state.batches).toBe(6)
  })
})
