import { describe, expect, it } from 'vitest'
import { contextualLinearBandit, makeGridWorld } from '../../methods/rl/index.ts'
import { DYNA_MAZE, randomMdpEnv } from '../../testing/rl-tabular.ts'
import { lstsq } from '../numerics/linalg/index.ts'
import { foldIn, key, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { recorder, runAlgorithm, tracePolicy } from '../trace/index.ts'
import type { Transition } from './agent.ts'
import type { Env } from './env.ts'
import { mdpFromEnv, policyEvaluation } from './mdp.ts'
import {
  clippedIps,
  directMethod,
  doublyRobust,
  fittedQEvaluation,
  importanceWeights,
  ips,
  type LoggedBandit,
  snips,
  tabularFqe,
  tabularFqeAlgorithm,
} from './off-policy.ts'
import { boltzmannProbabilities } from './policies.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

describe('contextual-bandit estimators are unbiased for the true value (from the environment oracle)', () => {
  const hidden = [0.6, 0.0, -0.8]
  const env = contextualLinearBandit.create({ arms: 3, dimension: 3, noise: 0.5, theta: hidden })
  const oracle = env.oracle
  if (oracle === undefined) throw new Error('no oracle')
  const theta = [0.5, -0.4, 0.3]
  const target = (x: readonly (readonly number[])[]): number[] =>
    boltzmannProbabilities(
      x.map((f) => f.reduce((acc, v, i) => acc + v * (theta[i] ?? 0), 0)),
      0.2,
    )
  // Logged data from a uniformly random behaviour policy; the truth is the target's expected reward on these contexts.
  const dataset = (seed: number, n: number): { data: LoggedBandit<readonly (readonly number[])[]>; truth: number } => {
    let r = env.reset(key(seed))
    const contexts: (readonly (readonly number[])[])[] = []
    const actions: number[] = []
    const rewards: number[] = []
    let truth = 0
    for (let t = 0; t < n; t++) {
      const k = foldIn(key(seed), t)
      const a = Math.min(2, Math.floor((uniform(foldIn(k, 9), []).data[0] ?? 0) * 3))
      const step = env.step(r.state, a, k)
      const x = r.observation
      contexts.push(x)
      actions.push(a)
      rewards.push(step.reward)
      truth += target(x).reduce((acc, p, b) => acc + p * oracle.expectedReward(r.state, b), 0) / n
      r = { state: step.state, observation: step.observation, info: {} }
    }
    return { data: { contexts, actions, rewards, propensities: new Array<number>(n).fill(1 / 3) }, truth }
  }
  const wrongModel = (): number => 0.3
  const exact = (x: readonly (readonly number[])[], a: number): number =>
    (x[a] ?? []).reduce((acc, v, i) => acc + v * (hidden[i] ?? 0), 0)
  const runs = Array.from({ length: 150 }, (_, i) => {
    const { data, truth } = dataset(i, 40)
    return {
      truth,
      ips: ips(data, target).value,
      dr: doublyRobust(data, target, wrongModel).value,
      exactDr: doublyRobust(data, target, exact).value,
      // With rewards shifted by 5, IPS pays for the offset and the self-normalised estimate does not.
      shiftedIps: ips({ ...data, rewards: data.rewards.map((x) => x + 5) }, target).value - 5,
      shiftedSnips: snips({ ...data, rewards: data.rewards.map((x) => x + 5) }, target).value - 5,
      snips: snips(data, target).value,
      se: ips(data, target).standardError ?? 0,
    }
  })
  const errorOf = (name: 'ips' | 'dr' | 'snips'): { bias: number; se: number } => {
    const errors = runs.map((r) => r[name] - r.truth)
    const m = mean(errors)
    const sd = Math.sqrt(mean(errors.map((e) => (e - m) ** 2)))
    return { bias: m, se: sd / Math.sqrt(errors.length) }
  }

  it('IPS and doubly robust (with a wrong reward model) average to the truth', () => {
    for (const name of ['ips', 'dr'] as const) {
      const { bias, se } = errorOf(name)
      expect(Math.abs(bias)).toBeLessThan(3 * se + 1e-3)
    }
  })

  it('the reported standard errors match the spread of IPS across datasets', () => {
    const spread = Math.sqrt(mean(runs.map((r) => (r.ips - r.truth - errorOf('ips').bias) ** 2)))
    expect(mean(runs.map((r) => r.se)) / spread).toBeGreaterThan(0.75)
    expect(mean(runs.map((r) => r.se)) / spread).toBeLessThan(1.33)
  })

  it('SNIPS and doubly robust have lower variance than IPS; the exact model makes DM exact', () => {
    const spreadOf = (name: 'ips' | 'dr' | 'snips' | 'exactDr' | 'shiftedIps' | 'shiftedSnips'): number =>
      Math.sqrt(mean(runs.map((r) => (r[name] - r.truth) ** 2)))
    expect(spreadOf('shiftedSnips')).toBeLessThan(spreadOf('shiftedIps') / 2)
    const { data, truth } = dataset(999, 40)
    expect(directMethod(data, target, exact).value).toBeCloseTo(truth, 12)
    expect(spreadOf('exactDr')).toBeLessThan(spreadOf('ips'))
  })
})

describe('bandit estimators on small data', () => {
  const data: LoggedBandit<number> = {
    contexts: [0, 1, 0, 1],
    actions: [0, 1, 1, 0],
    rewards: [1, 0.5, 0, 2],
    propensities: [0.5, 0.25, 0.5, 0.75],
  }
  const target = (x: number): number[] => (x === 0 ? [1, 0] : [0.5, 0.5])
  it('weights, clipping and the agreement of DR with IPS under a zero model', () => {
    expect(importanceWeights(data, target)).toEqual([2, 2, 0, 2 / 3])
    expect(clippedIps(data, target, { maxWeight: 1 }).value).toBeCloseTo((1 + 0.5 + 0 + (2 / 3) * 2) / 4, 12)
    expect(clippedIps(data, target, { maxWeight: 100 }).value).toBeCloseTo(ips(data, target).value, 12)
    expect(doublyRobust(data, target, () => 0).value).toBeCloseTo(ips(data, target).value, 12)
    expect(directMethod(data, target, (x, a) => x + a).value).toBeCloseTo((0 + 1.5 + 0 + 1.5) / 4, 12)
    expect(ips(data, target).interval).toHaveLength(2)
  })

  it('rejects malformed logs', () => {
    expect(() => ips({ ...data, propensities: [0, 1, 1, 1] }, target)).toThrow(/propensities/)
    expect(() => ips({ ...data, rewards: [1] }, target)).toThrow(/length/)
    expect(() => ips({ contexts: [], actions: [], rewards: [], propensities: [] }, target)).toThrow(/no logged/)
    expect(() => snips(data, () => [0, 0])).toThrow(/probability 0/)
  })
})

/** Logged transitions of a uniformly random behaviour policy. */
function logged(env: Env<number, number, number>, steps: number, seed: number): Transition<number, number>[] {
  const nA = env.actionSpace.kind === 'discrete' ? env.actionSpace.n : 1
  const out: Transition<number, number>[] = []
  let r = env.reset(key(seed))
  for (let t = 0; t < steps; t++) {
    const k = foldIn(key(seed), t)
    const a = Math.min(nA - 1, Math.floor((uniform(foldIn(k, 7), []).data[0] ?? 0) * nA))
    const step = env.step(r.state, a, k)
    out.push({
      observation: r.observation,
      action: a,
      reward: step.reward,
      nextObservation: step.observation,
      terminated: step.terminated,
      truncated: false,
    })
    r = step.terminated ? env.reset(foldIn(k, 3)) : { state: step.state, observation: step.observation, info: {} }
  }
  return out
}

describe('fitted Q evaluation', () => {
  it('tabular FQE on a deterministic maze equals exact policy evaluation', () => {
    const maze = makeGridWorld({ map: DYNA_MAZE, slip: 0, stepReward: -0.1, goalReward: 1 })
    const data = logged(maze, 6000, 1)
    const policy = Array.from({ length: 54 }, () => [0.4, 0.3, 0.2, 0.1])
    const exact = policyEvaluation(mdpFromEnv(maze, 0.9), policy)
    const fqe = tabularFqe(data, policy, { nStates: 54, nActions: 4, gamma: 0.9, tol: 1e-12 })
    expect(fqe.converged).toBe(true)
    // Walls and the goal are never left, so only cells the data visits are compared.
    const visited = new Set(data.map((t) => t.observation))
    for (const s of visited) expect(Math.abs((fqe.V[s] ?? 0) - (exact[s] ?? 0))).toBeLessThan(1e-9)
  })

  it('tabular FQE on a stochastic MDP approaches policy evaluation with enough data', () => {
    const env = randomMdpEnv(5, 4, 2, 0.1)
    const policy = [0, 1, 1, 0]
    const exact = policyEvaluation(mdpFromEnv(env, 0.8), policy)
    const fqe = tabularFqe(logged(env, 20_000, 2), policy, {
      nStates: 4,
      nActions: 2,
      gamma: 0.8,
      initial: [0.25, 0.25, 0.25, 0.25],
    })
    for (const [s, v] of exact.entries()) expect(Math.abs((fqe.V[s] ?? 0) - v)).toBeLessThan(0.05)
    expect(fqe.value).toBeCloseTo(mean(fqe.V), 12)
  })

  it('generic FQE with least squares on one-hot features reproduces tabular FQE', () => {
    const env = randomMdpEnv(6, 3, 2, 0.2)
    const data = logged(env, 400, 3)
    const policy = [
      [0.2, 0.8],
      [0.5, 0.5],
      [1, 0],
    ]
    const tab = tabularFqe(data, policy, { nStates: 3, nActions: 2, gamma: 0.7, tol: 1e-13 })
    const fit = (inputs: readonly (readonly number[])[], targets: readonly number[]) => {
      const w = toFlat(
        lstsq(
          inputs.map((r) => [...r]),
          [...targets],
        ),
      )
      return (xs: readonly (readonly number[])[]): number[] =>
        xs.map((x) => x.reduce((acc, v, i) => acc + v * (w[i] ?? 0), 0))
    }
    const oneHot = (s: number, a: number): number[] => Array.from({ length: 6 }, (_, j) => (j === s * 2 + a ? 1 : 0))
    const rec = recorder()
    const result = fittedQEvaluation(data, (s: number) => policy[s] ?? [], {
      nActions: 2,
      features: oneHot,
      fit,
      gamma: 0.7,
      tol: 1e-12,
      initialObservations: [0, 1, 2],
      tracer: rec,
      policy: tracePolicy({ metrics: ['delta', 'startValue'] }),
    })
    expect(result.converged).toBe(true)
    for (let s = 0; s < 3; s++) for (let a = 0; a < 2; a++) expect(result.q(s, a)).toBeCloseTo(tab.Q[s]?.[a] ?? 0, 9)
    expect(result.value.value).toBeCloseTo(mean(tab.V), 9)
    expect(rec.run().events.some((e) => e.type === 'metric')).toBe(true)
  })

  it('runs as a traced algorithm and rejects pairs outside the table', () => {
    const t = { observation: 0, action: 0, reward: 1, nextObservation: 0, terminated: false, truncated: false }
    const result = runAlgorithm(tabularFqeAlgorithm([t], [0], { nStates: 1, nActions: 1, gamma: 0.5 }), {
      maxSteps: 100,
    })
    expect(result.state.Q[0]).toBeCloseTo(2, 9)
    expect(() => tabularFqe([{ ...t, observation: 3 }], [0], { nStates: 1, nActions: 1 })).toThrow(/outside/)
  })
})
