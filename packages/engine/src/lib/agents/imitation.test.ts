import { describe, expect, it, vi } from 'vitest'
import { hazardGrid, reach } from '../../testing/rl-continuous.ts'
import { argmaxOf } from '../base/index.ts'
import { frozenLake, makeGridWorld } from '../environments/index.ts'
import { key } from '../random/index.ts'
import {
  evaluatePolicy,
  greedyPolicyOf,
  make,
  mdpFromEnv,
  oneHotObservation,
  runEpisode,
  timeLimit,
  trainAgent,
  valueIteration,
} from '../rl/index.ts'
import { recorder, tracePolicy } from '../trace/index.ts'
import {
  dagger,
  daggerAlgorithm,
  featureExpectations,
  gail,
  maxEntIrl,
  softValueIteration,
  stateVisitationFrequencies,
} from './imitation.ts'

// Learning runs take seconds alone; the timeout only guards against a heavily loaded machine.
vi.setConfig({ testTimeout: 60_000 })

type Vec = readonly number[]
const oneHot = (n: number, i: number): number[] => Array.from({ length: n }, (_, j) => (j === i ? 1 : 0))

describe('DAgger', () => {
  it('matches the expert on the dry FrozenLake', () => {
    const raw = frozenLake.create({ isSlippery: false })
    const expertPolicy = valueIteration(mdpFromEnv(raw, 0.9)).policy
    const env = make(frozenLake, { isSlippery: false }, { wrappers: [(e) => oneHotObservation(e as never) as never] })
    const expert = (o: Vec): number => expertPolicy[argmaxOf(o)] ?? 0
    const rec = recorder()
    const result = dagger(env as never, expert, {
      iterations: 4,
      tracer: rec,
      policy: tracePolicy({ metrics: ['agreement', 'beta'] }),
    })
    expect(result.iterations).toBe(4)
    expect(result.agreement).toBe(1)
    const learned = evaluatePolicy(env as never, (o: Vec) => result.policy(o, key(0)), { episodes: 1 })
    expect(learned.meanReturn).toBe(1)
    expect(learned.lengths[0]).toBe(6)
    const betas = rec
      .run()
      .events.filter((e) => e.type === 'metric' && (e as { name: string }).name === 'beta')
      .map((e) => (e as { value: number }).value)
    expect(betas.slice(0, 3)).toEqual([1, 0.5, 0.25])
  })

  it('imitates a continuous controller by regression, visiting the states the learner reaches', () => {
    const env = timeLimit(reach(), 10)
    const expert = (o: Vec): number[] => [Math.max(-1, Math.min(1, -4 * (o[0] ?? 0)))]
    const result = dagger(env, expert, { iterations: 4, epochs: 100, learningRate: 1e-2, hidden: [16] })
    const learned = evaluatePolicy(env, result.policy, { episodes: 20, seed: 1 }).meanReturn
    const reference = evaluatePolicy(env, expert, { episodes: 20, seed: 1 }).meanReturn
    expect(learned).toBeGreaterThan(reference - 0.2)
    expect(result.dataset.length).toBe(4 * 2 * 10)
    expect(daggerAlgorithm(env, expert).name).toBe('dagger')
  })
})

describe('GAIL', () => {
  // The hazard grid with an expert that walks around the hazards (down, right, right, right, up).
  const action: Readonly<Record<number, number>> = { 0: 2, 1: 2, 2: 2, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0 }
  const env = oneHotObservation(timeLimit(hazardGrid(), 20))
  const expert = (o: Vec): number => action[argmaxOf(o)] ?? 0
  const demonstrations = runEpisode(env, expert, key(0))
    .observations.slice(0, 5)
    .map((o) => ({
      observation: o,
      action: expert(o),
    }))
  const path = [0, 4, 5, 6, 7]
  const options = { hidden: [16], learningRate: 3e-3, nEpochs: 4, nSteps: 128, batchSize: 32 } as const

  it('improves imitation of the expert from the discriminator reward alone', () => {
    const agent = gail<number>(env, demonstrations, {
      ...options,
      rewardType: 'airl',
      discriminatorLearningRate: 1e-2,
    })
    const agreement = (s: ReturnType<typeof agent.init>): number =>
      path.filter((st) => agent.greedy(s, oneHot(8, st)) === action[st]).length / path.length
    const before = agreement(agent.init(key(0)))
    const result = trainAgent(env, agent, { maxSteps: 3000, seed: 0, checkpoints: false })
    expect(agreement(result.agent)).toBeGreaterThan(before + 0.5)
    const episode = runEpisode(env, greedyPolicyOf(agent, result.agent), key(1))
    expect(episode.return).toBeCloseTo(0.6, 12)
    for (const k of ['discriminatorLoss', 'expertAccuracy', 'policyAccuracy', 'imitationReward', 'policyLoss'])
      expect(Number.isFinite(result.agent.stats[k])).toBe(true)
  })

  it('trains with the -log(1 - D) reward and needs demonstrations', () => {
    const agent = gail<number>(env, demonstrations, { ...options, nSteps: 32, batchSize: 16 })
    const s = trainAgent(env, agent, { maxSteps: 64, checkpoints: false }).agent
    expect(s.stats['imitationReward']).toBeGreaterThan(0)
    expect(s.policy.updates).toBe(2)
    expect(() => gail(env, [])).toThrow(/demonstrations/)
  })
})

describe('maximum-entropy IRL', () => {
  const env = makeGridWorld({ map: ['S...', '....', '...G'], slip: 0, stepReward: -0.1 })
  const mdp = mdpFromEnv(env, 0.95)
  const expert = valueIteration(mdp).policy
  const trajectories = [0, 1, 2].map((i) => {
    const ep = runEpisode(env, (o: number) => expert[o] ?? 0, key(i))
    return ep.states.slice(0, ep.length)
  })

  it("recovers a reward whose optimal policy matches the expert's", () => {
    const result = maxEntIrl(mdp, trajectories, { learningRate: 0.5, maxIter: 300 })
    const learned = valueIteration({
      ...mdp,
      rewards: Float64Array.from(
        { length: mdp.nStates * mdp.nActions },
        (_, j) => result.reward[Math.floor(j / 4)] ?? 0,
      ),
    }).policy
    for (const s of new Set(trajectories.flat())) expect(learned[s]).toBe(expert[s])
    expect(result.gradientNorm).toBeLessThan(0.05)
    // The recovered reward makes the expert's path the most visited one under the soft-optimal policy.
    expect(result.svf[1] ?? 0).toBeGreaterThan(result.svf[4] ?? 0)
  })

  it('offers the finite-horizon visitation frequencies and the soft values of known rewards', () => {
    const finite = maxEntIrl(mdp, trajectories, { horizon: 5, maxIter: 20 })
    expect(finite.iterations).toBe(20)
    const { policy, V, Q } = softValueIteration(mdp, new Array<number>(12).fill(0))
    for (const row of policy) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(V[0]).toBeGreaterThan(Q[0]?.[0] ?? Infinity)
    const svf = stateVisitationFrequencies(mdp, policy, { horizon: 1 })
    expect(svf[0]).toBe(1)
    expect(featureExpectations([[0, 1]], [oneHot(2, 0), oneHot(2, 1)], { gamma: 0.5 })).toEqual([1, 0.5])
  })
})
