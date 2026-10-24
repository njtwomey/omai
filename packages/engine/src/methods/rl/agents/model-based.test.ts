import { describe, expect, it } from 'vitest'
import { argmaxOf } from '../../../lib/base/index.ts'
import { adam } from '../../../lib/optim/index.ts'
import { foldIn, key, normal } from '../../../lib/random/index.ts'
import { evaluatePolicy, greedyPolicyOf, make, oneHotObservation, trainAgent } from '../../../lib/rl/index.ts'
import { tensor, toFlat } from '../../../lib/tensor/index.ts'
import { doubleIntegrator, frozenLake, pointMass } from '../environments/index.ts'
import { gradientStep } from './_common.ts'
import { dreamerNets, dreamerParams, imagine, type SequenceBatch, worldPass } from './_dreamer.ts'
import { memberLoss, particleReturns, probabilisticNet, trainEnsemble } from './_ensemble.ts'
import { latentEnv, muZeroNets, muZeroParams, searchFrom, unrollLoss, unrollTargets } from './_muzero.ts'
import { dreamer, mbpo, muZero, pets } from './model-based.ts'
import { dqn } from './value-based.ts'

const LONG = 120_000

describe('probabilistic ensembles', () => {
  // A linear system: next = obs + 0.1 action, reward = -obs^2.
  const data = (() => {
    const z = toFlat(normal(key(7), [200, 2]))
    const inputs = Array.from({ length: 100 }, (_, i) => [z[2 * i] ?? 0, z[2 * i + 1] ?? 0])
    return { inputs, targets: inputs.map(([o = 0, a = 0]) => [0.1 * a, -o * o]) }
  })()

  it('fit Gaussian likelihoods on bootstrap resamples and sample trajectories in batches', () => {
    const net = probabilisticNet(1, 1, [16], key(0))
    const opt = adam({ learningRate: 1e-2 })
    const members = [net.init(key(1)), net.init(key(2))]
    const r = trainEnsemble(
      net,
      members,
      members.map((m) => opt.init(m.params)),
      data,
      {
        optimizer: opt,
        gradientSteps: 150,
        batchSize: 32,
      },
      key(3),
    )
    const [m0, m1] = r.members
    const [i0] = members
    if (m0 === undefined || m1 === undefined || i0 === undefined) throw new Error('missing members')
    // Training lowers the likelihood loss (the untrained weights scored with the fitted normaliser).
    expect(memberLoss(net, m0, data)).toBeLessThan(memberLoss(net, { ...i0, norm: m0.norm }, data))
    // Each member saw its own resample, so they disagree a little.
    const a = toFlat(net.model.predict(m0, tensor([[0.5]]), tensor([[1]])).mean)
    const b = toFlat(net.model.predict(m1, tensor([[0.5]]), tensor([[1]])).mean)
    expect(a).not.toEqual(b)
    expect(a[0]).toBeCloseTo(0.1, 1)
    const plans = [
      [[1], [1]],
      [[-1], [-1]],
      [[0], [0]],
    ]
    const returns = particleReturns({ model: net.model, members: r.members }, [0.5], plans, 3, key(4), {
      encode: (u) => [...u],
      reward: (o) => -((o[0] ?? 0) ** 2),
    })
    expect(returns).toHaveLength(3)
    // Pushing towards zero beats pushing away from it.
    expect(returns[1]).toBeGreaterThan(returns[0] ?? 0)
  })
})

describe('PETS', () => {
  it(
    'improves on random actions on the double integrator within two episodes',
    () => {
      const env = make(doubleIntegrator, { maxForce: 2 }, { maxEpisodeSteps: 100 })
      const agent = pets(env as never, {
        hidden: [16, 16],
        members: 3,
        population: 24,
        particles: 2,
        horizon: 8,
        iterations: 2,
        warmup: 100,
        trainEvery: 100,
        gradientSteps: 150,
        learningRate: 3e-3,
        batchSize: 32,
        // PETS plans with the task's known reward (the environment's, from the observation and the clipped action).
        reward: (o, a) => -((o[0] ?? 0) ** 2 + (o[1] ?? 0) ** 2 + Math.min(4, (a[0] ?? 0) ** 2)),
      })
      const r = trainAgent(env, agent, { episodes: 2, seed: 1 })
      // The first episode acts at random while collecting data; the second plans in the trained ensemble.
      expect(r.agent.trainings).toBe(2)
      expect(r.returns[1]).toBeGreaterThan((r.returns[0] ?? 0) / 10)
      expect(r.returns[1]).toBeGreaterThan(-200)
    },
    LONG,
  )
})

describe('MBPO', () => {
  const lake = () =>
    make(frozenLake, { isSlippery: false }, { wrappers: [(e) => oneHotObservation(e as never) as never] })
  const base = {
    hidden: [32],
    learningRate: 5e-3,
    gamma: 0.9,
    batchSize: 32,
    learningStarts: 64,
    targetUpdateInterval: 50,
    explorationSteps: 800,
    epsilonEnd: 0.05,
    bufferSize: 2000,
  } as const
  const holes = new Set([5, 7, 11, 12, 15])

  it(
    'with DQN learns the dry lake faster than plain DQN',
    () => {
      let model = 0
      let plain = 0
      for (const seed of [1, 2, 3]) {
        const env = lake()
        // The learner sees four transitions per real step (one real, three imagined): stretch its schedules to match.
        const learner = dqn(env as never, { ...base, explorationSteps: 800 * 4, targetUpdateInterval: 200 })
        const agent = mbpo(env as never, learner, {
          // One member keeps the test fast; the ensemble machinery is the same for any size.
          members: 1,
          hidden: [32],
          gradientSteps: 150,
          learningRate: 2e-2,
          warmup: 100,
          trainEvery: 100,
          rollouts: 3,
          deterministic: true,
          // MBPO uses known termination functions; one-hot predictions are projected to the nearest cell.
          terminated: (o) => holes.has(argmaxOf(o)),
          project: (o) => o.map((_, j) => (j === argmaxOf(o) ? 1 : 0)),
        })
        const rm = trainAgent(env as never, agent, { maxSteps: 500, seed })
        model += evaluatePolicy(env as never, greedyPolicyOf(agent, rm.agent), { episodes: 1 }).meanReturn
        expect(rm.agent.modelTransitions).toBeGreaterThan(1000)
        const reference = dqn(env as never, base)
        const rp = trainAgent(env as never, reference, { maxSteps: 500, seed })
        plain += evaluatePolicy(env as never, greedyPolicyOf(reference, rp.agent), { episodes: 1 }).meanReturn
      }
      expect(model).toBeGreaterThan(plain)
    },
    LONG,
  )
})

describe('Dreamer', () => {
  const nets = dreamerNets(
    2,
    { size: 1, discrete: false, low: [-1], high: [1] },
    { embed: 8, deter: 8, stoch: 3, hidden: 16 },
    key(0),
  )
  const z = (t: number, shape: number[]) => normal(foldIn(key(5), t), shape)
  // Sequences of a fixed linear system so the world model has something to learn.
  const batch: SequenceBatch = (() => {
    const B = 6
    const L = 5
    let obs = toFlat(z(0, [B, 2]))
    const observations = []
    const actions = []
    const rewards = []
    const continuations = []
    for (let t = 0; t < L; t++) {
      const a = toFlat(z(100 + t, [B])).map((v) => Math.tanh(v))
      observations.push(tensor(Array.from({ length: B }, (_, b) => [obs[2 * b] ?? 0, obs[2 * b + 1] ?? 0])))
      actions.push(tensor(a.map((v) => [v])))
      rewards.push(tensor(Array.from({ length: B }, (_, b) => -((obs[2 * b] ?? 0) ** 2))))
      continuations.push(tensor(new Array<number>(B).fill(1)))
      obs = obs.map((v, i) => (i % 2 === 0 ? v + 0.1 * (obs[i + 1] ?? 0) : v + 0.1 * (a[(i - 1) / 2] ?? 0)))
    }
    return { observations, actions, rewards, continuations }
  })()
  const kl = { klBalance: 0.8, freeNats: 0.1, klScale: 1 }

  it('fits sequences by the KL-balanced ELBO', () => {
    const opt = adam({ learningRate: 1e-2 })
    let world = dreamerParams(nets).world
    let state: unknown = opt.init(world)
    const first = toFlat(worldPass(nets, world, batch, key(1), kl).loss)[0] ?? 0
    for (let i = 0; i < 25; i++) {
      const s = gradientStep((p) => worldPass(nets, p as never, batch, key(1), kl).loss, world, opt, state)
      world = s.params as never
      state = s.optState
    }
    const pass = worldPass(nets, world, batch, key(1), kl)
    expect(toFlat(pass.loss)[0]).toBeLessThan(first)
    expect(pass.h).toHaveLength(5)
    expect(pass.z[0]?.shape).toEqual([6, 3])
    expect(toFlat(pass.kl)[0]).toBeGreaterThanOrEqual(0.1)
  })

  it('imagines rollouts with lambda-returns of the predicted rewards and discounts', () => {
    const params = dreamerParams(nets)
    const im = imagine(nets, params, z(1, [4, 8]), z(2, [4, 3]), 3, key(2), { gamma: 0.9, lambda: 0.5 })
    expect(im.features).toHaveLength(4)
    expect(im.returns[0]?.shape).toEqual([4])
    // R_k = r_k + d_k ((1 - lambda) v_{k+1} + lambda R_{k+1}), R_H = v_H.
    const r = im.rewards.map(toFlat)
    const d = im.discounts.map(toFlat)
    const v = im.values.map(toFlat)
    let next = v[3] ?? []
    for (let k = 2; k >= 0; k--) {
      next = next.map((x, i) => (r[k]?.[i] ?? 0) + (d[k]?.[i] ?? 0) * (0.5 * (v[k + 1]?.[i] ?? 0) + 0.5 * x))
      for (const [i, x] of next.entries()) expect(toFlat(im.returns[k] ?? tensor([]))[i]).toBeCloseTo(x, 12)
    }
    expect(toFlat(im.actions[0] ?? tensor([])).every((a) => Math.abs(a) <= 1)).toBe(true)
  })

  it(
    'trains its world model, actor and critic as an agent on continuous and discrete tasks',
    () => {
      const env = make(pointMass, { margin: 0.3 }, { maxEpisodeSteps: 30 })
      const agent = dreamer(env as never, {
        hidden: 16,
        warmup: 40,
        trainEvery: 10,
        batchSize: 4,
        sequenceLength: 6,
        horizon: 5,
      })
      const r = trainAgent(env, agent, { episodes: 3, seed: 0 })
      expect(r.agent.updates).toBeGreaterThan(3)
      expect(Number.isFinite(r.agent.worldLoss) && Number.isFinite(r.agent.criticLoss)).toBe(true)
      const a = agent.greedy(r.agent, [0.1, 0.1, 0, 0, 0, 0]) as readonly number[]
      expect(a.every((x) => Math.abs(x) <= 1)).toBe(true)
      const lake = make(
        frozenLake,
        { isSlippery: false },
        { wrappers: [(e) => oneHotObservation(e as never) as never] },
      )
      const discrete = dreamer<number>(lake as never, {
        hidden: 16,
        warmup: 20,
        trainEvery: 10,
        batchSize: 4,
        sequenceLength: 4,
        horizon: 4,
      })
      const rd = trainAgent(lake as never, discrete, { maxSteps: 60, seed: 0 })
      expect(rd.agent.updates).toBeGreaterThan(0)
      expect([0, 1, 2, 3]).toContain(
        discrete.greedy(
          rd.agent,
          new Array<number>(16).fill(0).map((_, i) => (i === 0 ? 1 : 0)),
        ),
      )
    },
    LONG,
  )
})

describe('MuZero', () => {
  const nets = muZeroNets(3, 2, 6, 16, key(0))

  it('builds n-step value, reward and policy targets with masks past the end', () => {
    const steps = [0, 1, 2].map((i) => ({
      observation: [i, 0, 0],
      action: i % 2,
      reward: i + 1,
      policy: [0.25, 0.75],
      rootValue: 10 * i,
    }))
    const t = unrollTargets({ steps, final: 0, terminated: true }, 1, 3, 2, 0.5, 2)
    // From position 1: 2 + 0.5 * 3 + 0.25 * 0 (the episode ends after two steps).
    expect(t.values).toEqual([2 + 0.5 * 3, 3, 0, 0])
    expect(t.valueMask).toEqual([1, 1, 1, 1])
    expect(t.rewards).toEqual([2, 3, 0])
    expect(t.policyMask).toEqual([1, 1, 0, 0])
    expect(t.actions).toEqual([1, 0, 0])
    const bootstrap = unrollTargets({ steps, final: 0, terminated: true }, 0, 0, 1, 0.5, 2)
    expect(bootstrap.values).toEqual([1 + 0.5 * 10])
    const truncated = unrollTargets({ steps, final: 4, terminated: false }, 2, 1, 3, 0.5, 2)
    expect(truncated.values).toEqual([3 + 0.5 * 4, 0])
    expect(truncated.valueMask).toEqual([1, 0])
  })

  it('searches its learned model with lib/rl mcts and lowers the unrolled loss', () => {
    const p = muZeroParams(nets)
    const env = latentEnv(nets, p)
    expect(env.step([0, 0, 0, 0, 0, 0], 1, key(0)).observation).toHaveLength(6)
    const r = searchFrom(nets, p, [1, 0, 0], { simulations: 10, c: 1.25, gamma: 0.9, depth: 3, key: key(1) })
    expect(r.visits.reduce((a, b) => a + b, 0)).toBe(10)
    const steps = [0, 1, 2, 3].map((i) => ({
      observation: [i / 3, 1 - i / 3, 0],
      action: i % 2,
      reward: i === 3 ? 1 : 0,
      policy: [i % 2, 1 - (i % 2)],
      rootValue: 0,
    }))
    const ep = { steps, final: 0, terminated: true }
    const obs = steps.map((s) => s.observation)
    const targets = steps.map((_, t) => unrollTargets(ep, t, 2, 3, 0.9, 2))
    const opt = adam({ learningRate: 1e-2 })
    let params = p as unknown
    let state: unknown = opt.init(params)
    const first = toFlat(unrollLoss(nets, p, obs, targets, 2))[0] ?? 0
    for (let i = 0; i < 30; i++) {
      const s = gradientStep((q) => unrollLoss(nets, q as never, obs, targets, 2), params, opt, state)
      params = s.params
      state = s.optState
    }
    expect(toFlat(unrollLoss(nets, params as never, obs, targets, 2))[0]).toBeLessThan(first / 2)
  })

  it(
    'plays, stores searched episodes and trains as an agent',
    () => {
      const lake = make(
        frozenLake,
        { isSlippery: false },
        { wrappers: [(e) => oneHotObservation(e as never) as never] },
      )
      const agent = muZero(lake as never, { hidden: 16, simulations: 8, warmup: 16, trainEvery: 2, batchSize: 8 })
      const r = trainAgent(lake as never, agent, { maxSteps: 120, seed: 0 })
      expect(r.agent.updates).toBeGreaterThan(10)
      expect(r.agent.episodes.length).toBeGreaterThan(0)
      const policy = r.agent.episodes[0]?.steps[0]?.policy ?? []
      expect(policy.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      expect(Number.isFinite(r.agent.loss)).toBe(true)
    },
    LONG,
  )
})
