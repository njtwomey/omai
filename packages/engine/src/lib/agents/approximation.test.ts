import { describe, expect, it } from 'vitest'
import type { Json } from '../base/index.ts'
import {
  BAIRD_BEHAVIOUR,
  BAIRD_FEATURES,
  BAIRD_GAMMA,
  BAIRD_INITIAL_WEIGHTS,
  BAIRD_TARGET,
  makeBairdCounterexample,
  makeRandomWalk,
  mountainCar,
  randomWalkValues,
} from '../environments/index.ts'
import { dgesv } from '../numerics/linalg/index.ts'
import { adam, sgd } from '../optim/index.ts'
import { foldIn, key } from '../random/index.ts'
import { lambdaReturns, trainAgent } from '../rl/index.ts'
import { stateFromJson, stateToJson } from '../trace/index.ts'
import {
  actionTileFeatures,
  emphaticTd,
  featureValue,
  gradientTd,
  type LinearTdState,
  matrixFeatures,
  oneHotFeatures,
  semiGradientSarsa,
  semiGradientTd,
  tileCoder,
  tileFeatures,
} from './approximation.ts'
import { tdLambda } from './tabular.ts'

const rms = (a: readonly number[], b: readonly number[]): number =>
  Math.sqrt(a.reduce((s, x, i) => s + (x - (b[i] ?? 0)) ** 2, 0) / a.length)

describe('tile coding', () => {
  it('activates one tile per tiling, overlapping for near points and disjoint for far ones', () => {
    const coder = tileCoder({ low: [0, 0], high: [1, 1], tilings: 8, tiles: 10 })
    expect(coder.size).toBe(8 * 11 * 11)
    const a = coder.indices([0.5, 0.5])
    expect(a).toHaveLength(8)
    expect(new Set(a).size).toBe(8)
    const near = coder.indices([0.52, 0.5]).filter((i) => a.includes(i)).length
    const far = coder.indices([0.9, 0.1]).filter((i) => a.includes(i)).length
    expect(near).toBeGreaterThanOrEqual(6)
    expect(far).toBe(0)
    expect(coder.indices([0.5, 0.5])).toEqual(a)
  })

  it('displaces the tilings asymmetrically, symmetrically or not at all', () => {
    const base = { low: [0, 0], high: [1, 1], tilings: 4, tiles: 4 }
    const code = (offsets: 'asymmetric' | 'symmetric' | 'none'): number[] =>
      tileCoder({ ...base, offsets }).indices([0.4, 0.4])
    const cells = 25
    const decode = (i: number): [number, number] => [Math.floor((i % cells) / 5), (i % cells) % 5]
    expect(
      code('none')
        .map(decode)
        .every(([x, y]) => x === 1 && y === 1),
    ).toBe(true)
    expect(
      code('symmetric')
        .map(decode)
        .every(([x, y]) => x === y),
    ).toBe(true)
    expect(
      code('asymmetric')
        .map(decode)
        .some(([x, y]) => x !== y),
    ).toBe(true)
  })

  it('hashes into a fixed memory and clips points outside the bounds', () => {
    const hashed = tileCoder({ low: [0, 0, 0], high: [1, 1, 1], tilings: 8, tiles: 6, memory: 64 })
    expect(hashed.size).toBe(64)
    for (const x of [
      [0.1, 0.2, 0.3],
      [0.9, 0.8, 0.7],
      [2, -3, 0.5],
    ]) {
      for (const i of hashed.indices(x)) expect(i >= 0 && i < 64).toBe(true)
    }
    expect(hashed.indices([2, 2, 2])).toEqual(hashed.indices([1, 1, 1]))
    expect(hashed.indices([0.1, 0.2, 0.3])).toEqual(hashed.indices([0.1, 0.2, 0.3]))
    expect(() => tileCoder({ low: [0], high: [0] })).toThrow()
  })

  it('shifts action features into the block of the action', () => {
    const coder = tileCoder({ low: [0], high: [1], tilings: 2, tiles: 2 })
    const f = actionTileFeatures(coder, 3)
    expect(f.size).toBe(3 * coder.size)
    expect(f.features([0.3], 2).indices).toEqual(coder.indices([0.3]).map((i) => 2 * coder.size + i))
  })
})

// A fixed behaviour trajectory of the 19-state walk, for comparing learners step by step.
function walkTransitions(
  n: number,
  steps: number,
): readonly { o: number; a: number; r: number; o2: number; end: boolean }[] {
  const env = makeRandomWalk({ n, leftReward: -1 })
  let r = env.reset(key(0))
  const out: { o: number; a: number; r: number; o2: number; end: boolean }[] = []
  for (let t = 0; t < steps; t++) {
    const a = t % 3 === 0 ? 0 : 1
    const s = env.step(r.state, a, foldIn(key(1), t))
    out.push({ o: r.observation, a, r: s.reward, o2: s.observation, end: s.terminated })
    r = s.terminated ? env.reset(foldIn(key(2), t)) : { state: s.state, observation: s.observation, info: {} }
  }
  return out
}

describe('semi-gradient TD(lambda)', () => {
  it('equals the tabular TD(lambda) agent with one-hot features, for every trace kind', () => {
    const n = 19
    const env = makeRandomWalk({ n, leftReward: -1 })
    const steps = walkTransitions(n, 400)
    for (const trace of ['accumulating', 'replacing', 'dutch'] as const) {
      const tabular = tdLambda(env, {
        alpha: 0.1,
        gamma: 0.95,
        lambda: 0.8,
        trace: trace === 'replacing' ? 'replacing' : trace,
      })
      const linear = semiGradientTd(env, { features: oneHotFeatures(n), alpha: 0.1, gamma: 0.95, lambda: 0.8, trace })
      let a = tabular.init(key(0))
      let b = linear.init(key(0))
      for (const t of steps) {
        const tr = {
          observation: t.o,
          action: t.a,
          reward: t.r,
          nextObservation: t.o2,
          terminated: t.end,
          truncated: false,
        }
        a = tabular.learn(a, tr, key(0))
        b = linear.learn(b, tr, key(0))
      }
      for (let s = 0; s < n; s++)
        expect(b.w[s] ?? Number.NaN, `${trace} state ${s}`).toBeCloseTo(a.V[s] ?? Number.NaN, 12)
    }
  })

  it('has offline updates that equal the forward view of the lambda returns', () => {
    const n = 7
    const env = makeRandomWalk({ n, leftReward: -1 })
    const w0 = Array.from({ length: n }, (_, i) => 0.3 * Math.sin(i + 1))
    const gamma = 0.9
    const lambda = 0.6
    const alpha = 0.05
    // One episode, as transitions, ending in termination.
    const states = [3, 4, 3, 2, 3, 4, 5, 4, 5]
    const rewards = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    const agent = semiGradientTd(env, {
      features: oneHotFeatures(n),
      alpha,
      gamma,
      lambda,
      offline: true,
      initialWeights: w0,
    })
    let s = agent.init(key(0))
    for (let t = 0; t < states.length; t++) {
      const end = t === states.length - 1
      s = agent.learn(
        s,
        {
          observation: states[t] ?? 0,
          action: 1,
          reward: rewards[t] ?? 0,
          nextObservation: end ? (states[t] ?? 0) : (states[t + 1] ?? 0),
          terminated: end,
          truncated: false,
        },
        key(0),
      )
      if (!end) expect(Array.from(s.w)).toEqual(w0) // nothing moves until the episode ends
    }
    const v = states.map((x) => w0[x] ?? 0)
    const next = states.map((_, t) => (t === states.length - 1 ? 0 : (w0[states[t + 1] ?? 0] ?? 0)))
    const dones = states.map((_, t) => (t === states.length - 1 ? 1 : 0))
    const g = lambdaReturns(rewards, next, dones, gamma, lambda)
    const expected = w0.slice()
    for (const [t, x] of states.entries()) expected[x] = (expected[x] ?? 0) + alpha * ((g[t] ?? 0) - (v[t] ?? 0))
    for (let i = 0; i < n; i++) expect(s.w[i] ?? Number.NaN).toBeCloseTo(expected[i] ?? Number.NaN, 12)
  })

  it('learns the true values of the 19-state walk with tile coding', () => {
    const n = 19
    const truth = randomWalkValues({ n, leftReward: -1 })
    const env = makeRandomWalk({ n, leftReward: -1 })
    // One tile per state: the tabular case through the tile coder.
    const exact = tileFeatures<number>(tileCoder({ low: [0], high: [n], tilings: 1, tiles: n, offsets: 'none' }))
    const run = (
      features: typeof exact,
      lambda: number,
      trace: 'replacing' | 'accumulating' | 'dutch',
      alpha: number,
    ) => {
      const agent = semiGradientTd(env, { features, alpha, gamma: 1, lambda, trace })
      const r = trainAgent(env, agent, { episodes: 2000, checkpoints: false, seed: 4 })
      return Array.from({ length: n }, (_, s) => featureValue(r.agent.w, features.features(s)))
    }
    expect(rms(run(exact, 0.8, 'replacing', 0.02), truth)).toBeLessThan(0.07)
    expect(rms(run(exact, 0.8, 'dutch', 0.02), truth)).toBeLessThan(0.07)
    // Coarse overlapping tiles generalise: five tiles of about four states, four tilings; error is bounded by the blur.
    const coarse = tileFeatures<number>(tileCoder({ low: [0], high: [n], tilings: 4, tiles: 5 }))
    const blurred = run(coarse, 0.5, 'accumulating', 0.02)
    expect(rms(blurred, truth)).toBeLessThan(0.2)
    expect(rms(blurred, truth)).toBeLessThan(rms(new Array(n).fill(0), truth) / 3)
  })

  it('takes an optimiser as the update rule: sgd reproduces the plain step, adam learns', () => {
    const n = 19
    const env = makeRandomWalk({ n, leftReward: -1 })
    const steps = walkTransitions(n, 300)
    const make = (extra: object) =>
      semiGradientTd(env, { features: oneHotFeatures(n), alpha: 0.05, gamma: 1, lambda: 0.7, ...extra })
    const plain = make({})
    const viaSgd = make({ optimiser: sgd({ learningRate: 0.05 }) })
    let a = plain.init(key(0))
    let b = viaSgd.init(key(0))
    for (const t of steps) {
      const tr = {
        observation: t.o,
        action: t.a,
        reward: t.r,
        nextObservation: t.o2,
        terminated: t.end,
        truncated: false,
      }
      a = plain.learn(a, tr, key(0))
      b = viaSgd.learn(b, tr, key(0))
    }
    for (let s = 0; s < n; s++) expect(b.w[s] ?? Number.NaN).toBeCloseTo(a.w[s] ?? Number.NaN, 10)
    const truth = randomWalkValues({ n, leftReward: -1 })
    const r = trainAgent(env, make({ optimiser: adam({ learningRate: 0.02 }) }), {
      episodes: 400,
      checkpoints: false,
      seed: 1,
    })
    expect(rms(Array.from(r.agent.w), truth)).toBeLessThan(rms(new Array(n).fill(0), truth) / 2)
    expect(() =>
      semiGradientTd(env, { features: oneHotFeatures(n), trace: 'dutch', optimiser: sgd({ learningRate: 1 }) }),
    ).toThrow()
  })

  it('has state that survives a round trip through JSON', () => {
    const n = 5
    const env = makeRandomWalk({ n })
    const agent = semiGradientTd(env, {
      features: oneHotFeatures(n),
      alpha: 0.1,
      gamma: 1,
      lambda: 0.5,
      optimiser: sgd({ learningRate: 0.1 }),
    })
    const r = trainAgent(env, agent, { episodes: 5, checkpoints: false, seed: 0 })
    const json: Json = stateToJson(r.agent)
    const back = stateFromJson(json) as LinearTdState
    expect(Array.from(back.w)).toEqual(Array.from(r.agent.w))
    expect(back.steps).toBe(r.agent.steps)
  })
})

describe('off-policy learning with linear features', () => {
  const env = makeBairdCounterexample()
  const base = {
    features: matrixFeatures(BAIRD_FEATURES),
    behaviour: () => BAIRD_BEHAVIOUR,
    target: () => BAIRD_TARGET,
    gamma: BAIRD_GAMMA,
    initialWeights: BAIRD_INITIAL_WEIGHTS,
  }
  const values = (w: Float64Array): number[] => BAIRD_FEATURES.map((f) => f.reduce((a, x, j) => a + x * (w[j] ?? 0), 0))

  it("makes semi-gradient TD diverge on Baird's counterexample while gradient-TD stays bounded", () => {
    const run = (agent: ReturnType<typeof semiGradientTd>, steps: number): LinearTdState =>
      trainAgent(env, agent, { maxSteps: steps, checkpoints: false, seed: 1 }).agent
    const norms = [1000, 2500, 5000].map(
      (steps) => run(semiGradientTd(env, { ...base, alpha: 0.01 }), steps).weightNorm,
    )
    expect(norms[0]).toBeGreaterThan(100)
    expect(norms[1]).toBeGreaterThan(norms[0] as number)
    expect(norms[2]).toBeGreaterThan(1e4 * (norms[0] as number))
    for (const variant of ['tdc', 'gtd2'] as const) {
      const agent = gradientTd(env, { ...base, alpha: 0.005, beta: 0.05, variant })
      const s = trainAgent(env, agent, { maxSteps: 5000, checkpoints: false, seed: 1 }).agent
      const v = values(s.w)
      expect(s.weightNorm, variant).toBeLessThan(15)
      expect(Math.max(...v.map(Math.abs)), variant).toBeLessThan(3)
      // The expected TD error of the solid action is small: the values have settled onto the Bellman fixed point manifold.
      for (let i = 0; i < 6; i++) expect(Math.abs(BAIRD_GAMMA * (v[6] ?? 0) - (v[i] ?? 0)), variant).toBeLessThan(0.05)
    }
  })

  it('converges in expectation under emphatic weighting where uniform weighting diverges', () => {
    // The expected sweep w += alpha sum_s m(s) delta(s) x(s) with the emphasis m = (I - gamma P^T)^-1 d (Sutton and
    // Barto, 11.8) against the same sweep with every state weighted equally (the semi-gradient DP of Figure 11.2).
    const g = BAIRD_GAMMA
    const sweep = (emphasis: readonly number[], alpha: number, steps: number): number => {
      const w = Float64Array.from(BAIRD_INITIAL_WEIGHTS)
      for (let k = 0; k < steps; k++) {
        const v = values(w)
        const next = w.slice()
        for (let s = 0; s < 7; s++) {
          const delta = g * (v[6] ?? 0) - (v[s] ?? 0)
          for (let j = 0; j < 8; j++)
            next[j] = (next[j] ?? 0) + alpha * (emphasis[s] ?? 0) * delta * (BAIRD_FEATURES[s]?.[j] ?? 0)
        }
        w.set(next)
      }
      return Math.max(...values(w).map(Math.abs))
    }
    const mu = new Array<number>(7).fill(1 / 7)
    // Under the target policy every state moves to state 6, so P[s][6] = 1; m solves (I - gamma P^T) m = mu.
    const a = new Float64Array(49)
    for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) a[i * 7 + j] = (i === j ? 1 : 0) - (i === 6 ? g : 0)
    const emphasis = Array.from(dgesv(a, Float64Array.from(mu), 7))
    expect(emphasis[6]).toBeCloseTo((1 / 7 + (g * 6) / 7) / (1 - g), 8)
    expect(sweep(mu, 0.5, 2000)).toBeGreaterThan(1e3)
    expect(sweep(emphasis, 0.002, 5000)).toBeLessThan(1e-3)
  })

  it('has emphatic TD whose sampled expected update matches the emphatic weighting on the random walk', () => {
    const n = 5
    const walk = makeRandomWalk({ n })
    const phi = [
      [1, 0, 0.5],
      [0.5, 1, 0],
      [0, 1, 1],
      [1, 0.2, 1],
      [0.3, 0.3, 1],
    ]
    const w0 = [0.2, -0.1, 0.4]
    const behaviour = [0.5, 0.5]
    const target = [0.3, 0.7]
    const P = (p: readonly number[]): number[][] => {
      const out = Array.from({ length: n }, () => new Array<number>(n).fill(0))
      for (let s = 0; s < n; s++) {
        for (const [a, d] of [
          [0, -1],
          [1, 1],
        ] as const) {
          const row = out[s] as number[]
          if (s + d >= 0 && s + d < n) row[s + d] = (row[s + d] ?? 0) + (p[a] ?? 0)
        }
      }
      return out
    }
    const solveT = (p: number[][], d: readonly number[]): number[] => {
      const a = new Float64Array(n * n)
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) a[i * n + j] = (i === j ? 1 : 0) - (p[j]?.[i] ?? 0)
      return Array.from(dgesv(a, Float64Array.from(d), n))
    }
    const Pb = P(behaviour)
    const Pt = P(target)
    const visits = solveT(Pb, [0, 0, 1, 0, 0])
    const m = solveT(Pt, visits)
    const v = phi.map((f) => f.reduce((a, x, j) => a + x * (w0[j] ?? 0), 0))
    const expected = [0, 0, 0]
    for (let s = 0; s < n; s++) {
      let delta = (target[1] ?? 0) * (s === n - 1 ? 1 : 0) - (v[s] ?? 0)
      for (let t = 0; t < n; t++) delta += (Pt[s]?.[t] ?? 0) * (v[t] ?? 0)
      for (let j = 0; j < 3; j++) expected[j] = (expected[j] ?? 0) + (m[s] ?? 0) * delta * (phi[s]?.[j] ?? 0)
    }
    const alpha = 1e-9
    const episodes = 6000
    const agent = emphaticTd(walk, {
      features: matrixFeatures(phi),
      behaviour: () => behaviour,
      target: () => target,
      gamma: 1,
      alpha,
      initialWeights: w0,
    })
    const sampled = Array.from(trainAgent(walk, agent, { episodes, checkpoints: false, seed: 3 }).agent.w).map(
      (x, j) => (x - (w0[j] ?? 0)) / (alpha * episodes),
    )
    const error = Math.hypot(...sampled.map((x, j) => x - (expected[j] ?? 0)))
    expect(error / Math.hypot(...expected)).toBeLessThan(0.1)
  })

  it('has gradient-TD reach the TD fixed point on-policy, the true values of the walk', () => {
    const n = 5
    const walk = makeRandomWalk({ n })
    const truth = randomWalkValues({ n })
    for (const variant of ['tdc', 'gtd2'] as const) {
      const agent = gradientTd(walk, { features: oneHotFeatures(n), alpha: 0.05, beta: 0.1, gamma: 1, variant })
      const r = trainAgent(walk, agent, { episodes: 3000, checkpoints: false, seed: 2 })
      expect(rms(Array.from(r.agent.w), truth), variant).toBeLessThan(0.05)
    }
  })
})

describe('semi-gradient Sarsa(lambda)', () => {
  it('learns Mountain Car with tile coding and shortens its episodes', () => {
    const env = mountainCar.create()
    const coder = tileCoder({ low: [-1.2, -0.07], high: [0.6, 0.07], tilings: 8, tiles: 8 })
    const agent = semiGradientSarsa(env, {
      features: actionTileFeatures(coder, 3),
      alpha: 0.5 / 8,
      gamma: 1,
      lambda: 0.9,
      trace: 'replacing',
      epsilon: 0,
    })
    const r = trainAgent(env, agent, { episodes: 120, checkpoints: false, seed: 5 })
    const early = r.lengths.slice(0, 5).reduce((a, b) => a + b, 0) / 5
    const late = r.lengths.slice(-20).reduce((a, b) => a + b, 0) / 20
    expect(early).toBeGreaterThan(190)
    expect(late).toBeLessThan(0.7 * early)
    expect(late).toBeLessThan(160)
  })

  it('updates a step once its successor action is known and ends episodes cleanly', () => {
    const env = makeRandomWalk({ n: 3 })
    const features = { size: 6, features: (o: number, a: number) => ({ indices: [o * 2 + a] }) }
    const agent = semiGradientSarsa(env, { features, alpha: 0.5, gamma: 1, lambda: 0 })
    let s = agent.init(key(0))
    s = agent.learn(
      s,
      { observation: 1, action: 1, reward: 0, nextObservation: 2, terminated: false, truncated: false },
      key(0),
    )
    expect(Array.from(s.w)).toEqual([0, 0, 0, 0, 0, 0]) // nothing yet: the successor action is unknown
    s = agent.learn(
      s,
      { observation: 2, action: 1, reward: 1, nextObservation: 2, terminated: true, truncated: false },
      key(0),
    )
    // The terminal step moves q(2,1) by 0.5 * (1 - 0); the first one then bootstraps from the updated value: it was
    // processed before with q(2,1) = 0, so its target is 0 and nothing moved.
    expect(Array.from(s.w)).toEqual([0, 0, 0, 0, 0, 0.5])
    expect(s.episodes).toBe(1)
    expect(Array.from(s.z).every((x) => x === 0)).toBe(true)
  })
})
