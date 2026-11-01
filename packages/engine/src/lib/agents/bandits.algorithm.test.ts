import { describe, expect, it } from 'vitest'
import { bernoulliBandit, gaussianBandit } from '../environments/index.ts'
import { foldIn, key } from '../random/index.ts'
import { cumulativeRegret } from '../rl/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { banditAlgorithm, banditContract, banditStrategies, epsilonGreedyBandit, runBandit, ucb1 } from './bandits.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

describe('banditAlgorithm', () => {
  const env = bernoulliBandit.create({ probs: [0.2, 0.5, 0.8] })

  it('pulls once per step and runBandit gives what the old hand-written loop gave', () => {
    const agent = ucb1(env)
    const root = key(11)
    // The loop runBandit was before it became this algorithm.
    let g = agent.init(foldIn(root, 0))
    let r = env.reset(foldIn(root, 1))
    let state = r.state
    let observation = r.observation
    const states: (typeof state)[] = []
    const actions: number[] = []
    const rewards: number[] = []
    for (let t = 0; t < 60; t++) {
      const k = foldIn(foldIn(root, 2), t)
      const action = agent.act(g, observation, foldIn(k, 0))
      const step = env.step(state, action, foldIn(k, 1))
      states.push(state)
      actions.push(action)
      rewards.push(step.reward)
      g = agent.learn(
        g,
        {
          observation,
          action,
          reward: step.reward,
          nextObservation: step.observation,
          terminated: step.terminated,
          truncated: step.truncated,
        },
        foldIn(k, 2),
      )
      if (step.terminated || step.truncated) {
        r = env.reset(foldIn(k, 3))
        state = r.state
        observation = r.observation
      } else {
        state = step.state
        observation = step.observation
      }
    }
    const want = cumulativeRegret(env.oracle as NonNullable<typeof env.oracle>, states, actions)
    const got = runBandit(env, agent, { pulls: 60, key: root })
    expect(got.actions).toEqual(actions)
    expect(got.rewards).toEqual(rewards)
    expect(got.regret).toEqual(want)
  })

  it('regret never decreases, and the logs have one value per pull', () => {
    const rec = recorder()
    const run = runAlgorithm(banditAlgorithm(env, ucb1(env), { pulls: 40 }), {
      maxSteps: 100,
      seed: 3,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    expect(run.status).toBe('completed')
    expect(run.steps).toBe(40)
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    const regret = metric('bandit/regret')
    expect(regret).toHaveLength(40)
    expect(regret.every((x, i) => i === 0 || x >= (regret[i - 1] ?? 0) - 1e-12)).toBe(true)
    expect(metric('bandit/arm').every((a) => a >= 0 && a < 3)).toBe(true)
    const counts = rec.events.filter((e) => e.type === 'tensor' && e.name === 'bandit/counts')
    expect(counts).toHaveLength(40)
    expect(Object.keys(banditContract)).toHaveLength(5)
  })

  it('resumes from a JSON checkpoint to the same end', () => {
    const alg = banditAlgorithm(env, epsilonGreedyBandit(env), { pulls: 50, key: key(2) })
    const whole = runAlgorithm(alg, { maxSteps: 100, key: key(2) })
    const half = runAlgorithm(alg, { maxSteps: 20, key: key(2) })
    const stored = viaJson(alg.serialise?.(half.state) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 100, key: key(2), start: { state: stored, step: 20, serialised: true } })
    expect(rest.state.actions).toEqual(whole.state.actions)
    expect(rest.state.rewards).toEqual(whole.state.rewards)
    for (const [i, x] of whole.state.regret.entries()) expect(rest.state.regret[i]).toBeCloseTo(x, 12)
  })

  it('without a key derives its own from the run, deterministically', () => {
    const run = (seed: number): readonly number[] =>
      runAlgorithm(banditAlgorithm(env, ucb1(env), { pulls: 30 }), { maxSteps: 30, seed }).state.actions
    expect(run(5)).toEqual(run(5))
  })

  it('works on a Gaussian bandit and refuses an environment without an oracle', () => {
    const g = gaussianBandit.create({ means: [0, 1], sigma: 1 })
    expect(runBandit(g, ucb1(g), { pulls: 20, key: key(1) }).regret).toHaveLength(20)
    const noOracle = { ...env, oracle: undefined } as unknown as typeof env
    expect(() => banditAlgorithm(noOracle, ucb1(env))).toThrow(/no oracle/)
  })

  it('the strategy table builds each agent with its knobs and ranks UCB above uniform play at the end', () => {
    expect(banditStrategies.names).toEqual([
      'epsilon-greedy',
      'ucb1',
      'kl-ucb',
      'thompson-bernoulli',
      'exp3',
      'gradient',
    ])
    for (const name of banditStrategies.names) {
      const agent = banditStrategies.make({ kind: name })(env)
      const run = runBandit(env, agent, { pulls: 30, key: key(8) })
      expect(run.actions).toHaveLength(30)
    }
    const ucb = banditStrategies.make({ kind: 'ucb1', c: 2 })(env)
    expect(runBandit(env, ucb, { pulls: 800, key: key(4) }).regret.at(-1)).toBeLessThan(0.3 * 800 * 0.5)
    expect(banditStrategies.where((p) => p.principle === 'optimism')).toEqual(['ucb1', 'kl-ucb'])
  }, 60_000)
})
