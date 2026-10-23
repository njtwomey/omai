import { describe, expect, it } from 'vitest'
import { key } from '../../lib/random/index.ts'
import { evaluatePolicy, GymEnv, make, rolloutAlgorithm } from '../../lib/rl/index.ts'
import { contains } from '../../lib/space/index.ts'
import { type Tensor, tensor, toFlat } from '../../lib/tensor/index.ts'
import { episodes, frames, recorder, runAlgorithm, tracePolicy } from '../../lib/trace/index.ts'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type CartPoleState, cartPole } from './index.ts'

describe('CartPole against gymnasium', () => {
  for (const c of loadFixture('methods/rl/classic_control').cases) {
    it(c.name, () => {
      const env = cartPole.create()
      let state = toFlat(decode(c.inputs[0]) as Tensor) as unknown as CartPoleState
      const states: number[][] = []
      const rewards: number[] = []
      const terminated: number[] = []
      for (const a of c.params['actions'] as number[]) {
        const r = env.step(state, a, key(0))
        state = r.state
        states.push([...state])
        rewards.push(r.reward)
        terminated.push(r.terminated ? 1 : 0)
        if (r.terminated) break
      }
      const got =
        c.fn === 'cartPoleStates' ? tensor(states) : c.fn === 'cartPoleRewards' ? tensor(rewards) : tensor(terminated)
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('the gym interface', () => {
  it('resets reproducibly from a seed and returns gymnasium tuples', () => {
    const a = new GymEnv(cartPole)
    const b = new GymEnv(cartPole)
    const [oa] = a.reset({ seed: 42 })
    const [ob] = b.reset({ seed: 42 })
    expect(oa).toEqual(ob)
    expect(contains(a.observationSpace, [...oa])).toBe(true)
    const [obs, reward, terminated, truncated] = a.step(1)
    expect([obs.length, reward, terminated, truncated]).toEqual([4, 1, false, false])
  })

  it('truncates at the time limit and refuses to step a finished episode', () => {
    const env = new GymEnv(cartPole)
    env.reset({ seed: 0 })
    // Alternate pushes keep the pole up long enough to reach the 500-step limit.
    let last: ReturnType<typeof env.step> | undefined
    for (let t = 0; t < 500; t++) {
      const [o] = last ?? [[0, 0, 0, 0]]
      last = env.step(((o as readonly number[])[2] ?? 0) + 0.5 * ((o as readonly number[])[3] ?? 0) > 0 ? 1 : 0)
      if (last[2] || last[3]) break
    }
    expect(last?.[3]).toBe(true)
    expect(() => env.step(0)).toThrow(/reset/)
  })

  it('renders the state as a scene', () => {
    const env = new GymEnv(cartPole)
    env.reset({ seed: 1 })
    const s = env.render()
    expect([s.width, s.height]).toEqual([600, 400])
    expect(s.children.map((c) => c.id).filter(Boolean)).toEqual(['cart', 'pole', 'axle'])
  })
})

describe('rollouts', () => {
  const balance = (o: readonly number[]): number => ((o[2] ?? 0) + 0.5 * (o[3] ?? 0) > 0 ? 1 : 0)

  it('evaluate policies: a balancing controller solves the task, pushing one way does not', () => {
    const env = make(cartPole)
    expect(evaluatePolicy(env, balance, { episodes: 5, seed: 0 }).meanReturn).toBe(500)
    expect(evaluatePolicy(env, () => 1, { episodes: 5, seed: 0 }).meanReturn).toBeLessThan(20)
  })

  it('record episodes and rendered frames by policy', () => {
    const rec = recorder()
    runAlgorithm(
      rolloutAlgorithm(make(cartPole), () => 0, { episodes: 2 }),
      {
        maxSteps: 1000,
        seed: 3,
        tracer: rec,
        policy: tracePolicy({ frames: true }),
      },
    )
    const eps = episodes(rec.run())
    expect(eps).toHaveLength(2)
    expect(eps.every((e) => e.terminated)).toBe(true)
    expect(frames(rec.run()).length).toBeGreaterThan(eps.reduce((n, e) => n + e.length, 0))
  })
})
