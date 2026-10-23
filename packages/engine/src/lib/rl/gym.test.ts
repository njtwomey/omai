import { describe, expect, it } from 'vitest'
import * as environments from '../../methods/rl/index.ts'
import { cartPole, frozenLake, mountainCar } from '../../methods/rl/index.ts'
import { contains } from '../space/index.ts'
import { isEnvDefinition, unwrapped } from './env.ts'
import { GymEnv, listEnvironments, make } from './gym.ts'
import { episodeStatistics } from './wrappers.ts'

describe('make and the registry', () => {
  it('lists every environment definition of a module, sorted by id', () => {
    const ids = listEnvironments(environments).map((d) => d.id)
    expect(ids).toEqual([...ids].sort())
    for (const id of ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v1', 'FrozenLake-v1', 'Taxi-v4']) {
      expect(ids).toContain(id)
    }
    expect(listEnvironments({ x: 1, y: 'no', cartPole }).map((d) => d.id)).toEqual(['CartPole-v1'])
    expect(isEnvDefinition({ id: 'a', description: 'b' })).toBe(false)
  })

  it('makes by id, applies the time limit, overrides or disables it, and wraps', () => {
    const registry = listEnvironments(environments)
    expect(make('MountainCar-v0', {}, { registry }).spec.maxEpisodeSteps).toBe(200)
    expect(make(cartPole, {}, { maxEpisodeSteps: 7 }).spec.maxEpisodeSteps).toBe(7)
    expect(make(cartPole, {}, { maxEpisodeSteps: null }).wrapped).toBeUndefined()
    const wrapped = make(mountainCar, {}, { wrappers: [(e) => episodeStatistics(e)] })
    expect(unwrapped(wrapped).spec.id).toBe('MountainCar-v0')
    expect(() => make('Nope-v0', {}, { registry })).toThrow(/no environment/)
    expect(() => make('CartPole-v1')).toThrow(/registry/)
  })
})

describe('GymEnv', () => {
  it('resets reproducibly from a seed and returns gymnasium tuples', () => {
    const a = new GymEnv(cartPole)
    const b = new GymEnv(cartPole)
    const [oa] = a.reset({ seed: 42 })
    const [ob] = b.reset({ seed: 42 })
    expect(oa).toEqual(ob)
    expect(contains(a.observationSpace, [...oa])).toBe(true)
    const [obs, reward, terminated, truncated] = a.step(1)
    expect([obs.length, reward, terminated, truncated]).toEqual([4, 1, false, false])
    expect(a.elapsedSteps).toBe(1)
    // A second reset without a seed continues the stream and starts a different episode.
    const [oc] = a.reset()
    expect(oc).not.toEqual(oa)
  })

  it('truncates at the time limit and refuses to step a finished episode', () => {
    const env = new GymEnv(cartPole)
    env.reset({ seed: 0 })
    let last: ReturnType<typeof env.step> | undefined
    for (let t = 0; t < 500; t++) {
      const [o] = last ?? [[0, 0, 0, 0]]
      last = env.step(((o as readonly number[])[2] ?? 0) + 0.5 * ((o as readonly number[])[3] ?? 0) > 0 ? 1 : 0)
      if (last[2] || last[3]) break
    }
    expect(last?.[3]).toBe(true)
    expect(() => env.step(0)).toThrow(/reset/)
  })

  it('samples actions from the action space and renders the state', () => {
    const env = new GymEnv(frozenLake, { isSlippery: false })
    env.reset({ seed: 1 })
    const a = env.sampleAction()
    expect(contains(env.actionSpace, a)).toBe(true)
    const s = new GymEnv(cartPole)
    s.reset({ seed: 1 })
    expect(
      s
        .render()
        .children.map((c) => c.id)
        .filter(Boolean),
    ).toEqual(['cart', 'pole', 'axle'])
  })

  it('wraps an already built environment and carries statistics across resets', () => {
    const env = new GymEnv(episodeStatistics(make(cartPole)))
    env.reset({ seed: 3 })
    for (let i = 0; i < 2; i++) {
      let done = false
      while (!done) {
        const r = env.step(1)
        done = r[2] || r[3]
        if (done) expect(r[4]['episode']).toBeDefined()
      }
      env.reset()
    }
    expect((env.state as { episodes: number }).episodes).toBe(2)
  })
})
