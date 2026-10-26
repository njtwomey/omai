import { describe, expect, it } from 'vitest'
import { cartPole } from '../environments/index.ts'
import { key } from '../random/index.ts'
import { GymVectorEnv, vectorEnv } from './vector.ts'
import { timeLimit } from './wrappers.ts'

describe('vector environments', () => {
  it('step copies with independent keys and auto-reset on the same step', () => {
    const venv = vectorEnv(timeLimit(cartPole.create(), 5), 3)
    let r = venv.reset(key(0))
    expect(new Set(r.observations.map((o) => o.join())).size).toBe(3)
    let state = r.state
    let finals = 0
    for (let t = 0; t < 6; t++) {
      const s = venv.step(state, [1, 0, 1], key(t))
      finals += s.finalObservations.filter((o) => o !== undefined).length
      state = s.state
    }
    expect(finals).toBeGreaterThanOrEqual(3)
    expect(() => venv.step(state, [1], key(0))).toThrow(/expected 3/)
    r = venv.reset(key(0))
    expect(r.infos).toHaveLength(3)
  })

  it('reset on the next step in next-step mode, and step a stateful vector env', () => {
    const venv = vectorEnv(timeLimit(cartPole.create(), 2), 2, { autoReset: 'next-step' })
    let state = venv.reset(key(0)).state
    state = venv.step(state, [1, 1], key(1)).state
    const end = venv.step(state, [1, 1], key(2))
    expect(end.truncated).toEqual([true, true])
    const after = venv.step(end.state, [1, 1], key(3))
    expect(after.rewards).toEqual([0, 0])
    const gym = new GymVectorEnv(cartPole.create(), 2)
    const [obs] = gym.reset({ seed: 0 })
    expect(obs).toHaveLength(2)
    expect(gym.step([0, 1])[1]).toEqual([1, 1])
    expect(gym.numEnvs).toBe(2)
    expect(() => new GymVectorEnv(cartPole.create(), 1).step([0])).toThrow(/reset/)
    expect(() => vectorEnv(cartPole.create(), 0)).toThrow(/positive/)
  })
})
