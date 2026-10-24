import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { key } from '../random/index.ts'
import { scene } from '../scene/index.ts'
import { box } from '../space/index.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import type { Env } from './env.ts'
import { normalizeObservation } from './observation-wrappers.ts'
import { normalizeReward, rescaleAction } from './wrappers.ts'

/** Replays scripted observations, rewards and terminations (the oracle's `Scripted` environment). */
function scripted(
  observations: readonly (readonly number[])[],
  rewards: readonly number[],
  terminated: readonly number[],
  low = -1,
  high = 1,
): Env<number, readonly number[], readonly number[]> & { readonly actions: (readonly number[])[] } {
  const actions: (readonly number[])[] = []
  return {
    actions,
    spec: { id: 'Scripted-v0', description: 'scripted' },
    observationSpace: box(-Infinity, Infinity, [observations[0]?.length ?? 0]),
    actionSpace: box(low, high, [2]),
    reset: () => ({ state: 1, observation: observations[0] ?? [], info: {} }),
    resetFrom: (t) => ({ state: t + 1, observation: observations[t] ?? [], info: {} }),
    step: (t, action) => {
      actions.push(action)
      return {
        state: t + 1,
        observation: observations[t] ?? [],
        reward: rewards[t - 1] ?? 0,
        terminated: (terminated[t - 1] ?? 0) === 1,
        truncated: false,
        info: {},
      }
    },
    render: () => scene(1, 1, []),
  }
}

describe('wrappers against gymnasium', () => {
  const cases = loadFixture('rl/wrappers').cases
  for (const c of cases) {
    it(c.name, () => {
      const inputs = c.inputs.map((x) => toArray(decode(x) as Tensor))
      if (c.fn === 'normalizeReward') {
        const rewards = inputs[0] as number[]
        const terms = inputs[1] as number[]
        const obs = Array.from({ length: rewards.length + 1 }, () => [0])
        const env = normalizeReward(scripted(obs, rewards, terms), { gamma: c.params['gamma'] as number })
        let s = env.reset(key(0)).state
        const got: number[] = []
        for (let t = 0; t < rewards.length; t++) {
          const r = env.step(s, [0, 0], key(t))
          got.push(r.reward)
          s = r.state
        }
        expectMatches(tensor(got), c.output, 1e-12, 1e-12, c.name)
      } else if (c.fn.startsWith('normalizeObservation')) {
        const obs = inputs[0] as number[][]
        const env = normalizeObservation(scripted(obs, new Array<number>(obs.length).fill(0), []))
        const r0 = env.reset(key(0))
        let s = r0.state
        const seen: (readonly number[])[] = [r0.observation]
        for (let t = 0; t + 1 < obs.length; t++) {
          const r = env.step(s, [0, 0], key(t))
          seen.push(r.observation)
          s = r.state
        }
        const got =
          c.fn === 'normalizeObservation' ? seen : c.fn === 'normalizeObservationMean' ? s.stats.mean : s.stats.var
        expectMatches(tensor(got as number[] | number[][]), c.output, c.rtol, c.atol, c.name)
      } else {
        const actions = inputs[0] as number[][]
        const inner = scripted([[0]], [], [], c.params['low'] as number, c.params['high'] as number)
        const env = rescaleAction(inner, -1, 1)
        let s = env.reset(key(0)).state
        for (const a of actions) s = env.step(s, a, key(0)).state
        expectMatches(tensor(inner.actions as number[][]), c.output, 1e-12, 1e-15, c.name)
      }
    })
  }
})
