import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import {
  cliffWalkingWorld,
  destination,
  type FrozenLakeOptions,
  frozenLakeWorld,
  taxiWorld,
  transitionModel,
  type WorldTask,
} from './index.ts'

function task(p: Readonly<Record<string, unknown>>): WorldTask {
  if (p['env'] === 'CliffWalking-v1') return cliffWalkingWorld({ isSlippery: p['is_slippery'] === true })
  const options: FrozenLakeOptions = {
    ...(typeof p['map_name'] === 'string' ? { mapName: p['map_name'] as '4x4' | '8x8' } : {}),
    ...(typeof p['is_slippery'] === 'boolean' ? { isSlippery: p['is_slippery'] } : {}),
    ...(typeof p['success_rate'] === 'number' ? { successRate: p['success_rate'] } : {}),
    ...(Array.isArray(p['reward_schedule'])
      ? { rewardSchedule: p['reward_schedule'] as [number, number, number] }
      : {}),
    ...(Array.isArray(p['desc']) ? { desc: p['desc'] as string[] } : {}),
  }
  return frozenLakeWorld(options)
}

describe('textbook worlds against gymnasium', () => {
  for (const c of loadFixture('worlds/textbook').cases) {
    it(c.name, () => {
      if (c.fn === 'taxiMoves') {
        const t = taxiWorld()
        for (const [cell = 0, action = 0, next = 0] of c.params['moves'] as number[][]) {
          const act = t.options.actions?.actions[action]
          const to = destination(t.world, cell, act?.kind === 'move' ? act.direction : -1)
          expect(to < 0 ? cell : to, `cell ${cell} action ${action}`).toBe(next)
        }
        return
      }
      const t = task(c.params)
      const m = transitionModel(t.world, t.rules, t.options)
      if (c.fn === 'initial') {
        expectMatches(tensor(Array.from(m.initial)), c.output, 1e-15, 1e-15, c.name)
        return
      }
      const reference = toArray(decode(c.output) as Tensor) as number[][]
      const pairs: [number, number][] = []
      for (const [s = 0, a = 0] of reference) {
        const last = pairs[pairs.length - 1]
        if (last === undefined || last[0] !== s || last[1] !== a) pairs.push([s, a])
      }
      const rows = pairs.flatMap(([s, a]) =>
        m.outcomes(s, a).map((o) => [s, a, o.probability, o.state, o.reward, o.terminated ? 1 : 0]),
      )
      expect(rows.length).toBe(reference.length)
      expectMatches(tensor(rows), c.output, 1e-15, 1e-15, c.name)
    })
  }
})
