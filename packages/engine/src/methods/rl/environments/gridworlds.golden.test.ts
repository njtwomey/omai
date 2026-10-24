import { describe, expect, it } from 'vitest'
import type { Env } from '../../../lib/rl/index.ts'
import { type Tensor, tensor, toArray } from '../../../lib/tensor/index.ts'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { modelRows } from '../../../testing/rl.ts'
import { type FrozenLakeOptions, makeCliffWalking, makeFrozenLake } from './gridworlds.ts'

function build(p: Readonly<Record<string, unknown>>): Env<number, number, number> {
  if (p['env'] === 'CliffWalking-v1') return makeCliffWalking({ isSlippery: p['is_slippery'] === true })
  const options: FrozenLakeOptions = {
    ...(typeof p['map_name'] === 'string' ? { mapName: p['map_name'] as '4x4' | '8x8' } : {}),
    ...(typeof p['is_slippery'] === 'boolean' ? { isSlippery: p['is_slippery'] } : {}),
    ...(typeof p['success_rate'] === 'number' ? { successRate: p['success_rate'] } : {}),
    ...(Array.isArray(p['reward_schedule'])
      ? { rewardSchedule: p['reward_schedule'] as [number, number, number] }
      : {}),
    ...(Array.isArray(p['desc']) ? { desc: p['desc'] as string[] } : {}),
  }
  return makeFrozenLake(options)
}

describe('grid worlds against gymnasium', () => {
  for (const c of loadFixture('methods/rl/environments/gridworlds').cases) {
    it(c.name, () => {
      const env = build(c.params)
      if (c.fn === 'table') {
        const reference = toArray(decode(c.output) as Tensor) as number[][]
        const rows = modelRows(env, reference)
        expect(rows.length).toBe(reference.length)
        expectMatches(tensor(rows), c.output, 1e-15, 1e-15, c.name)
      } else {
        const initial = new Array<number>(env.model?.states.length ?? 0).fill(0)
        for (const { probability, state } of env.model?.initial ?? []) initial[state] = probability
        expectMatches(tensor(initial), c.output, 1e-15, 1e-15, c.name)
      }
    })
  }
})
