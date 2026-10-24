import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { validatePlan } from '../../lib/search/index.ts'
import { planBlocks } from './planning.ts'
import { blocksWorld, randomBlocks, SUSSMAN, towersOf } from './planning-domains.ts'

describe('planBlocks', () => {
  it('solves the Sussman anomaly in six actions and plays search and plan', () => {
    const r = planBlocks(SUSSMAN.initial, SUSSMAN.goal)
    expect(r.status).toBe('solved')
    expect(r.plan).toHaveLength(6)
    expect(validatePlan(blocksWorld(SUSSMAN.initial, SUSSMAN.goal), r.plan).valid).toBe(true)
    const [search, plan] = recordsOf(r.exposition, 'boardSteps')
    expect(search?.steps.length).toBeGreaterThan(1)
    expect(plan?.steps).toHaveLength(7)
    expect(plan?.steps.at(-1)?.labels?.filter((l) => l !== '')).toEqual(['A', 'B', 'C'])
  })

  it('reads towers back and plans random problems with every search', () => {
    expect(towersOf(blocksWorld([['B', 'A'], ['C']], []).initial)).toEqual({
      towers: [['B', 'A'], ['C']],
      holding: null,
    })
    const { initial, goal } = randomBlocks(key(5), 5)
    for (const search of ['a-star', 'greedy', 'breadth-first'] as const) {
      const r = planBlocks(initial, goal, { search })
      expect(r.status).toBe('solved')
      expect(validatePlan(blocksWorld(initial, goal), r.plan).valid).toBe(true)
    }
  })
})
