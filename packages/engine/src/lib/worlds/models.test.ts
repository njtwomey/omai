import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { isWhiteboxError } from '../base/index.ts'
import { key } from '../random/index.ts'
import {
  ACTIONS,
  deterministicModel,
  fromAscii,
  lockedDoors,
  pickups,
  rewardOn,
  rules,
  sampleInitial,
  sampleOutcome,
  slip,
  slipUniform,
  stepReward,
  terminateOn,
  terrainCost,
  transitionModel,
} from './index.ts'

const valueError = (f: () => unknown): boolean => {
  try {
    f()
  } catch (e) {
    return isWhiteboxError(e, 'value')
  }
  return false
}

describe('transition models', () => {
  it('numbers states by slot, heading and items, and inverts the numbering', () => {
    const w = fromAscii(['SK.', '.#D', '..G'])
    const m = transitionModel(w, rules(lockedDoors(), pickups(), terminateOn('goals')), { actions: ACTIONS.turning })
    expect([m.nStates, m.nActions, m.tracked]).toEqual([9 * 4 * 2, 3, [0]])
    for (let i = 0; i < m.nStates; i++) expect(m.stateIndex(m.state(i))).toBe(i)
    expect(m.state(m.stateIndex({ cell: 4, items: 1, heading: 2 }))).toEqual({ cell: 4, items: 1, heading: 2 })
    expect(Array.from(m.initial).indexOf(1)).toBe(m.stateIndex({ cell: 0, items: 0, heading: 1 }))
    expect(valueError(() => m.state(-1))).toBe(true)
  })

  it('renumbers passable cells, starts uniformly over non-terminal cells, and merges equal outcomes', () => {
    const w = fromAscii(['.#.', '..G'])
    const m = transitionModel(w, rules(slipUniform(0.4), terminateOn('goals')), {
      cells: 'passable',
      initial: 'uniform',
      merge: true,
    })
    expect([m.nStates, m.cells]).toEqual([5, [0, 2, 3, 4, 5]])
    expect(Array.from(m.initial)).toEqual([0.25, 0.25, 0.25, 0.25, 0])
    // From the top-left corner up (0.4) and left (0.2) bump and right (0.2) hits a wall, so 0.8 stays; 0.2 goes down.
    const os = m.outcomes(0, 0)
    expect(os.map((o) => [o.state, Number(o.probability.toFixed(12))])).toEqual([
      [0, 0.8],
      [2, 0.2],
    ])
    expect(valueError(() => m.stateIndex({ cell: 1, items: 0, heading: -1 }))).toBe(true)
  })

  it('absorbs terminal and impassable states unless told not to', () => {
    const w = fromAscii(['S#G'])
    const r = rules(stepReward(-1), terminateOn('goals'))
    const m = transitionModel(w, r)
    expect(m.outcomes(1, 0)).toEqual([{ probability: 1, state: 1, reward: 0, terminated: true, cost: 0 }])
    expect([m.terminal(0), m.terminal(1), m.terminal(2)]).toEqual([false, true, true])
    const open = transitionModel(w, r, { absorbing: false })
    expect(open.outcomes(2, 3)[0]?.state).toBe(2)
    expect(valueError(() => transitionModel(fromAscii(['...']), []))).toBe(true)
  })

  it('samples outcomes and starts in proportion to their probabilities', () => {
    const m = transitionModel(
      fromAscii(['S..', '...']),
      rules(
        slip([
          [0, 0.25],
          [1, 0.75],
        ]),
      ),
    )
    let right = 0
    for (let i = 0; i < 400; i++) if (sampleOutcome(m, 4, 0, key(i)).state === 5) right++
    expect(right / 400).toBeGreaterThan(0.65)
    expect(right / 400).toBeLessThan(0.85)
    const two = transitionModel(fromAscii(['S.S']), [])
    const counts = [0, 0, 0]
    for (let i = 0; i < 200; i++) {
      const s = sampleInitial(two, key(i))
      counts[s] = (counts[s] ?? 0) + 1
    }
    expect(counts[1]).toBe(0)
    expect(Math.abs((counts[0] ?? 0) - 100)).toBeLessThan(30)
  })

  it('gives every action a distribution that sums to one (property)', () => {
    const chars = '..~#HSGK$D>'.split('')
    fc.assert(
      fc.property(
        fc.array(fc.array(fc.constantFrom(...chars), { minLength: 4, maxLength: 4 }), { minLength: 3, maxLength: 3 }),
        fc.double({ min: 0, max: 1, noNaN: true }),
        (grid, success) => {
          const rows = grid.map((r) => r.join(''))
          rows[0] = `S${rows[0]?.slice(1)}`
          const w = fromAscii(rows)
          if (w.items.filter((it) => it.kind !== 'door').length > 6) return
          const fail = (1 - success) / 2
          const m = transitionModel(
            w,
            rules(
              slip([
                [-1, fail],
                [0, success],
                [1, fail],
              ]),
              lockedDoors(),
              pickups(),
              stepReward(-1),
              terminateOn('hazards'),
              terminateOn('goals'),
            ),
            { actions: ACTIONS.urdl },
          )
          for (let s = 0; s < m.nStates; s += 3) {
            for (let a = 0; a < m.nActions; a++) {
              const os = m.outcomes(s, a)
              const total = os.reduce((t, o) => t + o.probability, 0)
              expect(total).toBeCloseTo(1, 12)
              for (const o of os) expect(o.state >= 0 && o.state < m.nStates).toBe(true)
            }
          }
        },
      ),
      { numRuns: 40 },
    )
  })
})

describe('deterministic models', () => {
  it('is a search problem: actions that move, results, costs, rewards and goals', () => {
    const w = fromAscii(['S~.', '#.G'])
    const m = deterministicModel(w, rules(terrainCost(), rewardOn('goals', 10), terminateOn('goals')))
    expect(m.actions(m.initial)).toEqual([1])
    const water = m.result(m.initial, 1)
    expect([water.cell, m.stepCost(m.initial, 1, water), m.label(water)]).toEqual([1, 5, '(0, 1)'])
    const below = m.result(water, 2)
    expect([m.reward(below, 1), m.isGoal(m.result(below, 1)), m.actions(m.result(below, 1))]).toEqual([10, true, []])
    expect(m.isTerminal(m.result(below, 1))).toBe(true)
    expect(m.key(below)).toBe(4)
    expect(m.starts).toEqual([m.initial])
  })

  it('labels headings and items, and refuses branching actions', () => {
    const w = fromAscii(['SK.'])
    const m = deterministicModel(w, rules(pickups()), { actions: ACTIONS.turning })
    expect(m.label(m.result(m.initial, 2))).toBe('(0, 1) heading 1 items 1')
    const slippery = deterministicModel(
      w,
      rules(
        slip([
          [0, 0.5],
          [1, 0.5],
        ]),
      ),
    )
    expect(valueError(() => slippery.result(slippery.initial, 0))).toBe(true)
    expect(valueError(() => deterministicModel(fromAscii(['..']), []))).toBe(true)
  })
})
