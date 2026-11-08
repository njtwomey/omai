import { describe, expect, it } from 'vitest'
import { isOmaiError } from '../base/index.ts'
import {
  ACTIONS,
  type AgentState,
  applyRules,
  fromAscii,
  lockedDoors,
  oneWay,
  pickups,
  type Rule,
  rewardOn,
  rules,
  slip,
  slipUniform,
  slotActions,
  stepReward,
  teleport,
  terminateOn,
  terrainCost,
  trackedItems,
  wind,
  world,
} from './index.ts'

const at = (cell: number, items = 0, heading = -1): AgentState => ({ cell, items, heading })
const summary = (es: readonly { probability: number; cell: number; reward: number; terminated: boolean }[]) =>
  es.map((e) => [e.probability, e.cell, e.reward, e.terminated])

describe('rules', () => {
  it('composes rewards, replacement, termination and teleports in order', () => {
    const w = fromAscii(['...', 'SCG'])
    const r = rules(
      stepReward(-1),
      rewardOn({ terrain: 'cliff' }, -100, { replace: true }),
      teleport({ terrain: 'cliff' }, 'start'),
      terminateOn('goals'),
    )
    expect(summary(applyRules(w, r, at(3), 1))).toEqual([[1, 3, -100, false]])
    expect(summary(applyRules(w, r, at(2), 2))).toEqual([[1, 5, -1, true]])
    expect(summary(applyRules(w, r, at(0), 0))).toEqual([[1, 0, -1, false]])
    expect(summary(applyRules(w, rules(teleport([1], 4)), at(0), 1))).toEqual([[1, 4, 0, false]])
    expect(summary(applyRules(w, rules(teleport([1], (c) => c + 1)), at(0), 1))).toEqual([[1, 2, 0, false]])
  })

  it('slips around the action ring and drops impossible branches', () => {
    const w = fromAscii(['...', '...', '...'])
    const frozen = slip([
      [-1, 1 / 3],
      [0, 1 / 3],
      [1, 1 / 3],
    ])
    // FrozenLake's order: left (0) slips to up (3), left, down (1).
    const left = applyRules(w, [frozen], at(4), 0, { actions: ACTIONS.ldru })
    expect(left.map((e) => e.cell)).toEqual([1, 3, 7])
    const aima = slip([
      [0, 1],
      [1, 0],
    ])
    expect(applyRules(w, [aima], at(4), 0).map((e) => e.cell)).toEqual([1])
    expect(() => slip([[0, 0.5]])).toThrow()
    const noisy = applyRules(w, [slipUniform(0.7)], at(4), 1)
    expect(noisy.map((e) => [e.cell, Number(e.probability.toFixed(12))])).toEqual([
      [1, 0.1],
      [5, 0.7],
      [7, 0.1],
      [3, 0.1],
    ])
  })

  it('pushes moves with wind, clamped to the grid, and with stochastic gusts', () => {
    const w = fromAscii(['...', '...', '...'])
    const strengths = [0, 1, 2]
    expect(applyRules(w, [wind(strengths)], at(7), 1).map((e) => e.cell)).toEqual([5])
    expect(applyRules(w, [wind(strengths)], at(8), 3).map((e) => e.cell)).toEqual([1])
    const gusty = applyRules(w, [wind(strengths, { stochastic: true })], at(7), 1)
    expect(gusty.map((e) => [e.cell, e.probability])).toEqual([
      [8, 1 / 3],
      [5, 1 / 3],
      [2, 1 / 3],
    ])
    // A column without wind and an edge-clamped diagonal (windy grid world semantics).
    expect(applyRules(w, [], at(0), 4, { actions: ACTIONS.king, edges: 'clamp' }).map((e) => e.cell)).toEqual([1])
    expect(applyRules(w, [], at(0), 4, { actions: ACTIONS.king }).map((e) => e.cell)).toEqual([0])
  })

  it('collects pickups and opens doors with keys of their colour', () => {
    const w = fromAscii(['SKD$G'])
    expect(trackedItems(w)).toEqual([0, 2])
    const r = rules(lockedDoors(), pickups())
    expect(summary(applyRules(w, r, at(1, 0), 1))).toEqual([[1, 1, 0, false]])
    const got = applyRules(w, r, at(0), 1)[0]
    expect([got?.cell, got?.items]).toEqual([1, 1])
    expect(applyRules(w, r, at(1, 1), 1)[0]?.cell).toBe(2)
    const coin = applyRules(w, r, at(2, 1), 1)[0]
    expect([coin?.items, coin?.reward]).toEqual([3, 1])
    expect(applyRules(w, rules(pickups({ reward: 5 })), at(2, 1), 1)[0]?.reward).toBe(5)
    // A key of another colour does not open the door.
    const red = fromAscii(['Sk.D'], { legend: { k: { terrain: 'floor', item: { kind: 'key', colour: 'red' } } } })
    expect(applyRules(red, r, at(2, 1), 1)[0]?.cell).toBe(2)
  })

  it('forbids entering one-way cells against their direction and costs terrain', () => {
    const w = fromAscii(['.~.'])
    expect(applyRules(w, [oneWay({ 1: 1 })], at(2), 3)[0]?.cell).toBe(2)
    expect(applyRules(w, [oneWay({ 1: 1 })], at(0), 1)[0]?.cell).toBe(1)
    expect(applyRules(w, [terrainCost()], at(0), 1)[0]?.cost).toBe(5)
    expect(applyRules(w, [terrainCost()], at(0), 3)[0]?.cost).toBe(1)
    const k = fromAscii(['...', '...'])
    expect(applyRules(k, [terrainCost()], at(0), 5, { actions: ACTIONS.king })[0]?.cost).toBe(Math.SQRT2)
    expect(applyRules(k, [], at(0), 8, { actions: ACTIONS.kingStay })[0]?.cost).toBe(0)
  })

  it('turns and moves forward with headings, on squares and hexagons', () => {
    const w = fromAscii(['...', '...'])
    const turnRight = applyRules(w, [], at(0, 0, 1), 1, { actions: ACTIONS.turning })[0]
    expect([turnRight?.cell, turnRight?.heading]).toEqual([0, 2])
    expect(applyRules(w, [], at(0, 0, 0), 0, { actions: ACTIONS.turning })[0]?.heading).toBe(3)
    expect(applyRules(w, [], at(0, 0, 1), 2, { actions: ACTIONS.turning })[0]?.cell).toBe(1)
    const h = fromAscii(['...', '...'], { topology: 'hex' })
    expect(applyRules(h, [], at(0, 0, 5), 1, { actions: ACTIONS.turning })[0]?.heading).toBe(0)
    expect(applyRules(h, [], at(0), 1)[0]?.cell).toBe(3)
  })

  it('moves along neighbour slots on graphs and rejects unknown actions', () => {
    const g = world({ topology: { kind: 'graph', n: 3, neighbours: [[1, 2], [0], [0]] } })
    expect(slotActions(g).names).toEqual(['slot 0', 'slot 1'])
    expect(applyRules(g, [], at(0), 1)[0]?.cell).toBe(2)
    expect(applyRules(g, [], at(1), 1)[0]?.cell).toBe(1)
    try {
      applyRules(g, [], at(0), 7)
      expect.unreachable()
    } catch (e) {
      expect(isOmaiError(e, 'value')).toBe(true)
    }
  })

  it('branches in entry effects multiply probabilities', () => {
    const w = fromAscii(['...'])
    const split: Rule = {
      name: 'split',
      enter: (e) => [
        { ...e, probability: 0.25 },
        { ...e, cell: 2, probability: 0.75 },
      ],
    }
    const out = applyRules(
      w,
      [
        slip([
          [0, 0.5],
          [2, 0.5],
        ]),
        split,
      ],
      at(1),
      1,
    )
    expect(out.map((e) => [e.cell, e.probability])).toEqual([
      [2, 0.125],
      [2, 0.375],
      [0, 0.125],
      [2, 0.375],
    ])
  })
})
