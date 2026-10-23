import { describe, expect, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { mdpFromEnv, valueIteration } from '../../../lib/rl/index.ts'
import {
  BLACKJACK_DECK,
  blackjack,
  blackjackStep,
  blackjackSum,
  makeNChain,
  makeRiverSwim,
  makeTaxi,
  taxiActionMask,
  taxiDecode,
  taxiEncode,
  taxiTransitions,
} from './toy-text.ts'

describe('Blackjack', () => {
  it('counts aces as 11 when they do not bust and pays naturals by the rules', () => {
    expect(blackjackSum([1, 1, 9])).toBe(21)
    expect(blackjackSum([1, 10, 5])).toBe(16)
    const natural = { player: [1, 10], dealer: [10, 7] }
    expect(blackjackStep(natural, 0, () => 2).reward).toBe(1)
    expect(blackjackStep(natural, 0, () => 2, { natural: true }).reward).toBe(1.5)
    expect(blackjackStep({ player: [1, 10], dealer: [1, 10] }, 0, () => 2, { sab: true }).reward).toBe(0)
    expect(blackjackStep({ player: [10, 9], dealer: [10, 6] }, 1, () => 5)).toEqual({
      state: { player: [10, 9, 5], dealer: [10, 6] },
      reward: -1,
      terminated: true,
    })
  })

  it('draws cards from an infinite deck, about 4/13 of them tens', () => {
    const env = blackjack.create()
    let tens = 0
    for (let i = 0; i < 1000; i++) tens += env.reset(key(i)).state.player.filter((c) => c === 10).length
    expect(Math.abs(tens / 2000 - 4 / 13)).toBeLessThan(0.03)
    expect(BLACKJACK_DECK).toHaveLength(13)
    expect(env.step(env.reset(key(3)).state, 0, key(4)).terminated).toBe(true)
  })
})

describe('Taxi', () => {
  it('encodes and decodes states and masks actions', () => {
    for (let s = 0; s < 500; s += 37) expect(taxiEncode(...taxiDecode(s))).toBe(s)
    expect(taxiActionMask(taxiEncode(4, 3, 4, 3))).toEqual([0, 1, 1, 0, 0, 1])
    expect(taxiTransitions(taxiEncode(2, 2, 1, 0), 4)[0]?.reward).toBe(-10)
  })

  it('solves the delivery task by value iteration', () => {
    const env = makeTaxi()
    const solution = valueIteration(mdpFromEnv(env, 1), { maxIter: 1000 })
    // At R with the passenger there, bound for G: pick up (-1), eight moves around the wall (-8), drop off (+20).
    const s = taxiEncode(0, 0, 0, 1)
    expect(solution.V[s]).toBe(20 - 9)
  })

  it('models a fickle passenger who changes destination once', () => {
    const env = makeTaxi({ ficklePassenger: true })
    expect(env.model?.states.length).toBe(1000)
    const s = taxiEncode(0, 0, 4, 1) + 500
    const moved = env.model?.transitions(s, 0) ?? []
    expect(moved).toHaveLength(3)
    expect(moved.map((o) => taxiDecode(o.state)[3]).sort()).toEqual([0, 2, 3])
    expect(moved.every((o) => o.state < 500)).toBe(true)
    expect(env.observe?.(s)).toBe(s - 500)
    const stay = env.model?.transitions(s, 1) ?? []
    expect(stay[0]?.state).toBe(s)
  })
})

describe('NChain and RiverSwim', () => {
  it('slip with the configured probability', () => {
    const env = makeNChain({ slip: 0 })
    expect(env.model?.transitions(0, 0)).toEqual([{ probability: 1, state: 1, reward: 0, terminated: false }])
    let back = 0
    const slippy = makeNChain({ slip: 0.2 })
    for (let i = 0; i < 2000; i++) back += slippy.step(2, 0, key(i)).state === 0 ? 1 : 0
    expect(Math.abs(back / 2000 - 0.2)).toBeLessThan(0.03)
  })

  it('make RiverSwim hard: the optimal value of the left bank is far above resting there', () => {
    const env = makeRiverSwim()
    const v = valueIteration(mdpFromEnv(env, 0.95)).V
    expect(v[0]).toBeGreaterThan(0.005 / (1 - 0.95))
  })
})
