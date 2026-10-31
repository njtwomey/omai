import { describe, expect, it } from 'vitest'
import { type Key, key, uniform } from '../random/index.ts'
import { mdpFromEnv, policyEvaluation, runEpisode } from '../rl/index.ts'
import {
  BAIRD_BEHAVIOUR,
  BAIRD_FEATURES,
  BAIRD_TARGET,
  BAIRD_TRUTH,
  INFINITE_VARIANCE_VALUE,
  makeBairdCounterexample,
  makeInfiniteVariance,
  makeRandomWalk,
  randomWalkValues,
} from './prediction.ts'

const uniformPolicy = (_o: number, k: Key): number => ((uniform(k, []).data[0] ?? 0) < 0.5 ? 0 : 1)

describe('the random walk', () => {
  it('has the closed-form values of the books, equal to the exact solve of its own model', () => {
    for (const options of [{ n: 5 }, { n: 19, leftReward: -1 }, { n: 7, rightReward: 3 }]) {
      const mdp = mdpFromEnv(makeRandomWalk(options), 1)
      const exact = policyEvaluation(mdp, new Array(options.n).fill([0.5, 0.5]) as number[][])
      for (const [i, v] of randomWalkValues(options).entries()) expect(v).toBeCloseTo(exact[i] ?? Number.NaN, 10)
    }
    expect(randomWalkValues({ n: 5 }).map((v) => Math.round(v * 6))).toEqual([1, 2, 3, 4, 5])
  })

  it('solves the walk with jumps by the linear system: antisymmetric and increasing', () => {
    const v = randomWalkValues({ n: 21, jump: 4, leftReward: -1 })
    for (let i = 0; i < 21; i++) expect((v[i] ?? 0) + (v[20 - i] ?? 0)).toBeCloseTo(0, 10)
    for (let i = 1; i < 21; i++) expect(v[i] ?? 0).toBeGreaterThan(v[i - 1] ?? 0)
  })

  it('pays the value of the middle state on average when a random policy runs it', () => {
    const env = makeRandomWalk({ n: 5 })
    let total = 0
    const episodes = 3000
    for (let e = 0; e < episodes; e++) total += runEpisode(env, uniformPolicy, key(e)).return
    expect(total / episodes).toBeCloseTo(randomWalkValues({ n: 5 })[2] ?? Number.NaN, 1)
    expect(Math.abs(total / episodes - 0.5)).toBeLessThan(0.04)
  })
})

describe("Baird's counterexample", () => {
  it('has the features of the book and zero true values that the weights represent exactly', () => {
    expect(BAIRD_FEATURES).toHaveLength(7)
    expect(BAIRD_FEATURES[0]).toEqual([2, 0, 0, 0, 0, 0, 0, 1])
    expect(BAIRD_FEATURES[6]).toEqual([0, 0, 0, 0, 0, 0, 1, 2])
    expect(BAIRD_TRUTH.values).toEqual([0, 0, 0, 0, 0, 0, 0])
    expect(BAIRD_BEHAVIOUR[0]).toBeCloseTo(6 / 7, 12)
    expect(BAIRD_TARGET).toEqual([0, 1])
  })

  it('moves to a uniform upper state on the dashed action and to the lower state on the solid one', () => {
    const m = makeBairdCounterexample().model
    expect(m?.transitions(3, 1)).toEqual([{ probability: 1, state: 6, reward: 0, terminated: false }])
    const dashed = m?.transitions(6, 0) ?? []
    expect(dashed.map((o) => o.state)).toEqual([0, 1, 2, 3, 4, 5])
    expect(dashed.every((o) => Math.abs(o.probability - 1 / 6) < 1e-12 && !o.terminated)).toBe(true)
  })
})

describe('the infinite-variance process', () => {
  it('pays 1 under the always-left policy', () => {
    const env = makeInfiniteVariance()
    let total = 0
    for (let e = 0; e < 500; e++) total += runEpisode(env, () => 0, key(e)).return
    expect(total / 500).toBe(INFINITE_VARIANCE_VALUE)
  })
})
