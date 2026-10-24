import { describe, expect, it } from 'vitest'
import { blackjack, cartPole, frozenLake, mountainCarContinuous, pendulum } from '../../methods/rl/index.ts'
import { maxOf } from '../base/index.ts'
import { foldIn, key } from '../random/index.ts'
import type { BoxSpace } from '../space/index.ts'
import { type Env, resetAfter } from './env.ts'
import {
  flattenObservation,
  frameStack,
  normalizeObservation,
  oneHotObservation,
  pixelObservation,
  timeAwareObservation,
} from './observation-wrappers.ts'
import {
  actionRepeat,
  clipAction,
  clipReward,
  episodeStatistics,
  normalizeReward,
  rescaleAction,
  runningMeanStd,
  stickyAction,
  timeLimit,
  transformAction,
  transformObservation,
  transformReward,
  updateRunningMeanStd,
} from './wrappers.ts'

function run<S, O, A>(env: Env<S, O, A>, action: A, steps: number, seed = 0): { states: S[]; last: S } {
  let s = env.reset(key(seed)).state
  const states = [s]
  for (let t = 0; t < steps; t++) {
    const r = env.step(s, action, foldIn(key(seed), t))
    s = r.terminated || r.truncated ? resetAfter(env, r.state, foldIn(key(seed + 1), t)).state : r.state
    states.push(s)
  }
  return { states, last: s }
}

describe('time limits and episode statistics', () => {
  it('truncates exactly at the limit, never on a terminal step', () => {
    const env = timeLimit(pendulum.create(), 3)
    let s = env.reset(key(0)).state
    const flags: boolean[] = []
    for (let t = 0; t < 3; t++) {
      const r = env.step(s, [0], key(t))
      flags.push(r.truncated)
      s = r.state
    }
    expect(flags).toEqual([false, false, true])
    const term = timeLimit(cartPole.create(), 1)
    const r = term.step({ inner: [0, 0, 0.3, 0], t: 0 }, 1, key(0))
    expect([r.terminated, r.truncated]).toEqual([true, false])
    expect(() => timeLimit(cartPole.create(), 0)).toThrow(/at least 1/)
  })

  it('records returns and lengths in info and keeps the last episodes across resets', () => {
    const env = episodeStatistics(timeLimit(cartPole.create(), 50), { bufferLength: 2 })
    const { last } = run(env, 1, 60)
    expect(last.episodes).toBeGreaterThanOrEqual(3)
    expect(last.returns.length).toBe(2)
    expect(last.returns).toEqual(last.lengths)
    let s = env.reset(key(5)).state
    for (;;) {
      const r = env.step(s, 0, key(9))
      s = r.state
      if (r.terminated || r.truncated) {
        expect(r.info['episode']).toEqual({ r: s.episodeReturn, l: s.episodeLength })
        break
      }
    }
  })
})

describe('reward and action wrappers', () => {
  it('normalises rewards by the running variance of the discounted return', () => {
    const env = normalizeReward(cartPole.create(), { gamma: 0.5 })
    const { last } = run(env, 1, 30)
    expect(last.stats.count).toBeGreaterThan(30)
    const frozen = normalizeReward(cartPole.create(), { update: false })
    expect(frozen.step(frozen.reset(key(0)).state, 1, key(1)).reward).toBeCloseTo(1 / Math.sqrt(1 + 1e-8), 12)
    expect(() => normalizeReward(cartPole.create(), { gamma: 2 })).toThrow(/gamma/)
  })

  it('transforms and clips rewards, observations and actions', () => {
    const base = cartPole.create()
    const s = base.reset(key(0)).state
    expect(transformReward(base, (r) => -r).step(s, 1, key(1)).reward).toBe(-1)
    expect(clipReward(base, -0.25, 0.25).step(s, 1, key(1)).reward).toBe(0.25)
    expect(transformObservation(base, (o) => o[0]).step(s, 1, key(1)).observation).toBe(
      base.step(s, 1, key(1)).observation[0],
    )
    const named = transformAction(base, (a: 'left' | 'right') => (a === 'right' ? 1 : 0))
    expect(named.step(s, 'right', key(1)).state).toEqual(base.step(s, 1, key(1)).state)
    expect(() => clipReward(base, 1, 0)).toThrow(/max/)
    // Models pass through observation wrappers but not reward or action wrappers.
    expect(transformObservation(frozenLake.create(), (o) => o).model).toBeDefined()
    expect(transformReward(frozenLake.create(), (r) => r).model).toBeUndefined()
  })

  it('clips and rescales box actions', () => {
    const env = mountainCarContinuous.create()
    const s = env.reset(key(0)).state
    const clipped = clipAction(env)
    expect(clipped.step(s, [5], key(1)).state).toEqual(env.step(s, [1], key(1)).state)
    const rescaled = rescaleAction(pendulum.create(), 0, 1)
    expect((rescaled.actionSpace as BoxSpace).low).toBe(0)
    const p = pendulum.create()
    const ps = p.reset(key(2)).state
    expect(rescaled.step(ps, [0.75], key(0)).state).toEqual(p.step(ps, [1], key(0)).state)
    expect(() => rescaleAction(p, 1, 1)).toThrow(/min < max/)
    expect(() => clipAction(cartPole.create() as never)).toThrow(/box/)
  })

  it('repeats actions, summing rewards and stopping at the end of an episode', () => {
    const env = actionRepeat(cartPole.create(), 3)
    const s = env.reset(key(0)).state
    expect(env.step(s, 1, key(1)).reward).toBe(3)
    const r = env.step([0, 0, 0.2, 1], 0, key(1))
    expect(r.terminated).toBe(true)
    expect(r.reward).toBeLessThan(3)
    expect(() => actionRepeat(cartPole.create(), 0)).toThrow(/positive/)
  })

  it('makes actions sticky: with p = 1 the first action repeats forever, with p = 0 never', () => {
    const always = stickyAction(cartPole.create(), 1)
    let s = always.reset(key(0)).state
    for (let t = 0; t < 4; t++) s = always.step(s, t === 0 ? 1 : 0, key(t)).state
    expect(s.last).toBe(1)
    const never = stickyAction(cartPole.create(), 0)
    let n = never.reset(key(0)).state
    n = never.step(n, 1, key(0)).state
    n = never.step(n, 0, key(1)).state
    expect(n.last).toBe(0)
    expect(never.spec.deterministic).toBe(false)
  })
})

describe('observation wrappers', () => {
  it('normalises observations with running statistics carried by resetFrom', () => {
    const env = normalizeObservation(timeLimit(cartPole.create(), 20))
    const { last } = run(env, 1, 50)
    // One update per step and per reset: more than the 51 observations of a single episode.
    expect(last.stats.count).toBeGreaterThan(51)
    const stats = updateRunningMeanStd(runningMeanStd(2, 0), [
      [1, 2],
      [3, 6],
    ])
    expect(stats.mean).toEqual([2, 4])
    expect(stats.var).toEqual([1, 4])
    expect(updateRunningMeanStd(stats, [])).toBe(stats)
  })

  it('stacks frames, padding with the first observation', () => {
    const env = frameStack(cartPole.create(), 3)
    const r = env.reset(key(0))
    expect(r.observation).toEqual([r.observation[2], r.observation[2], r.observation[2]])
    const r1 = env.step(r.state, 1, key(1))
    expect(r1.observation[0]).toEqual(r.observation[0])
    expect(r1.observation[2]).not.toEqual(r.observation[2])
    expect(env.observationSpace.kind).toBe('tuple')
    const zero = frameStack(cartPole.create(), 2, { padding: { value: [0, 0, 0, 0] } })
    expect(zero.reset(key(0)).observation[0]).toEqual([0, 0, 0, 0])
  })

  it('flattens tuples, one-hot encodes discrete observations and appends time', () => {
    expect(flattenObservation(blackjack.create()).reset(key(0)).observation.length).toBe(3)
    const oh = oneHotObservation(frozenLake.create())
    expect(oh.reset(key(0)).observation).toEqual([1, ...new Array<number>(15).fill(0)])
    expect(() => oneHotObservation(cartPole.create() as never)).toThrow(/discrete/)
    const ta = timeAwareObservation(cartPole.create(), { horizon: 10 })
    const s0 = ta.reset(key(0))
    const s1 = ta.step(s0.state, 1, key(1))
    expect([s0.observation[4], s1.observation[4]]).toEqual([0, 0.1])
  })

  it('renders pixel observations, in colour or grey', () => {
    const env = pixelObservation(cartPole.create(), { width: 60, height: 40 })
    const o = env.reset(key(0)).observation
    expect(o.shape).toEqual([40, 60, 3])
    expect(maxOf(Array.from(o.data).slice(0, 3))).toBe(255)
    const grey = pixelObservation(cartPole.create(), { width: 30, height: 20, grayscale: true })
    expect(grey.step(grey.reset(key(0)).state, 1, key(1)).observation.shape).toEqual([20, 30, 1])
  })
})
