import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { foldIn, key } from '../random/index.ts'
import { item, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  boltzmann,
  boltzmannProbabilities,
  categoricalPolicy,
  deterministicPolicy,
  epsilonGreedy,
  epsilonGreedyProbabilities,
  gaussianNoise,
  gaussianPolicy,
  logProbOf,
  ornsteinUhlenbeckNoise,
} from './policies.ts'
import { discountedReturns, gae, nStepReturns, retrace, retraceTraces, tdErrors } from './returns.ts'

describe('exploration policies', () => {
  it('sample epsilon-greedy and Boltzmann actions with their probabilities', () => {
    const q = [1, 3, 2]
    const counts = [0, 0, 0]
    for (let i = 0; i < 6000; i++) {
      const a = epsilonGreedy(q, 0.3, key(i))
      counts[a] = (counts[a] ?? 0) + 1
    }
    const p = epsilonGreedyProbabilities(q, 0.3)
    for (const [a, c] of counts.entries()) expect(Math.abs(c / 6000 - (p[a] ?? 0))).toBeLessThan(0.02)
    const b = [0, 0, 0]
    for (let i = 0; i < 6000; i++) {
      const a = boltzmann(q, 1, key(i))
      b[a] = (b[a] ?? 0) + 1
    }
    const pb = boltzmannProbabilities(q, 1)
    for (const [a, c] of b.entries()) expect(Math.abs(c / 6000 - (pb[a] ?? 0))).toBeLessThan(0.02)
    expect(() => epsilonGreedy([], 0.1, key(0))).toThrow(/no actions/)
    expect(() => boltzmannProbabilities(q, 0)).toThrow(/temperature/)
  })
})

describe('policy heads', () => {
  it('sample categorical actions and expose log-probabilities and entropy', () => {
    const head = categoricalPolicy([Math.log(0.2), Math.log(0.8)])
    let ones = 0
    for (let i = 0; i < 2000; i++) ones += item(head.sample(key(i)))
    expect(Math.abs(ones / 2000 - 0.8)).toBeLessThan(0.03)
    expect(item(head.mode())).toBe(1)
    expect(logProbOf(head, 0)).toBeCloseTo(Math.log(0.2), 12)
    const { action, logProb } = head.sampleWithLogProb(key(3))
    expect(item(logProb)).toBeCloseTo(item(head.logProb(action)), 14)
    expect(() => head.rsample(key(0))).toThrow(/reparameterisation/)
  })

  it('reparameterise Gaussian samples so gradients flow to the mean and scale', () => {
    const k = key(4)
    const g = grad((m: Value) => sum(gaussianPolicy(m, [0, 0]).rsample(k)))(tensor([0.5, -1]))
    expect(toFlat(g)).toEqual([1, 1])
    const squashed = gaussianPolicy([0.3], [-0.5], { squash: true, low: -2, high: 2 })
    for (let i = 0; i < 20; i++) {
      const a = item(squashed.sample(foldIn(k, i)))
      expect(Math.abs(a)).toBeLessThan(2)
    }
    const { action, logProb } = squashed.sampleWithLogProb(key(9))
    expect(item(logProb)).toBeCloseTo(item(squashed.logProb(action)), 6)
    expect(toFlat(squashed.mode())[0]).toBeCloseTo(2 * Math.tanh(0.3), 14)
    expect(item(gaussianPolicy([0, 0], [0, 0]).entropy())).toBeCloseTo(1 + Math.log(2 * Math.PI), 12)
  })

  it('give deterministic heads without densities', () => {
    const d = deterministicPolicy([1, 2])
    expect(toFlat(d.rsample(key(0)))).toEqual([1, 2])
    expect(toFlat(d.mode())).toEqual([1, 2])
    expect(() => d.logProb([1, 2])).toThrow(/density/)
  })
})

describe('noise processes', () => {
  it('draw Gaussian noise and mean-reverting Ornstein-Uhlenbeck noise', () => {
    const g = gaussianNoise(3, { sigma: 0.5 })
    const x = g.step(g.init(), key(0))
    expect(x).toHaveLength(3)
    const ou = ornsteinUhlenbeckNoise(1, { theta: 0.5, sigma: 0.1, dt: 0.1, initial: 5 })
    let v = ou.init()
    for (let i = 0; i < 300; i++) v = ou.step(v, key(i))
    expect(Math.abs(v[0] ?? 0)).toBeLessThan(1)
  })
})

describe('return edge cases', () => {
  it('check lengths and parameters', () => {
    expect(() => discountedReturns([1, 2], 0.9, { dones: [0] })).toThrow(/same length/)
    expect(() => nStepReturns([1], [0], [0], 0.9, 1)).toThrow(/need 2 values/)
    expect(() => nStepReturns([1], [0, 0], [0], 0.9, 0)).toThrow(/positive/)
    expect(() => retrace([1, 1], [1, 1], [0, 0], [], [])).toThrow(/entries/)
    expect(tdErrors([0], [0], [0], [true], 1)).toEqual([0])
    expect(gae([], [], [], 0).advantages).toEqual([])
    expect(retraceTraces([1], [0])).toEqual([1])
  })
})
