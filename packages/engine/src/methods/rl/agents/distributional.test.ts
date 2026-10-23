import { describe, expect, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { evaluatePolicy, greedyPolicyOf, make, oneHotObservation, trainAgent } from '../../../lib/rl/index.ts'
import { item, tensor } from '../../../lib/tensor/index.ts'
import { recorder, tracePolicy } from '../../../lib/trace/index.ts'
import { frozenLake } from '../environments/index.ts'
import {
  c51,
  categoricalDistribution,
  categoricalProjection,
  categoricalSupport,
  implicitQuantileNetwork,
  implicitQuantiles,
  iqn,
  qrDqn,
  quantileDistribution,
  quantileHuberLoss,
  rainbow,
  riskDistortion,
} from './distributional.ts'

// FrozenLake without ice, one-hot observations: the shortest path takes six steps and pays 1.
const lake = () =>
  make(frozenLake, { isSlippery: false }, { wrappers: [(e) => oneHotObservation(e as never) as never] })

const base = {
  hidden: [32],
  learningRate: 5e-3,
  gamma: 0.9,
  batchSize: 32,
  learningStarts: 64,
  targetUpdateInterval: 50,
  explorationSteps: 800,
  epsilonEnd: 0.05,
  bufferSize: 2000,
} as const

const solves = (agent: Parameters<typeof trainAgent>[1], steps: number, seed: number): number => {
  const env = lake()
  const result = trainAgent(env as never, agent, { maxSteps: steps, seed })
  return evaluatePolicy(env as never, greedyPolicyOf(agent, result.agent as never), { episodes: 1 }).meanReturn
}

describe('categorical projection', () => {
  it('keeps probability mass and shifts the mean by r + gamma E[Z]', () => {
    const z = categoricalSupport(-2, 2, 9)
    const p = [[0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0, 0]]
    const [m = []] = categoricalProjection(p, [0.3], [0.9], -2, 2)
    expect(m.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14)
    const meanOf = (q: readonly number[]) => q.reduce((a, pi, j) => a + pi * (z[j] ?? 0), 0)
    expect(meanOf(m)).toBeCloseTo(0.3 + 0.9 * meanOf(p[0] ?? []), 14)
  })

  it('clips to the support and collapses at terminal states', () => {
    const [clipped = []] = categoricalProjection([[0, 0, 1]], [5], [1], -1, 1)
    expect(clipped).toEqual([0, 0, 1])
    const [terminal = []] = categoricalProjection([[0.5, 0, 0.5]], [0], [0], -1, 1)
    expect(terminal).toEqual([0, 1, 0])
  })
})

describe('quantile losses and distortions', () => {
  it('weights under- and over-estimates asymmetrically', () => {
    expect(item(quantileHuberLoss([[0]], [[2]], [0.25]))).toBe(0.375)
    expect(item(quantileHuberLoss([[0]], [[-2]], [0.25]))).toBe(1.125)
    expect(item(quantileHuberLoss([[0]], [[2]], [0.25], { kappa: 0 }))).toBe(0.5)
    expect(quantileHuberLoss([[0], [1]], [[2], [1]], [0.5], { reduction: 'none' }).shape).toEqual([2])
  })

  it('distorts fractions for risk-sensitive policies', () => {
    expect(riskDistortion('neutral')([0.3])).toEqual([0.3])
    expect(riskDistortion({ kind: 'cvar', eta: 0.5 })([0.5])).toEqual([0.25])
    expect(riskDistortion({ kind: 'wang', eta: 0 })([0.3])[0]).toBeCloseTo(0.3, 12)
    expect(riskDistortion({ kind: 'wang', eta: -0.75 })([0.5])[0]).toBeLessThan(0.5)
    expect(riskDistortion({ kind: 'cpw', eta: 1 })([0.3])[0]).toBeCloseTo(0.3, 14)
    expect(riskDistortion({ kind: 'pow', eta: -2 })([0.5])[0]).toBeLessThan(0.5)
  })
})

describe('distributional agents', () => {
  it('C51 learns the dry lake and reports its statistics', { timeout: 20_000 }, () => {
    const env = lake()
    const agent = c51(env as never, { ...base, atoms: 21, vMin: -1, vMax: 1 })
    const rec = recorder()
    const result = trainAgent(env as never, agent, {
      maxSteps: 1500,
      seed: 0,
      tracer: rec,
      policy: tracePolicy({ metrics: ['loss', 'qValue'] }),
    })
    expect(evaluatePolicy(env as never, greedyPolicyOf(agent, result.agent), { episodes: 1 }).meanReturn).toBe(1)
    const metrics = rec
      .run()
      .events.filter((e) => e.type === 'metric')
      .map((e) => (e as { name: string }).name)
    expect(new Set(metrics)).toEqual(new Set(['loss', 'qValue']))
    const dist = categoricalDistribution(env as never, result.agent, oneHot16(0), {
      ...base,
      atoms: 21,
      vMin: -1,
      vMax: 1,
    })
    expect(dist.probabilities).toHaveLength(4)
    for (const row of dist.probabilities) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('QR-DQN learns the dry lake', { timeout: 20_000 }, () => {
    expect(solves(qrDqn(lake() as never, { ...base, quantiles: 16 }), 1500, 0)).toBe(1)
  })

  it('IQN learns the dry lake', { timeout: 20_000 }, () => {
    const opts = { ...base, batchSize: 16, trainFrequency: 2, quantiles: 8, targetQuantiles: 8, policySamples: 8 }
    expect(solves(iqn(lake() as never, { ...opts, embedding: 16 }), 1200, 1)).toBe(1)
  })

  it('Rainbow learns the dry lake', { timeout: 20_000 }, () => {
    const opts = { ...base, trainFrequency: 2, atoms: 21, vMin: -1, vMax: 1 }
    expect(solves(rainbow(lake() as never, opts), 1200, 1)).toBe(1)
  })

  it('reads out distributions of the right shapes', () => {
    const env = lake()
    const qs = qrDqn(env as never, { hidden: [4], quantiles: 5 })
    expect(
      quantileDistribution(env as never, qs.init(key(0)), oneHot16(3), { hidden: [4], quantiles: 5 }),
    ).toHaveLength(4)
    const opts = { hidden: [4], embedding: 3 }
    const s = iqn(env as never, opts).init(key(0))
    const z = implicitQuantiles(env as never, s, oneHot16(3), [0.1, 0.5, 0.9], opts)
    expect(z.map((r) => r.length)).toEqual([3, 3, 3, 3])
    expect(iqn(env as never, { distortion: { kind: 'cvar', eta: 0.25 } }).config).toMatchObject({
      distortion: { kind: 'cvar', eta: 0.25 },
    })
    const net = implicitQuantileNetwork(2, 3, { hidden: [4], embedding: 3 }, key(1))
    expect(net.apply(net.params, tensor([[1, 0]]), tensor([[0.2, 0.4]])).shape).toEqual([1, 2, 3])
  })
})

function oneHot16(i: number): number[] {
  const x = new Array<number>(16).fill(0)
  x[i] = 1
  return x
}
