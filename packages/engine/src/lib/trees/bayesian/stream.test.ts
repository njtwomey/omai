import { describe, expect, it } from 'vitest'
import { key, normal, uniform } from '../../random/index.ts'
import { add, mul, sliceAxis, sum, toFlat, type Value } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { growBayesTree } from './grow.ts'
import { normalGammaLeaf, regressionPrior } from './leaf.ts'
import { treePrior } from './prior.ts'
import { streamAlgorithm, streamInit, streamLeaves, streamTree, streamUpdate } from './stream.ts'
import { bayesData, leafStatistics } from './tree.ts'

const step = (n: number, k: number): { x: Value; y: Value } => {
  const x = uniform(key(k), [n], { low: 0, high: 1 })
  const clean = toFlat(x).map((v) => (v < 0.4 ? 0 : 3))
  return { x, y: add(clean, mul(normal(key(k + 1), [n]), 0.3)) }
}

const slices = (x: Value, y: Value, size: number): [Value, Value][] =>
  Array.from({ length: Math.ceil((x.shape[0] ?? 0) / size) }, (_, i) => [
    sliceAxis(x, 0, i * size, Math.min(x.shape[0] ?? 0, (i + 1) * size)),
    sliceAxis(y, 0, i * size, Math.min(y.shape[0] ?? 0, (i + 1) * size)),
  ])

describe('the streaming tree', () => {
  it('has the batch posterior after any sequence of batches, to rounding', () => {
    const { x, y } = step(50, 1)
    const model = normalGammaLeaf(regressionPrior(y))
    const batch = model.describe(toFlat(sum(model.statistics(y), 0)))
    for (const size of [1, 3, 7, 50]) {
      let state = streamInit(model, 1)
      for (const [bx, by] of slices(x, y, size)) state = streamUpdate(state, model, bx, by, { checkEvery: 1000 })
      const [leaf] = streamLeaves(state, model)
      expect(leaf?.n).toBe(50)
      expect(leaf?.value[0]).toBeCloseTo(batch.value[0] ?? 0, 12)
      expect(leaf?.sd).toBeCloseTo(batch.sd ?? 0, 12)
      expect(leaf?.precision).toBeCloseTo(batch.precision, 12)
    }
  })

  it('moves the mean by one over the precision of the evidence so far, as an adaptive step', () => {
    const { y } = step(30, 2)
    const model = normalGammaLeaf(regressionPrior(y))
    const prior = model.prior as { m0: number; kappa0: number }
    let mean = prior.m0
    const values = toFlat(y)
    values.forEach((v, t) => {
      const stepSize = 1 / (prior.kappa0 + t + 1)
      mean += stepSize * (v - mean)
    })
    const batch = model.describe(toFlat(sum(model.statistics(y), 0)))
    expect(mean).toBeCloseTo(batch.value[0] ?? 0, 12)
    expect(batch.stepSize).toBeCloseTo(1 / (prior.kappa0 + 30), 12)
  })

  it('splits when the evidence arrives, and its leaves hold the exact statistics of their rows', () => {
    const { x, y } = step(120, 3)
    const model = normalGammaLeaf(regressionPrior(y))
    let state = streamInit(model, 1)
    for (const [bx, by] of slices(x, y, 5)) state = streamUpdate(state, model, bx, by, { checkEvery: 10 })
    expect(state.splits.length).toBeGreaterThanOrEqual(1)
    const first = state.splits[0]
    expect(Math.abs((first?.threshold ?? 0) - 0.4)).toBeLessThan(0.12)
    const tree = streamTree(state)
    const data = bayesData(x, y, model)
    const batch = toFlat(leafStatistics(data, tree))
    const streamed = streamLeaves(state, model)
    expect(streamed.length).toBe(batch.length / 3)
    streamed.forEach((leaf, i) => {
      expect(leaf.n).toBe(batch[i * 3])
      const posterior = model.describe(batch.slice(i * 3, i * 3 + 3))
      expect(leaf.value[0]).toBeCloseTo(posterior.value[0] ?? 0, 10)
      expect(leaf.stepSize).toBeCloseTo(posterior.stepSize, 12)
    })
    // The rarely visited leaf has the larger step size.
    const small = streamed.reduce((a, b) => (a.n < b.n ? a : b))
    const large = streamed.reduce((a, b) => (a.n < b.n ? b : a))
    expect(small.stepSize).toBeGreaterThanOrEqual(large.stepSize)
  })

  it('asks once on the whole stream and then agrees with the batch grower', () => {
    const { x, y } = step(80, 4)
    const model = normalGammaLeaf(regressionPrior(y))
    const prior = treePrior({ maxDepth: 1 })
    const once = streamUpdate(streamInit(model, 1), model, x, y, { prior, checkEvery: 80 })
    const grown = growBayesTree(bayesData(x, y, model), { prior })
    expect(once.splits[0]?.threshold).toBe(grown.steps[0]?.threshold)
    expect(once.splits[0]?.gain).toBeCloseTo(grown.steps[0]?.gain ?? 0, 10)
  })

  it('runs as a stepped algorithm over the rows in order', () => {
    const { x, y } = step(60, 5)
    const data = bayesData(x, y, normalGammaLeaf(regressionPrior(y)))
    const run = runAlgorithm(streamAlgorithm(data, { batch: 6, checkEvery: 12 }), { maxSteps: 100 })
    expect(run.status).toBe('converged')
    expect(run.state.seen).toBe(60)
  })
})
