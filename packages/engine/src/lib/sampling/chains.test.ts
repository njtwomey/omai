import { describe, expect, it } from 'vitest'
import { standardNormal } from '../../testing/sampling.ts'
import { exponential } from '../dists/index.ts'
import { foldIn, key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import {
  metricSeries,
  policies,
  recorder,
  runAlgorithm,
  runTree,
  tensorSeries,
  validateAgainstSchema,
} from '../trace/index.ts'
import {
  chainsFromDraws,
  constrainChains,
  discardWarmup,
  flattenChains,
  parameterDraws,
  sampleChain,
  sampleChains,
  samplerAlgorithm,
  samplerSchema,
  thin,
} from './chains.ts'
import { randomWalkMetropolis } from './metropolis.ts'
import { targetFromDistribution } from './target.ts'

const kernel = randomWalkMetropolis(standardNormal(2), { scale: 1 })

describe('samplerAlgorithm', () => {
  it('records per-iteration statistics and positions that conform to its schema', () => {
    const rec = recorder()
    runAlgorithm(samplerAlgorithm(kernel, [0, 0]), { key: key(0), maxSteps: 6, tracer: rec, policy: policies.teaching })
    const run = rec.run()
    expect(metricSeries(run, 'accepted').values.length).toBe(6)
    expect(tensorSeries(run, 'position').length).toBe(7)
    expect(validateAgainstSchema(run.events, samplerSchema(2))).toEqual([])
  })
})

describe('sampleChain', () => {
  it('discards warm-up, thins, and optionally keeps the warm-up draws', () => {
    const r = sampleChain(kernel, [0, 0], { key: key(1), numDraws: 10, numWarmup: 5, thin: 3, keepWarmup: true })
    expect(r.count).toBe(10)
    expect(r.draws.length).toBe(20)
    expect(r.warmup?.count).toBe(5)
    expect(r.stats.acceptanceRate?.length).toBe(10)
    expect(r.stats.accepted?.every((a) => a === 0 || a === 1)).toBe(true)
  })

  it('is reproducible from its key', () => {
    const a = sampleChain(kernel, [0, 0], { key: key(2), numDraws: 20 })
    const b = sampleChain(kernel, [0, 0], { key: key(2), numDraws: 20 })
    expect(Array.from(a.draws)).toEqual(Array.from(b.draws))
    expect(() => sampleChain(kernel, [0, 0], { key: key(2), numDraws: -1 })).toThrow()
    expect(() => sampleChain(kernel, [0, 0], { key: key(2), numDraws: 1, thin: 0 })).toThrow()
  })
})

describe('sampleChains', () => {
  it('runs chains with independent keys from shared, per-chain, functional or default starting points', () => {
    const shared = sampleChains(kernel, { key: key(3), numChains: 3, numDraws: 15, init: [0.5, 0.5] })
    expect(shared.draws.shape).toEqual([3, 15, 2])
    const d = shared.draws.data
    expect(d[0]).not.toBe(d[30])
    const each = sampleChains(kernel, {
      key: key(3),
      numChains: 2,
      numDraws: 1,
      init: [
        [5, 5],
        [-5, -5],
      ],
    })
    expect(Math.abs((each.draws.data[0] ?? 0) - 5)).toBeLessThan(5)
    expect(Math.abs((each.draws.data[2] ?? 0) + 5)).toBeLessThan(5)
    const fn = sampleChains(kernel, {
      key: key(3),
      numChains: 2,
      numDraws: 1,
      init: (_, c) => [c * 10, 0],
      numWarmup: 0,
    })
    expect(Math.abs((fn.draws.data[2] ?? 0) - 10)).toBeLessThan(5)
    const random = sampleChains(kernel, { key: key(4), numChains: 4, numDraws: 2 })
    expect(random.parameters).toEqual([{}, {}, {}, {}])
    expect(() => sampleChains(kernel, { key: key(3), init: [1, 2, 3], numDraws: 1 })).toThrow()
    expect(() => sampleChains(kernel, { key: key(3), numChains: 3, init: [[1, 2]], numDraws: 1 })).toThrow()
    expect(() => sampleChains(kernel, { key: key(3), numChains: 0 })).toThrow()
  })

  it('traces a parent run with one child run per chain', () => {
    const rec = recorder()
    sampleChains(kernel, { key: key(5), numChains: 2, numDraws: 4, numWarmup: 2, tracer: rec })
    const [root] = runTree(rec.run())
    expect(root?.children.length).toBe(2)
    expect(root?.run.name).toBe('sampleChains')
  })

  it('is reproducible and differs between keys', () => {
    const a = sampleChains(kernel, { key: key(6), numChains: 2, numDraws: 5 })
    const b = sampleChains(kernel, { key: key(6), numChains: 2, numDraws: 5 })
    const c = sampleChains(kernel, { key: foldIn(key(6), 1), numChains: 2, numDraws: 5 })
    expect(toArray(a.draws)).toEqual(toArray(b.draws))
    expect(toArray(a.draws)).not.toEqual(toArray(c.draws))
  })
})

describe('chain utilities', () => {
  const c = chainsFromDraws(
    [
      [
        [1, 10],
        [2, 20],
        [3, 30],
      ],
      [
        [4, 40],
        [5, 50],
        [6, 60],
      ],
    ],
    {
      names: ['a', 'b'],
      stats: {
        divergent: [
          [0, 1, 0],
          [0, 0, 0],
        ],
      },
    },
  )

  it('extracts coordinates and pools chains', () => {
    expect(toArray(parameterDraws(c, 'a'))).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ])
    expect(toArray(flattenChains(c))).toEqual([
      [1, 10],
      [2, 20],
      [3, 30],
      [4, 40],
      [5, 50],
      [6, 60],
    ])
    expect(() => parameterDraws(c, 'z')).toThrow()
    expect(() => parameterDraws(c, 2)).toThrow()
  })

  it('thins and drops warm-up draws, statistics included', () => {
    const t = thin(c, 2)
    expect(toArray(parameterDraws(t, 1))).toEqual([
      [10, 30],
      [40, 60],
    ])
    expect(toArray(t.stats.divergent ?? c.draws)).toEqual([
      [0, 0],
      [0, 0],
    ])
    const w = discardWarmup(c, 1)
    expect(toArray(w.stats.divergent ?? c.draws)).toEqual([
      [1, 0],
      [0, 0],
    ])
    expect(() => thin(c, 0)).toThrow()
    expect(() => discardWarmup(c, 4)).toThrow()
  })

  it('validates external draws and maps chains to the constrained space', () => {
    expect(() => chainsFromDraws([1, 2, 3])).toThrow()
    expect(() => chainsFromDraws([[1, 2]], { names: ['a', 'b'] })).toThrow()
    expect(() => chainsFromDraws([[1, 2]], { stats: { energy: [[1, 2, 3]] } })).toThrow()
    const t = targetFromDistribution(exponential(2))
    const k = constrainChains(chainsFromDraws([[0, 1]]), t)
    expect(toArray(k.draws)).toEqual([[[1], [Math.E]]])
    expect(k.names).toEqual(t.names)
  })
})
