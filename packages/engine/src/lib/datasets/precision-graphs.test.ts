import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { linalg } from '../numerics/index.ts'
import { key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { datasetRegistry, gaussianGraph, makeGaussianGraph } from './index.ts'

const identity = (a: number[][], b: number[][]): number[][] =>
  a.map((r, i) => b[0]?.map((_, j) => r.reduce((s, v, k) => s + v * (b[k]?.[j] ?? 0), 0) - (i === j ? 1 : 0)) ?? [])

describe('makeGaussianGraph', () => {
  for (const structure of ['band', 'hub', 'random'] as const) {
    it(`${structure}: the precision is positive definite, exactly sparse on the graph and inverse to the covariance`, () => {
      const g = makeGaussianGraph(key(3), 9, { structure, edgeProbability: 0.3, hubs: 2, bandwidth: 2, strength: 0.4 })
      expect(Math.min(...toFlat(linalg.eigvalsh(g.precision)))).toBeGreaterThan(0)
      for (let a = 0; a < 9; a++) {
        expect(g.covariance[a]?.[a]).toBeCloseTo(1, 12)
        for (let b = 0; b < 9; b++) {
          if (a !== b && g.adjacency[a]?.[b] !== true) expect(g.precision[a]?.[b]).toBe(0)
          else if (a !== b) expect(Math.abs(g.precision[a]?.[b] ?? 0)).toBeGreaterThan(0)
          expect(g.precision[a]?.[b]).toBeCloseTo(g.precision[b]?.[a] ?? 0, 14)
        }
      }
      for (const row of identity(g.covariance, g.precision))
        for (const v of row) expect(Math.abs(v)).toBeLessThan(1e-10)
    })
  }

  it('the partial correlations are the precision scaled by its diagonal, and the edge list matches the adjacency', () => {
    const g = makeGaussianGraph(key(5), 7, { structure: 'random', edgeProbability: 0.4 })
    for (const [a, b] of g.edges) {
      const partial = -(g.precision[a]?.[b] ?? 0) / Math.sqrt((g.precision[a]?.[a] ?? 1) * (g.precision[b]?.[b] ?? 1))
      expect(g.partialCorrelations[a]?.[b]).toBeCloseTo(partial, 12)
      expect(g.adjacency[a]?.[b]).toBe(true)
      expect(a).toBeLessThan(b)
    }
    expect(g.edges.length).toBe(g.adjacency.flat().filter(Boolean).length / 2)
  })

  it('the size parameters shape the graph: a chain, hubs and the complete graph', () => {
    expect(makeGaussianGraph(key(0), 6, { structure: 'band', bandwidth: 1 }).edges.length).toBe(5)
    expect(makeGaussianGraph(key(0), 6, { structure: 'band', bandwidth: 2 }).edges.length).toBe(9)
    expect(makeGaussianGraph(key(0), 8, { structure: 'hub', hubs: 2 }).edges.length).toBe(6)
    expect(makeGaussianGraph(key(0), 5, { structure: 'random', edgeProbability: 1 }).edges.length).toBe(10)
    expect(makeGaussianGraph(key(0), 5, { structure: 'random', edgeProbability: 0 }).edges.length).toBe(0)
    expect(() => makeGaussianGraph(key(0), 1)).toThrow()
  })
})

describe('gaussianGraph', () => {
  it('samples from the planted model: the sample covariance approaches it, and graphSeed alone fixes the graph', () => {
    const a = generate(gaussianGraph, { key: key(1), knobs: { n: 20000, p: 5, structure: 'band' } })
    const b = generate(gaussianGraph, { key: key(2), knobs: { n: 50, p: 5, structure: 'band' } })
    expect(a.truth.extra?.['edges']).toEqual(b.truth.extra?.['edges'])
    expect(a.truth.extra?.['precision']).toEqual(b.truth.extra?.['precision'])
    const cov = a.truth.extra?.['covariance'] as number[][]
    const x = toArray(a.data.x) as number[][]
    for (let i = 0; i < 5; i++)
      for (let j = 0; j < 5; j++) {
        const s = x.reduce((t, r) => t + (r[i] ?? 0) * (r[j] ?? 0), 0) / x.length
        expect(Math.abs(s - (cov[i]?.[j] ?? 0))).toBeLessThan(0.05)
      }
    expect(
      generate(gaussianGraph, { key: key(1), knobs: { n: 20, p: 5, graphSeed: 4 } }).truth.extra?.['edges'],
    ).not.toEqual(generate(gaussianGraph, { key: key(1), knobs: { n: 20, p: 5, graphSeed: 5 } }).truth.extra?.['edges'])
  })

  it('the marginal knob pushes the latent Gaussian through a monotone map and keeps the latent as the target', () => {
    const cubic = generate(gaussianGraph, { key: key(1), knobs: { n: 30, p: 3, marginal: 'cubic' } })
    const exp = generate(gaussianGraph, { key: key(1), knobs: { n: 30, p: 3, marginal: 'exp' } })
    const latent = toFlat(cubic.data.y)
    expect(toFlat(cubic.data.x).every((v, i) => Math.abs(v - (latent[i] ?? 0) ** 3) < 1e-12)).toBe(true)
    expect(toFlat(exp.data.x).every((v, i) => Math.abs(v - Math.exp(toFlat(exp.data.y)[i] ?? 0)) < 1e-12)).toBe(true)
    expect(cubic.truth.extra?.['marginal']).toBe('cubic')
  })

  it('is registered as a tabular synthetic generator', () => {
    const entry = datasetRegistry.entries.find((e) => e.name === 'gaussianGraph')
    expect(entry?.form).toBe('generator')
    expect(entry?.modality).toBe('tabular')
    expect(entry?.real).toBe(false)
  })
})
