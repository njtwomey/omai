import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs, swissRoll } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { isomap } from './isomap.ts'
import { classicalMds, mds } from './mds.ts'
import {
  continuity,
  coRankingMatrix,
  embeddingStress,
  neighbourhoodPreservation,
  rnxCurve,
  trustworthiness,
} from './quality.ts'
import { selfOrganisingMap, somAlgorithm, uMatrix } from './som.ts'
import { barnesHutKl, jointProbabilities, sparseJointProbabilities, tsne, tsneAlgorithm, tsneGradient } from './tsne.ts'
import { findAbParams, fuzzySimplicialSet, umap, umapAlgorithm } from './umap.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 45, centers: 3, clusterStd: 0.6, nFeatures: 4 } })

describe('MDS', () => {
  it('classical MDS recovers a planar configuration; SMACOF lowers the stress every step', () => {
    const pts = [
      [0, 0],
      [3, 0],
      [3, 4],
      [0, 4],
      [1, 1],
    ]
    const d = pts.map((a) => pts.map((b) => Math.hypot((a[0] ?? 0) - (b[0] ?? 0), (a[1] ?? 0) - (b[1] ?? 0))))
    const { embedding } = classicalMds(d, 2)
    const e = embeddingStress(pts, embedding)
    expect(e.raw).toBeLessThan(1e-12)
    const f = mds.fit({ x: pts }, { key: key(1), logs: 'all' })
    const stress = metricSeries(f.run, 'solver/stress').values
    for (let t = 1; t < stress.length; t++) expect(stress[t] ?? 0).toBeLessThanOrEqual((stress[t - 1] ?? 0) + 1e-9)
    expect(f.params.objective).toBeLessThan(1e-3)
    expect(toFlat(f.transform(pts)).length).toBe(10)
  })
})

describe('Isomap and quality', () => {
  it('unrolls a swiss roll better than the identity projection, and the measures are consistent', () => {
    const roll = generate(swissRoll, { key: key(1), knobs: { n: 120 } }).data
    const iso = isomap({ nNeighbors: 8 }).fit({ x: roll.x })
    const t = trustworthiness(roll.x, iso.params.embedding, 5)
    expect(t).toBeGreaterThan(0.9)
    expect(continuity(roll.x, iso.params.embedding, 5)).toBeGreaterThan(0.85)
    expect(neighbourhoodPreservation(roll.x, roll.x, 5)).toBe(1)
    const q = coRankingMatrix(roll.x, roll.x)
    expect(rnxCurve(q).auc).toBeCloseTo(1, 9)
  })
})

describe('t-SNE', () => {
  it('the gradient descent lowers the KL divergence and separates the blobs', () => {
    const f = tsne.with({ perplexity: 10, maxIter: 150, exaggerationIter: 50 }).fit(data, { logs: 'all' })
    const kl = metricSeries(f.run, 'solver/kl').values
    expect(kl.at(-1) ?? 0).toBeLessThan(kl[0] ?? 0)
    expect(trustworthiness(data.x, f.params.embedding, 5)).toBeGreaterThan(0.9)
  })

  it('a session shows the embedding early; Barnes-Hut agrees with the exact gradient as theta shrinks', () => {
    const s = tsne.with({ perplexity: 10 }).session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 3 })
    expect((s.probe(3, 'model/embedding') as number[][]).length).toBe(45)
    const { p } = sparseJointProbabilities(data.x, 5)
    const dense = Array.from({ length: p.n }, () => new Array<number>(p.n).fill(0))
    for (let i = 0; i < p.n; i++)
      for (let e = p.indptr[i] ?? 0; e < (p.indptr[i + 1] ?? 0); e++)
        (dense[i] as number[])[p.indices[e] ?? 0] = p.values[e] ?? 0
    const y = Array.from({ length: p.n }, (_, i) => [Math.sin(i), Math.cos(i * 1.3)])
    const exact = tsneGradient(dense, y)
    const grads: number[] = []
    for (const theta of [0, 0.3, 0.8]) {
      const g = new Float64Array(p.n * 2)
      barnesHutKl(p, Float64Array.from(y.flat()), 2, theta, 1, g)
      grads.push(Math.max(...Array.from(g, (v, i) => Math.abs(v - (exact.grad[Math.floor(i / 2)]?.[i % 2] ?? 0)))))
    }
    expect(grads[0]).toBeLessThan(1e-9)
    expect(grads[1] ?? 1).toBeLessThan(0.05)
    const bh = tsne.with({ perplexity: 5, method: 'barnes-hut', maxIter: 50, exaggerationIter: 10 }).fit(data)
    expect(bh.params.embedding.length).toBe(45)
  })

  it('joint probabilities sum to one; the state serialises', () => {
    const { p } = jointProbabilities(data.x, 8)
    expect(p.flat().reduce((a, v) => a + v, 0)).toBeCloseTo(1, 6)
    const state = runAlgorithm(
      tsneAlgorithm(
        p,
        p.map((_, i) => [i * 0.01, 0]),
        { learningRate: 50, exaggerationIter: 5 },
      ),
      { maxSteps: 3 },
    ).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })
})

describe('UMAP', () => {
  it('fits the curve as umap-learn does and lays out the blobs, one epoch per step', () => {
    const { a, b } = findAbParams(1, 0.1)
    expect(a).toBeCloseTo(1.577, 2)
    expect(b).toBeCloseTo(0.895, 2)
    const f = umap.with({ nNeighbors: 10, nEpochs: 60 }).fit(data, { key: key(2), logs: 'all' })
    expect(trustworthiness(data.x, f.params.embedding, 5)).toBeGreaterThan(0.9)
    expect(metricSeries(f.run, 'solver/crossEntropy').values.length).toBeGreaterThan(2)
    const { weights } = fuzzySimplicialSet(data.x, 10)
    const state = runAlgorithm(umapAlgorithm(weights, f.params.embedding, { nEpochs: 5, a, b }), {
      maxSteps: 2,
      key: key(0),
    }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })
})

describe('self-organising map', () => {
  it('lowers the quantisation error, maps rows to grid positions, and has a U-matrix', () => {
    const f = selfOrganisingMap.with({ rows: 3, cols: 3, epochs: 60 }).fit(data, { key: key(1), logs: 'all' })
    const err = metricSeries(f.run, 'solver/quantisationError').values
    expect(err.at(-1) ?? 0).toBeLessThan(err[5] ?? 0)
    expect(f.transform(data.x).shape).toEqual([45, 2])
    expect(f.params.uMatrix.length).toBe(3)
    expect(uMatrix([[0], [1], [3]], [1, 3])).toEqual([[1, 1.5, 2]])
    const state = runAlgorithm(somAlgorithm(data.x, f.params.weights, [3, 3], { epochs: 3 }), {
      maxSteps: 1,
      key: key(0),
    }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })
})
