import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { analogyCorpus } from '../datasets/index.ts'
import { flatObjective } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { cooccurrence } from '../text/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { analogyAccuracy } from './evaluation.ts'
import { gloveAdaGradAlgorithm, gloveProblem, gloveSolvers, trainGlove } from './glove.ts'

const toy = [
  ['the', 'cat', 'sat', 'on', 'the', 'mat'],
  ['the', 'dog', 'sat', 'on', 'the', 'log'],
  ['a', 'cat', 'saw', 'a', 'dog'],
]

describe('the GloVe loss', () => {
  it('sums over the nonzero harmonic-weighted co-occurrences', () => {
    const p = gloveProblem(toy, { dimension: 3, window: 2 }, key(0))
    const m = cooccurrence(toy, { window: 2, weighting: 'harmonic' })
    const nonzero = m.rows.reduce((a, r) => a + r.filter((v) => v > 0).length, 0)
    expect(p.batches?.n).toBe(nonzero)
    expect(p.words).toEqual(m.words)
  })

  it('equals the weighted least-squares formula at the start', () => {
    const p = gloveProblem(toy, { dimension: 2, window: 2, xmax: 3, alpha: 0.75 }, key(1))
    const x = p.x0 as Float64Array
    const V = p.words.length
    let total = 0
    for (let e = 0; e < p.counts.length; e++) {
      const i = p.rows[e] as number
      const j = p.cols[e] as number
      const X = p.counts[e] as number
      let pred = (x[2 * V * 2 + i] ?? 0) + (x[2 * V * 2 + V + j] ?? 0)
      for (let c = 0; c < 2; c++) pred += (x[i * 2 + c] ?? 0) * (x[V * 2 + j * 2 + c] ?? 0)
      const f = X < 3 ? (X / 3) ** 0.75 : 1
      total += 0.5 * f * (pred - Math.log(X)) ** 2
    }
    expect(flatObjective(p.objective, x).value(x)).toBeCloseTo(total / p.counts.length, 12)
  })

  it('one AdaGrad update is a step along the autodiff gradient, scaled by the accumulator', () => {
    const p = gloveProblem(toy, { dimension: 3, window: 2 }, key(2))
    const alg = gloveAdaGradAlgorithm(p, { learningRate: 1e-7, epochs: 1 })
    const s0 = alg.init(key(0))
    const e = s0.order[0] as number
    const x0 = Float64Array.from(s0.x)
    const s1 = alg.step(s0, { t: 0, key: key(0) } as never)
    // The batch objective is the mean over one entry, i.e. that entry's loss.
    const { grad } = flatObjective((q) => p.objective(q, [e]), x0).valueAndGrad(x0)
    for (let i = 0; i < x0.length; i++) expect(((x0[i] ?? 0) - (s1.x[i] ?? 0)) / 1e-7).toBeCloseTo(grad[i] ?? 0, 4)
  })

  it('offers AdaGrad beside the generic solvers, which minimise the same loss', () => {
    expect(gloveSolvers.names[0]).toBe('gloveAdaGrad')
    const p = gloveProblem(toy, { dimension: 3, window: 2 }, key(3))
    const start = flatObjective(p.objective, p.x0).value(p.x0 as Float64Array)
    const finals = [
      { kind: 'lbfgs' as const, gradTol: 1e-12 },
      { kind: 'gloveAdaGrad' as const, epochs: 300, learningRate: 1 },
    ].map((solver) => trainGlove(toy, { dimension: 3, window: 2, solver, maxSteps: 12_000 }, key(3)).loss)
    for (const f of finals) expect(f).toBeLessThan(start * 0.2)
    // L-BFGS and a long AdaGrad run both fit this tiny problem almost exactly (the optimum is near zero)
    expect(Math.abs((finals[0] ?? 0) - (finals[1] ?? 0))).toBeLessThan(0.1 * start)
  })

  it('runs as a stepped algorithm that serialises', () => {
    const p = gloveProblem(toy, { dimension: 2, window: 1 }, key(4))
    const alg = gloveAdaGradAlgorithm(p, { epochs: 2, batch: 4 })
    const r = runAlgorithm(alg, { seed: 0, maxSteps: 10_000 })
    expect(r.steps).toBe(Math.ceil((p.batches?.n ?? 0) / 4) * 2)
    const back = alg.deserialise?.(alg.serialise?.(r.state) as never)
    expect(Array.from(back?.accumulated ?? [])).toEqual(Array.from(r.state.accumulated))
  })
})

describe('GloVe on the planted corpus', () => {
  it('recovers the planted analogies', () => {
    const { data, truth } = generate(analogyCorpus, {
      key: key(5),
      knobs: { concepts: 4, attributes: 3, contexts: 3, sentences: 2500, noise: 0.15 },
    })
    const sentences = (data.documents ?? []).map((d) => d.split(' '))
    const quads = truth.extra?.['analogies'] as [string, string, string, string][]
    const targets = truth.extra?.['targets'] as string[]
    const m = trainGlove(sentences, { dimension: 16, window: 4, solver: { kind: 'gloveAdaGrad', epochs: 25 } }, key(1))
    const acc = analogyAccuracy(m.vectors, quads, { candidates: targets })
    expect(acc.total).toBe(quads.length)
    expect(acc.accuracy).toBeGreaterThan(0.9)
  })
})
