import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { gaussianGraph } from '../datasets/index.ts'
import { kFold } from '../learn/model-selection/index.ts'
import { key } from '../random/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { toArray } from '../tensor/index.ts'
import {
  gaussianLogLikelihood,
  graphicalLassoEbic,
  type PenaltySelector,
  selectGraphicalLassoPenalty,
} from './graph-selection.ts'

const planted = (seed: number, n: number, p: number, strength = 0.4): number[][] =>
  toArray(
    generate(gaussianGraph, { key: key(seed), knobs: { n, p, structure: 'random', edgeProbability: 0.3, strength } })
      .data.x,
  ) as number[][]

describe('the chosen penalty is the argmax (or argmin) of its criterion curve', () => {
  it('for cross-validation and EBIC, with ties to the smaller penalty, for any data and folds', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 10_000 }),
        fc.integer({ min: 40, max: 120 }),
        fc.integer({ min: 3, max: 6 }),
        fc.constantFrom<PenaltySelector>('cv', 'ebic'),
        fc.integer({ min: 2, max: 5 }),
        (seed, n, p, method, folds) => {
          const x = planted(seed, n, p)
          const out = selectGraphicalLassoPenalty(x, { method, cv: kFold(folds), alphas: 6 })
          const better = method === 'cv' ? Math.max(...out.criterion) : Math.min(...out.criterion)
          expect(out.criterion[out.bestIndex]).toBe(better)
          expect(out.alpha).toBe(out.alphas[out.bestIndex])
          // Ties go to the smaller penalty: no later (smaller) penalty has the same criterion.
          expect(out.criterion.slice(out.bestIndex + 1).includes(better)).toBe(false)
          expect([...out.alphas].sort((a, b) => b - a)).toEqual(out.alphas)
        },
      ),
      { numRuns: 15 },
    )
  })
})

describe('EBIC with gamma = 0 is the BIC', () => {
  it('equals -2 log-likelihood + edges log n, whatever the precision, and picks the lowest BIC', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 10_000 }), fc.integer({ min: 50, max: 150 }), (seed, n) => {
        const x = planted(seed, n, 5)
        const s = toArray(empiricalCovariance(x)) as number[][]
        const out = selectGraphicalLassoPenalty(x, { method: 'ebic', gamma: 0, alphas: 8 })
        const bic = out.precisions.map((precision) => {
          let edges = 0
          for (let a = 0; a < 5; a++) for (let b = a + 1; b < 5; b++) if (precision[a]?.[b] !== 0) edges++
          return -2 * n * gaussianLogLikelihood(s, precision) + edges * Math.log(n)
        })
        for (const [i, c] of out.criterion.entries()) expect(c).toBeCloseTo(bic[i] ?? 0, 7)
        expect(out.alpha).toBe(out.alphas[bic.indexOf(Math.min(...bic))])
        for (const precision of out.precisions)
          expect(graphicalLassoEbic(s, precision, n, 0)).toBeLessThanOrEqual(graphicalLassoEbic(s, precision, n, 0.5))
      }),
      { numRuns: 12 },
    )
  })
})

describe('StARS on a clean case', () => {
  it('starts at zero instability and the monotone curve never decreases; the raw curve rises as the penalty falls', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 1000 }), (seed) => {
        const x = planted(seed, 400, 6, 0.45)
        const out = selectGraphicalLassoPenalty(x, { method: 'stars', nSubsamples: 12 }, key(seed))
        expect(out.instability[0]).toBe(0)
        for (let i = 1; i < out.criterion.length; i++)
          expect(out.criterion[i] ?? 0).toBeGreaterThanOrEqual(out.criterion[i - 1] ?? 0)
        // Monotone-ish: the raw instability of the sparse third of the grid is below that of the dense two thirds' peak.
        const third = Math.floor(out.instability.length / 3)
        expect(Math.max(...out.instability.slice(0, third))).toBeLessThanOrEqual(
          Math.max(...out.instability.slice(third)),
        )
        expect(out.edges[0]).toBe(0)
        expect(out.edges.at(-1) ?? 0).toBeGreaterThanOrEqual(out.edges[0] ?? 0)
      }),
      { numRuns: 6 },
    )
  })

  it('every instability is a probability-scale quantity in [0, 0.5]', () => {
    const out = selectGraphicalLassoPenalty(planted(3, 200, 5), { method: 'stars', nSubsamples: 8 }, key(1))
    for (const v of out.instability) {
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThanOrEqual(0.5)
    }
  })
})
