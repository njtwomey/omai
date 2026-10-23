import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, tensor, toArray } from '../../tensor/index.ts'
import * as P from '../index.ts'

const vec = (x: unknown): number[] => toArray(decode(x) as Tensor) as number[]
const mat = (x: unknown): number[][] => toArray(decode(x) as Tensor) as number[][]

describe('Gaussian belief propagation against numpy', () => {
  for (const c of loadFixture('pgm/inference/gaussian_bp').cases) {
    it(c.name, () => {
      if (c.fn === 'tree') {
        const q = mat(c.inputs[0])
        const names = q.map((_, i) => `v${i}`)
        const field = P.gmrf(names, vec(c.inputs[1]), q)
        const observed = (c.params['observed'] as number[]).map((i) => names[i] ?? '')
        const x = vec(c.inputs[2])
        const evidence = Object.fromEntries(observed.map((v, i) => [v, x[i] ?? 0]))
        for (const schedule of ['flooding', 'sequential'] as const) {
          const r = P.gaussianBeliefPropagation(field, { evidence, schedule })
          expect(r.converged).toBe(true)
          const free = names.filter((v) => !observed.includes(v))
          expectMatches(
            tensor([...free.map((v) => r.means[v] ?? 0), ...free.map((v) => r.variances[v] ?? 0)]),
            c.output,
            c.rtol,
            c.atol,
            c.name,
          )
        }
        return
      }
      const q = mat(c.inputs[0])
      const names = q.map((_, i) => `v${i}`)
      const field = P.gmrfInformation(names, q, vec(c.inputs[1]))
      expect(P.isWalkSummable(field)).toBe(true)
      const r = P.gaussianBeliefPropagation(field, { tol: 1e-14 })
      expect(r.converged).toBe(true)
      expectMatches(tensor(names.map((v) => r.means[v] ?? 0)), c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('Gaussian belief propagation', () => {
  it('solves linear-Gaussian networks with evidence (exact means, loopy variances close)', () => {
    const g = P.linearGaussianBayesNet([
      { name: 'x1', parents: [], weights: [], intercept: 1, variance: 4 },
      { name: 'x2', parents: ['x1'], weights: [0.5], intercept: -5, variance: 4 },
      { name: 'x3', parents: ['x2'], weights: [-1], intercept: 4, variance: 3 },
      { name: 'x4', parents: ['x1', 'x3'], weights: [0.25, 2], intercept: 0, variance: 1 },
    ])
    const r = P.gaussianBeliefPropagation(g, { evidence: { x4: 1 }, damping: 0.3, maxIter: 2000 })
    expect(r.converged).toBe(true)
    const exact = P.linearGaussianPosterior(g, ['x1', 'x2', 'x3'], { x4: 1 })
    for (const [i, v] of ['x1', 'x2', 'x3'].entries()) {
      expect(r.means[v]).toBeCloseTo(exact.mean[i] ?? 0, 8)
      expect(Math.abs((r.variances[v] ?? 0) - (exact.covariance[i]?.[i] ?? 0))).toBeLessThan(
        0.5 * (exact.covariance[i]?.[i] ?? 1),
      )
    }
  })

  it('detects divergence on a field that is not walk-summable', () => {
    const q = [
      [1, 0.6, 0.6],
      [0.6, 1, 0.6],
      [0.6, 0.6, 1],
    ]
    const field = P.gmrfInformation(['a', 'b', 'c'], q, [1, 1, 1])
    expect(P.isWalkSummable(field)).toBe(false)
    const r = P.gaussianBeliefPropagation(field, { maxIter: 500 })
    expect(r.converged).toBe(false)
  })
})
