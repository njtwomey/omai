import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { grad } from '../../autodiff/index.ts'
import { minOf } from '../../base/index.ts'
import { key, normal } from '../../random/index.ts'
import { sum, type Tensor, type TensorLike, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as I from './index.ts'

/** A Gaussian blurring problem (the fixture's has width 0.03; the behaviour tests use the more ill-posed 0.06). */
function problem(n: number, noise: readonly number[], width = 0.06): { a: number[][]; x: number[]; b: number[] } {
  const t = Array.from({ length: n }, (_, i) => (i + 0.5) / n)
  const a = t.map((ti) => t.map((tj) => Math.exp(-((ti - tj) ** 2) / (2 * width ** 2)) / n))
  const x = t.map((ti) => Math.sin(2 * Math.PI * ti) + 0.5 * Math.exp(-((ti - 0.6) ** 2) / 0.005))
  const b = a.map((row, i) => row.reduce((s, v, j) => s + v * (x[j] ?? 0), 0) + (noise[i] ?? 0))
  return { a, x, b }
}

describe('regularisation against numpy and scipy', () => {
  for (const c of loadFixture('numerics/inverse/inverse').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params
      const data = I.spectralData(t[0] as Tensor, t[1] as Tensor)
      const actual =
        c.fn === 'tikhonov'
          ? I.tikhonov(data, p['lam'] as number)
          : c.fn === 'general'
            ? I.tikhonovGeneral(t[0] as Tensor, t[1] as Tensor, p['lam'] as number)
            : c.fn === 'generalSecond'
              ? I.tikhonovGeneral(t[0] as Tensor, t[1] as Tensor, p['lam'] as number, { l: I.secondDifference(40) })
              : c.fn === 'tsvd'
                ? I.tsvd(data, p['k'] as number)
                : c.fn === 'gcvFunction'
                  ? I.gcv(data, 40, { lambdas: t[2] as Tensor }).criterion
                  : tensor(I.regularise(I.lsqrAlgorithm(t[0] as Tensor, t[1] as Tensor), p['iterations'] as number).x)
      expectMatches(actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('regularisation behaviour', () => {
  // White noise: a deterministic sinusoid has almost no component on the small singular vectors.
  const noise = toFlat(normal(key(7), [40])).map((v) => 1e-3 * v)
  const { a, x, b } = problem(40, noise)
  const data = I.spectralData(a, b)
  const delta = Math.sqrt(noise.reduce((s, v) => s + v * v, 0))
  const error = (est: readonly number[]): number => Math.sqrt(est.reduce((s, v, i) => s + (v - (x[i] ?? 0)) ** 2, 0))

  it('the parameter-choice rules give errors far below the unregularised solution', () => {
    const naive = error(toFlat(I.tsvd(data, 40)))
    for (const lambda of [
      I.lCurve(data).lambda,
      I.gcv(data, 40).lambda,
      I.discrepancyPrinciple(data, delta),
      I.quasiOptimality(data).lambda,
    ]) {
      expect(error(toFlat(I.tikhonov(data, lambda)))).toBeLessThan(naive / 100)
    }
  })

  it('the L-curve corner matches the discrete curvature maximum', () => {
    const lambdas = toFlat(I.lambdaGrid(data, 200))
    const path = I.tikhonovPath(data, b, lambdas)
    const lr = path.map((q) => Math.log(q.residualNorm))
    const le = path.map((q) => Math.log(q.solutionNorm))
    let best = 1
    let bestK = Number.NEGATIVE_INFINITY
    for (let i = 1; i + 1 < lambdas.length; i++) {
      const dx = (lr[i + 1] ?? 0) - (lr[i - 1] ?? 0)
      const dy = (le[i + 1] ?? 0) - (le[i - 1] ?? 0)
      const ddx = (lr[i + 1] ?? 0) - 2 * (lr[i] ?? 0) + (lr[i - 1] ?? 0)
      const ddy = (le[i + 1] ?? 0) - 2 * (le[i] ?? 0) + (le[i - 1] ?? 0)
      // The grid runs from large to small lambda, which reverses the orientation of the curve.
      const k = -(2 * (dx * ddy - dy * ddx)) / (dx * dx + dy * dy) ** 1.5
      if (k > bestK) {
        bestK = k
        best = i
      }
    }
    const corner = I.lCurve(data, { lambdas }).lambda
    expect(Math.abs(Math.log10(corner) - Math.log10(lambdas[best] ?? 1))).toBeLessThan(0.3)
  })

  it('the parameter-choice rules and the path take the parameters as tensor-likes and return tensors (migration, D-009)', () => {
    const grid = toFlat(I.lambdaGrid(data, 12))
    const rules: [string, (l: TensorLike) => Tensor][] = [
      ['lCurve', (l) => I.lCurve(data, { lambdas: l }).criterion as Tensor],
      ['gcv', (l) => I.gcv(data, 40, { lambdas: l }).criterion as Tensor],
      ['quasiOptimality', (l) => I.quasiOptimality(data, { lambdas: l }).criterion as Tensor],
      ['tikhonovPath', (l) => I.tikhonovPath(data, b, l)[3]?.x as Tensor],
    ]
    for (const [name, f] of rules) {
      const expected = toFlat(f(grid))
      expect(toFlat(f(Float64Array.from(grid))), name).toEqual(expected)
      expect(toFlat(f(tensor(grid))), name).toEqual(expected)
      expect(() => f(tensor([grid, grid])), name).toThrow(expect.objectContaining({ code: 'shape' }))
      expect(() => grad((l: Value) => sum(f(l as TensorLike)))(tensor(grid)), name).toThrow(
        expect.objectContaining({ code: 'not-differentiable' }),
      )
    }
    expect(I.lambdaGrid(data, 12).shape).toEqual([12])
  })

  it('iterative methods semi-converge: the error falls and then rises', () => {
    for (const algorithm of [I.cglsAlgorithm(a, b, { xTrue: x }), I.lsqrAlgorithm(a, b, { xTrue: x })]) {
      const r = I.regularise(algorithm, 40)
      const best = minOf(r.errors)
      expect(best).toBeLessThan((r.errors[0] ?? 0) / 2)
      expect(r.errors.at(-1) ?? 0).toBeGreaterThan(best * 2)
      for (let i = 1; i < r.residualNorms.length; i++)
        expect(r.residualNorms[i] ?? 0).toBeLessThanOrEqual((r.residualNorms[i - 1] ?? 0) * (1 + 1e-9))
    }
    const lw = I.regularise(I.landweberAlgorithm(a, b, { xTrue: x }), 200)
    expect(lw.errors.at(-1) ?? 0).toBeLessThan(lw.errors[0] ?? 0)
    // CGLS and LSQR agree in exact arithmetic.
    const c1 = I.regularise(I.cglsAlgorithm(a, b), 6).x
    const c2 = I.regularise(I.lsqrAlgorithm(a, b), 6).x
    for (const [i, v] of c1.entries()) expect(v).toBeCloseTo(c2[i] ?? 0, 6)
  })
})
