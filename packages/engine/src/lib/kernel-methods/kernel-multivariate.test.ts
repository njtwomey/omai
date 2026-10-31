/**
 * Tests for kernel CCA and kernel PLS: the planted shared signal found by kernel CCA where linear CCA sees none, the
 * agreement of the canonical correlation with the correlation of the variates, and kernel PLS equal to linear PLS for a
 * linear kernel and to ridge-like fits growing with the number of components.
 */

import { describe, expect, it } from 'vitest'
import { plsRegression } from '../decomposition/index.ts'
import { linearKernel, rbfKernel } from '../kernels/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { normals, uniformRows } from './_testdata.ts'
import { kernelCca, kernelPlsRegression } from './index.ts'

const correlation = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  const n = a.length
  const ma = Array.from(a).reduce((s, v) => s + v, 0) / n
  const mb = Array.from(b).reduce((s, v) => s + v, 0) / n
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    sab += ((a[i] ?? 0) - ma) * ((b[i] ?? 0) - mb)
    saa += ((a[i] ?? 0) - ma) ** 2
    sbb += ((b[i] ?? 0) - mb) ** 2
  }
  return sab / Math.sqrt(saa * sbb)
}

describe('kernel canonical correlation analysis', () => {
  const n = 120
  const z = uniformRows(n, 1, 101, -1, 1).map((r) => r[0] ?? 0)
  const e = normals(2 * n, 102)
  // The blocks share z through nonlinear, uncorrelated views: x = z, y = z^2 (zero linear correlation by symmetry).
  const x = z.map((v, i) => [v + 0.05 * (e[i] ?? 0)])
  const y = z.map((v, i) => [v * v + 0.05 * (e[n + i] ?? 0)])

  it('finds the nonlinear shared signal that linear CCA misses', () => {
    const linear = kernelCca(x, y, { kernelX: linearKernel(), kernelY: linearKernel(), regularisation: 1e-6 })
    const rbf = kernelCca(x, y, {
      kernelX: rbfKernel({ lengthScale: 0.6 }),
      kernelY: rbfKernel({ lengthScale: 0.3 }),
      regularisation: 0.05,
    })
    expect(linear.correlations[0] ?? 1).toBeLessThan(0.3)
    expect(rbf.correlations[0] ?? 0).toBeGreaterThan(0.9)
    const a = toFlat(rbf.transformX(x))
    const b = toFlat(rbf.transformY(y))
    expect(Math.abs(correlation(a, b))).toBeGreaterThan(0.9)
  })

  it('has a first canonical correlation equal to the correlation of the variates when the ridge vanishes', () => {
    const cca = kernelCca(x, y, {
      kernelX: rbfKernel({ lengthScale: 1 }),
      kernelY: rbfKernel({ lengthScale: 1 }),
      regularisation: 1e-9,
    })
    const r = Math.abs(correlation(toFlat(cca.transformX(x)), toFlat(cca.transformY(y))))
    expect(r).toBeCloseTo(cca.correlations[0] ?? 0, 3)
  })

  it('has regularised correlations that fall as the ridge grows, in [0, 1], and several components', () => {
    const small = kernelCca(x, y, { regularisation: 0.01, nComponents: 3 })
    const large = kernelCca(x, y, { regularisation: 10, nComponents: 3 })
    expect(small.correlations.length).toBe(3)
    for (const c of small.correlations) expect(c).toBeLessThanOrEqual(1 + 1e-9)
    expect(large.correlations[0] ?? 1).toBeLessThan(small.correlations[0] ?? 0)
    for (let i = 1; i < 3; i++)
      expect(small.correlations[i] ?? 0).toBeLessThanOrEqual((small.correlations[i - 1] ?? 0) + 1e-12)
  })

  it('refuses blocks of different length and a non-positive ridge', () => {
    expect(() => kernelCca(x, y.slice(1))).toThrow()
    expect(() => kernelCca(x, y, { regularisation: 0 })).toThrow()
  })
})

describe('kernel partial least squares', () => {
  const n = 60
  const x = uniformRows(n, 3, 111, -1, 1)
  const noise = normals(2 * n, 112)
  const y2 = x.map((r, i) => [
    (r[0] ?? 0) - 2 * (r[1] ?? 0) + 0.05 * (noise[i] ?? 0),
    (r[2] ?? 0) + (r[0] ?? 0) + 0.05 * (noise[n + i] ?? 0),
  ])
  const y1 = y2.map((r) => r[0] ?? 0)
  const q = [
    [0.2, -0.3, 0.5],
    [-0.8, 0.1, 0.9],
  ]

  for (const [label, y] of [
    ['one target', y1],
    ['two targets', y2],
  ] as const) {
    for (const nComponents of [1, 2, 3]) {
      it(`is linear PLS regression for the linear kernel (${label}, ${nComponents} components)`, () => {
        const kernelFit = kernelPlsRegression({ kernel: 'linear', gamma: 1, nComponents }).fit({ x, y })
        const linearFit = plsRegression.with({ nComponents, scale: false, tol: 1e-13, maxIter: 20000 }).fit({ x, y })
        const a = toFlat(kernelFit.predict(q))
        const linearPredict = linearFit.predict
        if (linearPredict === undefined) throw new Error('no predict')
        const b = toFlat(linearPredict(q))
        expect(b.length).toBe(a.length)
        for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i] ?? 0, 6)
      })
    }
  }

  it('fits better as components are added, each step of the tape lowering the remaining targets', () => {
    const fit = kernelPlsRegression({ kernel: 'rbf', gamma: 0.5, nComponents: 5 }).fit({ x, y: y1 })
    const residual = metricSeries(fit.run, 'solver/residual').values
    for (let i = 1; i < residual.length; i++) expect(residual[i] ?? 0).toBeLessThan((residual[i - 1] ?? 0) + 1e-9)
    const one = kernelPlsRegression({ kernel: 'rbf', gamma: 0.5, nComponents: 1 }).fit({ x, y: y1 })
    expect(fit.score({ x, y: y1 })).toBeGreaterThan(one.score({ x, y: y1 }))
  })

  it('gives the model with t components as fittedAt(t), and revives from its parameters', () => {
    const est = kernelPlsRegression({ kernel: 'rbf', gamma: 0.5, nComponents: 3 })
    const s = est.session({ x, y: y1 }, { mode: 'eager' })
    const two = s.fittedAt(2)
    const direct = kernelPlsRegression({ kernel: 'rbf', gamma: 0.5, nComponents: 2 }).fit({ x, y: y1 })
    expect(toFlat(two.predict(q))[0]).toBeCloseTo(toFlat(direct.predict(q))[0] ?? 0, 9)
    const fit = est.fit({ x, y: y1 })
    const back = est.revive(JSON.parse(JSON.stringify(fit.params)))
    expect(toFlat(back.predict(q))[1]).toBeCloseTo(toFlat(fit.predict(q))[1] ?? 0, 12)
  })

  it('refuses more components than rows', () => {
    expect(() => kernelPlsRegression({ nComponents: 100 }).fit({ x, y: y1 })).toThrow()
  })
})
