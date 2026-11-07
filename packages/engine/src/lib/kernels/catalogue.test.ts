import { describe, expect, it } from 'vitest'
import { minEigenvalue } from '../../testing/kernels.ts'
import { grad, numericalGradient } from '../autodiff/index.ts'
import { sum, tensor, toFlat, type Value, zeros } from '../tensor/index.ts'
import {
  additiveChi2Kernel,
  arcCosineKernel,
  chi2Kernel,
  constantKernel,
  cosineKernel,
  cosineSimilarityKernel,
  exponentialKernel,
  gram,
  histogramIntersectionKernel,
  type Kernel,
  laplacianKernel,
  linearKernel,
  locallyPeriodicKernel,
  maternKernel,
  neuralTangentKernel,
  onDimensions,
  periodicKernel,
  polynomialKernel,
  powerKernel,
  productKernel,
  rationalQuadraticKernel,
  rbfKernel,
  scaleKernel,
  sigmoidKernel,
  skewedChi2Kernel,
  spectralMixtureKernel,
  sumKernel,
  theta,
  whiteNoiseKernel,
  withTheta,
} from './index.ts'

/** Deterministic points in [-2, 2]^d (a linear congruential sequence, so that the table is reproducible). */
function pointsOf(n: number, d: number, nonNegative = false): number[][] {
  let s = 1
  const next = (): number => {
    s = (s * 1664525 + 1013904223) % 4294967296
    return s / 4294967296
  }
  return Array.from({ length: n }, () => Array.from({ length: d }, () => (nonNegative ? next() * 2 : next() * 4 - 2)))
}

/** [name, kernel, the largest dimension in which it is guaranteed PSD (Infinity: always, 0: no guarantee), needs non-negative points]. */
const CATALOGUE: readonly (readonly [string, Kernel, number, boolean?])[] = [
  ['rbf', rbfKernel(), Infinity],
  ['matern 1/2', maternKernel({ nu: 0.5 }), Infinity],
  ['matern 3/2', maternKernel(), Infinity],
  ['matern 5/2', maternKernel({ nu: 2.5 }), Infinity],
  ['matern 3.3', maternKernel({ nu: 3.3 }), Infinity],
  ['matern inf', maternKernel({ nu: Number.POSITIVE_INFINITY }), Infinity],
  ['rational quadratic', rationalQuadraticKernel(), Infinity],
  ['exponential', exponentialKernel(), Infinity],
  ['gamma-exponential 1.5', exponentialKernel({ power: 1.5 }), Infinity],
  ['gamma-exponential 2', exponentialKernel({ power: 2 }), Infinity],
  ['laplacian', laplacianKernel(), Infinity],
  ['periodic', periodicKernel(), 1],
  ['locally periodic', locallyPeriodicKernel(), 1],
  ['cosine', cosineKernel(), 1],
  ['linear', linearKernel(), Infinity],
  ['polynomial', polynomialKernel(), Infinity],
  ['sigmoid', sigmoidKernel(), 0],
  ['cosine similarity', cosineSimilarityKernel(), Infinity],
  ['arc-cosine 0', arcCosineKernel({ order: 0 }), Infinity],
  ['arc-cosine 1', arcCosineKernel(), Infinity],
  ['arc-cosine 2 depth 2', arcCosineKernel({ order: 2, depth: 2 }), Infinity],
  ['ntk', neuralTangentKernel(), Infinity],
  ['nngp', neuralTangentKernel({ kind: 'nngp' }), Infinity],
  ['ntk erf', neuralTangentKernel({ activation: 'erf' }), Infinity],
  ['white', whiteNoiseKernel(), Infinity],
  ['constant', constantKernel(), Infinity],
  ['chi2', chi2Kernel(), Infinity, true],
  ['additive chi2', additiveChi2Kernel(), Infinity, true],
  ['skewed chi2', skewedChi2Kernel(), Infinity, true],
  ['histogram intersection', histogramIntersectionKernel(), Infinity, true],
  // The algebra: sums and products are as PSD as their weakest part, integer powers keep it, other powers lose it.
  ['rbf + periodic', sumKernel(rbfKernel(), periodicKernel()), 1],
  ['periodic * cosine', productKernel(periodicKernel(), cosineKernel()), 1],
  ['scaled periodic', scaleKernel(periodicKernel(), 2), 1],
  ['rbf + sigmoid', sumKernel(rbfKernel(), sigmoidKernel()), 0],
  ['rbf * linear', productKernel(rbfKernel(), linearKernel()), Infinity],
  ['linear squared', powerKernel(linearKernel(), 2), Infinity],
  ['periodic squared', powerKernel(periodicKernel(), 2), 1],
  ['rbf to the 1.5', powerKernel(rbfKernel(), 1.5), 0],
  ['periodic on one column', onDimensions(periodicKernel(), [1]), Infinity],
  ['periodic on two columns', onDimensions(periodicKernel(), [0, 1]), 0],
]

const SM = spectralMixtureKernel({
  weights: [0.5, 0.5],
  means: [
    [0.3, 0.1],
    [0.7, 0.2],
  ],
  scales: [
    [0.2, 0.2],
    [0.1, 0.3],
  ],
})

describe('the kernel catalogue', () => {
  it('is symmetric, agrees with its cross-covariance and its diagonal, and is PSD exactly where it says', () => {
    for (const [name, kernel, upTo, nonNegative] of CATALOGUE) {
      const guarantee = upTo === Infinity ? 'always' : 'conditional'
      expect([kernel.psd, kernel.psdDimension], `${name}: the declared flag`).toEqual([
        guarantee,
        upTo === Infinity || upTo === 0 ? undefined : upTo,
      ])
      const columns = ((kernel.config['dims'] ?? [0]) as readonly number[]).map((c) => c + 1)
      for (const d of [1, 2, 3].filter((v) => v >= Math.max(...columns))) {
        const x = pointsOf(8, d, nonNegative === true)
        const k = toFlat(gram(kernel, x))
        const n = 8
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++)
            expect(Math.abs((k[i * n + j] ?? 0) - (k[j * n + i] ?? 1)), `${name} d=${d}: symmetric`).toBe(0)
          expect(toFlat(kernel.diag(x))[i], `${name} d=${d}: diag`).toBeCloseTo(k[i * n + i] ?? Number.NaN, 10)
        }
        if (name !== 'white') {
          const cross = toFlat(kernel.k(x, x))
          expect(
            Math.max(...k.map((v, i) => Math.abs(v - (cross[i] ?? 0)))),
            `${name} d=${d}: gram is k(x, x)`,
          ).toBeLessThan(1e-9)
        }
        const trace = Array.from({ length: n }, (_, i) => k[i * n + i] ?? 0).reduce((a, b) => a + b, 0)
        const floor = minEigenvalue(gram(kernel, x) as Value)
        if (d <= upTo) expect(floor, `${name} d=${d}: PSD`).toBeGreaterThanOrEqual(-1e-9 * trace)
      }
    }
    const x = pointsOf(8, 2)
    expect(minEigenvalue(gram(SM, x) as Value)).toBeGreaterThanOrEqual(-1e-9 * 8)
  })
  it('the conditional kernels do fail where the flag stops guaranteeing them', () => {
    const negative = (kernel: Kernel, d: number): boolean =>
      minEigenvalue(gram(kernel, pointsOf(8, d)) as Value) < -1e-3
    expect([periodicKernel(), cosineKernel()].map((k) => negative(k, 2) || negative(k, 3))).toEqual([true, true])
    expect([1, 2, 3].map((d) => negative(sigmoidKernel(), d))).toContain(true)
    // The flag survives rebuilding the kernel from new hyperparameters.
    expect(periodicKernel().withParams(periodicKernel({ period: 2 }).params).psd).toBe('conditional')
  })
  it('has a finite gradient in the points and in theta at coincident points, equal to a central difference', () => {
    for (const [name, kernel, , nonNegative] of [...CATALOGUE, ['spectral mixture', SM, 1] as const]) {
      const p = pointsOf(3, 2, nonNegative === true) as [number[], number[], number[]]
      const x = tensor([p[0], p[0], p[1], p[2]])
      const objectives: [string, (v: Value) => Value, Value][] = [['x', (v) => sum(kernel.k(v)), x]]
      if (toFlat(theta(kernel)).length > 0)
        objectives.push(['theta', (t) => sum(withTheta(kernel, t).k(x)), theta(kernel)])
      for (const [what, f, at] of objectives) {
        const got = toFlat(grad(f)(at) as Value)
        const want = toFlat(numericalGradient(f, at))
        expect(got.every(Number.isFinite), `${name}: gradient in ${what} is finite`).toBe(true)
        got.forEach((g, i) => {
          expect(Math.abs(g - (want[i] ?? Number.NaN)), `${name}: d/d${what}[${i}]`).toBeLessThan(1e-5)
        })
      }
    }
  })
  it('gives [0, 0] for no points, [1, 1] for one, and a NaN only where the NaN point enters', () => {
    // Kernels that never read the coordinates (white noise, constant) have no NaN to show; the zero-order arc-cosine
    // kernel is 1 on the diagonal whatever the point is, so its own entry stays finite.
    const blind = new Set(['white', 'constant'])
    for (const [name, kernel, , nonNegative] of [...CATALOGUE, ['spectral mixture', SM, 1] as const]) {
      const d = 2
      expect(gram(kernel, zeros([0, d])).shape, `${name}: no points`).toEqual([0, 0])
      expect(gram(kernel, pointsOf(1, d, nonNegative === true)).shape, `${name}: one point`).toEqual([1, 1])
      const rows = pointsOf(3, d, nonNegative === true)
      const column = ((kernel.config['dims'] ?? [0]) as readonly number[])[0] ?? 0
      ;(rows[1] as number[])[column] = Number.NaN
      const k = toFlat(gram(kernel, rows))
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) {
          const touched = i === 1 || j === 1
          const poisoned = touched && !blind.has(name) && !(name === 'arc-cosine 0' && i === j)
          const v = k[i * 3 + j] ?? 0
          expect(poisoned ? Number.isNaN(v) : Number.isFinite(v), `${name}: entry (${i}, ${j}) is ${v}`).toBe(true)
        }
    }
  })
})
