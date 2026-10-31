import { describe, expect, it } from 'vitest'
import { gaussianNB, logisticRegression } from '../../testing/learn.ts'
import { solve } from '../numerics/linalg/index.ts'
import { foldIn, key, normal } from '../random/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import {
  classifierDensityRatio,
  kliep,
  kliepAlgorithm,
  kliepProblem,
  kliepSolvers,
  ulsif,
  ulsifLoo,
  ulsifProblem,
  ulsifSolvers,
} from './density-ratio.ts'

/** Training N(0, 1) and test N(shift, 1) in one dimension: the true ratio is exp(shift x - shift^2 / 2). */
function gaussianShift(seed: number, n: number, shift: number): { train: number[][]; test: number[][] } {
  const a = toFlat(normal(key(seed), [n]))
  const b = toFlat(normal(foldIn(key(seed), 1), [n]))
  return {
    train: Array.from(a, (v) => [v]),
    test: Array.from(b, (v) => [v + shift]),
  }
}

const truth = (x: number, shift: number): number => Math.exp(shift * x - (shift * shift) / 2)

/** Root mean squared error of estimated weights against the truth, relative to the root mean square of the truth. */
function relativeError(est: readonly number[], train: number[][], shift: number): number {
  // Within two standard deviations of the training density, where it has data to speak of.
  const keep = train.map(([x = 0], i) => ({ x, i })).filter(({ x }) => Math.abs(x) <= 2)
  const num = keep.reduce((a, { x, i }) => a + ((est[i] ?? 0) - truth(x, shift)) ** 2, 0)
  const den = keep.reduce((a, { x }) => a + truth(x, shift) ** 2, 0)
  return Math.sqrt(num / den)
}

/** Ignoring the shift entirely (every weight one) is the error to beat. */
const unweighted = (train: number[][], shift: number): number =>
  relativeError(
    train.map(() => 1),
    train,
    shift,
  )

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

describe('classifierDensityRatio', () => {
  it('recovers a planted Gaussian ratio with any base estimator', () => {
    const { train, test } = gaussianShift(1, 400, 1)
    for (const estimator of [undefined, logisticRegression({ C: 100 }), gaussianNB()]) {
      const r = classifierDensityRatio(train, test, estimator === undefined ? {} : { estimator })
      const w = Array.from(toFlat(r.trainWeights))
      expect(relativeError(w, train, 1), estimator?.name ?? 'lda').toBeLessThan(0.5 * unweighted(train, 1))
      expect(w.every((v) => v > 0)).toBe(true)
      expect(r.ess).toBeGreaterThan(1)
      expect(r.ess).toBeLessThanOrEqual(train.length)
    }
  })

  it('clips the probabilities and caps the weights, and applies the fitted ratio to new points', () => {
    const { train, test } = gaussianShift(2, 200, 2)
    const r = classifierDensityRatio(train, test, { estimator: logisticRegression({ C: 100 }), maxWeight: 5 })
    expect(Math.max(...toFlat(r.trainWeights))).toBeLessThanOrEqual(5)
    const at = toFlat(r.predictRatio([[-1], [0], [1]]))
    expect(at[0]).toBeLessThan(at[1] ?? 0)
    expect(at[1]).toBeLessThan(at[2] ?? 0)
  })
})

describe('KLIEP', () => {
  const { train, test } = gaussianShift(3, 300, 1)

  it('normalises the training weights to mean one and recovers the planted ratio', () => {
    const r = kliep(train, test, { sigma: 0.8 })
    const w = Array.from(toFlat(r.trainWeights))
    expect(mean(w)).toBeCloseTo(1, 9)
    expect(r.alpha.every((a) => a >= 0)).toBe(true)
    expect(relativeError(w, train, 1)).toBeLessThan(0.7 * unweighted(train, 1))
    expect(r.iterations).toBe(300)
    // L-BFGS on the unconstrained form stops on its gradient tolerance well inside the same budget.
    const lbfgs = kliep(train, test, { sigma: 0.8, solver: { kind: 'lbfgs', memory: 10, gradTol: 1e-8 } })
    expect(lbfgs.converged).toBe(true)
    expect(lbfgs.iterations).toBeLessThan(300)
    expect(lbfgs.objective).toBeGreaterThanOrEqual(r.objective - 1e-6)
  })

  it('reaches the same optimum with the projected gradient and with the generic smooth solvers', () => {
    const options = { sigma: 0.8, nCentres: 25, maxIterations: 3000 }
    const reference = kliep(train, test, options)
    const w0 = Array.from(toFlat(reference.trainWeights))
    for (const solver of [
      { kind: 'lbfgs', memory: 10, gradTol: 1e-9 },
      { kind: 'bfgs', damped: false, gradTol: 1e-9 },
      { kind: 'conjugateGradient', beta: 'polak-ribiere', restart: 0, gradTol: 1e-9 },
    ]) {
      const r = kliep(train, test, { ...options, solver })
      const w = Array.from(toFlat(r.trainWeights))
      expect(mean(w), solver.kind).toBeCloseTo(1, 9)
      // The same likelihood, and the same ratio at the training points.
      expect(r.objective, solver.kind).toBeCloseTo(reference.objective, 2)
      expect(
        Math.sqrt(mean(w.map((v, i) => (v - (w0[i] ?? 0)) ** 2))) / Math.sqrt(mean(w0.map((v) => v * v))),
        solver.kind,
      ).toBeLessThan(0.05)
    }
    expect(kliepSolvers.names).toEqual([
      'projectedGradient',
      ...['gradientDescent', 'conjugateGradient', 'newton', 'bfgs', 'lbfgs', 'coordinateDescent', 'gradient'],
    ])
  })

  it('is an objective over theta = log alpha whose minimiser normalises to the KLIEP solution', () => {
    const p = kliepProblem(train, test, { sigma: 0.8, nCentres: 10 })
    const f = p.objective
    const at0 = Number(toFlat(f(fromStorage(new Float64Array(10), [10], 'float64')))[0])
    // At theta = 0 (alpha uniform), the objective is log(sum b) - mean_j log(sum_l K(x'_j, c_l)).
    let sumB = 0
    for (const v of p.kernelMean) sumB += v
    let rows = 0
    for (let j = 0; j < test.length; j++) {
      let s = 0
      for (let l = 0; l < 10; l++) s += p.phiTest[j * 10 + l] ?? 0
      rows += Math.log(s) / test.length
    }
    expect(at0).toBeCloseTo(Math.log(sumB) - rows, 10)
    // Scale invariance: adding a constant to theta changes nothing.
    const shifted = Number(toFlat(f(fromStorage(new Float64Array(10).fill(2.5), [10], 'float64')))[0])
    expect(shifted).toBeCloseTo(at0, 10)
  })

  it('steps the chosen solver, logging its objective, and serialises the projected-gradient state', () => {
    const rec = recorder()
    const alg = kliepAlgorithm(train, test, { sigma: 0.8, nCentres: 20 })
    const run = runAlgorithm(alg, { maxSteps: 3, tracer: rec })
    expect(run.steps).toBe(3)
    const revived = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(run.state))))
    expect((revived as { iteration: number }).iteration).toBe(3)
    const lb = kliepAlgorithm(train, test, { sigma: 0.8, nCentres: 20, solver: { kind: 'lbfgs' } })
    expect(lb.name).toBeDefined()
    // Run under a tracer: every path the generic solver logs is in its contract, or the run would throw.
    expect(runAlgorithm(lb, { maxSteps: 2, tracer: recorder() }).steps).toBe(2)
    const ul = ulsif(train, test, {
      sigma: 0.8,
      lambda: 0.1,
      nCentres: 10,
      solver: { kind: 'lbfgs' },
      tracer: recorder(),
    })
    expect(ul.theta.length).toBe(10)
  })

  it('weights shift further from uniform as the shift grows (smaller effective sample size)', () => {
    const near = gaussianShift(4, 300, 0.5)
    const far = gaussianShift(4, 300, 1.5)
    expect(kliep(far.train, far.test, { sigma: 0.8 }).ess).toBeLessThan(
      kliep(near.train, near.test, { sigma: 0.8 }).ess,
    )
  })
})

describe('uLSIF', () => {
  const { train, test } = gaussianShift(5, 300, 1)

  it('recovers the planted ratio with the closed form and selects its width and penalty by leave-pair-out', () => {
    const r = ulsif(train, test, { nCentres: 60 })
    const w = Array.from(toFlat(r.trainWeights))
    expect(relativeError(w, train, 1)).toBeLessThan(0.7 * unweighted(train, 1))
    expect(Number.isFinite(r.looScore)).toBe(true)
    expect(w.every((v) => v >= 0)).toBe(true)
    expect(mean(w)).toBeGreaterThan(0.8)
    expect(mean(w)).toBeLessThan(1.2)
    // The selected pair has the lowest score of the grid.
    const fixed = ulsif(train, test, { nCentres: 60, sigma: r.sigma, lambda: r.lambda })
    expect(fixed.looScore).toBeCloseTo(r.looScore, 10)
    const worse = ulsif(train, test, { nCentres: 60, sigma: r.sigma, lambda: 1000 })
    expect(worse.looScore).toBeGreaterThan(r.looScore)
  })

  it('has the closed form as one mode of a slot whose generic solvers reach the same coefficients', () => {
    const options = { nCentres: 20, sigma: 0.8, lambda: 0.1 }
    const ref = ulsif(train, test, options)
    for (const solver of [
      { kind: 'newton', modification: 'cholesky', beta: 1e-3, gradTol: 1e-10 },
      { kind: 'lbfgs', memory: 10, gradTol: 1e-10 },
      { kind: 'conjugateGradient', beta: 'polak-ribiere', restart: 0, gradTol: 1e-10 },
    ]) {
      const r = ulsif(train, test, { ...options, solver, maxIterations: 2000 })
      for (const [i, t] of r.theta.entries()) expect(t, `${solver.kind} theta ${i}`).toBeCloseTo(ref.theta[i] ?? 0, 4)
    }
    expect(ulsifSolvers.names[0]).toBe('closedForm')
    // The closed form solves (H + lambda I) theta = h, and the objective's gradient vanishes there.
    const p = ulsifProblem(train, test, options)
    const b = p.h.length
    const A = Float64Array.from(p.H, (v, i) => (i % (b + 1) === 0 ? v + 0.1 : v))
    const theta = toFlat(solve(fromStorage(A, [b, b], 'float64'), fromStorage(p.h, [b], 'float64')))
    for (let i = 0; i < b; i++) {
      let row = 0
      for (let j = 0; j < b; j++) row += (A[i * b + j] ?? 0) * (theta[j] ?? 0)
      expect(row).toBeCloseTo(p.h[i] ?? 0, 10)
    }
  })

  it('bounds the relative ratio by 1 / alpha and shrinks the weights towards one', () => {
    const wide = gaussianShift(6, 300, 2)
    const plain = ulsif(wide.train, wide.test, { nCentres: 60, sigma: 0.8, lambda: 0.01 })
    const relative = ulsif(wide.train, wide.test, { nCentres: 60, sigma: 0.8, lambda: 0.01, alpha: 0.2 })
    const wp = Array.from(toFlat(plain.trainWeights))
    const wr = Array.from(toFlat(relative.trainWeights))
    expect(Math.max(...wr)).toBeLessThan(1.2 / 0.2)
    expect(Math.max(...wr)).toBeLessThan(Math.max(...wp))
    expect(relative.ess).toBeGreaterThan(plain.ess)
    expect(relative.alpha).toBe(0.2)
  })
})

describe('ulsifLoo', () => {
  it('equals the explicit leave-pair-out refits', () => {
    const b = 4
    const n = 7
    const z = toFlat(normal(key(9), [2, n, b]))
    const phiTr = Float64Array.from(z.slice(0, n * b), (v) => Math.exp(-0.5 * v * v))
    const phiTe = Float64Array.from(z.slice(n * b), (v) => Math.exp(-0.5 * (v - 0.5) ** 2))
    for (const [lambda, alpha] of [
      [0.1, 0],
      [0.5, 0],
      [0.1, 0.3],
    ] as const) {
      let total = 0
      for (let i = 0; i < n; i++) {
        const A = new Float64Array(b * b)
        const h = new Float64Array(b)
        for (let j = 0; j < n; j++) {
          if (j === i) continue
          for (let r = 0; r < b; r++) {
            h[r] = (h[r] ?? 0) + (phiTe[j * b + r] ?? 0) / (n - 1)
            for (let c = 0; c < b; c++) {
              A[r * b + c] =
                (A[r * b + c] ?? 0) +
                ((1 - alpha) * (phiTr[j * b + r] ?? 0) * (phiTr[j * b + c] ?? 0) +
                  alpha * (phiTe[j * b + r] ?? 0) * (phiTe[j * b + c] ?? 0)) /
                  (n - 1)
            }
          }
        }
        for (let r = 0; r < b; r++) A[r * b + r] = (A[r * b + r] ?? 0) + lambda
        const theta = toFlat(solve(fromStorage(A, [b, b], 'float64'), fromStorage(h, [b], 'float64'))).map((t) =>
          Math.max(0, t),
        )
        const fu = theta.reduce((a, t, r) => a + t * (phiTr[i * b + r] ?? 0), 0)
        const fv = theta.reduce((a, t, r) => a + t * (phiTe[i * b + r] ?? 0), 0)
        total += (0.5 * (1 - alpha) * fu * fu + 0.5 * alpha * fv * fv - fv) / n
      }
      expect(ulsifLoo(phiTr, phiTe, b, lambda, alpha), `lambda ${lambda}, alpha ${alpha}`).toBeCloseTo(total, 10)
    }
  })
})
