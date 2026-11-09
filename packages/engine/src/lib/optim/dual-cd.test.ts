import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import { type DualCdLoss, linearSvmDualCd, linearSvmDualCdAlgorithm } from './dual-cd.ts'
import { svcDual } from './smo.ts'

function blobs(seed: number, n: number, d: number, shift: number): { x: number[][]; y: number[] } {
  const z = toFlat(normal(key(seed), [n * d]))
  const y = Array.from({ length: n }, (_, i) => (i < n / 2 ? -1 : 1))
  return {
    x: y.map((label, i) =>
      Array.from({ length: d }, (_, j) => (z[i * d + j] ?? 0) + label * shift * (j === 0 ? 1 : 0.5)),
    ),
    y,
  }
}

describe('linear SVM by dual coordinate descent', () => {
  it('solves a one-dimensional separable problem exactly (margin 1 at x = 2 gives w = 1/2)', () => {
    const r = linearSvmDualCd([[2], [3], [-2], [-3]], [1, 1, -1, -1], 10, { fitIntercept: false, tol: 1e-12 })
    expect(r.coef[0]).toBeCloseTo(0.5, 9)
    expect(r.converged).toBe(true)
    expect(r.gap).toBeLessThan(1e-9)
  })

  it('the duality gap is nonnegative at every sweep and closes, for both losses', () => {
    const { x, y } = blobs(1, 80, 4, 0.5)
    for (const loss of ['l1', 'l2'] as const) {
      const alg = linearSvmDualCdAlgorithm(x, y, 1, { loss, tol: 1e-9 })
      const rec = recorder()
      const r = runAlgorithm(alg, { maxSteps: 5000, tracer: rec, key: key(2) })
      expect(r.state.converged).toBe(true)
      expect(r.state.primal - r.state.dual).toBeGreaterThanOrEqual(-1e-9)
      expect(r.state.primal - r.state.dual).toBeLessThan(1e-4 * Math.abs(r.state.primal))
      // The dual objective never decreases, sweep to sweep (coordinate ascent on the dual).
      const duals: number[] = []
      let s = alg.init(key(0))
      for (let t = 0; t < 30 && !s.converged; t++) {
        s = alg.step(s, {
          t,
          key: key(t),
          run: undefined as never,
          log: undefined as never,
          wants: undefined as never,
          emit: undefined as never,
          taps: undefined as never,
        })
        duals.push(s.dual)
      }
      for (let i = 1; i < duals.length; i++)
        expect(duals[i] as number).toBeGreaterThanOrEqual((duals[i - 1] as number) - 1e-9)
    }
  })

  it('agrees with SMO on the same problem: the symmetrised dataset has no intercept, so the primal objectives match', () => {
    // Duplicating every example as (-x, -y) leaves each margin y w.x unchanged and makes the optimal intercept zero, so SMO
    // (with its unregularised bias) on the 2n points solves the same problem as the intercept-free dual CD on the n points
    // with the penalty doubled.
    const { x, y } = blobs(3, 40, 3, 0.4)
    const x2 = [...x, ...x.map((r) => r.map((v) => -v))]
    const y2 = [...y, ...y.map((v) => -v)]
    const K = x2.map((a) => x2.map((b) => a.reduce((s, v, k) => s + v * (b[k] ?? 0), 0)))
    for (const loss of ['l1', 'l2'] as DualCdLoss[]) {
      const cSmo = 0.4
      const kernel = loss === 'l1' ? K : K.map((row, i) => row.map((v, j) => v + (i === j ? 1 / (2 * cSmo) : 0)))
      const smo = svcDual(kernel, y2, loss === 'l1' ? cSmo : 1e9, { tol: 1e-12 })
      const alpha = toFlat(smo.alpha)
      const w = [0, 0, 0]
      alpha.forEach((a, i) => {
        for (let j = 0; j < 3; j++) w[j] = (w[j] ?? 0) + a * (y2[i] ?? 0) * (x2[i]?.[j] ?? 0)
      })
      const cd = linearSvmDualCd(x, y, 2 * cSmo, { loss, fitIntercept: false, tol: 1e-9, maxSweeps: 100000 })
      expect(cd.converged).toBe(true)
      for (let j = 0; j < 3; j++) expect(cd.coef[j]).toBeCloseTo(w[j] ?? 0, 6)
      const primal = (ww: number[]): number => {
        const hinge = x.reduce(
          (a, r, i) =>
            a +
            Math.max(0, 1 - (y[i] ?? 0) * r.reduce((s, v, k) => s + v * (ww[k] ?? 0), 0)) ** (loss === 'l1' ? 1 : 2),
          0,
        )
        return 0.5 * ww.reduce((a, v) => a + v * v, 0) + 2 * cSmo * hinge
      }
      expect(primal(Array.from(cd.coef))).toBeCloseTo(primal(w), 8)
      expect(cd.primal).toBeCloseTo(primal(w), 6)
    }
  })

  it('shrinking changes the path, not the answer, and drops coordinates on a separable problem', () => {
    const { x, y } = blobs(4, 120, 3, 2.5)
    const a = linearSvmDualCd(x, y, 1, { tol: 1e-9, shrinking: true })
    const b = linearSvmDualCd(x, y, 1, { tol: 1e-9, shrinking: false })
    for (let j = 0; j < 3; j++) expect(a.coef[j]).toBeCloseTo(b.coef[j] ?? 0, 6)
    expect(a.intercept).toBeCloseTo(b.intercept, 6)
    // After a few sweeps most coordinates (the non-support vectors) have left the active set.
    const alg = linearSvmDualCdAlgorithm(x, y, 1, { tol: 1e-9 })
    const r = runAlgorithm(alg, { maxSteps: 6, key: key(1) })
    expect(r.state.active.length).toBeLessThan(120 * 0.5)
  })

  it('serialises: resuming from a checkpoint reproduces the uninterrupted run', () => {
    const { x, y } = blobs(5, 50, 3, 0.5)
    const alg = linearSvmDualCdAlgorithm(x, y, 1, { tol: 1e-8 })
    const full = runAlgorithm(alg, { maxSteps: 40, key: key(7) })
    const half = runAlgorithm(alg, { maxSteps: 5, key: key(7) })
    const restored = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(half.state))))
    const rest = runAlgorithm(alg, { maxSteps: 35, key: key(7), start: { state: restored, step: 5 } })
    expect(Array.from(rest.state.w)).toEqual(Array.from(full.state.w))
  })

  it('predicts the planted separator and rejects bad input', () => {
    const { x, y } = blobs(6, 200, 2, 1.5)
    const r = linearSvmDualCd(x, y, 1)
    const correct = x.filter(
      (row, i) =>
        (row[0] ?? 0) * (r.coef[0] ?? 0) + (row[1] ?? 0) * (r.coef[1] ?? 0) + r.intercept > 0 === (y[i] ?? 0) > 0,
    ).length
    expect(correct / 200).toBeGreaterThan(0.9)
    expect(() =>
      linearSvmDualCd(
        x,
        y.map((v) => v * 2 + (v > 0 ? 1 : 0)),
        1,
      ),
    ).not.toThrow()
    expect(() => linearSvmDualCd(x, y, 0)).toThrow()
    expect(() => linearSvmDualCd(x, new Array(200).fill(1), 1)).toThrow()
    expect(() => linearSvmDualCd(x, [1, -1], 1)).toThrow()
  })

  it('row weights multiply C: weight 2 on a row equals repeating it, weights must be positive', () => {
    const { x, y } = blobs(8, 30, 2, 1)
    const w = y.map((_, i) => (i % 3 === 0 ? 2 : 1))
    const weighted = linearSvmDualCd(x, y, 1, { weights: w, loss: 'l2', tol: 1e-10, maxSweeps: 5000 })
    const rep = x.flatMap((row, i) => (i % 3 === 0 ? [row, row] : [row]))
    const labels = y.flatMap((v, i) => (i % 3 === 0 ? [v, v] : [v]))
    const repeated = linearSvmDualCd(rep, labels, 1, { loss: 'l2', tol: 1e-10, maxSweeps: 5000 })
    for (const [j, v] of Array.from(weighted.coef).entries()) expect(v).toBeCloseTo(repeated.coef[j] ?? 0, 6)
    expect(weighted.intercept).toBeCloseTo(repeated.intercept, 6)
    expect(weighted.gap).toBeLessThan(1e-6)
    expect(() => linearSvmDualCd(x, y, 1, { weights: y.map(() => 0) })).toThrow(/positive/)
  })
})
