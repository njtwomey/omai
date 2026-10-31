import { describe, expect, it } from 'vitest'
import { lassoCoordinateDescent } from '../optim/sparse.ts'
import { foldIn, key, normal } from '../random/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import {
  forwardStepwiseInference,
  forwardStepwisePolyhedron,
  lassoPolyhedron,
  lassoSelectiveInference,
  polyhedralBounds,
  selectiveInference,
  truncatedNormalCdf,
} from './selective-inference.ts'

const Phi = (x: number): number => 0.5 * (1 + erf(x / Math.SQRT2))
function erf(x: number): number {
  // Abramowitz-Stegun 7.1.26 is not accurate enough; use a series/continued fraction via the complementary function.
  const t = 1 / (1 + 0.5 * Math.abs(x))
  const y =
    1 -
    t *
      Math.exp(
        -x * x -
          1.26551223 +
          t *
            (1.00002368 +
              t *
                (0.37409196 +
                  t *
                    (0.09678418 +
                      t *
                        (-0.18628806 +
                          t *
                            (0.27886807 +
                              t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
      )
  return x >= 0 ? y : -y
}

function design(seed: number, n: number, p: number): number[][] {
  const z = toFlat(normal(key(seed), [n * p]))
  return Array.from({ length: n }, (_, i) => Array.from({ length: p }, (_, j) => z[i * p + j] ?? 0))
}

/** The active set and signs of the lasso at y, by the library solver. */
function lassoSelection(x: number[][], y: number[], lambda: number): { active: number[]; signs: number[] } {
  const n = x.length
  const p = x[0]?.length ?? 0
  const fit = lassoCoordinateDescent(fromStorage(Float64Array.from(x.flat()), [n, p], 'float64'), y, lambda, {
    tol: 1e-14,
    maxIterations: 20000,
  })
  const c = toFlat(fit.coef)
  const active = c.flatMap((v, j) => (v !== 0 ? [j] : []))
  return { active, signs: active.map((j) => Math.sign(c[j] ?? 0)) }
}

describe('truncated normal distribution function', () => {
  it('matches the direct formula where it is well conditioned', () => {
    expect(truncatedNormalCdf(0, 0, 1, -Infinity, Infinity)).toBeCloseTo(0.5, 12)
    const direct = (Phi(1) - Phi(0)) / (Phi(2) - Phi(0))
    expect(truncatedNormalCdf(1, 0, 1, 0, 2)).toBeCloseTo(direct, 6)
    expect(truncatedNormalCdf(-0.5, 0, 2, -3, 1)).toBeCloseTo((Phi(-0.25) - Phi(-1.5)) / (Phi(0.5) - Phi(-1.5)), 6)
    expect(truncatedNormalCdf(5, 0, 1, 0, 2)).toBe(1)
    expect(truncatedNormalCdf(-5, 0, 1, 0, 2)).toBe(0)
    expect(() => truncatedNormalCdf(0, 0, 1, 2, 1)).toThrow()
  })

  it('stays accurate far in the tails, where the naive difference of CDFs is 0 / 0', () => {
    // Truncation to [10, 12]: the density decays like exp(-10 x), so the law is nearly exponential with rate 10.
    const f = truncatedNormalCdf(10.1, 0, 1, 10, 12)
    const expo = (1 - Math.exp(-1.0)) / (1 - Math.exp(-20))
    expect(Math.abs(f - expo)).toBeLessThan(0.02)
    const g = truncatedNormalCdf(-10.1, 0, 1, -12, -10)
    // The mirror image of the exponential law above: F(-10.1) = 1 - (1 - exp(-1)) / (1 - exp(-20)).
    expect(Math.abs(g - (1 - expo))).toBeLessThan(0.02)
    expect(truncatedNormalCdf(11, 0, 1, 10, 12)).toBeGreaterThan(truncatedNormalCdf(10.5, 0, 1, 10, 12))
    expect(Number.isFinite(truncatedNormalCdf(50, 0, 1, 40, Infinity))).toBe(true)
  })
})

describe('polyhedral lemma', () => {
  it('truncates along the contrast and leaves the orthogonal part alone', () => {
    const poly = { a: Float64Array.from([-1, 0, 1, 1]), b: Float64Array.from([-1, 6]), rows: 2, n: 2 }
    const b = polyhedralBounds(poly, [1, 0], [3, 2])
    // y_1 >= 1 and y_1 + y_2 <= 6 (y_2 = 2 fixed): 1 <= y_1 <= 4.
    expect(b.vMinus).toBeCloseTo(1, 12)
    expect(b.vPlus).toBeCloseTo(4, 12)
    expect(b.vZero).toBe(Number.POSITIVE_INFINITY)
  })

  it('rejects an observation outside the polyhedron', () => {
    const poly = { a: Float64Array.from([1]), b: Float64Array.from([0]), rows: 1, n: 1 }
    expect(() => selectiveInference(poly, [1], [2], { sigma: 1 })).toThrow()
  })

  it('p-value and interval are consistent: zero lies inside the interval exactly when p exceeds alpha', () => {
    const poly = { a: Float64Array.from([-1]), b: Float64Array.from([-1]), rows: 1, n: 1 }
    for (const yv of [1.1, 1.8, 2.5, 3.5]) {
      const r = selectiveInference(poly, [1], [yv], { sigma: 1 })
      expect(r.interval[0]).toBeLessThan(r.estimate)
      expect(r.interval[1]).toBeGreaterThan(r.estimate)
      expect(r.interval[0] <= 0 && 0 <= r.interval[1]).toBe(r.pValue > 0.05)
    }
  })
})

describe('lasso at a fixed penalty', () => {
  it('the polyhedron is exactly the set of y with the same active set and signs', () => {
    const n = 40
    const p = 6
    const x = design(1, n, p)
    const beta = [1.2, -0.9, 0, 0, 0, 0]
    const eps = toFlat(normal(key(2), [n]))
    const y = x.map((r, i) => r.reduce((a, v, j) => a + v * (beta[j] ?? 0), 0) + (eps[i] ?? 0))
    const lambda = 6
    const { active, signs } = lassoSelection(x, y, lambda)
    expect(active.length).toBeGreaterThan(0)
    const poly = lassoPolyhedron(x, active, signs, lambda)
    // The observed y satisfies A y <= b.
    for (let r = 0; r < poly.rows; r++) {
      let ay = 0
      for (let i = 0; i < n; i++) ay += (poly.a[r * n + i] ?? 0) * (y[i] ?? 0)
      expect(ay).toBeLessThanOrEqual((poly.b[r] ?? 0) + 1e-6)
    }
    // Along the contrast of the first active coefficient the selection changes exactly at the limits.
    const xe = x.map((row) => active.map((j) => row[j] ?? 0))
    const m = active.length
    const gram = Array.from({ length: m }, (_, a) =>
      Array.from({ length: m }, (_, b) => xe.reduce((s, row) => s + (row[a] ?? 0) * (row[b] ?? 0), 0)),
    )
    // eta = X_E (X_E'X_E)^-1 e_1, found by solving the small system with Gauss-Jordan.
    const eta = solveColumn(gram, xe)
    const bounds = polyhedralBounds(poly, eta, y)
    const ee = eta.reduce((a, v) => a + v * v, 0)
    const z = eta.reduce((a, v, i) => a + v * (y[i] ?? 0), 0)
    const shifted = (t: number): number[] => y.map((v, i) => v + ((t - z) * (eta[i] ?? 0)) / ee)
    const same = (t: number): boolean => {
      const s = lassoSelection(x, shifted(t), lambda)
      return s.active.join() === active.join() && s.signs.join() === signs.join()
    }
    expect(same(z)).toBe(true)
    const margin = 0.02
    if (Number.isFinite(bounds.vMinus)) {
      expect(same(bounds.vMinus + margin)).toBe(true)
      expect(same(bounds.vMinus - margin)).toBe(false)
    }
    if (Number.isFinite(bounds.vPlus)) {
      expect(same(bounds.vPlus - margin)).toBe(true)
      expect(same(bounds.vPlus + margin)).toBe(false)
    }
    expect(Number.isFinite(bounds.vMinus) || Number.isFinite(bounds.vPlus)).toBe(true)
  })

  const n = 100
  const p = 8
  const x = design(11, n, p)
  const lambda = 15

  it('selective p-values are uniform under the global null while the naive ones are far from uniform', () => {
    const selective: number[] = []
    const naive: number[] = []
    for (let rep = 0; rep < 700; rep++) {
      const y = Array.from(toFlat(normal(foldIn(key(21), rep), [n])))
      let r: ReturnType<typeof lassoSelectiveInference>
      try {
        r = lassoSelectiveInference(x, y, lambda, { sigma: 1 })
      } catch {
        continue
      }
      selective.push(...r.pValues)
      naive.push(...r.naivePValues)
    }
    expect(selective.length).toBeGreaterThan(300)
    const frac = (ps: number[], a: number): number => ps.filter((v) => v < a).length / ps.length
    expect(Math.abs(frac(selective, 0.05) - 0.05)).toBeLessThan(0.025)
    expect(Math.abs(frac(selective, 0.25) - 0.25)).toBeLessThan(0.06)
    expect(Math.abs(frac(selective, 0.5) - 0.5)).toBeLessThan(0.07)
    // Kolmogorov-Smirnov distance to the uniform law is within its 1% critical value.
    const sorted = [...selective].sort((a, b) => a - b)
    const ks = Math.max(
      ...sorted.map((v, i) => Math.max(Math.abs((i + 1) / sorted.length - v), Math.abs(i / sorted.length - v))),
    )
    expect(ks).toBeLessThan(1.63 / Math.sqrt(sorted.length))
    // The naive p-values reject far too often (the winner's curse).
    expect(frac(naive, 0.05)).toBeGreaterThan(0.3)
  })

  it('selective 90% intervals cover the selected-model coefficient at the nominal rate; naive intervals do not', () => {
    const beta = [1.0, -0.8, 0.6, 0, 0, 0, 0, 0]
    const mu = x.map((row) => row.reduce((a, v, j) => a + v * (beta[j] ?? 0), 0))
    let cover = 0
    let naiveCover = 0
    let total = 0
    for (let rep = 0; rep < 500; rep++) {
      const e = toFlat(normal(foldIn(key(31), rep), [n]))
      const y = mu.map((m, i) => m + (e[i] ?? 0))
      let r: ReturnType<typeof lassoSelectiveInference>
      try {
        r = lassoSelectiveInference(x, y, lambda, { sigma: 1, level: 0.9 })
      } catch {
        continue
      }
      // The target: the coefficient of each selected variable in the least-squares projection of mu on the selected set.
      const target = projectOnto(x, mu, r.variables)
      r.variables.forEach((_, k) => {
        total++
        const lo = r.intervals[k]?.[0] ?? 0
        const hi = r.intervals[k]?.[1] ?? 0
        if ((target[k] ?? 0) >= lo && (target[k] ?? 0) <= hi) cover++
        const nlo = r.naiveIntervals[k]?.[0] ?? 0
        const nhi = r.naiveIntervals[k]?.[1] ?? 0
        // The naive interval at level 0.9: rescale the 95% one by the ratio of critical values.
        const mid = r.estimates[k] ?? 0
        const half = ((nhi - nlo) / 2) * (1.6448536269514722 / 1.959963984540054)
        if ((target[k] ?? 0) >= mid - half && (target[k] ?? 0) <= mid + half) naiveCover++
      })
    }
    expect(total).toBeGreaterThan(500)
    expect(Math.abs(cover / total - 0.9)).toBeLessThan(0.04)
    expect(naiveCover / total).toBeLessThan(cover / total)
  }, 60000)

  it('estimates sigma from the full fit when it is not given and refuses to when n <= p + 1', () => {
    const y = Array.from(toFlat(normal(key(41), [n])))
    const a = lassoSelectiveInference(
      x,
      y.map((v, i) => v + 3 * (x[i]?.[0] ?? 0)),
      lambda,
    )
    expect(a.sigma).toBeGreaterThan(0.8)
    expect(a.sigma).toBeLessThan(1.25)
    expect(() =>
      lassoSelectiveInference(
        [
          [1, 0],
          [0, 1],
        ],
        [1, 2],
        0.1,
      ),
    ).toThrow()
  })
})

describe('forward stepwise', () => {
  const n = 80
  const p = 10
  const x = design(51, n, p)

  it('selects what an independent greedy search selects and the observed y is in the polyhedron', () => {
    const beta = [2, 0, -1.5, 0, 0, 0, 0, 0, 0, 0]
    const e = toFlat(normal(key(52), [n]))
    const y = x.map((r, i) => r.reduce((a, v, j) => a + v * (beta[j] ?? 0), 0) + (e[i] ?? 0))
    const path = forwardStepwisePolyhedron(x, y, 3)
    expect(path.variables.slice(0, 2).sort()).toEqual([0, 2])
    // An independent implementation: greedy by the reduction in residual sum of squares, with refits.
    expect(path.variables).toEqual(greedy(x, y, 3))
    for (let r = 0; r < path.polyhedron.rows; r++) {
      let ay = 0
      for (let i = 0; i < n; i++) ay += (path.polyhedron.a[r * n + i] ?? 0) * (y[i] ?? 0)
      expect(ay).toBeLessThanOrEqual(1e-9)
    }
  })

  it('selective p-values are uniform under the global null and the naive ones are not', () => {
    const selective: number[] = []
    const naive: number[] = []
    for (let rep = 0; rep < 500; rep++) {
      const y = Array.from(toFlat(normal(foldIn(key(61), rep), [n])))
      const r = forwardStepwiseInference(x, y, 2, { sigma: 1 })
      selective.push(...r.pValues)
      naive.push(...r.naivePValues)
    }
    const frac = (ps: number[], a: number): number => ps.filter((v) => v < a).length / ps.length
    expect(Math.abs(frac(selective, 0.05) - 0.05)).toBeLessThan(0.02)
    expect(Math.abs(frac(selective, 0.5) - 0.5)).toBeLessThan(0.05)
    expect(frac(naive, 0.05)).toBeGreaterThan(0.25)
  }, 60000)

  it('detects a real signal with a small selective p-value', () => {
    const e = toFlat(normal(key(71), [n]))
    const y = x.map((r, i) => 1.2 * (r[3] ?? 0) + (e[i] ?? 0))
    const r = forwardStepwiseInference(x, y, 1, { sigma: 1 })
    expect(r.variables).toEqual([3])
    expect(r.pValues[0]).toBeLessThan(0.01)
    expect(r.intervals[0]?.[0]).toBeGreaterThan(0)
  })

  it('rejects an impossible number of steps', () => {
    expect(() =>
      forwardStepwisePolyhedron(
        [
          [1, 2],
          [3, 4],
          [5, 7],
        ],
        [1, 2, 3],
        3,
      ),
    ).toThrow()
  })
})

/** The least-squares coefficients of mu on the columns `cols`. */
function projectOnto(x: number[][], mu: number[], cols: readonly number[]): number[] {
  const xe = x.map((row) => cols.map((j) => row[j] ?? 0))
  const m = cols.length
  const gram = Array.from({ length: m }, (_, a) =>
    Array.from({ length: m }, (_, b) => xe.reduce((s, row) => s + (row[a] ?? 0) * (row[b] ?? 0), 0)),
  )
  const rhs = Array.from({ length: m }, (_, a) => xe.reduce((s, row, i) => s + (row[a] ?? 0) * (mu[i] ?? 0), 0))
  return gaussSolve(gram, rhs)
}

/** X_E (X_E'X_E)^-1 e_1 for the first selected column. */
function solveColumn(gram: number[][], xe: number[][]): number[] {
  const m = gram.length
  const e1 = Array.from({ length: m }, (_, a) => (a === 0 ? 1 : 0))
  const w = gaussSolve(gram, e1)
  return xe.map((row) => row.reduce((s, v, a) => s + v * (w[a] ?? 0), 0))
}

function gaussSolve(a: number[][], b: number[]): number[] {
  const n = b.length
  const m = a.map((r, i) => [...r, b[i] ?? 0])
  for (let c = 0; c < n; c++) {
    let piv = c
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r]?.[c] ?? 0) > Math.abs(m[piv]?.[c] ?? 0)) piv = r
    ;[m[c], m[piv]] = [m[piv] as number[], m[c] as number[]]
    const d = m[c]?.[c] ?? 1
    m[c] = (m[c] as number[]).map((v) => v / d)
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = m[r]?.[c] ?? 0
      m[r] = (m[r] as number[]).map((v, j) => v - f * (m[c]?.[j] ?? 0))
    }
  }
  return m.map((r) => r[n] ?? 0)
}

/** Forward stepwise by refitting least squares on each candidate and taking the lowest residual sum of squares. */
function greedy(x: number[][], y: number[], steps: number): number[] {
  const chosen: number[] = []
  const p = x[0]?.length ?? 0
  for (let s = 0; s < steps; s++) {
    let best = -1
    let bestRss = Number.POSITIVE_INFINITY
    for (let j = 0; j < p; j++) {
      if (chosen.includes(j)) continue
      const cols = [...chosen, j]
      const beta = projectOnto(x, y, cols)
      const rss = x.reduce(
        (a, row, i) => a + ((y[i] ?? 0) - cols.reduce((t, c, k) => t + (row[c] ?? 0) * (beta[k] ?? 0), 0)) ** 2,
        0,
      )
      if (rss < bestRss) {
        bestRss = rss
        best = j
      }
    }
    chosen.push(best)
  }
  return chosen
}
