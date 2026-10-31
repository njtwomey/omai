import { describe, expect, it } from 'vitest'
import { eigh } from '../numerics/linalg/index.ts'
import { foldIn, key, multivariateNormal, normal } from '../random/index.ts'
import { fromStorage, primalOf, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  equicorrelatedScale,
  gaussianKnockoffs,
  knockoffFilter,
  knockoffScaleAlgorithm,
  knockoffScales,
  knockoffStatistics,
  knockoffThreshold,
} from './knockoffs.ts'

const ar1 = (p: number, rho: number): number[][] =>
  Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => rho ** Math.abs(i - j)))
const minEig = (a: number[][]): number => {
  const p = a.length
  return Math.min(...toFlat(primalOf(eigh(fromStorage(Float64Array.from(a.flat()), [p, p], 'float64')).eigenvalues)))
}

/** Rows from N(0, sigma) via its Cholesky factor. */
function gaussianRows(k: ReturnType<typeof key>, sigma: number[][], n: number): number[][] {
  const p = sigma.length
  // Lower Cholesky by hand for the small test matrices.
  const l = Array.from({ length: p }, () => new Array<number>(p).fill(0))
  for (let i = 0; i < p; i++)
    for (let j = 0; j <= i; j++) {
      let s = sigma[i]?.[j] ?? 0
      for (let c = 0; c < j; c++) s -= (l[i]?.[c] ?? 0) * (l[j]?.[c] ?? 0)
      ;(l[i] as number[])[j] = i === j ? Math.sqrt(s) : s / (l[j]?.[j] ?? 1)
    }
  const z = toFlat(
    multivariateNormal(k, new Array<number>(p).fill(0), fromStorage(Float64Array.from(l.flat()), [p, p], 'float64'), n),
  )
  return Array.from({ length: n }, (_, i) => Array.from({ length: p }, (_, j) => z[i * p + j] ?? 0))
}

describe('knockoff scales', () => {
  it('equicorrelated scale is min(2 lambda_min, 1) on the correlation scale', () => {
    expect(
      equicorrelatedScale(
        [
          [1, 0.5],
          [0.5, 1],
        ],
        1,
      ),
    ).toEqual([1, 1].map((v) => expect.closeTo(v, 12)))
    const s = equicorrelatedScale(
      [
        [1, 0.9],
        [0.9, 1],
      ],
      1,
    )
    expect(s[0]).toBeCloseTo(0.2, 12)
    // Variances rescale the scale: s_j = sigma_j^2 * min(2 lambda_min(R), 1).
    const t = equicorrelatedScale(
      [
        [4, 0.9 * 2 * 3],
        [0.9 * 2 * 3, 9],
      ],
      1,
    )
    expect(t[0]).toBeCloseTo(0.2 * 4, 12)
    expect(t[1]).toBeCloseTo(0.2 * 9, 12)
  })

  it('the coordinate scale is feasible, no smaller than equicorrelated, and large for the independent features', () => {
    const sigma = [
      [1, 0.95, 0, 0],
      [0.95, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ]
    const eq = knockoffScales(sigma)
    const co = knockoffScales(sigma, 'coordinate')
    expect(eq[0]).toBeCloseTo(0.1 * 0.999, 3)
    expect(co.reduce((a, b) => a + b, 0)).toBeGreaterThan(eq.reduce((a, b) => a + b, 0) + 1)
    expect(co[2]).toBeGreaterThan(0.95)
    for (const [j, v] of co.entries()) expect(v).toBeLessThanOrEqual(1 + 1e-12 + 0 * j)
    // 2 Sigma - diag(s) is positive semidefinite.
    const m = sigma.map((row, i) => row.map((v, j) => 2 * v - (i === j ? (co[i] ?? 0) : 0)))
    expect(minEig(m)).toBeGreaterThan(-1e-9)
  })

  it('runs as an algorithm whose scale never decreases and whose state serialises', () => {
    const sigma = ar1(5, 0.6)
    const alg = knockoffScaleAlgorithm(sigma)
    const r = runAlgorithm(alg, { maxSteps: 10 })
    const total = (s: readonly number[]): number => s.reduce((a, b) => a + b, 0)
    expect(total(r.state.s)).toBeGreaterThanOrEqual(total(alg.init(key(0)).s) - 1e-12)
    const back = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(r.state))))
    expect(back?.s).toEqual(r.state.s)
  })

  it('accepts a vector scale', () => {
    expect(
      knockoffScales(
        [
          [1, 0],
          [0, 1],
        ],
        [0.3, 0.4],
      ),
    ).toEqual([0.3, 0.4])
  })
})

describe('Gaussian knockoffs', () => {
  it('has the exchangeability second moments: cov(X, X~) = Sigma - D and cov(X~) = Sigma', () => {
    const sigma = ar1(4, 0.5)
    const x = gaussianRows(key(1), sigma, 40000)
    const ko = gaussianKnockoffs(x, { sigma, key: key(2) })
    const xt = toFlat(ko.xTilde)
    const p = 4
    const n = 40000
    let worst = 0
    for (let a = 0; a < p; a++)
      for (let b = 0; b < p; b++) {
        let cxx = 0
        let cxt = 0
        let ctt = 0
        for (let i = 0; i < n; i++) {
          cxx += ((x[i]?.[a] ?? 0) * (x[i]?.[b] ?? 0)) / n
          cxt += ((x[i]?.[a] ?? 0) * (xt[i * p + b] ?? 0)) / n
          ctt += ((xt[i * p + a] ?? 0) * (xt[i * p + b] ?? 0)) / n
        }
        const s = sigma[a]?.[b] ?? 0
        worst = Math.max(
          worst,
          Math.abs(cxx - s),
          Math.abs(ctt - s),
          Math.abs(cxt - (s - (a === b ? (ko.s[a] ?? 0) : 0))),
        )
      }
    expect(worst).toBeLessThan(0.03)
    expect(ko.s.every((v) => v > 0 && v <= 1)).toBe(true)
  })

  it('is deterministic given the key and differs across keys', () => {
    const x = gaussianRows(key(3), ar1(3, 0.2), 50)
    const a = toFlat(gaussianKnockoffs(x, { key: key(4) }).xTilde)
    const b = toFlat(gaussianKnockoffs(x, { key: key(4) }).xTilde)
    const c = toFlat(gaussianKnockoffs(x, { key: key(5) }).xTilde)
    expect(a).toEqual(b)
    expect(a).not.toEqual(c)
  })

  it('rejects a scale of the wrong length', () => {
    expect(() =>
      gaussianKnockoffs(
        [
          [0, 1],
          [1, 0],
          [2, 2],
        ],
        { scale: [0.1] },
      ),
    ).toThrow()
  })
})

describe('knockoff threshold', () => {
  it('finds the smallest t with (offset + #negatives) / #positives at most q', () => {
    const w = [5, 4, 3, 2, 1, -0.5, 0.2, 6, 7, 8, 9, 10]
    expect(knockoffThreshold(w, 0.2)).toBe(0.2)
    expect(knockoffThreshold([1, -1, 2, -2, 3], 0.2)).toBe(Number.POSITIVE_INFINITY)
    // Knockoff+ is stricter than knockoff by the offset.
    const v = [3, 3, 3, 3, -3, 2, 2, -1]
    expect(knockoffThreshold(v, 0.2, false)).toBeLessThanOrEqual(knockoffThreshold(v, 0.2, true))
    expect(() => knockoffThreshold(w, 1.5)).toThrow()
  })
})

describe('knockoff statistics are sign-symmetric for null features', () => {
  it('W_j is positive about half the time when y is independent of X (all three statistics)', () => {
    const sigma = ar1(8, 0.3)
    for (const statistic of ['correlation', 'lcd', 'lsm'] as const) {
      let pos = 0
      let nonzero = 0
      for (let rep = 0; rep < 40; rep++) {
        const x = gaussianRows(key(100 + rep), sigma, 120)
        const ko = gaussianKnockoffs(x, { sigma, key: key(200 + rep) })
        const y = toFlat(normal(key(300 + rep), [120]))
        const w = knockoffStatistics(x, ko.xTilde, y, {
          statistic,
          ...(statistic === 'lcd' ? { alpha: 0.05 } : { nAlphas: 20 }),
        })
        for (const v of w)
          if (v !== 0) {
            nonzero++
            if (v > 0) pos++
          }
      }
      expect(nonzero).toBeGreaterThan(100)
      expect(Math.abs(pos / nonzero - 0.5)).toBeLessThan(0.08)
    }
  })
})

describe('FDR control on a planted sparse regression', () => {
  const p = 30
  const n = 300
  const sigma = ar1(p, 0.3)
  const support = new Set([2, 6, 11, 15, 20, 26])
  const amplitude = 5 / Math.sqrt(n)

  function replicate(rep: number, statistic: 'lcd' | 'lsm' | 'correlation', q: number): { fdp: number; power: number } {
    const x = gaussianRows(key(1000 + rep), sigma, n)
    const eps = toFlat(normal(key(5000 + rep), [n]))
    const y = x.map(
      (row, i) =>
        row.reduce((a, v, j) => a + (support.has(j) ? (j % 2 === 0 ? 1 : -1) * amplitude * v : 0), 0) + (eps[i] ?? 0),
    )
    const r = knockoffFilter(x, y, {
      sigma,
      q,
      statistic,
      key: foldIn(key(9000), rep),
      ...(statistic === 'lcd' ? { alpha: 0.06 } : { nAlphas: 25 }),
    })
    const falses = r.selected.filter((j) => !support.has(j)).length
    const trues = r.selected.filter((j) => support.has(j)).length
    return { fdp: falses / Math.max(1, r.selected.length), power: trues / support.size }
  }

  it('controls the FDR at the nominal level over 100 replicates (lasso coefficient difference, knockoff+)', () => {
    const q = 0.2
    let fdr = 0
    let power = 0
    const reps = 100
    for (let rep = 0; rep < reps; rep++) {
      const r = replicate(rep, 'lcd', q)
      fdr += r.fdp / reps
      power += r.power / reps
    }
    expect(fdr).toBeLessThanOrEqual(q + 0.03)
    // Not trivially conservative (selecting nothing would also control it): there is power and the FDR is not tiny.
    expect(power).toBeGreaterThan(0.55)
    expect(fdr).toBeGreaterThan(0.04)
  })

  it('also controls it with the signed-maximum statistic and with the correlation statistic', () => {
    for (const statistic of ['lsm', 'correlation'] as const) {
      let fdr = 0
      let power = 0
      const reps = 50
      for (let rep = 0; rep < reps; rep++) {
        const r = replicate(500 + rep, statistic, 0.2)
        fdr += r.fdp / reps
        power += r.power / reps
      }
      expect(fdr).toBeLessThanOrEqual(0.2 + 0.05)
      expect(power).toBeGreaterThan(0.3)
    }
  }, 60000)
})
