import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { recordsOf } from '../exposition/index.ts'
import { linalg } from '../numerics/index.ts'
import { key, normal } from '../random/index.ts'
import { empiricalCovariance, ledoitWolf, oas } from '../stats/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  GraphicalLasso,
  graphicalLasso,
  graphicalLassoAlgorithm,
  graphicalLassoEstimator,
  graphicalLassoObjective,
} from './graphical-lasso.ts'
import { cStepAlgorithm, EllipticEnvelope, ellipticEnvelope, fastMcdAlgorithm, MinCovDet, minCovDet } from './robust.ts'
import { CovarianceEstimator, covarianceEstimator, mahalanobisDistances, squaredMahalanobis } from './shrinkage.ts'

const clean = toArray(normal(key(0), [60, 3])) as number[][]
const dirty = clean.map((r, i) => (i < 8 ? r.map((v) => v + 12) : r))

describe('covariance estimators', () => {
  it('empirical and shrinkage estimators give Mahalanobis scores, sessions and a likelihood', () => {
    for (const method of ['empirical', 'shrunk', 'ledoit-wolf', 'oas'] as const) {
      const est = covarianceEstimator({ method })
      const f = est.fit({ x: clean })
      const [rec] = recordsOf(f.exposition, 'anomalyScores')
      expect(rec?.scores.length).toBe(60)
      expect(Number.isFinite(f.score({ x: clean }))).toBe(true)
      if (method !== 'empirical') expect(f.params.shrinkage).toBeGreaterThan(0)
      const sess = est.session({ x: clean }, { mode: 'eager' })
      expect(sess.fittedAt(sess.t).params.location.length).toBe(3)
    }
    expect(new CovarianceEstimator().fit(clean).model.mahalanobis(clean).shape).toEqual([60])
    expect(
      mahalanobisDistances(
        [
          [0, 0],
          [3, 4],
        ],
        [0, 0],
        [
          [1, 0],
          [0, 1],
        ],
      ).shape,
    ).toEqual([2])
  })

  it('the MCD ignores gross outliers that pull the empirical mean and steps monotonically', () => {
    const emp = covarianceEstimator().fit({ x: dirty }).params.location
    const mcd = new MinCovDet().fit(dirty, undefined, { key: key(1) }).model
    expect(Math.hypot(...mcd.params.location)).toBeLessThan(0.6)
    expect(Math.hypot(...emp)).toBeGreaterThan(2)
    expect(mcd.params.support.slice(0, 8).every((s) => !s)).toBe(true)
    const scores = toFlat(mcd.mahalanobis(dirty))
    expect(Math.min(...scores.slice(0, 8))).toBeGreaterThan(Math.max(...scores.slice(8)))

    const alg = fastMcdAlgorithm(dirty, { nTrials: 10 })
    const run = runAlgorithm(alg, { key: key(1), maxSteps: 20 })
    expect(run.steps).toBeGreaterThanOrEqual(1)
    expect(alg.deserialise?.(alg.serialise?.(run.state) ?? null)).toEqual(run.state)

    const env = new EllipticEnvelope({ contamination: 8 / 60 }).fit(dirty, undefined, { key: key(1) })
    const preds = toFlat(env.predict(dirty))
    expect(preds.slice(0, 8)).toEqual(new Array(8).fill(-1))
    expect(
      ellipticEnvelope({ method: 'ledoit-wolf', contamination: 0.15 }).fit({ x: clean }).params.location.length,
    ).toBe(3)
    expect(minCovDet({ nTrials: 5 }).session({ x: dirty }, { key: key(1), mode: 'eager' }).t).toBeGreaterThanOrEqual(1)
  })

  it('the graphical lasso traces a path of partial correlations that shrinks with the penalty', () => {
    const m = new GraphicalLasso({ alpha: 0.05 }).fit(clean).model
    const [path] = recordsOf(m.exposition, 'regularisationPath')
    expect(path?.features.length).toBe(3)
    expect(path?.active?.[0]?.length ?? 1).toBeLessThanOrEqual(path?.active?.at(-1)?.length ?? 0)

    const alg = graphicalLassoAlgorithm(
      [
        [1, 0.4],
        [0.4, 1],
      ],
      0.1,
    )
    const out = runAlgorithm(alg, { maxSteps: 10 })
    expect(alg.deserialise?.(alg.serialise?.(out.state) ?? null)).toEqual(out.state)
    expect(graphicalLassoEstimator({ alpha: 0.05 }).session({ x: clean }, { mode: 'eager' }).t).toBeGreaterThanOrEqual(
      1,
    )
  })

  it('property: squared Mahalanobis distances under positive-definite precision are non-negative and zero at location', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -10, max: 10, noNaN: true }), { minLength: 2, maxLength: 2 }),
        fc.array(fc.double({ min: -10, max: 10, noNaN: true }), { minLength: 2, maxLength: 2 }),
        (loc, pt) => {
          const prec = [
            [2, 0.5],
            [0.5, 2],
          ]
          const [d0, d1] = squaredMahalanobis([loc, pt], loc, prec)
          expect(d0).toBeCloseTo(0, 12)
          expect(d1 ?? -1).toBeGreaterThanOrEqual(0)
        },
      ),
      { numRuns: 40 },
    )
  })
})

describe('the shrinkage estimators', () => {
  it('wrap the stats estimators and score the Gaussian log likelihood', () => {
    const emp = toArray(empiricalCovariance(clean)) as number[][]
    expect(covarianceEstimator().fit({ x: clean }).params.covariance).toEqual(emp)
    const lw = ledoitWolf(clean)
    const fitted = covarianceEstimator({ method: 'ledoit-wolf' }).fit({ x: clean })
    expect(fitted.params.shrinkage).toBe(lw.shrinkage)
    expect(fitted.params.covariance).toEqual(toArray(lw.covariance))
    expect(covarianceEstimator({ method: 'oas' }).fit({ x: clean }).params.shrinkage).toBe(oas(clean).shrinkage)
    // The mean log likelihood of N(0, I) data at the identity: -(d / 2) log(2 pi) - tr(S) / (2 d)... per row -0.5 (|x|^2 + d log 2 pi).
    const id = covarianceEstimator({ method: 'shrunk', shrinkage: 1, assumeCentered: true }).fit({ x: clean })
    const rows = clean.map((r) => r.reduce((a, v) => a + v * v, 0))
    const mu = rows.reduce((a, v) => a + v, 0) / rows.length
    const cov = id.params.covariance
    // Fully shrunk covariance is (tr(S) / d) I: the likelihood is that of a spherical Gaussian with variance mu / d.
    expect(cov[0]?.[0]).toBeCloseTo(mu / 3, 12)
    const v = mu / 3
    expect(id.score({ x: clean })).toBeCloseTo(-0.5 * (mu / v + 3 * Math.log(2 * Math.PI * v)), 10)
  })

  it('has Mahalanobis distances that are affine invariant', () => {
    const f = covarianceEstimator().fit({ x: clean })
    const scale = [2, 0.5, 3]
    const moved = clean.map((r) => r.map((v, j) => v * (scale[j] ?? 1) + 5))
    const g = covarianceEstimator().fit({ x: moved })
    const a = toFlat(f.mahalanobis(clean))
    const b = toFlat(g.mahalanobis(moved))
    for (let i = 0; i < a.length; i++) expect(b[i]).toBeCloseTo(a[i] ?? 0, 9)
  })
})

describe('FastMCD', () => {
  it('lowers the determinant at every concentration step', () => {
    const support = Array.from({ length: 32 }, (_, i) => i + 10)
    const logdets = [1, 2, 3, 4, 5].map(
      (n) => runAlgorithm(cStepAlgorithm(dirty, support), { maxSteps: n }).state.logdet,
    )
    for (let i = 1; i < logdets.length; i++) expect(logdets[i] ?? 0).toBeLessThanOrEqual((logdets[i - 1] ?? 0) + 1e-12)
    expect(logdets[0] ?? 0).toBeGreaterThan(logdets.at(-1) ?? 0)
  })

  it('is translation equivariant for a fixed key', () => {
    const a = minCovDet({ nTrials: 10 }).fit({ x: dirty }, { key: key(3) })
    const shifted = dirty.map((r) => r.map((v) => v + 7))
    const b = minCovDet({ nTrials: 10 }).fit({ x: shifted }, { key: key(3) })
    expect(b.params.support).toEqual(a.params.support)
    a.params.location.forEach((v, j) => {
      expect(b.params.location[j]).toBeCloseTo(v + 7, 9)
    })
    a.params.covariance.forEach((r, i) => {
      r.forEach((v, j) => {
        expect(b.params.covariance[i]?.[j]).toBeCloseTo(v, 9)
      })
    })
  })

  it('flags the planted outliers as the elliptic envelope and not as the empirical ellipse', () => {
    const robust = new EllipticEnvelope({ contamination: 8 / 60 }).fit(dirty, undefined, { key: key(1) })
    const empirical = new EllipticEnvelope({ contamination: 8 / 60, method: 'empirical' }).fit(dirty)
    const gap = (m: typeof robust): number => {
      const d = toFlat(m.model.mahalanobis(dirty))
      return Math.min(...d.slice(0, 8)) / Math.max(...d.slice(8))
    }
    // Masking: the outliers drag the empirical mean and inflate its covariance, so the robust ellipse separates more.
    expect(gap(robust)).toBeGreaterThan(gap(empirical))
    expect(toFlat(robust.predict(dirty)).filter((v) => v === -1).length).toBe(8)
  })
})

describe('the graphical lasso', () => {
  const s = [
    [1, 0.6, 0.2],
    [0.6, 1, 0.5],
    [0.2, 0.5, 1],
  ]

  it('is the inverse covariance with no penalty and a diagonal matrix with a large one', () => {
    const free = graphicalLasso(s, 0, { tol: 1e-12, enetTol: 1e-14, maxIter: 500 })
    const exact = toArray(linalg.inv(s)) as number[][]
    free.precision.forEach((r, i) => {
      r.forEach((v, j) => {
        expect(v).toBeCloseTo(exact[i]?.[j] ?? 0, 6)
      })
    })
    const sparse = graphicalLasso(s, 0.9)
    for (const [a, b] of [
      [0, 1],
      [1, 2],
      [0, 2],
    ] as const)
      expect(Math.abs(sparse.precision[a]?.[b] ?? 1)).toBe(0)
    // scikit-learn does not penalise the diagonal: the precision's diagonal is 1 / S_ii.
    expect(sparse.precision[0]?.[0]).toBeCloseTo(1, 10)
  })

  it('lowers the objective, closes the duality gap and serialises its state', () => {
    const alg = graphicalLassoAlgorithm(s, 0.1, { tol: 1e-9, enetTol: 1e-12 })
    let state = alg.init(key(0))
    let previous = state.objective
    for (let i = 0; i < 6; i++) {
      state = runAlgorithm(graphicalLassoAlgorithm(s, 0.1, { tol: 1e-9, enetTol: 1e-12 }), { maxSteps: i + 1 }).state
      expect(state.objective).toBeLessThanOrEqual(previous + 1e-9)
      previous = state.objective
    }
    expect(Math.abs(state.gap)).toBeLessThan(1e-6)
    expect(graphicalLassoObjective(s, state.precision, 0.1)).toBeCloseTo(state.objective, 12)
    expect(alg.deserialise?.(alg.serialise?.(state) ?? null)).toEqual(state)
  })

  it('has more edges as the penalty falls, and the estimator agrees with the function', () => {
    const edges = (a: number): number => {
      const p = graphicalLasso(s, a).precision
      return [p[0]?.[1], p[0]?.[2], p[1]?.[2]].filter((v) => Math.abs(v ?? 0) > 1e-8).length
    }
    // Edges only leave as the penalty rises, and none survive once it exceeds the largest |S_ij| (covariance screening).
    const counts = [0.01, 0.1, 0.3, 0.55, 0.61].map(edges)
    for (let i = 1; i < counts.length; i++) expect(counts[i] ?? 0).toBeLessThanOrEqual(counts[i - 1] ?? 0)
    expect(counts[0]).toBeGreaterThan(counts.at(-1) ?? 0)
    expect(counts.at(-1)).toBe(0)
    const x = clean
    const est = graphicalLassoEstimator({ alpha: 0.1, tol: 1e-8 }).fit({ x })
    const direct = graphicalLasso(toArray(empiricalCovariance(x)), 0.1, { tol: 1e-8, maxIter: 100 })
    expect(est.params.precision).toEqual(direct.precision)
  })
})
