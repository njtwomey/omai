import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blockCovarianceData, factorCovarianceData, wishartCovarianceData } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { empiricalCovariance, ledoitWolf, oas } from '../stats/index.ts'
import { toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { symmetricEigen } from './_matrix.ts'
import {
  shrinkagePath,
  shrinkagePathAlgorithm,
  shrinkageTargetMatrix,
  TargetShrinkageEstimator,
  targetShrinkage,
  targetShrinkageAlgorithm,
  targetShrinkageEstimator,
} from './shrinkage-targets.ts'

const X = [
  [1, 2, 0.5],
  [3, 6, 1],
  [5, 4, -1],
  [2, 2, 0],
  [4, 5, 2],
  [0, 1, 1.5],
]

describe('the targets', () => {
  it('has the sample variances on the diagonal of the diagonal, constant-correlation and single-factor targets', () => {
    const s = toArray(empiricalCovariance(X)) as number[][]
    for (const t of ['diagonal', 'constant-correlation', 'single-factor'] as const) {
      const f = shrinkageTargetMatrix(X, t)
      for (let i = 0; i < 3; i++) expect(f[i]?.[i]).toBeCloseTo(s[i]?.[i] ?? 0, 12)
    }
    expect(shrinkageTargetMatrix(X, 'diagonal')[0]?.[1]).toBe(0)
  })

  it('is a scaled identity with the mean variance for the identity target', () => {
    const s = toArray(empiricalCovariance(X)) as number[][]
    const mu = ((s[0]?.[0] ?? 0) + (s[1]?.[1] ?? 0) + (s[2]?.[2] ?? 0)) / 3
    expect(shrinkageTargetMatrix(X, 'identity')).toEqual([
      [mu, 0, 0],
      [0, mu, 0],
      [0, 0, mu],
    ])
  })

  it('has one correlation off the diagonal for the constant-correlation target', () => {
    const f = shrinkageTargetMatrix(X, 'constant-correlation')
    const r = (i: number, j: number): number => (f[i]?.[j] ?? 0) / Math.sqrt((f[i]?.[i] ?? 1) * (f[j]?.[j] ?? 1))
    expect(r(0, 1)).toBeCloseTo(r(0, 2), 12)
    expect(r(0, 1)).toBeCloseTo(r(1, 2), 12)
  })

  it('has rank-one off-diagonal covariances through the market for the single-factor target', () => {
    const f = shrinkageTargetMatrix(X, 'single-factor')
    // f_ij f_kl = f_il f_kj for distinct indices of a rank-one off-diagonal part.
    const four = shrinkageTargetMatrix(
      [
        [1, 2, 0.5, 1],
        [3, 6, 1, 0],
        [5, 4, -1, 2],
        [2, 2, 0, -1],
        [4, 5, 2, 3],
        [0, 1, 1.5, 1],
      ],
      'single-factor',
    )
    expect((four[0]?.[1] ?? 0) * (four[2]?.[3] ?? 0)).toBeCloseTo((four[0]?.[3] ?? 0) * (four[2]?.[1] ?? 0), 12)
    expect(f).toHaveLength(3)
  })
})

describe('targetShrinkage', () => {
  it('is the sample covariance at intensity 0 and the target at intensity 1, and linear between', () => {
    const s = toArray(empiricalCovariance(X)) as number[][]
    for (const target of ['identity', 'diagonal', 'constant-correlation', 'single-factor'] as const) {
      const f = shrinkageTargetMatrix(X, target)
      expect(targetShrinkage(X, { target, intensity: 0 }).covariance).toEqual(s)
      expect(targetShrinkage(X, { target, intensity: 1 }).covariance).toEqual(f)
      const half = targetShrinkage(X, { target, intensity: 0.25 }).covariance
      expect(half[0]?.[1]).toBeCloseTo(0.75 * (s[0]?.[1] ?? 0) + 0.25 * (f[0]?.[1] ?? 0), 12)
    }
  })

  it('reproduces the Ledoit-Wolf and OAS intensities of scikit-learn for the identity target', () => {
    expect(targetShrinkage(X, { target: 'identity', intensity: 'analytic' }).shrinkage).toBeCloseTo(
      ledoitWolf(X).shrinkage,
      12,
    )
    expect(targetShrinkage(X, { target: 'identity', intensity: 'oas' }).shrinkage).toBeCloseTo(oas(X).shrinkage, 12)
    const rb = targetShrinkage(X, { target: 'identity', intensity: 'rblw' }).shrinkage
    expect(rb).toBeGreaterThan(0)
    expect(rb).toBeLessThanOrEqual(1)
  })

  it('gives an analytic intensity in [0, 1] and a positive semidefinite covariance for every target', () => {
    for (const seed of [1, 2, 3]) {
      const { data } = generate(factorCovarianceData, { key: key(seed), knobs: { n: 30, p: 12, factors: 2 } })
      for (const target of ['identity', 'diagonal', 'constant-correlation', 'single-factor'] as const) {
        const r = targetShrinkage(data.x, { target })
        expect(r.shrinkage).toBeGreaterThanOrEqual(0)
        expect(r.shrinkage).toBeLessThanOrEqual(1)
        expect(symmetricEigen(r.covariance).values[11]).toBeGreaterThan(-1e-10)
      }
    }
  })

  it('makes a singular sample covariance (more variables than rows) positive definite', () => {
    const { data } = generate(factorCovarianceData, { key: key(4), knobs: { n: 15, p: 25, factors: 2 } })
    for (const target of ['identity', 'diagonal', 'constant-correlation'] as const) {
      const r = targetShrinkage(data.x, { target })
      expect(r.shrinkage).toBeGreaterThan(0)
      expect(symmetricEigen(r.covariance).values[24]).toBeGreaterThan(1e-8)
    }
  })

  it('shrinks all the way to a target that is right and not at all to one that is far from the truth', () => {
    const equi = generate(blockCovarianceData, {
      key: key(5),
      knobs: { n: 4000, p: 10, blocks: 1, within: 0.5, between: 0 },
    }).data.x
    expect(targetShrinkage(equi, { target: 'constant-correlation' }).shrinkage).toBeGreaterThan(0.8)
    const blocks = generate(blockCovarianceData, {
      key: key(5),
      knobs: { n: 20000, p: 10, blocks: 2, within: 0.9, between: 0 },
    }).data.x
    expect(targetShrinkage(blocks, { target: 'constant-correlation' }).shrinkage).toBeLessThan(0.05)
    expect(targetShrinkage(blocks, { target: 'diagonal' }).shrinkage).toBeLessThan(0.05)
    const white = generate(factorCovarianceData, {
      key: key(6),
      knobs: { n: 600, p: 10, factors: 1, strength: 0.1, uniqueness: 1 },
    }).data.x
    expect(targetShrinkage(white, { target: 'diagonal' }).shrinkage).toBeGreaterThan(0.5)
  })

  it('shrinks towards the market with a high intensity when one factor with the same sign drives every variable', () => {
    const { data } = generate(blockCovarianceData, {
      key: key(7),
      knobs: { n: 80, p: 30, blocks: 1, within: 0.4, between: 0 },
    })
    expect(targetShrinkage(data.x, { target: 'single-factor' }).shrinkage).toBeGreaterThan(0.5)
  })

  it('rejects an intensity outside [0, 1], OAS for a non-identity target and a market without variance', () => {
    expect(() => targetShrinkage(X, { intensity: 1.5 })).toThrow(/intensity/)
    expect(() => targetShrinkage(X, { target: 'diagonal', intensity: 'oas' })).toThrow(/identity target only/)
    expect(() =>
      targetShrinkage(
        [
          [1, -1],
          [-1, 1],
          [2, -2],
        ],
        { target: 'single-factor' },
      ),
    ).toThrow(/market/)
  })

  it('improves the held-out likelihood over the sample covariance when the sample is short', () => {
    const train = generate(factorCovarianceData, { key: key(8), knobs: { n: 40, p: 25, factors: 2 } }).data.x
    const test = generate(factorCovarianceData, { key: key(8), knobs: { n: 800, p: 25, factors: 2 } }).data.x
    const path = shrinkagePath(train, {
      target: 'constant-correlation',
      intensities: [0, targetShrinkage(train).shrinkage, 1],
      heldOut: test,
    })
    expect(path[1]?.heldOut).toBeGreaterThan(path[0]?.heldOut ?? Number.POSITIVE_INFINITY)
  })
})

describe('shrinkagePath', () => {
  const { data } = generate(wishartCovarianceData, { key: key(9), knobs: { n: 60, p: 20, df: 25 } })
  const test = generate(wishartCovarianceData, { key: key(9), knobs: { n: 1000, p: 20, df: 25 } }).data.x

  it('has the sample condition number at 0 and falls monotonically towards the identity target', () => {
    const path = shrinkagePath(data.x, { target: 'identity', heldOut: test })
    expect(path).toHaveLength(21)
    expect(path[0]?.intensity).toBe(0)
    expect(path[20]?.intensity).toBe(1)
    for (let i = 1; i < path.length; i++)
      expect(path[i]?.conditionNumber).toBeLessThanOrEqual((path[i - 1]?.conditionNumber ?? 0) + 1e-9)
    expect(path[20]?.conditionNumber).toBeCloseTo(1, 10)
    expect(path[20]?.spread).toBeCloseTo(0, 10)
    expect(path[0]?.spread).toBeGreaterThan(path[10]?.spread ?? 0)
  })

  it('has a held-out likelihood that rises and then falls again, peaking inside (0, 1)', () => {
    const path = shrinkagePath(data.x, { target: 'identity', heldOut: test })
    const scores = path.map((p) => p.heldOut)
    const best = scores.indexOf(Math.max(...scores))
    expect(best).toBeGreaterThan(0)
    expect(best).toBeLessThan(20)
    expect(scores[best]).toBeGreaterThan(scores[0] ?? 0)
    expect(scores[best]).toBeGreaterThan(scores[20] ?? 0)
  })

  it('cross-validates the likelihood without held-out rows, and is -Infinity for a singular sample at intensity 0', () => {
    const wide = generate(factorCovarianceData, { key: key(2), knobs: { n: 20, p: 30, factors: 2 } }).data.x
    const path = shrinkagePath(wide, { target: 'diagonal', intensities: [0, 0.2, 0.6, 1], folds: 4 })
    expect(path[0]?.heldOut).toBe(Number.NEGATIVE_INFINITY)
    expect(path[0]?.conditionNumber).toBe(Number.POSITIVE_INFINITY)
    expect(Number.isFinite(path[2]?.heldOut ?? Number.NaN)).toBe(true)
    expect(Number.isFinite(path[3]?.conditionNumber ?? Number.NaN)).toBe(true)
  })

  it('is a stepped algorithm with one intensity per step and the spectrum on the tape', () => {
    const out = runAlgorithm(shrinkagePathAlgorithm(data.x, { intensities: [0, 0.5, 1], heldOut: test }), {
      maxSteps: 10,
    })
    expect(out.steps).toBe(3)
    expect(out.state.points.map((p) => p.intensity)).toEqual([0, 0.5, 1])
    expect(out.state.points[0]?.eigenvalues).toHaveLength(20)
  })

  it('rejects intensities outside [0, 1]', () => {
    expect(() => shrinkagePath(X, { intensities: [0, 2] })).toThrow(/intensities/)
    expect(() => shrinkagePath(X, { intensities: [] })).toThrow(/intensities/)
  })
})

describe('targetShrinkageEstimator', () => {
  const { data } = generate(factorCovarianceData, { key: key(3), knobs: { n: 50, p: 15, factors: 2 } })

  it('fits the shrunk covariance with its precision, the intensity and the verbs', () => {
    const fit = targetShrinkageEstimator({ target: 'constant-correlation' }).fit({ x: data.x })
    const direct = targetShrinkage(data.x, { target: 'constant-correlation' })
    expect(fit.params.shrinkage).toBeCloseTo(direct.shrinkage, 12)
    expect(fit.params.covariance[2]?.[3]).toBeCloseTo(direct.covariance[2]?.[3] ?? 0, 12)
    expect(Number.isFinite(fit.score({ x: data.x }))).toBe(true)
    expect(fit.exposition.map((r) => r.kind)).toEqual(['curves', 'anomalyScores'])
  })

  it('takes a fixed intensity and the other rules', () => {
    expect(
      targetShrinkageEstimator({ target: 'diagonal', rule: 'fixed', intensity: 0.3 }).fit({ x: data.x }).params
        .shrinkage,
    ).toBe(0.3)
    const lw = targetShrinkageEstimator({ target: 'identity', rule: 'analytic' }).fit({ x: data.x }).params.shrinkage
    expect(lw).toBeCloseTo(ledoitWolf(data.x).shrinkage, 12)
    expect(() => targetShrinkageEstimator({ target: 'diagonal', rule: 'oas' }).fit({ x: data.x })).toThrow(
      /identity target only/,
    )
  })

  it('runs as a one-step algorithm and as a class', () => {
    const out = runAlgorithm(targetShrinkageAlgorithm(data.x), { maxSteps: 3 })
    expect(out.steps).toBe(1)
    expect(out.state.shrinkage).toBeGreaterThanOrEqual(0)
    expect(new TargetShrinkageEstimator({ target: 'single-factor' }).fit(data.x).model.params.location).toHaveLength(15)
  })
})
