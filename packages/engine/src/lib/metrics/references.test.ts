import { describe, expect, it } from 'vitest'
import { beta, gamma, laplace, logNormal, normal } from '../dists/index.ts'
import * as M from './index.ts'

// Reference values of metrics that no golden file runs: the cluster validity indices (scikit-learn), the proper scoring
// rules (closed forms and scipy.integrate.quad of (F(x) - 1{x >= y})^2 for the CRPS), the energy score and the
// precision-recall curve (scikit-learn). One-off snippets, not fixtures.
const points = [
  [0, 0],
  [0.5, 1],
  [1, 0.2],
  [0.3, 0.4],
  [5, 5],
  [5.5, 4.6],
  [4.8, 5.9],
  [5.2, 5.1],
  [10, 0],
  [9.5, 1],
  [10.4, -0.6],
  [9.9, 0.3],
]
const tight = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
const mixed = [0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2]
const value = (r: { estimate: { value: number } }): number => r.estimate.value

describe('metrics against reference values', () => {
  const rows: readonly (readonly [string, () => number, number])[] = [
    ['silhouetteScore of three tight clusters', () => value(M.silhouetteScore(points, tight)), 0.8717662732786109],
    [
      'silhouetteScore of a split that cuts across them',
      () => value(M.silhouetteScore(points, mixed)),
      0.15507483150718207,
    ],
    [
      'calinskiHarabaszScore of three tight clusters',
      () => value(M.calinskiHarabaszScore(points, tight)),
      275.1568381430362,
    ],
    [
      'calinskiHarabaszScore of the cutting split',
      () => value(M.calinskiHarabaszScore(points, mixed)),
      7.514149417852521,
    ],
    [
      'daviesBouldinScore of three tight clusters',
      () => value(M.daviesBouldinScore(points, tight)),
      0.1406561180527208,
    ],
    ['daviesBouldinScore of the cutting split', () => value(M.daviesBouldinScore(points, mixed)), 25.769761384762138],
    [
      'dunnIndex: least distance between clusters over the largest diameter',
      () => value(M.dunnIndex(points, tight)),
      2.9314624012750046,
    ],
    ['crps of normal(1, 2) at 0.3', () => value(M.crps(normal(1, 2), [0.3])), 0.5641451322181058],
    ['crps of gamma(2, 1.5) at 1.7', () => value(M.crps(gamma(2, 1.5), [1.7])), 0.34036210707366243],
    ['crps of logNormal(0, 0.5) at 1.2', () => value(M.crps(logNormal(0, 0.5), [1.2])), 0.1504342079490752],
    ['crps of laplace(0, 1) at -0.4', () => value(M.crps(laplace(0, 1), [-0.4])), 0.320320045619905],
    ['crps of beta(2, 3) at 0.35', () => value(M.crps(beta(2, 3), [0.35])), 0.053491910714285704],
    ['logScore of normal(1, 2) at 0.3', () => value(M.logScore(normal(1, 2), [0.3])), 1.6733357137646179],
    ['logScore of gamma(2, 1.5) at 1.7', () => value(M.logScore(gamma(2, 1.5), [1.7])), 1.208441532721501],
    [
      'logScore of beta(2, 3) at 0.35 (a density above 1: negative)',
      () => value(M.logScore(beta(2, 3), [0.35])),
      -0.5735186931044141,
    ],
    [
      'sphericalScore of normal(1, 2) at 0.3: density over the L2 norm',
      () => value(M.sphericalScore(normal(1, 2), [0.3])),
      0.49957074377524596,
    ],
    [
      'sphericalScore of gamma(2, 1.5) at 1.7 (by quadrature)',
      () => value(M.sphericalScore(gamma(2, 1.5), [1.7])),
      0.4877136119215788,
    ],
    [
      'energyScore of an ensemble of five points in the plane',
      () =>
        value(
          M.energyScore(
            [
              [0.1, 0.2],
              [1.0, 0.5],
              [0.4, -0.3],
              [0.8, 0.9],
              [-0.2, 0.1],
            ],
            [0.3, 0.4],
          ),
        ),
      0.24478258616667514,
    ],
  ]

  // What scikit-learn returns when a class is missing: zero_division = 0 for the ratios, 0 for the correlation, and NaN for
  // a kappa or an AUC that has nothing to compare.
  const cases: Readonly<Record<string, readonly [number[], number[]]>> = {
    'no positives at all': [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ],
    'no predicted positive': [
      [0, 1, 1, 0],
      [0, 0, 0, 0],
    ],
    'no true positive': [
      [0, 0, 0, 0],
      [0, 1, 1, 0],
    ],
    'only positives': [
      [1, 1, 1],
      [1, 1, 1],
    ],
  }
  const N = Number.NaN
  const expected: readonly (readonly [
    string,
    (a: number[], b: number[]) => { estimate: { value: number } },
    readonly number[],
  ])[] = [
    // in the order of `cases`
    ['f1Score', M.f1Score, [0, 0, 0, 1]],
    ['precisionScore', M.precisionScore, [0, 0, 0, 1]],
    ['recallScore', M.recallScore, [0, 0, 0, 1]],
    ['jaccardScore', M.jaccardScore, [0, 0, 0, 1]],
    ['matthewsCorrcoef', M.matthewsCorrcoef, [0, 0, 0, 0]],
    ['cohenKappaScore', M.cohenKappaScore, [N, 0, 0, N]],
    ['balancedAccuracyScore', M.balancedAccuracyScore, [1, 0.5, 0.5, 1]],
    ['accuracyScore', M.accuracyScore, [1, 0.5, 0.5, 1]],
  ]
  it('agree with scikit-learn, scipy and the closed forms, including single-class and empty-positive inputs', () => {
    for (const [name, v, expected] of rows) expect(v(), name).toBeCloseTo(expected, 7)
    const curve = M.precisionRecallCurve([0, 0, 1, 1, 0, 1, 1, 0], [0.1, 0.4, 0.35, 0.8, 0.5, 0.45, 0.9, 0.4])
    expect(curve.precision, 'precision').toEqual([0.5, 4 / 7, 0.5, 0.75, 2 / 3, 1, 1, 1])
    expect(curve.recall, 'recall').toEqual([1, 1, 0.75, 0.75, 0.5, 0.5, 0.25, 0])
    expect(curve.thresholds, 'thresholds').toEqual([0.1, 0.35, 0.4, 0.45, 0.5, 0.8, 0.9])
    for (const [name, metric, want] of expected)
      Object.entries(cases).forEach(([label, [a, b]], i) => {
        const got = value(metric(a, b))
        const w = want[i] as number
        if (Number.isNaN(w)) expect(got, `${name}, ${label}`).toBeNaN()
        else expect(got, `${name}, ${label}`).toBeCloseTo(w, 12)
      })
    for (const y of [
      [1, 1, 1],
      [0, 0, 0],
    ])
      expect(value(M.rocAucScore(y, [0.2, 0.5, 0.9])), 'rocAucScore with one class').toBeNaN()
  })
})
