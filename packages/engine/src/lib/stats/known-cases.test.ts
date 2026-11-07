import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import * as S from './index.ts'
import * as M from './smoothing/index.ts'
import * as T from './tests/index.ts'

const mean = (x: readonly number[]): number => x.reduce((a, b) => a + b, 0) / x.length
const variance = (x: readonly number[]): number => x.reduce((a, b) => a + (b - mean(x)) ** 2, 0) / (x.length - 1)
const flat = (v: Parameters<typeof toFlat>[0]): number[] => Array.from(toFlat(v))

// Small cases whose answer is a count: the p-value of an exact or permutation test is the share of the arrangements at least
// as extreme as the one observed, so scipy (permutation_test, fisher_exact, mannwhitneyu, wilcoxon, binomtest) and hand
// enumeration give the same number as the engine.
describe('exact and permutation tests on small cases', () => {
  const x3 = [1, 2, 3]
  const y3 = [4, 5, 6]
  const pairedA = [1.1, 2.4, 3.3, 4.9]
  const pairedB = [2.0, 2.9, 4.5, 6.1]
  const alt = ['two-sided', 'less', 'greater'] as const
  const exact = (samples: number[][], type: 'independent' | 'samples', alternative: (typeof alt)[number]): number =>
    S.permutationTest(samples, S.meanDifference, { permutationType: type, alternative, key: key(0) }).pValue
  const rows: readonly (readonly [string, () => number, number])[] = [
    // 20 splits of 6 values into 3 and 3; the observed one is the most extreme in one direction (2 of 20 in absolute value).
    ...alt.map(
      (a, i) =>
        [
          `permutation, independent samples, ${a}`,
          () => exact([x3, y3], 'independent', a),
          [0.1, 0.05, 1][i] as number,
        ] as const,
    ),
    // 2^4 = 16 sign flips of the paired differences.
    ...alt.map(
      (a, i) =>
        [
          `permutation, paired signs, ${a}`,
          () => exact([pairedA, pairedB], 'samples', a),
          [0.125, 0.0625, 1][i] as number,
        ] as const,
    ),
    [
      'permutation, pairings: Pearson r of 1..5 against 1,3,2,5,4, greater (120 orders)',
      () =>
        S.permutationTest(
          [
            [1, 2, 3, 4, 5],
            [1, 3, 2, 5, 4],
          ],
          (a, b) => S.pearson(a, b).statistic,
          { permutationType: 'pairings', alternative: 'greater', key: key(0) },
        ).pValue,
      8 / 120,
    ],
    ...alt.map(
      (a, i) =>
        [
          `fisherExact [[3, 1], [1, 3]], ${a}`,
          () =>
            T.fisherExact(
              [
                [3, 1],
                [1, 3],
              ],
              { alternative: a },
            ).pValue,
          [0.48571428571428565, 0.9857142857142858, 0.24285714285714283][i] as number,
        ] as const,
    ),
    ...alt.map(
      (a, i) =>
        [
          `fisherExact [[8, 2], [1, 5]], ${a}`,
          () =>
            T.fisherExact(
              [
                [8, 2],
                [1, 5],
              ],
              { alternative: a },
            ).pValue,
          [0.034965034965034975, 0.9991258741258742, 0.024475524475524483][i] as number,
        ] as const,
    ),
    ...alt.map(
      (a, i) =>
        [
          `mannWhitneyU exact, [1, 2, 3] against [4, 5, 6], ${a}`,
          () => T.mannWhitneyU(x3, y3, { method: 'exact', alternative: a }).pValue,
          [0.1, 0.05, 1][i] as number,
        ] as const,
    ),
    ...alt.map(
      (a, i) =>
        [
          `mannWhitneyU exact, sizes 4 and 3, ${a}`,
          () => T.mannWhitneyU([1.5, 3.1, 4.2, 7.7], [2.2, 2.9, 5.1], { method: 'exact', alternative: a }).pValue,
          [0.8571428571428571, 0.6857142857142857, 0.42857142857142855][i] as number,
        ] as const,
    ),
    ...alt.map(
      (a, i) =>
        [
          `wilcoxon exact, six differences, ${a}`,
          () => T.wilcoxon([1.5, -0.5, 2.5, 3.5, -4.5, 5.5], undefined, { method: 'exact', alternative: a }).pValue,
          [0.4375, 0.84375, 0.21875][i] as number,
        ] as const,
    ),
    ...alt.map(
      (a, i) =>
        [
          `binomTest(7, 10, 0.5), ${a}`,
          () => T.binomTest(7, 10, 0.5, { alternative: a }).pValue,
          [0.34375, 0.9453125, 0.171875][i] as number,
        ] as const,
    ),
    ...alt.map(
      (a, i) =>
        [
          `binomTest(2, 12, 0.3), ${a}`,
          () => T.binomTest(2, 12, 0.3, { alternative: a }).pValue,
          [0.5291598783249996, 0.2528153478549998, 0.914974950051][i] as number,
        ] as const,
    ),
    // Discordant pairs 1 and 6: twice the binomial tail P(X <= 1), n = 7, p = 1/2, is 2 * 8 / 128.
    [
      'mcnemar exact, discordant pairs 1 and 6',
      () =>
        T.mcnemar(
          [
            [10, 1],
            [6, 20],
          ],
          { exact: true },
        ).pValue,
      0.125,
    ],
  ]

  // The multiple-testing wrappers against the formulas (Bonferroni m p; Holm and Hochberg step down and up on the sorted
  // p-values; Benjamini-Hochberg p m / rank, and Yekutieli times the harmonic number), scipy.stats.false_discovery_control
  // for the two FDR rows.
  const p = [0.01, 0.04, 0.03, 0.005, 0.2, 0.9, 0.015]
  const adjusted: readonly (readonly [string, (p: number[]) => unknown, readonly number[]])[] = [
    ['bonferroni', T.bonferroni, [0.07, 0.28, 0.21, 0.035, 1, 1, 0.105]],
    ['holm', T.holm, [0.06, 0.12, 0.12, 0.035, 0.4, 0.9, 0.075]],
    ['hochberg', T.hochberg, [0.06, 0.12, 0.12, 0.035, 0.4, 0.9, 0.075]],
    ['benjaminiHochberg', T.benjaminiHochberg, [0.035, 0.056, 0.0525, 0.035, 0.23333333333333336, 0.9, 0.035]],
    ['benjaminiYekutieli', T.benjaminiYekutieli, [0.09075, 0.1452, 0.136125, 0.09075, 0.605, 1, 0.09075]],
  ]
  it('give the share of arrangements at least as extreme, and the multiple-testing wrappers adjust p-values as Bonferroni, Holm, Hochberg, Benjamini-Hochberg and Benjamini-Yekutieli do', () => {
    for (const [name, p, expected] of rows) expect(p(), name).toBeCloseTo(expected, 12)
    expect(S.permutationTest([x3, y3], S.meanDifference, { key: key(0) }).method, 'small samples are enumerated').toBe(
      'exact enumeration',
    )
    for (const [name, adjust, expected] of adjusted) {
      const got = flat(adjust(p) as Parameters<typeof toFlat>[0])
      for (const [i, v] of got.entries()) expect(v, `${name}[${i}]`).toBeCloseTo(expected[i] as number, 12)
    }
    const method = {
      bonferroni: 'bonferroni',
      hochberg: 'hochberg',
      benjaminiHochberg: 'benjamini-hochberg',
      benjaminiYekutieli: 'benjamini-yekutieli',
    } as const
    for (const [name, adjust] of adjusted.filter(([n]) => n in method))
      expect(flat(adjust(p) as Parameters<typeof toFlat>[0]), `${name} is multipleTests`).toEqual(
        flat(T.multipleTests(p, { method: method[name as keyof typeof method] }).pValuesCorrected),
      )
    expect(T.multipleTests(p, { method: 'holm', alpha: 0.1 }).reject).toEqual([
      true,
      false,
      false,
      true,
      false,
      false,
      true,
    ])
  })
})

describe('effect sizes and helpers on closed forms', () => {
  const x = [1, 2, 3, 4, 6]
  const y = [3, 4, 5, 6, 9, 10]
  const [nx, ny] = [x.length, y.length]
  const pooled = Math.sqrt(((nx - 1) * variance(x) + (ny - 1) * variance(y)) / (nx + ny - 2))
  const d = (mean(x) - mean(y)) / pooled
  const diff = x.map((v, i) => v - (y[i] as number))
  const groups = [x, y, [2, 2, 7, 8]]
  const all = groups.flat()
  const ssb = groups.reduce((a, g) => a + g.length * (mean(g) - mean(all)) ** 2, 0)
  const sst = all.reduce((a, v) => a + (v - mean(all)) ** 2, 0)
  const msw = (sst - ssb) / (all.length - groups.length)
  const dominance = x.flatMap((a) => y.map((b) => Math.sign(a - b))).reduce((a, b) => a + b, 0) / (nx * ny)
  const rows: readonly (readonly [string, () => number, number])[] = [
    ['cohensD: mean difference over the pooled standard deviation', () => S.cohensD(x, y).value, d],
    [
      'cohensD against a constant: (mean - mu) / sd',
      () => S.cohensD(x, 2).value,
      (mean(x) - 2) / Math.sqrt(variance(x)),
    ],
    [
      'cohensD paired: over the root mean of the two variances',
      () => S.cohensD(x, y.slice(0, 5), { paired: true }).value,
      (mean(x) - mean(y.slice(0, 5))) / Math.sqrt((variance(x) + variance(y.slice(0, 5))) / 2),
    ],
    ['hedgesG: d times 1 - 3 / (4 N - 9)', () => S.hedgesG(x, y).value, d * (1 - 3 / (4 * (nx + ny) - 9))],
    [
      'glassDelta: over the control standard deviation',
      () => S.glassDelta(x, y).value,
      (mean(x) - mean(y)) / Math.sqrt(variance(y)),
    ],
    [
      'cohensH: 2 asin sqrt p1 - 2 asin sqrt p2',
      () => S.cohensH(0.7, 0.4).value,
      2 * Math.asin(Math.sqrt(0.7)) - 2 * Math.asin(Math.sqrt(0.4)),
    ],
    [
      'cramersV of [[10, 20], [20, 10]]: sqrt(chi2 / n) with chi2 = 20 / 3',
      () =>
        S.cramersV([
          [10, 20],
          [20, 10],
        ]).value,
      Math.sqrt(20 / 3 / 60),
    ],
    [
      'cramersV of a 2 x 3 table: sqrt(chi2 / (n (min(r, c) - 1)))',
      () =>
        S.cramersV([
          [10, 20, 30],
          [30, 20, 10],
        ]).value,
      Math.sqrt(((2 * (10 - 20) ** 2) / 20 + 2 * 0 + (2 * (30 - 20) ** 2) / 20) / 120),
    ],
    ['cliffsDelta: P(X > Y) - P(X < Y) over all pairs', () => S.cliffsDelta(x, y).value, dominance],
    ['etaSquared: SS between over SS total', () => S.etaSquared(groups).value, ssb / sst],
    [
      'omegaSquared: (SSB - (k - 1) MSW) / (SST + MSW)',
      () => S.omegaSquared(groups).value,
      (ssb - (groups.length - 1) * msw) / (sst + msw),
    ],
    [
      'rank-biserial correlation of mannWhitneyU is 2 U / (n m) - 1',
      () => T.mannWhitneyU(x, y).effectSize?.value ?? Number.NaN,
      (2 * x.flatMap((a) => y.map((b) => (a > b ? 1 : a === b ? 0.5 : 0) as number)).reduce((a, b) => a + b, 0)) /
        (nx * ny) -
        1,
    ],
    [
      'trimmedVariance of 10 values, 20% cut each end',
      () => S.trimmedVariance([1, 2, 3, 4, 100, 7, 5, 6, 9, 8], 0.2),
      3.5,
    ],
    [
      'trimmedVariance with ddof 0',
      () => S.trimmedVariance([1, 2, 3, 4, 100, 7, 5, 6, 9, 8], 0.2, { ddof: 0 }),
      2.9166666666666665,
    ],
    // The biased HSIC with linear kernels in one dimension is the square of the (biased) covariance: tr(K H L H) / n^2.
    [
      'hsic with linear kernels is the squared covariance',
      () =>
        T.hsic(x, diff, {
          kernel: (a, b) => (a[0] as number) * (b[0] as number),
          kernelY: (a, b) => (a[0] as number) * (b[0] as number),
        }),
      (x.reduce((s, v, i) => s + (v - mean(x)) * ((diff[i] as number) - mean(diff)), 0) / nx) ** 2,
    ],
  ]

  it('equal the formulas, and the report helpers, the renamed designs and the shrinkage penalty do what they say', () => {
    for (const [name, value, expected] of rows) expect(value(), name).toBeCloseTo(expected, 10)
    const t = {
      test: 'Welch t test',
      statistic: 2.34567,
      pValue: 0.0312,
      df: [4.5, 7],
      alternative: 'two-sided',
    } as const
    expect(T.formatTest(t)).toBe('Welch t test: statistic 2.346 (df 4.5, 7), p = 0.0312, two-sided')
    expect(T.formatTest({ test: t.test, statistic: t.statistic, pValue: 1.234e-5, alternative: t.alternative })).toBe(
      'Welch t test: statistic 2.346, p = 1.23e-5, two-sided',
    )
    expect(T.formatTest({ ...t, pValue: Number.NaN, statistic: Number.POSITIVE_INFINITY, df: 3 })).toBe(
      'Welch t test: statistic Infinity (df 3), two-sided',
    )
    expect(
      T.formatTest({ ...t, effectSize: { name: "Cohen's d", value: 0.5, interval: [0.1, 0.9], level: 0.9 } }),
    ).toBe("Welch t test: statistic 2.346 (df 4.5, 7), p = 0.0312, two-sided, Cohen's d 0.5 (90% CI 0.1 to 0.9)")
    expect([0.0499, 0.05, 0.0501, Number.NaN].map((pValue) => T.rejects({ ...t, pValue }))).toEqual([
      true,
      false,
      false,
      false,
    ])
    expect(T.rejects({ ...t, pValue: 0.0312 }, 0.01)).toBe(false)

    // The names kept for the older spellings run the same code.
    const k = key(4)
    expect(flat(S.completelyRandomised(k, 12))).toEqual(flat(S.simpleRandomisation(k, 12)))
    expect(flat(S.blockedDesign(k, 12, { blockSize: 4 }))).toEqual(
      flat(S.permutedBlockRandomisation(k, 12, { blockSize: 4 })),
    )
    const cov = [
      [0, 1],
      [1, 3],
      [0, 2],
      [1, 1],
      [5, 5],
      [4, 6],
    ]
    expect(S.matchedPairs(cov)).toEqual(S.matchedPairsDesign(cov))
    expect(S.fullFactorial([2, 3])).toEqual(S.fullFactorialDesign([2, 3]))
    expect(S.fractionalFactorial(4, ['D=ABC'])).toEqual(S.fractionalFactorialDesign(4, ['D=ABC']))
    // The shrunk penalty has no null space: its null eigenvalues become epsilon times the smallest positive one.
    const penalty = M.differencePenalty(6, 2)
    const eigenvalues = flat(M.nullSpace(penalty).eigenvalues).sort((a, b) => a - b)
    const smallest = eigenvalues.find((e) => e > 1e-9) as number
    const shrunk = flat(M.nullSpace(M.shrinkagePenalty(penalty, { epsilon: 0.1 }), { tol: 0 }).eigenvalues).sort(
      (a, b) => a - b,
    )
    expect(M.nullSpace(M.shrinkagePenalty(penalty)).dimension).toBe(0)
    expect(shrunk[0]).toBeCloseTo(0.1 * smallest, 8)
    expect(shrunk[1]).toBeCloseTo(0.1 * smallest, 8)
    expect(shrunk[2]).toBeCloseTo(smallest, 8)
  })
})
