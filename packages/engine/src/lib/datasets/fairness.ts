/**
 * Datasets for fairness audits and mitigation: synthetic credit decisions with a protected group and controllable bias
 * (representation imbalance, a real base-rate gap, historical label bias, noisier measurements of the minority and a
 * proxy feature), the Statlog German credit data, and Titanic survival with sex and age as protected attributes.
 */

import { type ClassificationData, type DatasetGenerator, type Knobs, xyClasses } from '../data/index.ts'
import { ndtr } from '../numerics/special/index.ts'
import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  GERMAN_CREDIT_CLASSES,
  GERMAN_CREDIT_FEATURES,
  GERMAN_CREDIT_SHA256,
  GERMAN_CREDIT_SHAPE,
  GERMAN_CREDIT_X,
  GERMAN_CREDIT_Y,
} from './_data/german-credit.data.ts'
import { TITANIC_COLUMNS, TITANIC_SHA256 } from './_data/titanic.data.ts'
import { monteCarloBayesError, withLazyBayesError } from './_mixtures.ts'
import { labels, matrix, stamp } from './_util.ts'

/** The group names of {@link biasedCredit}'s `group` column. */
export const CREDIT_GROUPS: readonly string[] = ['majority', 'minority']

/** The parameters of {@link biasedCredit}, read from its knobs. */
interface CreditModel {
  readonly share: number
  readonly gap: number
  readonly bias: number
  readonly proxy: number
  readonly sigma: readonly [number, number]
  readonly noise: number
}

function creditModel(knobs: Knobs): CreditModel {
  return {
    share: knobs['minorityShare'] as number,
    gap: knobs['baseRateGap'] as number,
    bias: knobs['labelBias'] as number,
    proxy: knobs['proxyStrength'] as number,
    sigma: [1, knobs['minorityNoise'] as number],
    noise: knobs['noise'] as number,
  }
}

/** Rows of the credit model: group, qualification, features, clean and recorded labels. */
function drawCredit(k: Key, n: number, m: CreditModel) {
  const u = uniform(foldIn(k, 0), [n]).data
  const zq = normal(foldIn(k, 1), [n]).data
  const z = normal(foldIn(k, 2), [n, 3]).data
  const e = normal(foldIn(k, 3), [n]).data
  const flip = uniform(foldIn(k, 4), [n]).data
  const d = 4
  const x = new Float64Array(n * d)
  const group: number[] = []
  const quality: number[] = []
  const clean: number[] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const a = (u[i] ?? 1) < m.share ? 1 : 0
    const q = -m.gap * a + (zq[i] ?? 0)
    const s = m.sigma[a]
    x[i * d] = q + s * (z[3 * i] ?? 0)
    x[i * d + 1] = q + s * (z[3 * i + 1] ?? 0)
    x[i * d + 2] = m.proxy * a + (z[3 * i + 2] ?? 0)
    x[i * d + 3] = a
    const ok = q + m.noise * (e[i] ?? 0) > 0 ? 1 : 0
    group.push(a)
    quality.push(q)
    clean.push(ok)
    // Historical bias: a share of the minority's good outcomes was recorded as bad.
    y.push(ok === 1 && a === 1 && (flip[i] ?? 1) < m.bias ? 0 : ok)
  }
  return { x: fromStorage(x, [n, d], 'float64'), group, quality, clean, y }
}

/**
 * P(y = 1 | x) of the credit model: the qualification's Gaussian posterior given the two noisy measurements, pushed
 * through the outcome noise, thinned by the label bias; without the group column, mixed over the group's posterior
 * given the measurements and the proxy.
 */
function creditPosterior(m: CreditModel, withGroup: boolean): (x: Value) => Value {
  // The argument of the normal CDF giving P(clean outcome | measurements, group a).
  const argument = (x1: number, x2: number, a: number): number => {
    const s2 = (m.sigma[a] ?? 1) ** 2
    const precision = 1 + 2 / s2
    const mean = (-m.gap * a + (x1 + x2) / s2) / precision
    return mean / Math.sqrt(1 / precision + m.noise ** 2)
  }
  // log p(x1, x2, x3 | a) up to a shared constant: the measurements' bivariate normal and the proxy's normal.
  const logLik = (x1: number, x2: number, x3: number, a: number): number => {
    const s2 = (m.sigma[a] ?? 1) ** 2
    const d1 = x1 + m.gap * a
    const d2 = x2 + m.gap * a
    const quad = (d1 * d1 + d2 * d2 - (d1 + d2) ** 2 / (s2 + 2)) / s2
    const proxy = x3 - m.proxy * a
    return -0.5 * (quad + Math.log(s2 * (s2 + 2)) + proxy * proxy)
  }
  return (x) => {
    const v = toFlat(x)
    const d = withGroup ? 4 : 3
    const n = v.length / d
    // Each row's posterior weight on the minority (its group when seen) and both groups' CDF arguments.
    const w1 = new Float64Array(n)
    const args = new Float64Array(2 * n)
    for (let i = 0; i < n; i++) {
      const [x1, x2, x3] = [v[i * d] ?? 0, v[i * d + 1] ?? 0, v[i * d + 2] ?? 0]
      if (withGroup) w1[i] = v[i * d + 3] ?? 0
      else {
        const l0 = Math.log(1 - m.share) + logLik(x1, x2, x3, 0)
        const l1 = Math.log(m.share) + logLik(x1, x2, x3, 1)
        w1[i] = 1 / (1 + Math.exp(l0 - l1))
      }
      args[2 * i] = argument(x1, x2, 0)
      args[2 * i + 1] = argument(x1, x2, 1)
    }
    const cdf = toFlat(ndtr(fromStorage(args, [2 * n], 'float64')))
    const out = new Float64Array(2 * n)
    for (let i = 0; i < n; i++) {
      const w = w1[i] ?? 0
      const p = (1 - w) * (cdf[2 * i] ?? 0) + w * (1 - m.bias) * (cdf[2 * i + 1] ?? 0)
      out[2 * i] = 1 - p
      out[2 * i + 1] = p
    }
    return fromStorage(out, [n, 2], 'float64')
  }
}

/**
 * Credit decisions with a protected group and controllable bias, for fairness audits and mitigation: each applicant
 * belongs to the minority group with probability `minorityShare` (representation imbalance) and has a latent
 * qualification `q ~ N(-baseRateGap * group, 1)` (a real difference in the target); two measurements of it,
 * `income` and `repayment`, carry noise of standard deviation 1 for the majority and `minorityNoise` for the
 * minority (less informative data about a group); `postcode` is a pure proxy, `N(proxyStrength * group, 1)`, that
 * says nothing about qualification; the clean outcome is `q + noise * e > 0`, and the recorded label drops a share
 * `labelBias` of the minority's good outcomes to bad (historical bias in the labels). The last column is the group (0
 * majority, 1 minority), named in `meta.sensitive`; a model fitted without it can still learn the group through the
 * proxy. The truth is exact: the posterior of the recorded label, a Monte Carlo Bayes error, the group, qualification
 * and clean labels per row, each group's clean and recorded base rates, and `extra.unawarePosterior`, the posterior
 * from the first three columns alone (the best a model without the group can do).
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { biasedCredit } from 'mdml-engine/datasets'
 * generate(biasedCredit, { knobs: { n: 10 } }).data.featureNames // => ['income', 'repayment', 'postcode', 'group']
 * ```
 */
export const biasedCredit: DatasetGenerator<ClassificationData> = {
  name: 'biasedCredit',
  kind: 'classification',
  description:
    'Credit outcomes with a protected group: base-rate gap, label bias, under-representation, noisier data and a proxy.',
  knobs: dict({
    n: int(10, 1_000_000, { label: 'samples', description: 'The number of applicants.', default: 1000 }),
    minorityShare: real(0.02, 0.98, {
      label: 'minority share',
      description: 'The probability an applicant belongs to the minority group (representation).',
      default: 0.3,
    }),
    baseRateGap: real(0, 3, {
      label: 'qualification gap',
      description: 'How far the minority’s mean qualification sits below the majority’s (a real base-rate gap).',
      default: 0.4,
    }),
    labelBias: real(0, 0.95, {
      label: 'label bias',
      description: 'The share of the minority’s good outcomes recorded as bad (historical bias in the labels).',
      default: 0.25,
    }),
    proxyStrength: real(0, 5, {
      label: 'proxy strength',
      description: 'How far apart the groups’ postcode distributions sit, in standard deviations.',
      default: 1.5,
    }),
    minorityNoise: real(0.2, 5, {
      label: 'minority measurement noise',
      description: 'The standard deviation of the minority’s measurement noise (the majority’s is 1).',
      default: 1.5,
    }),
    noise: real(0.01, 5, {
      label: 'outcome noise',
      description: 'The standard deviation of the outcome noise beyond the qualification.',
      default: 0.5,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const m = creditModel(knobs)
    const draw = drawCredit(foldIn(k, 0), n, m)
    const featureNames = ['income', 'repayment', 'postcode', 'group']
    const clean = Array.from(toFlat(ndtr([0, -m.gap / Math.sqrt(1 + m.noise ** 2)])))
    const posterior = creditPosterior(m, true)
    return {
      data: xyClasses(draw.x, labels(draw.y), {
        ...stamp('biasedCredit', biasedCredit.description, knobs),
        featureNames,
        classes: ['default', 'repaid'],
        sensitive: { group: CREDIT_GROUPS },
      }),
      truth: withLazyBayesError(
        {
          posterior,
          relevantFeatures: [0, 1, 3],
          latents: {
            group: draw.group,
            qualification: draw.quality,
            cleanLabels: draw.clean,
          },
          extra: {
            groups: CREDIT_GROUPS,
            cleanBaseRates: clean,
            recordedBaseRates: clean.map((r, a) => r * (1 - m.bias * a)),
            unawarePosterior: creditPosterior(m, false),
          },
        },
        () => monteCarloBayesError(posterior, (key, size) => drawCredit(key, size, m).x, foldIn(k, 1)),
      ),
    }
  },
}

/**
 * The Statlog (German credit) data: 1000 loan applicants described by twelve attributes (checking and savings
 * balances and employment length as ordinal codes 0, 1, ... in the order of the documentation, the loan's duration in
 * months, amount in Deutsche Mark and instalment rate, age, existing credits, dependents, home ownership, foreign
 * worker, and `female`, derived from the personal status), each a good or bad credit risk (Hofmann, 1994). `female`
 * is the protected attribute (`meta.sensitive`). From the UCI repository (CC BY 4.0).
 *
 * @returns The dataset.
 * @concept datasets
 * @example
 * ```ts
 * import { germanCredit } from 'mdml-engine/datasets'
 * germanCredit().x.shape // => [1000, 12]
 * ```
 */
export function germanCredit(): ClassificationData {
  const [n, d] = GERMAN_CREDIT_SHAPE
  return xyClasses(
    fromStorage(Float64Array.from(GERMAN_CREDIT_X), [n, d], 'float64'),
    fromStorage(Int32Array.from(GERMAN_CREDIT_Y), [n], 'int32'),
    {
      name: 'germanCredit',
      description: '1000 loan applicants with twelve attributes, each a good or bad credit risk (Hofmann, 1994).',
      url: 'https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data',
      citation: 'Hofmann (1994), Statlog (German Credit Data), UCI Machine Learning Repository.',
      licence: 'CC BY 4.0 (UCI Machine Learning Repository)',
      sha256: GERMAN_CREDIT_SHA256,
      source: 'UCI Statlog (German credit), german.data',
      featureNames: GERMAN_CREDIT_FEATURES,
      classes: GERMAN_CREDIT_CLASSES,
      sensitive: { female: ['male', 'female'] },
    },
  )
}

/**
 * Survival of the 2201 people aboard the Titanic as a classification problem with a protected attribute: one-hot
 * travelling class (`first`, `second`, `third`; crew is all zeros), `child` and `female` (the protected attribute,
 * `meta.sensitive`), and whether each survived. From R's `datasets::Titanic` (see `titanic`).
 *
 * @returns The dataset.
 * @concept datasets
 * @example
 * ```ts
 * import { titanicSurvival } from 'mdml-engine/datasets'
 * titanicSurvival().featureNames // => ['first', 'second', 'third', 'child', 'female']
 * ```
 */
export function titanicSurvival(): ClassificationData {
  const freq = TITANIC_COLUMNS['Freq'] as readonly number[]
  const col = (name: string): readonly string[] => TITANIC_COLUMNS[name] as readonly string[]
  const x: number[] = []
  const y: number[] = []
  for (const [i, f] of freq.entries()) {
    const c = col('Class')[i]
    const r = [c === '1st' ? 1 : 0, c === '2nd' ? 1 : 0, c === '3rd' ? 1 : 0]
    r.push(col('Age')[i] === 'Child' ? 1 : 0, col('Sex')[i] === 'Female' ? 1 : 0)
    for (let k = 0; k < f; k++) {
      x.push(...r)
      y.push(col('Survived')[i] === 'Yes' ? 1 : 0)
    }
  }
  return xyClasses(matrix(x, y.length, 5) as Tensor, labels(y), {
    name: 'titanicSurvival',
    description: 'Class, age and sex of the 2201 people aboard the Titanic, and who survived (R datasets::Titanic).',
    url: 'https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/Titanic.html',
    citation: 'Dawson (1995), The unusual episode data revisited, Journal of Statistics Education 3(3).',
    licence: 'Public domain (British Board of Trade, 1912); table from R (GPL-2 | GPL-3)',
    sha256: TITANIC_SHA256,
    source: 'R datasets::Titanic via Rdatasets',
    featureNames: ['first', 'second', 'third', 'child', 'female'],
    classes: ['died', 'survived'],
    sensitive: { female: ['male', 'female'] },
  })
}
