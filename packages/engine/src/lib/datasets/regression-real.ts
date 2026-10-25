/**
 * Small real regression datasets: the diabetes data of least angle regression, Anscombe's quartet, Linnerud, Motor
 * Trend cars, and California housing as a pinned remote asset. Also CDC growth charts as LMS parameters, with a
 * generator of children's measurements whose conditional quantiles are known exactly.
 */

import { MdmlError } from '../base/index.ts'
import {
  type DatasetGenerator,
  type DatasetRef,
  gunzip,
  parseCsv,
  type RegressionData,
  type TableData,
  table,
  untar,
  xy,
} from '../data/index.ts'
import { gaussHermite } from '../numerics/quadrature/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { foldIn, normal, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, item, type Tensor, toFlat, type Value } from '../tensor/index.ts'
import { ANSCOMBE_COLUMNS, ANSCOMBE_SHA256 } from './_data/anscombe.data.ts'
import { DIABETES_FEATURES, DIABETES_SHA256, DIABETES_SHAPE, DIABETES_X, DIABETES_Y } from './_data/diabetes.data.ts'
import { GROWTH_COLUMNS, GROWTH_SHA256 } from './_data/growth-charts.data.ts'
import { LINNERUD_FEATURES, LINNERUD_SHA256, LINNERUD_TARGETS, LINNERUD_X, LINNERUD_Y } from './_data/linnerud.data.ts'
import { MTCARS_COLUMNS, MTCARS_SHA256 } from './_data/mtcars.data.ts'
import { matrix, stamp, vector } from './_util.ts'

/** Options of {@link diabetes}. */
export interface DiabetesOptions {
  /**
   * Centre each feature and scale it to unit sum of squares, as sklearn's default `load_diabetes()` does (default
   * false: the original units).
   */
  readonly scaled?: boolean
}

/**
 * The diabetes data of Efron, Hastie, Johnstone and Tibshirani (2004, "Least angle regression"): ten baseline variables
 * (age, sex, body mass index, blood pressure and six blood serum measurements) for 442 patients, and a measure of
 * disease progression one year later. Embedded from scikit-learn (`load_diabetes(scaled=False)`).
 *
 * @param options - Whether to scale the features as sklearn does by default.
 * @returns The dataset.
 * @concept datasets
 * @example
 * ```ts
 * import { diabetes } from 'mdml-engine/datasets'
 * diabetes().x.shape // => [442, 10]
 * ```
 */
export function diabetes(options: DiabetesOptions = {}): RegressionData {
  const [n, d] = DIABETES_SHAPE
  const x = Float64Array.from(DIABETES_X)
  if (options.scaled === true) {
    for (let j = 0; j < d; j++) {
      let mean = 0
      for (let i = 0; i < n; i++) mean += x[i * d + j] ?? 0
      mean /= n
      let ss = 0
      for (let i = 0; i < n; i++) ss += ((x[i * d + j] ?? 0) - mean) ** 2
      const scale = Math.sqrt(ss)
      for (let i = 0; i < n; i++) x[i * d + j] = ((x[i * d + j] ?? 0) - mean) / scale
    }
  }
  return xy(fromStorage(x, [n, d], 'float64'), fromStorage(Float64Array.from(DIABETES_Y), [n], 'float64'), {
    name: 'diabetes',
    description: 'Disease progression one year after baseline for 442 diabetes patients (Efron et al., 2004).',
    source: 'scikit-learn load_diabetes',
    url: 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html',
    licence: 'BSD-3-Clause (scikit-learn)',
    citation: 'Efron, Hastie, Johnstone and Tibshirani (2004), Least angle regression, Annals of Statistics 32(2).',
    sha256: DIABETES_SHA256,
    featureNames: DIABETES_FEATURES,
    targetNames: ['progression'],
  })
}

/**
 * One of Anscombe's four data sets (Anscombe, 1973): eleven points each with nearly identical means, variances,
 * correlation and least-squares line, but very different shapes.
 *
 * @param set - Which data set, 1 to 4.
 * @returns The dataset.
 * @throws MdmlError with code `value` for a set outside 1 to 4.
 * @example
 * ```ts
 * import { anscombe } from 'mdml-engine/datasets'
 * anscombe(4).x.shape // => [11, 1]
 * ```
 */
export function anscombe(set: 1 | 2 | 3 | 4): RegressionData {
  if (![1, 2, 3, 4].includes(set)) throw new MdmlError('value', `anscombe: set ${set} is not 1 to 4`)
  const x = ANSCOMBE_COLUMNS[`x${set}`] as readonly number[]
  const y = ANSCOMBE_COLUMNS[`y${set}`] as readonly number[]
  return xy(matrix(x, x.length, 1), vector(y), {
    name: `anscombe${set}`,
    description: `Anscombe's quartet, data set ${set} (Anscombe, 1973).`,
    source: 'R datasets::anscombe via Rdatasets',
    url: 'https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/anscombe.html',
    citation: 'Anscombe (1973), Graphs in statistical analysis, The American Statistician 27(1).',
    licence: 'Public domain data; table from R (GPL-2 | GPL-3)',
    sha256: ANSCOMBE_SHA256,
    featureNames: ['x'],
    targetNames: ['y'],
  })
}

/**
 * The Linnerud data (Tenenhaus, 1998): three exercises (chin-ups, sit-ups, jumps) and three physiological measurements
 * (weight, waist, pulse) of 20 middle-aged men, a small multi-output regression set. Embedded from scikit-learn's
 * `load_linnerud`.
 *
 * @returns The dataset with targets `[20, 3]`.
 * @example
 * ```ts
 * import { linnerud } from 'mdml-engine/datasets'
 * linnerud().y.shape // => [20, 3]
 * ```
 */
export function linnerud(): RegressionData {
  return xy(matrix(LINNERUD_X, 20, 3), matrix(LINNERUD_Y, 20, 3), {
    name: 'linnerud',
    description: 'Exercise and physiological measurements of 20 men (Tenenhaus, 1998).',
    source: 'scikit-learn load_linnerud',
    citation: 'Tenenhaus (1998), La regression PLS: theorie et pratique, Technip.',
    licence: 'BSD-3-Clause (scikit-learn)',
    sha256: LINNERUD_SHA256,
    featureNames: LINNERUD_FEATURES,
    targetNames: LINNERUD_TARGETS,
  })
}

const MTCARS_META = {
  name: 'mtcars',
  description: 'Fuel consumption and ten aspects of design and performance of 32 cars (Motor Trend, 1974).',
  source: 'R datasets::mtcars via Rdatasets',
  url: 'https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/mtcars.html',
  citation: 'Henderson and Velleman (1981), Building multiple regression models interactively, Biometrics 37(2).',
  licence: 'Public domain data; table from R (GPL-2 | GPL-3)',
  sha256: MTCARS_SHA256,
}

/**
 * Motor Trend road tests of 32 cars (1973-74 models) as a table: miles per gallon, cylinders, displacement,
 * horsepower, rear axle ratio, weight, quarter-mile time, engine shape, transmission, gears, carburettors and the
 * model name. The target is `mpg`.
 *
 * @returns The table.
 * @example
 * ```ts
 * import { mtcars } from 'mdml-engine/datasets'
 * mtcars().target // => 'mpg'
 * ```
 */
export function mtcars(): TableData {
  return table(MTCARS_COLUMNS as Readonly<Record<string, readonly (number | null)[] | readonly (string | null)[]>>, {
    target: 'mpg',
    text: ['model'],
    meta: MTCARS_META,
  })
}

/**
 * The mtcars data as numeric regression: miles per gallon from the other ten measurements.
 *
 * @returns The dataset.
 * @example
 * ```ts
 * import { mtcarsRegression } from 'mdml-engine/datasets'
 * mtcarsRegression().featureNames.length // => 10
 * ```
 */
export function mtcarsRegression(): RegressionData {
  const names = ['cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
  const cols = names.map((nm) => MTCARS_COLUMNS[nm] as readonly number[])
  const n = cols[0]?.length ?? 0
  const x = Float64Array.from(
    { length: n * names.length },
    (_, i) => cols[i % names.length]?.[Math.floor(i / names.length)] ?? 0,
  )
  return xy(matrix(x, n, names.length), vector(MTCARS_COLUMNS['mpg'] as readonly number[]), {
    ...MTCARS_META,
    featureNames: names,
    targetNames: ['mpg'],
  })
}

// ---------------------------------------------------------------------------------------------------------------
// California housing (remote)

const HOUSING_FEATURES = [
  'MedInc',
  'HouseAge',
  'AveRooms',
  'AveBedrms',
  'Population',
  'AveOccup',
  'Latitude',
  'Longitude',
]

/**
 * Parse StatLib's `cal_housing.tgz` into sklearn's `fetch_california_housing` form: eight features per block group
 * (median income, median house age, average rooms, bedrooms and occupants per household, population, latitude,
 * longitude) and the median house value in units of $100,000.
 *
 * @param bytes - The gzipped tar archive.
 * @returns The dataset.
 * @throws MdmlError with code `value` when the archive has no `cal_housing.data`.
 * @example
 * ```ts
 * import { parseCaliforniaHousing } from 'mdml-engine/datasets'
 * typeof parseCaliforniaHousing // => 'function'
 * ```
 */
export async function parseCaliforniaHousing(bytes: Uint8Array): Promise<RegressionData> {
  const files = untar(await gunzip(bytes))
  const entry = [...files.entries()].find(([name]) => name.endsWith('cal_housing.data'))
  if (entry === undefined) throw new MdmlError('value', 'parseCaliforniaHousing: no cal_housing.data in the archive')
  const rows = parseCsv(new TextDecoder().decode(entry[1]), { header: false }).rows.map((r) => r.map(Number))
  const n = rows.length
  const x = new Float64Array(n * 8)
  const y = new Float64Array(n)
  for (const [i, r] of rows.entries()) {
    const [lon = 0, lat = 0, age = 0, rooms = 0, bedrooms = 0, population = 0, households = 1, income = 0, value = 0] =
      r
    const feats = [
      income,
      age,
      rooms / households,
      bedrooms / households,
      population,
      population / households,
      lat,
      lon,
    ]
    for (const [j, v] of feats.entries()) x[i * 8 + j] = v
    y[i] = value / 100_000
  }
  return xy(matrix(x, n, 8), vector(y), {
    name: 'californiaHousing',
    description: 'Median house values of 20640 California block groups in the 1990 census (Pace and Barry, 1997).',
    source: 'StatLib cal_housing.tgz (as fetched by scikit-learn)',
    url: 'https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html',
    citation: 'Pace and Barry (1997), Sparse spatial autoregressions, Statistics and Probability Letters 33(3).',
    licence: 'Public domain (US Census data)',
    featureNames: HOUSING_FEATURES,
    targetNames: ['MedHouseVal'],
  })
}

/**
 * The California housing data, a licence-clean replacement for the Boston housing set, as a remote asset (442 kB,
 * pinned by the checksum scikit-learn uses): load it with `load(californiaHousing, { cache })`.
 *
 * @example
 * ```ts
 * import { californiaHousing } from 'mdml-engine/datasets'
 * californiaHousing.kind // => 'remote'
 * ```
 */
export const californiaHousing: DatasetRef<RegressionData> = {
  kind: 'remote',
  name: 'californiaHousing',
  url: 'https://ndownloader.figshare.com/files/5976036',
  sha256: 'aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681',
  size: 441_963,
  licence: 'Public domain (US Census data)',
  parse: parseCaliforniaHousing,
}

// ---------------------------------------------------------------------------------------------------------------
// Growth charts

/** The quantity a growth chart describes. */
export type GrowthMeasure = 'weight' | 'stature'

/** The LMS parameters of a growth chart by age in months. */
export interface GrowthChart {
  readonly measure: GrowthMeasure
  readonly sex: 'male' | 'female'
  readonly ageMonths: readonly number[]
  readonly L: readonly number[]
  readonly M: readonly number[]
  readonly S: readonly number[]
  readonly unit: string
}

/**
 * A CDC 2000 growth chart (weight-for-age in kg or stature-for-age in cm, ages 2 to 20) as LMS parameters: the
 * Box-Cox power `L`, median `M` and coefficient of variation `S` by age (Kuczmarski et al., 2002). Public domain.
 *
 * @param measure - `weight` or `stature`.
 * @param sex - `male` or `female`.
 * @returns The chart.
 * @example
 * ```ts
 * import { growthChart } from 'mdml-engine/datasets'
 * growthChart('stature', 'female').ageMonths[0] // => 24
 * ```
 */
export function growthChart(measure: GrowthMeasure, sex: 'male' | 'female'): GrowthChart {
  const m = GROWTH_COLUMNS['measure'] as readonly string[]
  const s = GROWTH_COLUMNS['sex'] as readonly string[]
  const rows = m.flatMap((v, i) => (v === measure && s[i] === sex ? [i] : []))
  const pick = (name: string) => rows.map((i) => (GROWTH_COLUMNS[name] as readonly number[])[i] ?? 0)
  return {
    measure,
    sex,
    ageMonths: pick('Agemos'),
    L: pick('L'),
    M: pick('M'),
    S: pick('S'),
    unit: measure === 'weight' ? 'kg' : 'cm',
  }
}

function interpolate(xs: readonly number[], ys: readonly number[], x: number): number {
  if (x <= (xs[0] ?? x)) return ys[0] ?? Number.NaN
  for (let i = 1; i < xs.length; i++) {
    const a = xs[i - 1] ?? 0
    const b = xs[i] ?? 0
    if (x <= b) return (ys[i - 1] ?? 0) + ((x - a) / (b - a)) * ((ys[i] ?? 0) - (ys[i - 1] ?? 0))
  }
  return ys[ys.length - 1] ?? Number.NaN
}

/**
 * The value at standard normal score `z` of a growth chart at an age (LMS parameters interpolated linearly in age):
 * `M (1 + L S z)^(1 / L)`, or `M exp(S z)` when `L = 0`.
 *
 * @param chart - The chart.
 * @param ageMonths - Age in months.
 * @param z - The score (for a percentile `p`, `z = Φ^-1(p)`).
 * @returns The measurement.
 * @example
 * ```ts
 * import { growthChart, growthValue } from 'mdml-engine/datasets'
 * const chart = growthChart('weight', 'male')
 * growthValue(chart, 24, 0) === chart.M[0] // => true
 * ```
 */
export function growthValue(chart: GrowthChart, ageMonths: number, z: number): number {
  const l = interpolate(chart.ageMonths, chart.L, ageMonths)
  const m = interpolate(chart.ageMonths, chart.M, ageMonths)
  const s = interpolate(chart.ageMonths, chart.S, ageMonths)
  return Math.abs(l) < 1e-12 ? m * Math.exp(s * z) : m * (1 + l * s * z) ** (1 / l)
}

/**
 * Children's measurements drawn from a CDC growth chart: ages uniform from 2 to 20 years and each measurement the
 * chart's LMS distribution at that age, so the conditional quantiles are known exactly (`truth.quantile`) and the
 * conditional mean is computed by Gauss-Hermite quadrature.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { growthCurves } from 'mdml-engine/datasets'
 * generate(growthCurves, { knobs: { n: 20, measure: 'stature' } }).data.targetNames // => ['stature (cm)']
 * ```
 */
export const growthCurves: DatasetGenerator<RegressionData> = {
  name: 'growthCurves',
  kind: 'regression',
  description: 'Weights or statures of children aged 2 to 20 drawn from the CDC 2000 growth charts.',
  knobs: dict({
    n: int(1, 100_000, { label: 'children', description: 'The number of children measured.', default: 200 }),
    measure: choice(['weight', 'stature'], {
      label: 'measure',
      description: 'Weight (kg) or stature (cm).',
      default: 'weight',
    }),
    sex: choice(['male', 'female'], { label: 'sex', description: 'Which CDC chart to draw from.', default: 'female' }),
    minAge: real(24, 240, {
      description: 'The youngest age drawn, in months.',
      default: 24,
      label: 'youngest age (months)',
    }),
    maxAge: real(24, 240, {
      description: 'The oldest age drawn, in months.',
      default: 240,
      label: 'oldest age (months)',
    }),
  }),
  generate: (k, knobs) => {
    const chart = growthChart(knobs['measure'] as GrowthMeasure, knobs['sex'] as 'male' | 'female')
    const n = knobs['n'] as number
    const ages = uniform(foldIn(k, 0), [n], { low: knobs['minAge'] as number, high: knobs['maxAge'] as number }).data
    const z = normal(foldIn(k, 1), [n]).data
    const y = Float64Array.from(ages, (a, i) => growthValue(chart, a, z[i] ?? 0))
    const rule = gaussHermite(40)
    const nodes = Array.from(rule.nodes.data as ArrayLike<number>)
    const weights = Array.from(rule.weights.data as ArrayLike<number>)
    const mean = (age: number) =>
      nodes.reduce(
        (s, x, i) => s + ((weights[i] ?? 0) / Math.sqrt(Math.PI)) * growthValue(chart, age, Math.SQRT2 * x),
        0,
      )
    const ageOf = (v: Value): number[] => toFlat(v)
    const label = `${chart.measure} (${chart.unit})`
    return {
      data: xy(matrix(ages, n, 1), vector(y), {
        ...stamp('growthCurves', growthCurves.description, knobs),
        sha256: GROWTH_SHA256,
        licence: 'Public domain (CDC)',
        url: 'https://www.cdc.gov/growthcharts/percentile_data_files.htm',
        featureNames: ['age (months)'],
        targetNames: [label],
      }),
      truth: {
        regressionFunction: (v) => vector(ageOf(v).map(mean)) as Tensor,
        quantile: (v, q) => {
          const zq = item(ndtri(q))
          return vector(ageOf(v).map((a) => growthValue(chart, a, zq))) as Tensor
        },
        relevantFeatures: [0],
        latents: { z: vector(z) },
        extra: { chart },
      },
    }
  },
}
