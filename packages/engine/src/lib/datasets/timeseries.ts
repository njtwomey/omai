/**
 * Time series, synthetic and real: ARMA processes, sinusoids in noise, seasonal series with known trend and seasonal
 * components, and subsampling to uneven times; the airline passengers, sunspots and coal-mining disasters series; and
 * an ECG record shipped with SciPy as a pinned remote asset.
 */

import { MdmlError } from '../base/index.ts'
import {
  type DatasetGenerator,
  type DatasetRef,
  type EventData,
  events,
  type Generated,
  parseNpy,
  type TimeSeriesData,
  timeSeries,
  unzip,
} from '../data/index.ts'
import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import { bool, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, toFlat, type Value } from '../tensor/index.ts'
import { AIR_PASSENGERS_COLUMNS, AIR_PASSENGERS_SHA256 } from './_data/air-passengers.data.ts'
import { COAL_COLUMNS, COAL_SHA256 } from './_data/coal.data.ts'
import { SUNSPOT_SHA256, SUNSPOT_VALUES, SUNSPOT_YEARS } from './_data/sunspots.data.ts'
import { stamp, vector } from './_util.ts'

/** The psi weights of the MA(infinity) form of an ARMA process: `x_t = sum_j psi_j e_{t-j}`. */
function psiWeights(ar: readonly number[], ma: readonly number[], count: number): number[] {
  const psi = new Array<number>(count).fill(0)
  psi[0] = 1
  for (let j = 1; j < count; j++) {
    let s = ma[j - 1] ?? 0
    for (let i = 1; i <= ar.length && i <= j; i++) s += (ar[i - 1] ?? 0) * (psi[j - i] ?? 0)
    psi[j] = s
  }
  return psi
}

/** Whether all roots of `1 - c_1 z - ... - c_p z^p` lie outside the unit circle (by the Schur-Cohn test). */
function stableAr(coeffs: readonly number[]): boolean {
  // Step-down recursion on a(z) = 1 - sum c_i z^i: every reflection coefficient must have modulus below one.
  let a = coeffs.map((c) => -c)
  for (let p = a.length; p > 0; p--) {
    const k = a[p - 1] ?? 0
    if (Math.abs(k) >= 1) return false
    const next: number[] = []
    for (let i = 0; i < p - 1; i++) next.push(((a[i] ?? 0) - k * (a[p - 2 - i] ?? 0)) / (1 - k * k))
    a = next
  }
  return true
}

/** Options of {@link armaProcess}. */
export interface ArmaOptions {
  /** Autoregressive coefficients `phi_1..phi_p` in `x_t = sum phi_i x_{t-i} + e_t + sum theta_j e_{t-j}`. */
  readonly ar?: readonly number[]
  /** Moving-average coefficients `theta_1..theta_q`. */
  readonly ma?: readonly number[]
  readonly n?: number
  /** Innovation standard deviation (default 1). */
  readonly sigma?: number
  /** The process mean (default 0). */
  readonly mean?: number
  /** Steps simulated and discarded first (default 200). */
  readonly burnIn?: number
}

/**
 * Simulate a Gaussian ARMA(p, q) process. The truth carries the coefficients, the innovations, the theoretical
 * autocovariance (from the psi weights), the spectral density and whether the process is stationary and invertible.
 *
 * @param key - The random key.
 * @param options - Coefficients, length, noise, mean and burn-in.
 * @returns The series and its truth.
 * @example
 * ```ts
 * import { armaProcess } from 'mdml-engine/datasets'
 * import { key } from 'mdml-engine/random'
 * armaProcess(key(0), { ar: [0.5], n: 10 }).truth.extra?.['stationary'] // => true
 * ```
 */
export function armaProcess(key: Key, options: ArmaOptions = {}): Generated<TimeSeriesData> {
  const ar = options.ar ?? []
  const ma = options.ma ?? []
  const n = options.n ?? 200
  const sigma = options.sigma ?? 1
  const mu = options.mean ?? 0
  const burn = options.burnIn ?? 200
  const total = n + burn
  const e = Float64Array.from(normal(key, [total]).data, (v) => sigma * v)
  const x = new Float64Array(total)
  for (let t = 0; t < total; t++) {
    let s = e[t] ?? 0
    for (const [i, phi] of ar.entries()) s += phi * (t - i - 1 >= 0 ? (x[t - i - 1] ?? 0) : 0)
    for (const [j, theta] of ma.entries()) s += theta * (t - j - 1 >= 0 ? (e[t - j - 1] ?? 0) : 0)
    x[t] = s
  }
  const series = Float64Array.from(x.subarray(burn), (v) => v + mu)
  const psi = psiWeights(ar, ma, 2000)
  const acvf = (h: number) => {
    let s = 0
    for (let j = 0; j + h < psi.length; j++) s += (psi[j] ?? 0) * (psi[j + h] ?? 0)
    return sigma * sigma * s
  }
  const spectral = (w: number) => {
    const poly = (c: readonly number[], sign: number) => {
      let re = 1
      let im = 0
      for (const [k, v] of c.entries()) {
        re += sign * v * Math.cos((k + 1) * w)
        im -= sign * v * Math.sin((k + 1) * w)
      }
      return re * re + im * im
    }
    return ((sigma * sigma) / (2 * Math.PI)) * (poly(ma, 1) / poly(ar, -1))
  }
  const name = `arma(${ar.length},${ma.length})`
  return {
    data: timeSeries(series, {
      meta: { name, description: `A Gaussian ${name.toUpperCase()} process.`, source: 'mdml-js generator' },
    }),
    truth: {
      noiseScale: sigma,
      latents: { innovations: vector(e.subarray(burn)) },
      extra: {
        ar,
        ma,
        mean: mu,
        stationary: stableAr(ar),
        invertible: stableAr(ma.map((t) => -t)),
        autocovariance: Array.from({ length: 21 }, (_, h) => acvf(h)),
        acvf,
        spectralDensity: spectral,
      },
    },
  }
}

/**
 * An ARMA(2, 2) process with its coefficients as knobs (zero coefficients drop terms), a generator front end to
 * {@link armaProcess}.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { arma } from 'mdml-engine/datasets'
 * generate(arma, { knobs: { n: 50, phi1: 0.9 } }).data.values.shape // => [50, 1]
 * ```
 */
export const arma: DatasetGenerator<TimeSeriesData> = {
  name: 'arma',
  kind: 'timeseries',
  description: 'A Gaussian ARMA(2, 2) process.',
  knobs: dict({
    n: int(1, 100_000, { label: 'length', description: 'The number of time steps.', default: 200 }),
    phi1: real(-2, 2, { label: 'φ₁', description: 'The first autoregressive coefficient.', default: 0.6 }),
    phi2: real(-1, 1, { label: 'φ₂', description: 'The second autoregressive coefficient.', default: 0 }),
    theta1: real(-2, 2, { label: 'θ₁', description: 'The first moving-average coefficient.', default: 0 }),
    theta2: real(-1, 1, { label: 'θ₂', description: 'The second moving-average coefficient.', default: 0 }),
    sigma: real(0.001, 100, {
      label: 'innovation σ',
      description: 'The standard deviation of the white-noise innovations.',
      default: 1,
    }),
    mean: real(-1000, 1000, { label: 'mean', description: 'The mean level of the process.', default: 0 }),
  }),
  generate: (k, knobs) => {
    const trim = (c: number[]) => {
      let m = c.length
      while (m > 0 && c[m - 1] === 0) m--
      return c.slice(0, m)
    }
    const g = armaProcess(k, {
      ar: trim([knobs['phi1'] as number, knobs['phi2'] as number]),
      ma: trim([knobs['theta1'] as number, knobs['theta2'] as number]),
      n: knobs['n'] as number,
      sigma: knobs['sigma'] as number,
      mean: knobs['mean'] as number,
    })
    return { data: { ...g.data, meta: { ...g.data.meta, ...stamp('arma', arma.description, knobs) } }, truth: g.truth }
  },
}

/**
 * Sinusoids in white noise: `y(t) = sum_k a_k sin(2π f_k t + φ_k) + noise`, sampled evenly or at uneven (uniform
 * random, sorted) times. The truth is the clean signal as a function of time and the frequencies, amplitudes and
 * phases, for spectral estimation (periodograms, Lomb-Scargle).
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { sinusoids } from 'mdml-engine/datasets'
 * (generate(sinusoids, { knobs: { components: 2 } }).truth.extra?.['frequencies'] as number[]).length // => 2
 * ```
 */
export const sinusoids: DatasetGenerator<TimeSeriesData> = {
  name: 'sinusoids',
  kind: 'timeseries',
  description: 'A sum of sinusoids with random frequencies, amplitudes and phases in white noise.',
  knobs: dict({
    n: int(2, 100_000, { label: 'length', description: 'The number of samples.', default: 256 }),
    components: int(1, 20, { label: 'components', description: 'The number of sinusoids summed.', default: 2 }),
    samplingRate: real(0.01, 10_000, {
      description: 'Samples per unit time.',
      default: 32,
      label: 'samples per unit time',
    }),
    noise: real(0, 10, {
      label: 'noise standard deviation',
      description: 'The standard deviation of the white noise.',
      default: 0.5,
    }),
    uneven: bool({
      description: 'Sample at uneven (jittered) times instead of a regular grid.',
      default: false,
      label: 'uneven sampling',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const rate = knobs['samplingRate'] as number
    const m = knobs['components'] as number
    const noise = knobs['noise'] as number
    const duration = n / rate
    // Frequencies between a few cycles per record and a third of the Nyquist frequency.
    const freqs = Array.from(uniform(foldIn(k, 0), [m], { low: 3 / duration, high: rate / 6 }).data)
    const amps = Array.from(uniform(foldIn(k, 1), [m], { low: 0.5, high: 2 }).data)
    const phases = Array.from(uniform(foldIn(k, 2), [m], { low: 0, high: 2 * Math.PI }).data)
    const signal = (t: number) =>
      freqs.reduce((s, f, i) => s + (amps[i] ?? 0) * Math.sin(2 * Math.PI * f * t + (phases[i] ?? 0)), 0)
    const times =
      knobs['uneven'] === true
        ? Float64Array.from(uniform(foldIn(k, 3), [n], { low: 0, high: duration }).data).sort()
        : Float64Array.from({ length: n }, (_, i) => i / rate)
    const z = normal(foldIn(k, 4), [n]).data
    const y = Float64Array.from(times, (t, i) => signal(t) + noise * (z[i] ?? 0))
    return {
      data: timeSeries(y, {
        times,
        ...(knobs['uneven'] === true ? {} : { samplingRate: rate }),
        meta: stamp('sinusoids', sinusoids.description, knobs),
      }),
      truth: {
        regressionFunction: (t: Value) => vector(toFlat(t).map(signal)) as Tensor,
        noiseScale: noise,
        bayesError: noise * noise,
        extra: { frequencies: freqs, amplitudes: amps, phases },
      },
    }
  },
}

/**
 * A seasonal series with known components for decomposition methods (STL, classical decomposition, seasonal ARIMA):
 * a linear trend, a seasonal pattern of `harmonics` sinusoids with period `period`, and AR(1) noise. The components
 * are `truth.latents.trend`, `seasonal` and `remainder`.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { seasonalSeries } from 'mdml-engine/datasets'
 * generate(seasonalSeries, { knobs: { period: 7 } }).data.period // => 7
 * ```
 */
export const seasonalSeries: DatasetGenerator<TimeSeriesData> = {
  name: 'seasonalSeries',
  kind: 'timeseries',
  description: 'Linear trend plus a periodic seasonal pattern plus AR(1) noise.',
  knobs: dict({
    n: int(2, 100_000, { label: 'length', description: 'The number of time steps.', default: 144 }),
    period: int(2, 1000, { label: 'period', description: 'The length of the season in steps.', default: 12 }),
    harmonics: int(1, 10, {
      label: 'harmonics',
      description: 'Fourier harmonics making up the seasonal pattern.',
      default: 2,
    }),
    trend: real(-10, 10, { description: 'The linear trend added per step.', default: 0.05, label: 'trend per step' }),
    amplitude: real(0, 100, {
      description: 'The amplitude of the seasonal pattern.',
      default: 2,
      label: 'seasonal amplitude',
    }),
    noise: real(0, 100, {
      label: 'noise standard deviation',
      description: 'The standard deviation of the AR(1) noise innovations.',
      default: 0.5,
    }),
    phi: real(-0.99, 0.99, {
      description: 'The AR(1) coefficient of the noise.',
      default: 0.5,
      label: 'noise AR(1) coefficient',
    }),
    multiplicative: bool({
      description: 'Make the seasonal swing proportional to the level.',
      default: false,
      label: 'seasonality grows with the level',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const p = knobs['period'] as number
    const h = knobs['harmonics'] as number
    const slope = knobs['trend'] as number
    const amp = knobs['amplitude'] as number
    const noise = knobs['noise'] as number
    const phi = knobs['phi'] as number
    const coef = Array.from(normal(foldIn(k, 0), [h, 2]).data)
    const shape = (t: number) => {
      let s = 0
      for (let j = 0; j < h; j++)
        s +=
          ((coef[2 * j] ?? 0) * Math.cos((2 * Math.PI * (j + 1) * t) / p) +
            (coef[2 * j + 1] ?? 0) * Math.sin((2 * Math.PI * (j + 1) * t) / p)) /
          (j + 1)
      return s
    }
    let scale = 0
    for (let t = 0; t < p; t++) scale = Math.max(scale, Math.abs(shape(t)))
    const e = normal(foldIn(k, 1), [n]).data
    const trend = Float64Array.from({ length: n }, (_, t) => 10 + slope * t)
    const seasonal = Float64Array.from({ length: n }, (_, t) => {
      const s = (amp * shape(t)) / (scale || 1)
      return knobs['multiplicative'] === true ? (s * (trend[t] ?? 0)) / 10 : s
    })
    const remainder = new Float64Array(n)
    for (let t = 0; t < n; t++)
      remainder[t] = phi * (t > 0 ? (remainder[t - 1] ?? 0) : 0) + noise * Math.sqrt(1 - phi * phi) * (e[t] ?? 0)
    const y = Float64Array.from({ length: n }, (_, t) => (trend[t] ?? 0) + (seasonal[t] ?? 0) + (remainder[t] ?? 0))
    return {
      data: timeSeries(y, {
        samplingRate: 1,
        period: p,
        meta: stamp('seasonalSeries', seasonalSeries.description, knobs),
      }),
      truth: {
        latents: { trend: vector(trend), seasonal: vector(seasonal), remainder: vector(remainder) },
        noiseScale: noise,
        extra: { period: p, slope, phi },
      },
    }
  },
}

/**
 * Keep a random subset of a series' time steps (in order), turning an evenly sampled series into an unevenly sampled
 * one with the same times.
 *
 * @param data - The series.
 * @param key - The random key.
 * @param fraction - The fraction of steps kept (default 0.5).
 * @returns The thinned series (without a sampling rate).
 * @throws MdmlError with code `value` for a fraction outside (0, 1].
 * @example
 * ```ts
 * import { timeSeries } from 'mdml-engine/data'
 * import { subsampleSeries } from 'mdml-engine/datasets'
 * import { key } from 'mdml-engine/random'
 * subsampleSeries(timeSeries([1, 2, 3, 4]), key(0), 0.5).values.shape // => [2, 1]
 * ```
 */
export function subsampleSeries(data: TimeSeriesData, key: Key, fraction = 0.5): TimeSeriesData {
  if (!(fraction > 0 && fraction <= 1)) throw new MdmlError('value', 'subsampleSeries: fraction must be in (0, 1]')
  const n = data.times.shape[0] ?? 0
  const m = Math.max(1, Math.round(fraction * n))
  const u = uniform(key, [n]).data
  const keep = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => (u[a] ?? 0) - (u[b] ?? 0))
    .slice(0, m)
    .sort((a, b) => a - b)
  const c = data.values.shape[1] ?? 1
  const values = Float64Array.from(
    keep.flatMap((i) => Array.from(data.values.data.slice(i * c, (i + 1) * c) as ArrayLike<number>)),
  )
  const { samplingRate: _rate, ...rest } = data
  return {
    ...rest,
    times: vector(keep.map((i) => data.times.data[i] ?? 0)),
    values: fromStorage(values, [m, c], 'float64'),
    ...(data.segments === undefined
      ? {}
      : {
          segments: fromStorage(
            Int32Array.from(keep, (i) => data.segments?.data[i] ?? 0),
            [m],
            'int32',
          ),
        }),
  }
}

/**
 * Monthly totals of international airline passengers (thousands), January 1949 to December 1960 (Box and Jenkins'
 * series G): trend, growing seasonality and a period of 12. Times are decimal years.
 *
 * @returns The series.
 * @example
 * ```ts
 * import { airPassengers } from 'mdml-engine/datasets'
 * airPassengers().values.shape // => [144, 1]
 * ```
 */
export function airPassengers(): TimeSeriesData {
  return timeSeries(AIR_PASSENGERS_COLUMNS['value'] as readonly number[] as number[], {
    times: AIR_PASSENGERS_COLUMNS['time'] as readonly number[] as number[],
    samplingRate: 12,
    period: 12,
    channelNames: ['passengers'],
    meta: {
      name: 'airPassengers',
      description: 'Monthly international airline passengers in thousands, 1949 to 1960 (Box and Jenkins).',
      source: 'R datasets::AirPassengers via Rdatasets',
      url: 'https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html',
      citation: 'Box, Jenkins and Reinsel (1976), Time Series Analysis, Forecasting and Control, series G.',
      licence: 'Public domain data; series from R (GPL-2 | GPL-3)',
      sha256: AIR_PASSENGERS_SHA256,
    },
  })
}

/**
 * Yearly mean sunspot numbers, 1700 to 2008, with their roughly 11-year cycle.
 *
 * @returns The series.
 * @example
 * ```ts
 * import { sunspots } from 'mdml-engine/datasets'
 * sunspots().times.data[0] // => 1700
 * ```
 */
export function sunspots(): TimeSeriesData {
  return timeSeries(SUNSPOT_VALUES as number[], {
    times: SUNSPOT_YEARS as number[],
    samplingRate: 1,
    channelNames: ['sunspots'],
    meta: {
      name: 'sunspots',
      description: 'Yearly mean sunspot numbers, 1700 to 2008.',
      source: 'statsmodels datasets.sunspots',
      url: 'https://www.statsmodels.org/stable/datasets/generated/sunspots.html',
      citation: 'Hoyt and Schatten (1998); NOAA National Geophysical Data Center.',
      licence: 'Public domain (NOAA)',
      sha256: SUNSPOT_SHA256,
    },
  })
}

const COAL_META = {
  description: 'Dates of the 191 explosions that killed ten or more British coal miners, 1851 to 1962 (Jarrett, 1979).',
  source: 'boot::coal via Rdatasets',
  url: 'https://vincentarelbundock.github.io/Rdatasets/doc/boot/coal.html',
  citation: 'Jarrett (1979), A note on the intervals between coal-mining disasters, Biometrika 66(1).',
  licence: 'Public domain data; dataset from the R package boot (GPL-2)',
  sha256: COAL_SHA256,
}

/**
 * British coal-mining disasters as event times (decimal years) over the window 1851 to 1962, the classic changepoint
 * example (the rate drops around 1890).
 *
 * @returns The events.
 * @example
 * ```ts
 * import { coalMiningDisasters } from 'mdml-engine/datasets'
 * coalMiningDisasters().times?.shape // => [191]
 * ```
 */
export function coalMiningDisasters(): EventData {
  const dates = COAL_COLUMNS['date'] as readonly number[]
  return events({ times: dates as number[], window: [1851, 1963], meta: { name: 'coalMiningDisasters', ...COAL_META } })
}

/**
 * British coal-mining disasters as yearly counts, 1851 to 1962.
 *
 * @returns The series of counts.
 * @example
 * ```ts
 * import { coalMiningCounts } from 'mdml-engine/datasets'
 * coalMiningCounts().values.shape // => [112, 1]
 * ```
 */
export function coalMiningCounts(): TimeSeriesData {
  const counts = new Array<number>(112).fill(0)
  for (const d of COAL_COLUMNS['date'] as readonly number[]) {
    const y = Math.floor(d) - 1851
    if (y >= 0 && y < 112) counts[y] = (counts[y] ?? 0) + 1
  }
  return timeSeries(counts, {
    times: counts.map((_, i) => 1851 + i),
    samplingRate: 1,
    channelNames: ['disasters'],
    meta: { name: 'coalMiningCounts', ...COAL_META },
  })
}

/**
 * Parse scipy's `ecg.dat` (an `.npz` archive holding `ecg.npy`) into millivolts: `(raw - 1024) / 200`, sampled at
 * 360 Hz.
 *
 * @param bytes - The archive.
 * @returns The series.
 * @throws MdmlError with code `value` when the archive has no `ecg.npy`.
 * @example
 * ```ts
 * import { parseElectrocardiogram } from 'mdml-engine/datasets'
 * typeof parseElectrocardiogram // => 'function'
 * ```
 */
export async function parseElectrocardiogram(bytes: Uint8Array): Promise<TimeSeriesData> {
  const files = await unzip(bytes)
  const npy = files.get('ecg.npy')
  if (npy === undefined) throw new MdmlError('value', 'parseElectrocardiogram: no ecg.npy in the archive')
  const { values } = parseNpy(npy)
  return timeSeries(
    Float64Array.from(values, (v) => (v - 1024) / 200),
    {
      samplingRate: 360,
      channelNames: ['mV'],
      meta: {
        name: 'electrocardiogram',
        description: 'Five minutes of an electrocardiogram (MIT-BIH arrhythmia record 208) at 360 Hz, in millivolts.',
        source: 'scipy.datasets.electrocardiogram',
        url: 'https://github.com/scipy/dataset-ecg',
        citation: 'Moody and Mark (2001), The impact of the MIT-BIH Arrhythmia Database, IEEE EMB Magazine 20(3).',
        licence: 'ODC-By 1.0 (PhysioNet)',
      },
    },
  )
}

/**
 * An electrocardiogram record (MIT-BIH record 208 as shipped with SciPy, 108000 samples at 360 Hz) as a remote asset
 * pinned by its SHA-256: load it with `load(electrocardiogram, { cache })`.
 *
 * @example
 * ```ts
 * import { electrocardiogram } from 'mdml-engine/datasets'
 * electrocardiogram.kind // => 'remote'
 * ```
 */
export const electrocardiogram: DatasetRef<TimeSeriesData> = {
  kind: 'remote',
  name: 'electrocardiogram',
  url: 'https://raw.githubusercontent.com/scipy/dataset-ecg/main/ecg.dat',
  sha256: 'f20ad3365fb9b7f845d0e5c48b6fe67081377ee466c3a220b7f69f35c8958baf',
  size: 119_035,
  licence: 'ODC-By 1.0 (PhysioNet)',
  parse: parseElectrocardiogram,
}
