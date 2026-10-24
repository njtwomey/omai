import { type DatasetGenerator, type EventData, events, type Generated } from '../data/index.ts'
import { exponential, foldIn, type Key, poisson, uniform } from '../random/index.ts'
import { choice, dict, real } from '../space/index.ts'
import { labels, matrix, stamp } from './_util.ts'

/** Draw a Poisson count. */
function count(k: Key, mean: number): number {
  return poisson(k, mean, [1]).data[0] ?? 0
}

/** Sorted uniform times on [0, T]. */
function uniformTimes(k: Key, n: number, horizon: number): number[] {
  return Array.from(uniform(k, [n], { low: 0, high: horizon }).data).sort((a, b) => a - b)
}

/**
 * A homogeneous Poisson process on `[0, horizon]`: the count is Poisson with mean `rate × horizon` and the times are
 * uniform. The truth gives the intensity, the compensator and the exact log-likelihood of the realisation.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { poissonProcess } from 'whitebox-engine/datasets'
 * generate(poissonProcess, { knobs: { rate: 2, horizon: 10 } }).data.window // => [0, 10]
 * ```
 */
export const poissonProcess: DatasetGenerator<EventData> = {
  name: 'poissonProcess',
  kind: 'events',
  description: 'A homogeneous Poisson process on [0, horizon].',
  knobs: dict({
    rate: real(0.001, 10_000, { label: 'rate', description: 'The mean number of events per unit time.', default: 1 }),
    horizon: real(0.01, 100_000, {
      label: 'horizon',
      description: 'The length of the observation window.',
      default: 100,
    }),
  }),
  generate: (k, knobs) => {
    const rate = knobs['rate'] as number
    const horizon = knobs['horizon'] as number
    const times = uniformTimes(foldIn(k, 1), count(foldIn(k, 0), rate * horizon), horizon)
    return {
      data: events({ times, window: [0, horizon], meta: stamp('poissonProcess', poissonProcess.description, knobs) }),
      truth: {
        extra: {
          intensity: () => rate,
          compensator: (t: number) => rate * t,
          logLikelihood: times.length * Math.log(rate) - rate * horizon,
          rate,
        },
      },
    }
  },
}

/** The intensity shapes of {@link inhomogeneousPoisson}. */
export type IntensityShape = 'sinusoidal' | 'piecewise' | 'ramp'

/**
 * An inhomogeneous Poisson process on `[0, horizon]` drawn by Lewis-Shedler thinning, with a sinusoidal intensity
 * `base (1 + amplitude sin(2π t / period))`, a piecewise-constant intensity (rate changes at known times, also
 * `truth.changepoints`) or a linear ramp. The truth gives the intensity, its integral and the log-likelihood.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { inhomogeneousPoisson } from 'whitebox-engine/datasets'
 * const g = generate(inhomogeneousPoisson, { knobs: { shape: 'piecewise', horizon: 30 } })
 * g.truth.changepoints // => [10, 20]
 * ```
 */
export const inhomogeneousPoisson: DatasetGenerator<EventData> = {
  name: 'inhomogeneousPoisson',
  kind: 'events',
  description: 'A Poisson process with a time-varying intensity, drawn by thinning.',
  knobs: dict({
    shape: choice(['sinusoidal', 'piecewise', 'ramp'], {
      label: 'intensity shape',
      description: 'A sinusoidal, piecewise-constant or ramp intensity.',
      default: 'sinusoidal',
    }),
    base: real(0.001, 10_000, { description: 'The base intensity.', default: 1, label: 'base rate' }),
    amplitude: real(0, 1, {
      description: 'The relative amplitude of the sinusoidal intensity.',
      default: 0.8,
      label: 'relative amplitude (sinusoidal)',
    }),
    period: real(0.001, 100_000, {
      label: 'period',
      description: 'The period of the sinusoidal intensity.',
      default: 25,
    }),
    horizon: real(0.01, 100_000, {
      label: 'horizon',
      description: 'The length of the observation window.',
      default: 100,
    }),
  }),
  generate: (k, knobs) => {
    const shape = knobs['shape'] as IntensityShape
    const base = knobs['base'] as number
    const amp = knobs['amplitude'] as number
    const period = knobs['period'] as number
    const horizon = knobs['horizon'] as number
    const breaks = [horizon / 3, (2 * horizon) / 3]
    const levels = [base, 3 * base, 0.5 * base]
    const intensity = (t: number): number => {
      if (shape === 'sinusoidal') return base * (1 + amp * Math.sin((2 * Math.PI * t) / period))
      if (shape === 'piecewise') return levels[t < (breaks[0] ?? 0) ? 0 : t < (breaks[1] ?? 0) ? 1 : 2] ?? base
      return (2 * base * t) / horizon
    }
    const compensator = (t: number): number => {
      if (shape === 'sinusoidal')
        return base * (t + ((amp * period) / (2 * Math.PI)) * (1 - Math.cos((2 * Math.PI * t) / period)))
      if (shape === 'piecewise') {
        const [b0 = 0, b1 = 0] = breaks
        const [l0 = 0, l1 = 0, l2 = 0] = levels
        return l0 * Math.min(t, b0) + l1 * Math.max(0, Math.min(t, b1) - b0) + l2 * Math.max(0, t - b1)
      }
      return (base * t * t) / horizon
    }
    const bound = shape === 'sinusoidal' ? base * (1 + amp) : shape === 'piecewise' ? 3 * base : 2 * base
    const candidates = uniformTimes(foldIn(k, 1), count(foldIn(k, 0), bound * horizon), horizon)
    const u = uniform(foldIn(k, 2), [candidates.length]).data
    const times = candidates.filter((t, i) => (u[i] ?? 1) * bound < intensity(t))
    const logLik = times.reduce((s, t) => s + Math.log(intensity(t)), 0) - compensator(horizon)
    return {
      data: events({
        times,
        window: [0, horizon],
        meta: stamp('inhomogeneousPoisson', inhomogeneousPoisson.description, knobs),
      }),
      truth: {
        ...(shape === 'piecewise' ? { changepoints: breaks } : {}),
        extra: { intensity, compensator, logLikelihood: logLik, shape },
      },
    }
  },
}

/** Simulate a multivariate exponential-kernel Hawkes process by its branching (cluster) representation. */
function simulateHawkes(
  k: Key,
  mu: readonly number[],
  alpha: readonly (readonly number[])[],
  beta: number,
  horizon: number,
): { times: number[]; marks: number[]; parents: number[] } {
  const kTypes = mu.length
  type Ev = { t: number; m: number; parent: number }
  const all: Ev[] = []
  for (const [m, rate] of mu.entries()) {
    for (const t of uniformTimes(foldIn(foldIn(k, 0), m), count(foldIn(foldIn(k, 1), m), rate * horizon), horizon))
      all.push({ t, m, parent: -1 })
  }
  // Breadth-first: each event of type l spawns Poisson(alpha[m][l]) children of type m at Exp(beta) delays.
  for (let i = 0, salt = 0; i < all.length && all.length < 1_000_000; i++) {
    const ev = all[i] as Ev
    for (let m = 0; m < kTypes; m++) {
      const kids = count(foldIn(foldIn(k, 2), salt++), alpha[m]?.[ev.m] ?? 0)
      if (kids === 0) continue
      const delays = exponential(foldIn(foldIn(k, 3), salt++), [kids], beta).data
      for (const d of delays) if (ev.t + d <= horizon) all.push({ t: ev.t + d, m, parent: i })
    }
  }
  const order = all.map((_, i) => i).sort((a, b) => (all[a]?.t ?? 0) - (all[b]?.t ?? 0))
  const position = new Map(order.map((i, p) => [i, p]))
  return {
    times: order.map((i) => all[i]?.t ?? 0),
    marks: order.map((i) => all[i]?.m ?? 0),
    parents: order.map((i) => {
      const p = all[i]?.parent ?? -1
      return p < 0 ? -1 : (position.get(p) ?? -1)
    }),
  }
}

/** The conditional intensity and exact log-likelihood of a multivariate exponential Hawkes process. */
function hawkesTruth(
  times: readonly number[],
  marks: readonly number[],
  mu: readonly number[],
  alpha: readonly (readonly number[])[],
  beta: number,
  horizon: number,
): { intensity: (t: number, type?: number) => number; logLikelihood: number } {
  const intensity = (t: number, type = 0): number => {
    let s = mu[type] ?? 0
    for (const [i, ti] of times.entries()) {
      if (ti >= t) break
      s += (alpha[type]?.[marks[i] ?? 0] ?? 0) * beta * Math.exp(-beta * (t - ti))
    }
    return s
  }
  // Recursive form: the excitation of each type decays by exp(-beta dt) between events.
  const kTypes = mu.length
  const excitation = new Array<number>(kTypes).fill(0)
  let last = 0
  let logLik = 0
  for (const [i, t] of times.entries()) {
    const decay = Math.exp(-beta * (t - last))
    for (let m = 0; m < kTypes; m++) excitation[m] = (excitation[m] ?? 0) * decay
    const type = marks[i] ?? 0
    logLik += Math.log((mu[type] ?? 0) + (excitation[type] ?? 0))
    for (let m = 0; m < kTypes; m++) excitation[m] = (excitation[m] ?? 0) + (alpha[m]?.[type] ?? 0) * beta
    last = t
  }
  let compensator = 0
  for (let m = 0; m < kTypes; m++) {
    compensator += (mu[m] ?? 0) * horizon
    for (const [i, t] of times.entries())
      compensator += (alpha[m]?.[marks[i] ?? 0] ?? 0) * (1 - Math.exp(-beta * (horizon - t)))
  }
  return { intensity, logLikelihood: logLik - compensator }
}

/**
 * A self-exciting Hawkes process with exponential kernel, `λ(t) = μ + Σ_{t_i < t} α β e^{-β (t - t_i)}`: each event
 * triggers on average `α` (the branching ratio, below 1 for stationarity) further events after Exp(β) delays.
 * Simulated by its branching representation, so the truth knows each event's parent (`-1` for immigrants), and gives
 * the conditional intensity of the realisation and its exact log-likelihood.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { hawkesProcess } from 'whitebox-engine/datasets'
 * generate(hawkesProcess, { knobs: { horizon: 50 } }).truth.extra?.['branchingRatio'] // => 0.5
 * ```
 */
export const hawkesProcess: DatasetGenerator<EventData> = {
  name: 'hawkesProcess',
  kind: 'events',
  description: 'A self-exciting Hawkes process with an exponential kernel.',
  knobs: dict({
    mu: real(0.001, 1000, {
      description: 'The background rate of events not triggered by others.',
      default: 0.5,
      label: 'background rate',
    }),
    alpha: real(0, 0.99, {
      description: 'The expected number of events each event triggers (below 1 for stability).',
      default: 0.5,
      label: 'branching ratio',
    }),
    beta: real(0.001, 1000, {
      description: 'How fast the excitement from an event decays.',
      default: 1,
      label: 'decay rate',
    }),
    horizon: real(0.01, 100_000, {
      label: 'horizon',
      description: 'The length of the observation window.',
      default: 100,
    }),
  }),
  generate: (k, knobs) => {
    const mu = knobs['mu'] as number
    const alpha = knobs['alpha'] as number
    const beta = knobs['beta'] as number
    const horizon = knobs['horizon'] as number
    const { times, marks, parents } = simulateHawkes(k, [mu], [[alpha]], beta, horizon)
    const t = hawkesTruth(times, marks, [mu], [[alpha]], beta, horizon)
    return {
      data: events({ times, window: [0, horizon], meta: stamp('hawkesProcess', hawkesProcess.description, knobs) }),
      truth: {
        latents: { parents: labels(parents) },
        extra: {
          intensity: (s: number) => t.intensity(s),
          logLikelihood: t.logLikelihood,
          branchingRatio: alpha,
          stationaryRate: mu / (1 - alpha),
          mu,
          beta,
        },
      },
    }
  },
}

/** Options of {@link multivariateHawkes}. */
export interface MultivariateHawkesOptions {
  /** Background rate of each type. */
  readonly mu: readonly number[]
  /** `alpha[k][l]`: the expected number of type-k events triggered by one type-l event. */
  readonly alpha: readonly (readonly number[])[]
  readonly beta?: number
  readonly horizon?: number
}

/**
 * A mutually exciting (multivariate) Hawkes process with a shared exponential decay: events carry their type as a
 * mark, and the truth has every event's parent, each type's conditional intensity and the exact log-likelihood.
 *
 * @param key - The random key.
 * @param options - Background rates, excitation matrix, decay and horizon.
 * @returns The marked events and their truth.
 * @example
 * ```ts
 * import { multivariateHawkes } from 'whitebox-engine/datasets'
 * import { key } from 'whitebox-engine/random'
 * multivariateHawkes(key(0), { mu: [0.2, 0.1], alpha: [[0.3, 0.2], [0, 0.4]], horizon: 20 }).data.markNames // => ['type0', 'type1']
 * ```
 */
export function multivariateHawkes(key: Key, options: MultivariateHawkesOptions): Generated<EventData> {
  const beta = options.beta ?? 1
  const horizon = options.horizon ?? 100
  const { times, marks, parents } = simulateHawkes(key, options.mu, options.alpha, beta, horizon)
  const t = hawkesTruth(times, marks, options.mu, options.alpha, beta, horizon)
  return {
    data: events({
      times,
      marks,
      window: [0, horizon],
      markNames: options.mu.map((_, i) => `type${i}`),
      meta: {
        name: 'multivariateHawkes',
        description: 'A mutually exciting Hawkes process.',
        source: 'whitebox-js generator',
      },
    }),
    truth: {
      latents: { parents: labels(parents) },
      extra: { intensity: t.intensity, logLikelihood: t.logLikelihood, mu: options.mu, alpha: options.alpha, beta },
    },
  }
}

/**
 * A spatial Poisson process on a rectangle: homogeneous, or with a Gaussian bump of intensity in the middle (drawn by
 * thinning). The truth gives the intensity at any location and its integral.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { spatialPoisson } from 'whitebox-engine/datasets'
 * generate(spatialPoisson, { knobs: { width: 2, height: 1 } }).data.region // => [[0, 2], [0, 1]]
 * ```
 */
export const spatialPoisson: DatasetGenerator<EventData> = {
  name: 'spatialPoisson',
  kind: 'events',
  description: 'A Poisson point process in a rectangle, homogeneous or with a central bump of intensity.',
  knobs: dict({
    rate: real(0.001, 100_000, {
      description: 'The base intensity per unit area.',
      default: 100,
      label: 'base intensity per unit area',
    }),
    bump: real(0, 100, {
      description: 'The peak of the extra central intensity, relative to the base (0: homogeneous).',
      default: 0,
      label: 'peak extra intensity relative to the base',
    }),
    width: real(0.01, 10_000, { label: 'width', description: 'The width of the rectangle.', default: 1 }),
    height: real(0.01, 10_000, { label: 'height', description: 'The height of the rectangle.', default: 1 }),
  }),
  generate: (k, knobs) => {
    const rate = knobs['rate'] as number
    const bump = knobs['bump'] as number
    const w = knobs['width'] as number
    const h = knobs['height'] as number
    const sd = Math.min(w, h) / 6
    const intensity = (x: number, y: number) =>
      rate * (1 + bump * Math.exp(-(((x - w / 2) ** 2 + (y - h / 2) ** 2) / (2 * sd * sd))))
    const bound = rate * (1 + bump)
    const m = count(foldIn(k, 0), bound * w * h)
    const xs = uniform(foldIn(k, 1), [m], { low: 0, high: w }).data
    const ys = uniform(foldIn(k, 2), [m], { low: 0, high: h }).data
    const u = uniform(foldIn(k, 3), [m]).data
    const pts: number[] = []
    for (let i = 0; i < m; i++) {
      if ((u[i] ?? 1) * bound < intensity(xs[i] ?? 0, ys[i] ?? 0)) pts.push(xs[i] ?? 0, ys[i] ?? 0)
    }
    // The bump is (almost exactly) a Gaussian inside the rectangle: its integral is 2π sd² (6 sd fits in the box).
    const total = rate * (w * h + bump * 2 * Math.PI * sd * sd)
    return {
      data: events({
        locations: matrix(pts, pts.length / 2, 2),
        region: [
          [0, w],
          [0, h],
        ],
        meta: stamp('spatialPoisson', spatialPoisson.description, knobs),
      }),
      truth: { extra: { intensity, expectedCount: total } },
    }
  },
}
