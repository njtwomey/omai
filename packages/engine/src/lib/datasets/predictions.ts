/**
 * Synthetic model outputs and outcomes with known structure: predicted probabilities with a chosen miscalibration
 * (over- or under-confidence, bias) for calibration metrics and recalibration, and survival data with known hazards
 * (Weibull or exponential proportional hazards, log-normal accelerated failure time) under censoring.
 */

import { softmaxOf } from '../base/index.ts'
import { type ClassificationData, type DatasetGenerator, type RegressionData, xy, xyClasses } from '../data/index.ts'
import { ndtr } from '../numerics/special/index.ts'
import { dirichlet, exponential, foldIn, type Key, normal, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { item } from '../tensor/index.ts'
import { labels, matrix, stamp, vector } from './_util.ts'

/** The calibration distortions of {@link miscalibratedPredictions}. */
export type Distortion = 'calibrated' | 'overconfident' | 'underconfident' | 'shifted'

/**
 * Model outputs with known miscalibration, for calibration metrics and recalibration methods: true class
 * probabilities drawn from a Dirichlet, labels drawn from them, and reported probabilities the true ones distorted in
 * log space (`overconfident`: log probabilities multiplied by `strength`, sharpening them; `underconfident`: divided by
 * it, flattening them; `shifted`: class 0's log probability raised by `log(strength)`). Features are the reported probabilities; the truth has the true probabilities, the distortion and
 * the population expected calibration error estimated on 100000 fresh draws with 15 bins.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { miscalibratedPredictions } from 'omai-engine/datasets'
 * generate(miscalibratedPredictions, { knobs: { classes: 3, n: 10 } }).data.x.shape // => [10, 3]
 * ```
 */
export const miscalibratedPredictions: DatasetGenerator<ClassificationData> = {
  name: 'miscalibratedPredictions',
  kind: 'classification',
  description: 'Reported class probabilities that distort the true ones, with the labels drawn from the truth.',
  knobs: dict({
    n: int(1, 10_000_000, { label: 'samples', description: 'The number of predictions.', default: 5000 }),
    classes: int(2, 100, { label: 'classes', description: 'The number of classes.', default: 2 }),
    distortion: choice(['calibrated', 'overconfident', 'underconfident', 'shifted'], {
      label: 'distortion',
      description: 'How the reported probabilities distort the true ones.',
      default: 'overconfident',
    }),
    strength: real(1, 10, {
      description: 'The temperature factor (over- or underconfidence) or the size of the shift.',
      default: 2,
      label: 'temperature factor or shift size',
    }),
    concentration: real(0.01, 100, {
      description: 'The Dirichlet concentration of the true probabilities: low values give confident truths.',
      default: 1,
      label: 'Dirichlet concentration of the true probabilities',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const kc = knobs['classes'] as number
    const strength = knobs['strength'] as number
    const distortion = knobs['distortion'] as Distortion
    const distort = (p: readonly number[]): number[] => {
      const logits = p.map((v) => Math.log(Math.max(v, 1e-300)))
      if (distortion === 'overconfident') return softmaxOf(logits.map((v) => v * strength))
      if (distortion === 'underconfident') return softmaxOf(logits.map((v) => v / strength))
      if (distortion === 'shifted') return softmaxOf(logits.map((v, c) => (c === 0 ? v + Math.log(strength) : v)))
      return [...p]
    }
    const draw = (key: Key, m: number) => {
      const truth = dirichlet(foldIn(key, 0), new Array<number>(kc).fill(knobs['concentration'] as number), m).data
      const u = uniform(foldIn(key, 1), [m]).data
      const y: number[] = []
      const reported: number[] = []
      for (let i = 0; i < m; i++) {
        const p = Array.from(truth.subarray(i * kc, (i + 1) * kc))
        let acc = 0
        let c = kc - 1
        for (const [j, pj] of p.entries()) {
          acc += pj
          if ((u[i] ?? 1) < acc) {
            c = j
            break
          }
        }
        y.push(c)
        reported.push(...distort(p))
      }
      return { truth, y, reported }
    }
    const sample = draw(foldIn(k, 0), n)
    // Population ECE (top-label, 15 equal-width bins), by Monte Carlo with the exact correctness probabilities,
    // computed on first call.
    let cached: number | undefined
    const expectedCalibrationError = (): number => {
      if (cached !== undefined) return cached
      const big = draw(foldIn(k, 1), 100_000)
      const bins = Array.from({ length: 15 }, () => ({ conf: 0, acc: 0, count: 0 }))
      for (let i = 0; i < 100_000; i++) {
        const r = big.reported.slice(i * kc, (i + 1) * kc)
        let top = 0
        for (let c = 1; c < kc; c++) if ((r[c] ?? 0) > (r[top] ?? 0)) top = c
        const conf = r[top] ?? 0
        const bin = bins[Math.min(14, Math.floor(conf * 15))] as { conf: number; acc: number; count: number }
        bin.conf += conf
        bin.acc += big.truth[i * kc + top] ?? 0
        bin.count++
      }
      cached = Math.round(bins.reduce((t, bin) => t + Math.abs(bin.acc - bin.conf) / 100_000, 0) * 1e4) / 1e4
      return cached
    }
    return {
      data: xyClasses(matrix(sample.reported, n, kc), labels(sample.y), {
        ...stamp('miscalibratedPredictions', miscalibratedPredictions.description, knobs),
        featureNames: Array.from({ length: kc }, (_, c) => `p${c}`),
      }),
      truth: {
        latents: { trueProbabilities: matrix(sample.truth, n, kc) },
        extra: { distortion, strength, expectedCalibrationError, distort },
      },
    }
  },
}

/** The event-time models of {@link survivalData}. */
export type SurvivalModel = 'weibull' | 'exponential' | 'lognormal'

/**
 * Survival data with known hazards: covariates `x ~ N(0, I)`, event times from a Weibull proportional-hazards model
 * (`h(t | x) = λ k t^(k-1) exp(β·x)`), its exponential special case, or a log-normal accelerated failure time model,
 * and independent exponential censoring at rate `censoring`. Targets are `[time, event]` (event 1 when observed). The
 * truth has the coefficients, the hazard and survival functions and the true event times.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { survivalData } from 'omai-engine/datasets'
 * generate(survivalData, { knobs: { n: 10 } }).data.targetNames // => ['time', 'event']
 * ```
 */
export const survivalData: DatasetGenerator<RegressionData> = {
  name: 'survivalData',
  kind: 'regression',
  description: 'Right-censored event times from a known proportional-hazards or AFT model.',
  knobs: dict({
    n: int(1, 10_000_000, { label: 'samples', description: 'The number of subjects.', default: 500 }),
    features: int(0, 100, {
      label: 'covariates',
      description: 'The number of covariates (0: one population with no covariates).',
      default: 3,
    }),
    model: choice(['weibull', 'exponential', 'lognormal'], {
      label: 'model',
      description: 'The baseline event-time distribution: Weibull, exponential or log-normal (AFT).',
      default: 'weibull',
    }),
    shape: real(0.1, 10, {
      description: 'The Weibull shape k (hazard rising above 1), or the log-normal σ.',
      default: 1.5,
      label: 'Weibull shape k (or log-normal sigma)',
    }),
    scale: real(0.0001, 1000, {
      description: 'The baseline hazard rate λ.',
      default: 0.1,
      label: 'baseline rate lambda',
    }),
    censoring: real(0, 100, {
      description: 'The rate of the exponential censoring times (0: no censoring).',
      default: 0.05,
      label: 'censoring rate',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const d = knobs['features'] as number
    const model = knobs['model'] as SurvivalModel
    const kShape = model === 'exponential' ? 1 : (knobs['shape'] as number)
    const lam = knobs['scale'] as number
    const beta = Array.from({ length: d }, (_, j) => (j % 2 === 0 ? 0.5 : -0.5) / Math.sqrt(1 + j))
    const x = normal(foldIn(k, 0), [n, d]).data
    const eta = (row: ArrayLike<number>) => beta.reduce((s, b, j) => s + b * (row[j] ?? 0), 0)
    const u = uniform(foldIn(k, 1), [n]).data
    const z = normal(foldIn(k, 2), [n]).data
    // Weibull PH: S(t) = exp(-lam t^k e^eta), so T = (-log U / (lam e^eta))^(1/k). Log-normal AFT: log T = mu - eta + sigma Z.
    const mu = -Math.log(lam)
    const times = Array.from({ length: n }, (_, i) => {
      const e = eta(x.subarray(i * d, (i + 1) * d))
      if (model === 'lognormal') return Math.exp(mu - e + kShape * (z[i] ?? 0))
      return (-Math.log(Math.max(u[i] ?? 0.5, 1e-300)) / (lam * Math.exp(e))) ** (1 / kShape)
    })
    const cens = knobs['censoring'] as number
    const c =
      cens > 0
        ? Array.from(exponential(foldIn(k, 3), [n], cens).data)
        : new Array<number>(n).fill(Number.POSITIVE_INFINITY)
    const observed = times.map((t, i) => Math.min(t, c[i] ?? t))
    const event = times.map((t, i) => (t <= (c[i] ?? Number.POSITIVE_INFINITY) ? 1 : 0))
    const survival = (t: number, row: ArrayLike<number>): number => {
      const e = eta(row)
      if (model === 'lognormal') {
        const w = (Math.log(t) - (mu - e)) / kShape
        return item(ndtr(-w))
      }
      return Math.exp(-lam * t ** kShape * Math.exp(e))
    }
    const hazard = (t: number, row: ArrayLike<number>): number => lam * kShape * t ** (kShape - 1) * Math.exp(eta(row))
    return {
      data: xy(
        matrix(x, n, d),
        matrix(
          observed.flatMap((t, i) => [t, event[i] ?? 0]),
          n,
          2,
        ),
        {
          ...stamp('survivalData', survivalData.description, knobs),
          targetNames: ['time', 'event'],
        },
      ),
      truth: {
        relevantFeatures: beta.map((_, j) => j),
        latents: { eventTimes: vector(times), censoringTimes: vector(c) },
        extra: {
          coefficients: beta,
          model,
          survival,
          ...(model === 'lognormal' ? {} : { hazard }),
          censoringRate: cens,
        },
      },
    }
  },
}
