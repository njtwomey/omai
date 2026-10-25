/**
 * A/B and A/B/n test data with known treatment effects: users allocated to a control and treatment arms over several
 * days, with a pre-experiment covariate for variance reduction by CUPED, for conversion, continuous or revenue metrics.
 */

import { type DatasetGenerator, type TableData, table } from '../data/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { categorical, foldIn, normal, randint, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { item } from '../tensor/index.ts'
import { stamp } from './_util.ts'

/** The outcome metrics of {@link abTest}. */
export type AbMetric = 'conversion' | 'continuous' | 'revenue'

/**
 * A/B (and A/B/n) test data with known effects: users are allocated to a control and `arms - 1` treatments, arrive on
 * one of `days` days, and have a pre-experiment value of the metric correlated `rho` with their outcome (for CUPED and
 * regression adjustment). Treatment `j` changes the metric by `lift × j / (arms - 1)` relative to control (so the last
 * arm has the full lift), optionally inflated by a novelty effect that decays over days. The metric is a conversion
 * (a latent Gaussian propensity thresholded), a continuous value, or zero-inflated log-normal revenue. The truth has
 * every arm's population mean, the absolute and relative effects, and the variance reduction CUPED can reach.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { abTest } from 'whitebox-engine/datasets'
 * generate(abTest, { knobs: { users: 100 } }).data.columns['arm']?.kind // => 'categorical'
 * ```
 */
export const abTest: DatasetGenerator<TableData> = {
  name: 'abTest',
  kind: 'table',
  description: 'Randomised experiment data with known treatment effects, pre-period covariates and arrival days.',
  knobs: dict({
    users: int(2, 10_000_000, { label: 'users', description: 'The number of users randomised.', default: 10_000 }),
    arms: int(2, 10, {
      label: 'arms',
      description: 'The number of experiment arms, the control included.',
      default: 2,
    }),
    metric: choice(['conversion', 'continuous', 'revenue'], {
      label: 'metric',
      description: 'The outcome: a conversion (0/1), a continuous measure or revenue (zero-inflated).',
      default: 'conversion',
    }),
    baseline: real(0.0001, 1000, {
      description: 'The control arm’s conversion rate or mean outcome.',
      default: 0.1,
      label: 'control conversion rate or mean',
    }),
    lift: real(-0.99, 10, {
      description: 'The relative effect of the last arm over control (arms in between interpolate).',
      default: 0.05,
      label: 'relative effect of the last arm',
    }),
    rho: real(0, 0.99, {
      description: 'The correlation between each user’s pre-period covariate and outcome (for CUPED).',
      default: 0.5,
      label: 'pre-period correlation',
    }),
    days: int(1, 365, { label: 'days', description: 'The number of days over which users arrive.', default: 14 }),
    novelty: real(0, 10, {
      description: 'An extra relative effect on the first day that decays over the experiment.',
      default: 0,
      label: 'extra relative effect on day 0, decaying',
    }),
    treatmentShare: real(0.01, 0.99, {
      description: 'The share of users assigned to the treatment arms together.',
      default: 0.5,
      label: 'share of users in treatments',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['users'] as number
    const arms = knobs['arms'] as number
    const metric = knobs['metric'] as AbMetric
    const base = knobs['baseline'] as number
    const lift = knobs['lift'] as number
    const rho = knobs['rho'] as number
    const days = knobs['days'] as number
    const novelty = knobs['novelty'] as number
    const share = knobs['treatmentShare'] as number
    const alloc = [1 - share, ...new Array<number>(arms - 1).fill(share / (arms - 1))]
    const arm = Array.from(categorical(foldIn(k, 0), alloc, [n]).data)
    const day = Array.from(randint(foldIn(k, 1), [n], 0, days).data)
    const latent = normal(foldIn(k, 2), [n]).data
    const preNoise = normal(foldIn(k, 3), [n]).data
    const postNoise = normal(foldIn(k, 4), [n]).data
    const armLift = (a: number) => (a === 0 ? 0 : (lift * a) / (arms - 1))
    const relative = (a: number, d: number) => armLift(a) * (1 + novelty * Math.exp(-d / 3))
    const sd = metric === 'continuous' ? base : 1
    // Both periods share the user latent; their correlation is rho.
    const pre = Array.from(
      { length: n },
      (_, i) => Math.sqrt(rho) * (latent[i] ?? 0) + Math.sqrt(1 - rho) * (preNoise[i] ?? 0),
    )
    const post = Array.from(
      { length: n },
      (_, i) => Math.sqrt(rho) * (latent[i] ?? 0) + Math.sqrt(1 - rho) * (postNoise[i] ?? 0),
    )
    // P(z > -ndtri(rate)) = rate for a standard normal z.
    const threshold = (rate: number) => -item(ndtri(rate))
    let y: number[]
    let preMetric: number[]
    let armMean: (a: number) => number
    if (metric === 'conversion') {
      const rate = (a: number, d: number) => Math.min(1, base * (1 + relative(a, d)))
      y = post.map((z, i) => (z > threshold(rate(arm[i] ?? 0, day[i] ?? 0)) ? 1 : 0))
      preMetric = pre.map((z) => (z > threshold(base) ? 1 : 0))
      armMean = (a) => Math.min(1, base * (1 + armLift(a)))
    } else if (metric === 'continuous') {
      y = post.map((z, i) => base * (1 + relative(arm[i] ?? 0, day[i] ?? 0)) + sd * z)
      preMetric = pre.map((z) => base + sd * z)
      armMean = (a) => base * (1 + armLift(a))
    } else {
      // Revenue: purchase with probability 0.2, then log-normal spend with mean `baseline` scaled by the lift.
      const buy = uniform(foldIn(k, 5), [n]).data
      const sigma = 1
      const mu = Math.log(base) - (sigma * sigma) / 2 - Math.log(0.2)
      y = post.map((z, i) =>
        (buy[i] ?? 1) < 0.2 ? Math.exp(mu + sigma * z) * (1 + relative(arm[i] ?? 0, day[i] ?? 0)) : 0,
      )
      preMetric = pre.map((z, i) => ((buy[(i + 1) % n] ?? 1) < 0.2 ? Math.exp(mu + sigma * z) : 0))
      armMean = (a) => base * (1 + armLift(a))
    }
    const names = ['control', ...Array.from({ length: arms - 1 }, (_, a) => `treatment${a + 1}`)]
    const means = names.map((_, a) => armMean(a))
    return {
      data: table(
        { arm: arm.map((a) => names[a] ?? 'control'), metric: y, pre: preMetric, day },
        { target: 'metric', levels: { arm: names }, meta: stamp('abTest', abTest.description, knobs) },
      ),
      truth: {
        effects: { ate: (means[arms - 1] ?? 0) - (means[0] ?? 0) },
        extra: {
          armMeans: means,
          absoluteEffects: means.map((m) => m - (means[0] ?? 0)),
          relativeEffects: names.map((_, a) => armLift(a)),
          allocation: alloc,
          cupedVarianceReduction: metric === 'continuous' ? rho * rho : undefined,
          novelty,
        },
      },
    }
  },
}
