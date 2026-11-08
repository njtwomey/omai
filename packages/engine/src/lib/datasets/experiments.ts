/**
 * A/B and A/B/n test data with known treatment effects: users allocated to a control and treatment arms over several
 * days, with a pre-experiment covariate for variance reduction by CUPED, for conversion, continuous or revenue metrics;
 * experiments with interference on a network, and two experiments in a row with carryover.
 */

import { type DatasetGenerator, type RegressionData, type TableData, table, xy } from '../data/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { categorical, foldIn, normal, randint, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { item } from '../tensor/index.ts'
import { matrix, stamp, vector } from './_util.ts'

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
 * import { generate } from 'omai-engine/data'
 * import { abTest } from 'omai-engine/datasets'
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

/**
 * An experiment on a network with interference: units on a ring lattice with random rewiring (a small-world graph),
 * assigned to treatment individually or in contiguous blocks, whose outcome depends on their own treatment and on the
 * share of their neighbours treated, `y = base + direct T + spillover S + noise`. Individual randomisation
 * mis-estimates the effect of treating everyone, since control units next to treated ones benefit too. Features are
 * `[treatment, exposure, degree, covariate]`. The truth has the direct effect, the spillover (the effect of all
 * neighbours treated), the total effect of treating everyone against no one, the edges and the clusters.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { networkExperiment } from 'omai-engine/datasets'
 * generate(networkExperiment, { knobs: { units: 40 } }).truth.extra?.['totalEffect'] // => 1.5
 * ```
 */
export const networkExperiment: DatasetGenerator<RegressionData> = {
  name: 'networkExperiment',
  kind: 'regression',
  description:
    'Treatment assigned on a network with spillover onto untreated neighbours; direct and total effects known.',
  knobs: dict({
    units: int(10, 1_000_000, { label: 'units', description: 'The number of units (nodes).', default: 500 }),
    neighbours: int(1, 50, {
      label: 'neighbours per side',
      description: 'Each node is linked to this many nodes on each side of the ring.',
      default: 2,
    }),
    rewiring: real(0, 1, {
      label: 'rewiring',
      description: 'The chance that an edge is moved to a random node.',
      default: 0.1,
    }),
    randomisation: choice(['unit', 'cluster'], {
      label: 'randomisation',
      description: 'Each unit is assigned alone, or blocks of consecutive nodes are assigned together.',
      default: 'unit',
    }),
    clusterSize: int(1, 10_000, {
      label: 'cluster size',
      description: 'Nodes per block under cluster randomisation.',
      default: 25,
    }),
    treated: real(0, 1, {
      label: 'treated share',
      description: 'The chance that a unit (or block) is treated.',
      default: 0.5,
    }),
    direct: real(-100, 100, {
      label: 'direct effect',
      description: 'The effect of a unit’s own treatment.',
      default: 1,
    }),
    spillover: real(-100, 100, {
      label: 'spillover',
      description: 'The effect on a unit of all its neighbours being treated.',
      default: 0.5,
    }),
    noise: real(0, 10, { label: 'noise standard deviation', description: 'The outcome noise.', default: 1 }),
  }),
  generate: (k, knobs) => {
    const n = knobs['units'] as number
    const half = Math.min(knobs['neighbours'] as number, Math.floor((n - 1) / 2))
    const rewire = knobs['rewiring'] as number
    const ur = uniform(foldIn(k, 0), [n * half, 2]).data
    const neighbours: Set<number>[] = Array.from({ length: n }, () => new Set<number>())
    const edges: [number, number][] = []
    for (let i = 0; i < n; i++)
      for (let s = 1; s <= half; s++) {
        const idx = i * half + (s - 1)
        let j = (i + s) % n
        if ((ur[2 * idx] ?? 1) < rewire) {
          j = Math.floor((ur[2 * idx + 1] ?? 0) * n)
          if (j === i) j = (i + 1) % n
        }
        if (!neighbours[i]?.has(j)) {
          neighbours[i]?.add(j)
          neighbours[j]?.add(i)
          edges.push([i, j])
        }
      }
    const size = knobs['randomisation'] === 'cluster' ? (knobs['clusterSize'] as number) : 1
    const blocks = Math.ceil(n / size)
    const ut = uniform(foldIn(k, 1), [blocks]).data
    const cluster = Array.from({ length: n }, (_, i) => Math.floor(i / size))
    const treatment = cluster.map((b) => ((ut[b] ?? 1) < (knobs['treated'] as number) ? 1 : 0))
    const exposure = treatment.map((_, i) => {
      const nb = [...(neighbours[i] ?? [])]
      return nb.length === 0 ? 0 : nb.reduce((a, j) => a + (treatment[j] ?? 0), 0) / nb.length
    })
    const covariate = normal(foldIn(k, 2), [n]).data
    const eps = normal(foldIn(k, 3), [n]).data
    const direct = knobs['direct'] as number
    const spill = knobs['spillover'] as number
    const sd = knobs['noise'] as number
    const y = treatment.map(
      (t, i) => (covariate[i] ?? 0) + direct * t + spill * (exposure[i] ?? 0) + sd * (eps[i] ?? 0),
    )
    const x = new Float64Array(n * 4)
    for (let i = 0; i < n; i++)
      x.set([treatment[i] ?? 0, exposure[i] ?? 0, neighbours[i]?.size ?? 0, covariate[i] ?? 0], i * 4)
    return {
      data: xy(matrix(x, n, 4), vector(y), {
        ...stamp('networkExperiment', networkExperiment.description, knobs),
        featureNames: ['treatment', 'exposure', 'degree', 'covariate'],
        targetNames: ['outcome'],
      }),
      truth: {
        noiseScale: sd,
        effects: { ate: direct + spill },
        extra: {
          directEffect: direct,
          spillover: spill,
          totalEffect: direct + spill,
          edges,
          clusters: cluster,
          exposureColumn: 1,
        },
      },
    }
  },
}

/**
 * Two experiments chained on one user population, with carryover: the first assigns treatment `T1` whose effect
 * persists (by a factor `persistence`) into the second period, and the second experiment's assignment `T2` repeats a
 * user's first arm with probability `stickiness` (users who stay in their arm). A naive comparison in the second
 * experiment mixes its own effect with the first's carryover. Features are `[t1, t2, pre]`; the outcome is the second
 * period's. The truth has both effects and the exact carryover bias of the naive difference in means.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { carryoverExperiment } from 'omai-engine/datasets'
 * generate(carryoverExperiment, { knobs: { users: 20, stickiness: 1, persistence: 0.5, firstEffect: 2 } }).truth.extra?.['naiveBias'] // => 1
 * ```
 */
export const carryoverExperiment: DatasetGenerator<RegressionData> = {
  name: 'carryoverExperiment',
  kind: 'regression',
  description: 'Two experiments in a row on one population, the first leaving a persistent effect in the second.',
  knobs: dict({
    users: int(4, 10_000_000, { label: 'users', description: 'The number of users.', default: 4000 }),
    firstEffect: real(-100, 100, {
      label: 'first effect',
      description: 'The first experiment’s effect in its own period.',
      default: 1,
    }),
    secondEffect: real(-100, 100, {
      label: 'second effect',
      description: 'The second experiment’s effect.',
      default: 0.5,
    }),
    persistence: real(0, 1, {
      label: 'persistence',
      description: 'The share of the first effect that remains in the second period.',
      default: 0.6,
    }),
    stickiness: real(0, 1, {
      label: 'stickiness',
      description: 'The chance that a user keeps the arm of the first experiment in the second.',
      default: 0.5,
    }),
    noise: real(0, 10, { label: 'noise standard deviation', description: 'The outcome noise.', default: 1 }),
  }),
  generate: (k, knobs) => {
    const n = knobs['users'] as number
    const stick = knobs['stickiness'] as number
    const u = uniform(foldIn(k, 0), [n, 3]).data
    const t1 = Array.from({ length: n }, (_, i) => ((u[3 * i] ?? 1) < 0.5 ? 1 : 0))
    const t2 = t1.map((t, i) => ((u[3 * i + 1] ?? 1) < stick ? t : (u[3 * i + 2] ?? 1) < 0.5 ? 1 : 0))
    const pre = normal(foldIn(k, 1), [n]).data
    const eps = normal(foldIn(k, 2), [n]).data
    const e1 = knobs['firstEffect'] as number
    const e2 = knobs['secondEffect'] as number
    const carry = (knobs['persistence'] as number) * e1
    const sd = knobs['noise'] as number
    const y = t2.map((t, i) => 0.8 * (pre[i] ?? 0) + e2 * t + carry * (t1[i] ?? 0) + sd * (eps[i] ?? 0))
    const x = new Float64Array(n * 3)
    for (let i = 0; i < n; i++) x.set([t1[i] ?? 0, t2[i] ?? 0, pre[i] ?? 0], i * 3)
    // P(T1 = 1 | T2 = 1) - P(T1 = 1 | T2 = 0) = stickiness for fair coins.
    return {
      data: xy(matrix(x, n, 3), vector(y), {
        ...stamp('carryoverExperiment', carryoverExperiment.description, knobs),
        featureNames: ['t1', 't2', 'pre'],
        targetNames: ['outcome'],
      }),
      truth: {
        noiseScale: sd,
        effects: { ate: e2 },
        extra: { firstEffect: e1, secondEffect: e2, carryover: carry, naiveBias: carry * stick },
      },
    }
  },
}
