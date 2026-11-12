/**
 * The campaigns of the periodic validation for the particle machine and the evidence estimators: every evidence estimator
 * against the closed-form evidence of the conjugate benchmarks (`evidence/conjugate`), the harmonic mean shown to fail where
 * bridge sampling does not (`evidence/harmonic-mean`), nested sampling against analytic Gaussian evidences with the
 * coverage of its error bar (`evidence/nested-gaussian`), the particle filters' evidence estimates against the Kalman filter,
 * unbiased on the natural scale (`evidence/particle-filter`), and particle marginal Metropolis-Hastings and particle Gibbs
 * against the exact posterior of the AR(1) coefficient (`evidence/pmcmc-ar1`). Each is a `Campaign` with a budget that
 * scales; `scripts/probprog-validate.ts` runs them after the others, and none runs in `make verify` or `make check`.
 */

import type { ParticleModel } from '../../dynamics/filtering/index.ts'
import { filtering } from '../../dynamics/index.ts'
import { normal as drawNormal, foldIn, type Key, uniform } from '../../random/index.ts'
import {
  bridgeSampling,
  harmonicMeanEvidence,
  type NestedResult,
  nestedSampler,
  proposal,
} from '../../sampling/index.ts'
import { add, arange, item, mul, sin, toFlat, type Value } from '../../tensor/index.ts'
import { type Config, session } from '../../trace/index.ts'
import { evidenceEstimators, type InferenceMethod, particleGibbsMethod, pmmhMethod } from '../inference/index.ts'
import type { ModelAnalysis } from '../inference/model-input.ts'
import type { Campaign, CampaignOutcome } from '../validation/runner.ts'
import { AR1_NOISE, ar1CoefficientFamily, ar1CoefficientPosterior, ar1Observations } from './ar1-parameters.ts'
import { betaBernoulli, gammaPoisson, normalNormal } from './conjugate.ts'

const scaled = (base: number, scale: number, min = 5): number => Math.max(min, Math.round(base * scale))
const seedOf = (key: Key, i = 0): number => Math.floor((uniform(foldIn(key, i), [1]).data[0] ?? 0) * 2 ** 30)
const cell = (v: unknown): string => String(v).replaceAll('|', '\\|')
const fix = (x: number, d = 3): string => (Number.isFinite(x) ? x.toFixed(d) : 'n/a')
const table = (head: readonly string[], rows: readonly (readonly string[])[]): string =>
  [
    `| ${head.join(' | ')} |`,
    `| ${head.map(() => '---').join(' | ')} |`,
    ...rows.map((r) => `| ${r.map(cell).join(' | ')} |`),
  ].join('\n')
const meanOf = (v: readonly number[]): number => v.reduce((a, b) => a + b, 0) / v.length
const sdOf = (v: readonly number[]): number => {
  const m = meanOf(v)
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, v.length - 1))
}

type AnyMethod = InferenceMethod<unknown, Config, string, ModelAnalysis>

/** What each estimator of the conjugate campaign is run with, at the budget scale, and the slack it is allowed beyond its stated error. */
function estimatorRuns(scale: number): readonly {
  readonly kind: string
  readonly config: Partial<Config>
  readonly steps: number
  readonly slack: number
  readonly note: string
  /** False: the row is reported, not asserted. */
  readonly assert?: boolean
}[] {
  return [
    {
      kind: 'bridge-sampling',
      config: { numDraws: scaled(1500, scale) },
      steps: 20000,
      slack: 0.03,
      note: 'slice-sampling posterior draws, fitted Gaussian',
    },
    {
      kind: 'stepping-stone',
      config: { numParticles: scaled(400, scale), targetEss: 0.95 },
      steps: 400,
      slack: 0.06,
      note: 'adaptive tempering ladder at target ESS 0.95',
    },
    {
      kind: 'thermodynamic-integration',
      config: { numParticles: scaled(400, scale), targetEss: 0.95 },
      steps: 400,
      slack: 0.1,
      note: 'the same ladder, trapezoid rule: a discretisation bias remains',
    },
    {
      kind: 'importance-evidence',
      config: { numBatches: scaled(40, scale), batchSize: 250 },
      steps: 400,
      slack: 0.03,
      note: 'pilot-fitted Gaussian proposal with a defensive prior share',
    },
    {
      kind: 'nested',
      config: { numLive: scaled(150, scale), innerSteps: 4 },
      steps: 100000,
      slack: 0.03,
      note: 'static nested sampling, slice sweeps',
    },
    {
      kind: 'tempered-smc',
      config: { numParticles: scaled(400, scale) },
      steps: 400,
      slack: 0.12,
      assert: false,
      note: 'the existing tempered SMC at its default ladder of two or three rungs (no error estimate): reported, not asserted',
    },
  ]
}

function conjugateCampaign(): Campaign {
  return {
    name: 'evidence/conjugate',
    description:
      'Every evidence estimator on the three conjugate benchmark models against their closed-form log evidence: each must be within four of its own standard errors plus a slack for the bias the error does not state.',
    run: ({ key, scale }): CampaignOutcome => {
      const rows: string[][] = []
      let failures = 0
      let total = 0
      for (const [mi, model] of [betaBernoulli, normalNormal, gammaPoisson].entries()) {
        const truth = model.truth().logEvidence ?? Number.NaN
        for (const [ei, run] of estimatorRuns(scale).entries()) {
          const entry = evidenceEstimators[run.kind]
          if (entry?.make === undefined) continue
          const method = entry.make(model.methodModel() as never) as unknown as AnyMethod
          const s = session(method as never, {
            steps: run.steps,
            mode: 'eager',
            seed: seedOf(key, mi * 10 + ei),
            config: run.config,
          })
          const z = method.posterior?.(s.state, s.config).logEvidence
          const err = (z?.value ?? Number.NaN) - truth
          const se = z?.standardError
          const ok = Math.abs(err) < 4 * (se !== undefined && Number.isFinite(se) ? se : 0) + run.slack
          if (run.assert !== false) {
            total += 1
            if (!ok) failures += 1
          }
          rows.push([
            model.name,
            run.kind,
            fix(z?.value ?? Number.NaN, 4),
            fix(truth, 4),
            fix(err, 4),
            se === undefined ? 'n/a' : fix(se, 4),
            se !== undefined && se > 0 ? fix(err / se, 2) : 'n/a',
            `${s.t} steps (${s.status})`,
            ok ? 'ok' : run.assert === false ? 'outside (reported)' : '**outside**',
          ])
        }
      }
      return {
        passed: failures === 0,
        verdict:
          failures === 0
            ? `${total} asserted estimates within their stated error (plus slack) of the closed-form log evidence`
            : `${failures} of ${total} estimates outside their stated error of the closed-form log evidence`,
        markdown: [
          "The truth is the closed-form evidence of `probprog/validation/exact.ts` (one run per cell; the error column is the estimate minus the truth, the standard error the estimator's own, z their ratio).",
          '',
          table(['model', 'estimator', 'estimate', 'truth', 'error', 'se', 'z', 'run', 'verdict'], rows),
          '',
          estimatorRuns(scale)
            .map((r) => `- \`${r.kind}\`: ${r.note}; slack ${r.slack}`)
            .join('\n'),
        ].join('\n'),
        budget: { models: 3, estimators: 6, scalePercent: Math.round(scale * 100) },
      }
    },
  }
}

function harmonicCampaign(): Campaign {
  return {
    name: 'evidence/harmonic-mean',
    description:
      'The harmonic mean estimator is refused by default; shown with its infinite variance acknowledged, on exact posterior draws of a diffuse-prior sharp-likelihood model, it fails where bridge sampling on the same draws does not.',
    run: ({ key, scale }): CampaignOutcome => {
      const reps = scaled(40, scale, 10)
      const draws = scaled(400, scale, 100)
      const rows: string[][] = []
      let refused = false
      try {
        harmonicMeanEvidence([-1, -2])
      } catch {
        refused = true
      }
      let harmonicWorse = true
      for (const s of [0.5, 0.1, 0.02]) {
        // Prior N(0, 10^2), likelihood exp(-theta^2 / (2 s^2)): the posterior is N(0, v), the draws below are exact.
        const s0 = 10
        const v = 1 / (1 / (s0 * s0) + 1 / (s * s))
        const truth = 0.5 * Math.log(2 * Math.PI * s * s) - 0.5 * Math.log(2 * Math.PI * (s0 * s0 + s * s))
        const harmonic: number[] = []
        const bridge: number[] = []
        for (let i = 0; i < reps; i++) {
          const x = Float64Array.from(
            toFlat(drawNormal(foldIn(key, 1000 * Math.round(s * 100) + i), [draws])),
            (e) => Math.sqrt(v) * e,
          )
          harmonic.push(
            harmonicMeanEvidence(
              Array.from(x, (t) => (-0.5 * t * t) / (s * s)),
              { acknowledgeInfiniteVariance: true },
            ).value,
          )
          bridge.push(
            bridgeSampling(
              (p) =>
                (-0.5 * (p[0] ?? 0) ** 2) / (s * s) -
                0.5 * ((p[0] ?? 0) / s0) ** 2 -
                Math.log(s0 * Math.sqrt(2 * Math.PI)),
              x,
              1,
              { key: foldIn(key, 7_000_000 + 1000 * Math.round(s * 100) + i) },
            ).logEvidence,
          )
        }
        const rms = (a: number[]): number => Math.sqrt(a.reduce((p, q) => p + (q - truth) ** 2, 0) / a.length)
        const rh = rms(harmonic)
        const rb = rms(bridge)
        if (!(rh > 3 * rb)) harmonicWorse = false
        rows.push([
          fix(s, 2),
          fix(truth, 3),
          fix(meanOf(harmonic) - truth, 3),
          fix(rh, 3),
          fix(meanOf(bridge) - truth, 3),
          fix(rb, 3),
          fix(rh / rb, 1),
        ])
      }
      return {
        passed: refused && harmonicWorse,
        verdict: `${refused ? 'refused by default' : 'NOT refused by default'}; ${harmonicWorse ? "its error is more than three times bridge sampling's at every likelihood width" : 'it was not clearly worse than bridge sampling'}`,
        markdown: [
          `Prior N(0, 10^2), likelihood exp(-theta^2 / (2 s^2)) at three widths s; ${reps} replications of ${draws} exact posterior draws each. Bias and rms error of the log evidence against the exact one.`,
          '',
          table(
            ['likelihood sd s', 'log Z', 'harmonic bias', 'harmonic rms', 'bridge bias', 'bridge rms', 'rms ratio'],
            rows,
          ),
          '',
          "The harmonic mean is biased upwards (a few draws near the likelihood's peak dominate its average) and its error grows as the likelihood narrows against the prior; the stated standard error of the estimator does not show it.",
        ].join('\n'),
        budget: { replications: reps, draws, widths: 3 },
      }
    },
  }
}

const NESTED_PRIOR_SD = 3
const NESTED_SIGMA = 0.5

function nestedGaussian(
  d: number,
  numLive: number,
  seed: number,
  y: readonly number[],
): { result: NestedResult; truth: number } {
  const prior = proposal({
    dim: d,
    sample: (k) => drawNormal(k, [d]).data.map((v) => v * NESTED_PRIOR_SD),
    logProb: (x) =>
      Array.from(x).reduce(
        (a, v) => a - 0.5 * (v / NESTED_PRIOR_SD) ** 2 - Math.log(NESTED_PRIOR_SD * Math.sqrt(2 * Math.PI)),
        0,
      ),
  })
  const logL = (x: Float64Array): number =>
    (-0.5 * Array.from(x).reduce((a, v, i) => a + (v - (y[i] ?? 0)) ** 2, 0)) / NESTED_SIGMA ** 2
  const v = NESTED_PRIOR_SD ** 2 + NESTED_SIGMA ** 2
  const truth =
    0.5 * d * Math.log(2 * Math.PI * NESTED_SIGMA ** 2) -
    0.5 * d * Math.log(2 * Math.PI * v) -
    y.reduce((a, b) => a + b * b, 0) / (2 * v)
  return { result: nestedSampler(prior, logL, { seed, numLive }), truth }
}

function nestedCampaign(): Campaign {
  return {
    name: 'evidence/nested-gaussian',
    description:
      'Nested sampling against the analytic evidence of a Gaussian likelihood under a Gaussian prior in 1, 2, 5 and 10 dimensions: the error against the stated error over seeds, and how often the stated error covers the truth.',
    run: ({ key, scale }): CampaignOutcome => {
      const seeds = scaled(12, scale, 4)
      const rows: string[][] = []
      let ok = true
      for (const [d, numLive] of [
        [1, 150],
        [2, 150],
        [5, 150],
        [10, 200],
      ] as const) {
        const y = toFlat(mul(sin(add(arange(d), 1)), 0.9))
        const errs: number[] = []
        const ses: number[] = []
        let covered = 0
        let evaluations = 0
        for (let i = 0; i < seeds; i++) {
          const { result, truth } = nestedGaussian(d, scaled(numLive, scale, 40), seedOf(key, d * 1000 + i), y)
          errs.push(result.logZ - truth)
          ses.push(result.logZError)
          if (Math.abs(result.logZ - truth) < 2 * result.logZError) covered += 1
          evaluations += result.evaluations / seeds
        }
        const coverage = covered / seeds
        // The error bar covers at about its nominal 95 percent, allowing for the few seeds and the volume-estimate noise.
        const good = coverage >= 0.7 && Math.abs(meanOf(errs)) < 2 * meanOf(ses)
        if (!good) ok = false
        rows.push([
          String(d),
          String(scaled(numLive, scale, 40)),
          fix(meanOf(errs), 3),
          fix(sdOf(errs), 3),
          fix(meanOf(ses), 3),
          `${covered}/${seeds}`,
          String(Math.round(evaluations)),
          good ? 'ok' : '**off**',
        ])
      }
      return {
        passed: ok,
        verdict: ok
          ? 'the stated error covers the analytic evidence at its nominal rate in every dimension, with no bias beyond it'
          : 'the stated error of nested sampling does not cover the analytic evidence in some dimension',
        markdown: [
          `Prior N(0, ${NESTED_PRIOR_SD}^2 I), likelihood exp(-|x - y|^2 / (2 ${NESTED_SIGMA}^2)); ${seeds} seeds per dimension, slice sampling with 5 sweeps per replacement. Error is the estimate minus the analytic log Z; coverage counts |error| < 2 stated error.`,
          '',
          table(
            [
              'dim',
              'live points',
              'mean error',
              'sd of error',
              'mean stated error',
              'coverage',
              'likelihood evaluations',
              'verdict',
            ],
            rows,
          ),
          '',
          'The stated error is the information-based `sqrt(H / n)` of Skilling; its calibration is what this table checks. The cost grows with the information `H` (about the dimension times the log of the prior-to-posterior width ratio).',
        ].join('\n'),
        budget: { seeds, dimensions: 4 },
      }
    },
  }
}

function particleFilterCampaign(): Campaign {
  return {
    name: 'evidence/particle-filter',
    description:
      "The particle filters' likelihood estimates against the Kalman filter on the AR(1) benchmark: unbiased on the natural scale (the mean of Z-hat / Z over replications is one within its error) for the bootstrap, auxiliary and guided proposals, with the guided one the least variable.",
    run: ({ key, scale }): CampaignOutcome => {
      const reps = scaled(80, scale, 20)
      const particles = 100
      const phi = 0.8
      const Qv = AR1_NOISE.sigmaX ** 2
      const Rv = AR1_NOISE.sigmaY ** 2
      const model = {
        f: (x: Value) => mul(x, phi),
        h: (x: Value) => x,
        Q: [[Qv]],
        R: [[Rv]],
        x0: [0],
        P0: [[Qv / (1 - phi * phi)]],
      }
      const exact = item(
        filtering.kalmanLogLikelihood(
          { F: [[phi]], H: [[1]], Q: [[Qv]], R: [[Rv]], x0: model.x0, P0: model.P0 },
          ar1Observations,
        ),
      )
      const variants: readonly { name: string; model: ParticleModel; run: typeof filtering.bootstrapParticleFilter }[] =
        [
          { name: 'bootstrap', model: filtering.gaussianParticleModel(model), run: filtering.bootstrapParticleFilter },
          { name: 'auxiliary', model: filtering.gaussianParticleModel(model), run: filtering.auxiliaryParticleFilter },
          {
            name: 'guided (locally optimal)',
            model: filtering.gaussianGuidedModel(model, { observationMatrix: [[1]] }),
            run: filtering.bootstrapParticleFilter,
          },
        ]
      const rows: string[][] = []
      const sds: number[] = []
      let unbiased = true
      variants.forEach((v, vi) => {
        const logs = Array.from(
          { length: reps },
          (_, i) => v.run(v.model, ar1Observations, { particles, key: foldIn(key, vi * 100000 + i) }).logEvidence,
        )
        const ratios = logs.map((l) => Math.exp(l - exact))
        const se = sdOf(ratios) / Math.sqrt(reps)
        const ok = Math.abs(meanOf(ratios) - 1) < 4 * se + 0.02
        if (!ok) unbiased = false
        sds.push(sdOf(logs))
        rows.push([
          v.name,
          fix(meanOf(ratios), 3),
          fix(se, 3),
          fix(meanOf(logs) - exact, 3),
          fix(sdOf(logs), 3),
          ok ? 'ok' : '**biased**',
        ])
      })
      const guidedBest = (sds[2] ?? 1) < (sds[0] ?? 0)
      return {
        passed: unbiased && guidedBest,
        verdict: `${unbiased ? 'every proposal unbiased on the natural scale' : 'a proposal is biased'}; ${guidedBest ? 'the guided proposal is less variable than the bootstrap' : 'the guided proposal is NOT less variable'}`,
        markdown: [
          `AR(1) with phi = ${phi}, ${ar1Observations.length} observations, ${particles} particles, ${reps} replications. The reference is the Kalman filter's exact log likelihood ${fix(exact, 4)}.`,
          '',
          table(['proposal', 'mean Z-hat / Z', 'se', 'mean bias of log Z-hat', 'sd of log Z-hat', 'verdict'], rows),
          '',
          "The log of an unbiased estimate is biased low by about half its variance, which is why the bias of log Z-hat is negative and the ratio is the test. The auxiliary filter's first stage uses a point prediction of the state; with process noise as large as the observation noise it is the most variable of the three.",
        ].join('\n'),
        budget: { replications: reps, particles, proposals: 3 },
      }
    },
  }
}

function pmcmcCampaign(): Campaign {
  return {
    name: 'evidence/pmcmc-ar1',
    description:
      "Particle marginal Metropolis-Hastings and particle Gibbs (with and without ancestor sampling) against the exact posterior of the AR(1) coefficient, from the Kalman filter's marginal likelihood on a grid.",
    run: ({ key, scale }): CampaignOutcome => {
      const iterations = scaled(3000, scale, 300)
      const exact = ar1CoefficientPosterior()
      const runs: readonly {
        name: string
        method: AnyMethod
        config: Partial<Config>
      }[] = [
        {
          name: 'PMMH, 50 particles, random walk',
          method: pmmhMethod(ar1CoefficientFamily, ar1Observations) as unknown as AnyMethod,
          config: { numIterations: iterations, particles: 50, stepSize: 0.35 },
        },
        {
          name: 'PMMH, 50 particles, adaptive',
          method: pmmhMethod(ar1CoefficientFamily, ar1Observations) as unknown as AnyMethod,
          config: { numIterations: iterations, particles: 50, proposal: 'adaptive', stepSize: 0.35 },
        },
        {
          name: 'particle Gibbs, ancestor sampling, 24 particles',
          method: particleGibbsMethod(ar1CoefficientFamily, ar1Observations) as unknown as AnyMethod,
          config: { numIterations: iterations, particles: 24, thetaMoves: 3, stepSize: 0.3 },
        },
        {
          name: 'particle Gibbs, no ancestor sampling, 24 particles',
          method: particleGibbsMethod(ar1CoefficientFamily, ar1Observations) as unknown as AnyMethod,
          config: { numIterations: iterations, particles: 24, thetaMoves: 3, stepSize: 0.3, ancestor: 'none' },
        },
      ]
      const rows: string[][] = []
      let ok = true
      for (const [i, r] of runs.entries()) {
        const s = session(r.method as never, {
          steps: iterations,
          mode: 'eager',
          seed: seedOf(key, i),
          config: r.config,
        })
        const post = r.method.posterior?.(s.state, s.config)
        const row = post?.summary()[0]
        const ess = row?.ess ?? 1
        const mcse = exact.sd / Math.sqrt(ess)
        const dz = ((row?.mean ?? Number.NaN) - exact.mean) / mcse
        const sdRatio = (row?.sd ?? Number.NaN) / exact.sd
        const good = Math.abs(dz) < 4 && Math.abs(sdRatio - 1) < 0.3
        if (!good) ok = false
        const state = s.state as { accepted?: number; attempted?: number; t: number; refresh?: number }
        const accept =
          state.attempted === undefined ? (state.accepted ?? 0) / state.t : (state.accepted ?? 0) / state.attempted
        rows.push([
          r.name,
          fix(row?.mean ?? Number.NaN, 3),
          fix(exact.mean, 3),
          fix(dz, 2),
          fix(row?.sd ?? Number.NaN, 3),
          fix(exact.sd, 3),
          fix(ess, 0),
          fix(accept, 2),
          state.refresh === undefined ? '-' : fix(state.refresh, 2),
          good ? 'ok' : '**off**',
        ])
      }
      return {
        passed: ok,
        verdict: ok
          ? 'every chain reproduces the exact posterior mean (within four Monte Carlo standard errors) and spread (within 30 percent)'
          : 'a chain does not reproduce the exact posterior of the coefficient',
        markdown: [
          `AR(1) benchmark data (${ar1Observations.length} observations), uniform prior on phi in (-0.99, 0.99), known noise. Exact posterior of phi from the Kalman marginal likelihood on a grid: mean ${fix(exact.mean, 4)}, sd ${fix(exact.sd, 4)}. ${iterations} iterations each, first 20 percent discarded; z is the mean error in Monte Carlo standard errors (the sd over the square root of the chain's ESS).`,
          '',
          table(
            ['method', 'mean', 'exact mean', 'z', 'sd', 'exact sd', 'ESS', 'accept', 'path refresh', 'verdict'],
            rows,
          ),
          '',
          'Path refresh is the fraction of times at which a conditional SMC sweep changed the state path: ancestor sampling keeps it near 1 with few particles, while without it the early path stays stuck to the reference (the path degeneracy that ancestor sampling removes).',
        ].join('\n'),
        budget: { iterations, chains: 4 },
      }
    },
  }
}

/**
 * The campaigns for the particle machine and the evidence estimators, in a fixed order (campaign `i` runs under the key the
 * runner derives from its position after the others, so adding one at the end changes no other table).
 *
 * @returns The campaigns.
 * @concept marginal-likelihood
 * @index function Periodic validation of the evidence estimators, nested sampling, particle filters and particle MCMC.
 * @example
 * ```ts
 * import { evidenceCampaigns } from 'omai-engine/probprog/models'
 * evidenceCampaigns().map((c) => c.name)[0] // => 'evidence/conjugate'
 * ```
 */
export function evidenceCampaigns(): Campaign[] {
  return [conjugateCampaign(), harmonicCampaign(), nestedCampaign(), particleFilterCampaign(), pmcmcCampaign()]
}
