/**
 * The campaigns of the periodic validation: calibration of NUTS, of variational guides (and of deliberately wrong
 * ones, which must fail), agreement of methods with the grid posterior, the funnel under non-centring, and Geweke
 * tests of the handler-layer kernels. Each is a `Campaign` with a budget that scales; the script
 * `scripts/probprog-validate.ts` runs them and writes the tables, and `make probprog-validate` calls it. None of this
 * runs in `make verify` or `make check`.
 */

import { bernoulli, halfCauchy, halfNormal, normal } from '../../dists/index.ts'
import { adam } from '../../optim/index.ts'
import { normal as drawNormal, foldIn, type Key } from '../../random/index.ts'
import { add, item, maximum, mul, tensor } from '../../tensor/index.ts'
import { laplaceApproximation, mcmc, svi } from '../engines/index.ts'
import { autoDelta, autoNormal } from '../guides.ts'
import { type KernelState, randomWalkKernel, resimulationKernel } from '../inference/index.ts'
import { type Model, observe, plate, sample } from '../primitives.ts'
import { locScale, reparam } from '../reparam.ts'
import { agree, agreementMarkdown, agreementWithin, fromPosterior } from './agree.ts'
import { gewekeTest } from './geweke.ts'
import { gridPosterior } from './grid.ts'
import type { Campaign, CampaignOutcome } from './runner.ts'
import { type SbcResult, sbc, sbcMarkdown } from './sbc.ts'

const scaled = (base: number, scale: number, min = 10): number => Math.max(min, Math.round(base * scale))
const z = (k: Key): number => item(drawNormal(k, []))

// m ~ N(0, 1), y ~ N(m, 1): the posterior of m is N(y / 2, 1 / 2).
const simulateNormal = (k: Key): { data: number; truth: Record<string, number> } => {
  const m = z(foldIn(k, 0))
  return { data: m + z(foldIn(k, 1)), truth: { m } }
}
const normalProgram =
  (y: number): Model =>
  () => {
    const m = sample('m', normal(0, 1))
    observe('y', normal(m, 1), y)
  }

function calibrated(
  name: string,
  description: string,
  fit: (y: number, k: Key) => Record<string, number[]>,
  replications: number,
  mustFail: boolean,
): Campaign {
  return {
    name,
    description,
    run: ({ key, scale }): CampaignOutcome => {
      const reps = scaled(replications, scale)
      const r = sbc({ simulate: simulateNormal, fit, key, replications: reps, numRanks: 19 })
      return {
        passed: r.passes !== mustFail,
        verdict: mustFail
          ? `${r.passes ? 'NOT detected' : 'detected'}: the deliberately wrong method ${r.passes ? 'passed' : 'failed'} calibration`
          : `${r.passes ? 'calibrated' : 'NOT calibrated'} over ${reps} replications`,
        markdown: sbcMarkdown(r),
        budget: { replications: reps, ranks: 19 },
      }
    },
  }
}

const nutsFit =
  (numWarmup: number) =>
  (y: number, k: Key): Record<string, number[]> => {
    // 190 draws, thinned evenly to the 19 ranked: consecutive NUTS draws are autocorrelated, which breaks the uniformity of the ranks (SBC needs independent draws).
    const post = mcmc(normalProgram(y), undefined, { key: k, numChains: 1, numWarmup, numDraws: 190 })
    return { m: Array.from(post.draws?.['m']?.data ?? []) }
  }

const sviFit =
  (kind: 'normal' | 'delta', numSteps: number) =>
  (y: number, k: Key): Record<string, number[]> => {
    const model = normalProgram(y)
    const r = svi(model, undefined, {
      key: k,
      guide: kind === 'delta' ? autoDelta(model) : autoNormal(model),
      numSteps,
      optimizer: adam({ learningRate: (t: number) => 0.15 / (1 + t / 30) }),
    })
    return { m: Array.from(r.posterior.sample(foldIn(k, 9), 19)['m']?.data ?? []) }
  }

/** A Gaussian guide that is too narrow by a factor: the exact posterior with its spread scaled. */
const scaledGuide =
  (factor: number) =>
  (y: number, k: Key): Record<string, number[]> => ({
    m: Array.from({ length: 19 }, (_, i) => y / 2 + factor * Math.SQRT1_2 * z(foldIn(k, i))),
  })

// Eight schools in miniature: four groups, non-centred, with the group effects as quantities.
const GROUPS = 4
function hierarchical(y: readonly number[]): Model {
  return reparam(
    () => {
      const mu = sample('mu', normal(0, 1))
      const tau = sample('tau', halfNormal(1))
      plate('g', GROUPS, () => {
        // A warm-up trajectory can push tau to underflow: floor it so that the scale stays valid.
        const theta = sample('theta', normal(mu, maximum(tau, 1e-9)))
        observe('y', normal(theta, 0.5), y)
      })
    },
    { theta: locScale(0) },
  )
}

const logisticX = Array.from({ length: 16 }, (_, i) => -1.5 + (3 * i) / 15)
const logisticY = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1]
const logistic: Model = () => {
  const b0 = sample('b0', normal(0, 2))
  const b1 = sample('b1', normal(0, 2))
  plate('n', logisticX.length, () => {
    observe('y', bernoulli({ logits: add(b0, mul(b1, tensor(logisticX))) }), logisticY)
  })
}

/**
 * The campaigns, in a fixed order (campaign `i` runs under the key derived from the run's key by `i`, so adding one at
 * the end changes no other).
 *
 * @returns The campaigns.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { validationCampaigns } from 'omai-engine/probprog/validation'
 * validationCampaigns().map((c) => c.name)[0] // => 'sbc/nuts-conjugate'
 * ```
 */
export function validationCampaigns(): Campaign[] {
  return [
    calibrated(
      'sbc/nuts-conjugate',
      'NUTS on the normal-normal program: ranks of the true mean among 19 draws (190 thinned by 10) must be uniform.',
      nutsFit(80),
      100,
      false,
    ),
    calibrated(
      'sbc/svi-normal-guide',
      'SVI with the mean-field normal guide on a Gaussian posterior, where the guide family is exact.',
      sviFit('normal', 120),
      100,
      false,
    ),
    calibrated(
      'sbc/wrong-guide-delta',
      'SVI with a point-estimate (delta) guide: it must FAIL calibration (all its draws are one number).',
      sviFit('delta', 120),
      40,
      true,
    ),
    calibrated(
      'sbc/wrong-guide-narrow',
      'The exact posterior with its spread scaled by 0.4, a too-confident guide: it must FAIL calibration.',
      scaledGuide(0.4),
      100,
      true,
    ),
    {
      name: 'sbc/nuts-hierarchical',
      description:
        'NUTS on a four-group hierarchical model, non-centred by `reparam`: calibration of mu, tau and the first group effect.',
      run: ({ key, scale }) => {
        const reps = scaled(40, scale)
        const result: SbcResult = sbc({
          key,
          replications: reps,
          numRanks: 19,
          simulate: (k) => {
            const mu = z(foldIn(k, 0))
            const tau = Math.abs(z(foldIn(k, 1)))
            const theta = Array.from({ length: GROUPS }, (_, g) => mu + tau * z(foldIn(k, 2 + g)))
            const y = theta.map((t, g) => t + 0.5 * z(foldIn(k, 10 + g)))
            return { data: y, truth: { mu, tau, 'theta[0]': theta[0] ?? 0 } }
          },
          fit: (y, k) => {
            const post = mcmc(hierarchical(y), undefined, { key: k, numChains: 1, numWarmup: 200, numDraws: 380 })
            const thin = (xs: ArrayLike<number>): number[] =>
              Array.from({ length: 19 }, (_, i) => xs[i * 20] ?? Number.NaN)
            const theta = post.draws?.['theta']
            const first = Array.from({ length: 380 }, (_, i) => theta?.data[i * GROUPS] ?? Number.NaN)
            return {
              mu: thin(post.draws?.['mu']?.data ?? []),
              tau: thin(post.draws?.['tau']?.data ?? []),
              'theta[0]': thin(first),
            }
          },
        })
        return {
          passed: result.passes,
          verdict: result.passes
            ? `calibrated over ${reps} replications`
            : 'NOT calibrated (draws thinned by 20; NUTS ranks may be autocorrelated)',
          markdown: sbcMarkdown(result),
          budget: { replications: reps, ranks: 19, warmup: 200, draws: 380 },
        }
      },
    },
    {
      name: 'agreement/logistic-grid',
      description:
        'Logistic regression with two coefficients: NUTS, the Laplace approximation and mean-field SVI against the grid posterior.',
      run: ({ key, scale }) => {
        const grid = gridPosterior(logistic, undefined, { points: 61 })
        const draws = scaled(1000, scale, 200)
        const nuts = mcmc(logistic, undefined, { key, numChains: 2, numWarmup: draws, numDraws: draws })
        const lap = laplaceApproximation(logistic, undefined, { key })
        const vi = svi(logistic, undefined, {
          key,
          numSteps: scaled(1500, scale, 200),
          optimizer: adam({ learningRate: (t: number) => 0.05 / (1 + t / 300) }),
        })
        const table = agree(
          [
            fromPosterior('nuts', nuts),
            fromPosterior('laplace', lap, { key: foldIn(key, 1) }),
            fromPosterior('svi', vi.posterior, { key: foldIn(key, 2) }),
          ],
          { oracle: Object.fromEntries(grid.marginals().map((m) => [m.name, m])) },
        )
        const nutsBad = agreementWithin(
          { ...table, rows: table.rows.filter((r) => r.method === 'nuts') },
          { ks: 0.08, meanError: 0.15 },
        )
        return {
          passed: nutsBad.length === 0,
          verdict:
            nutsBad.length === 0
              ? `NUTS agrees with the grid (log evidence ${grid.logEvidence.toFixed(4)}); Laplace and SVI are reported as they are`
              : `NUTS differs from the grid on ${nutsBad.map((b) => b.row.quantity).join(', ')}`,
          markdown: agreementMarkdown(table),
          budget: { draws, gridPoints: grid.points, gridEvaluations: grid.evaluations },
        }
      },
    },
    {
      name: 'funnel/non-centred',
      description:
        "A hierarchical funnel (Rubin's eight schools, centred) against the same model non-centred by `reparam`: divergences and the smallest tau reached.",
      run: ({ key, scale }) => {
        const Y = [28, 8, -3, 7, -1, 1, 18, 12]
        const SIGMA = [15, 10, 16, 11, 9, 11, 10, 18]
        const schools: Model = () => {
          const mu = sample('mu', normal(0, 5))
          const tau = sample('tau', halfCauchy(5))
          plate('J', 8, () => {
            const theta = sample('theta', normal(mu, tau))
            observe('y', normal(theta, tensor(SIGMA)), Y)
          })
        }
        const draws = scaled(400, scale, 100)
        const rows: string[] = [
          '| parametrisation | divergences | smallest tau | chains x draws |',
          '| --- | --- | --- | --- |',
        ]
        const out: { div: number; minTau: number }[] = []
        for (const [name, m] of [
          ['centred', schools],
          ['non-centred', reparam(schools, { theta: locScale(0) })],
        ] as const) {
          const post = mcmc(m, undefined, { key, numChains: 2, numWarmup: Math.round(draws * 0.75), numDraws: draws })
          const tau = Array.from(post.draws?.['tau']?.data ?? [])
          const d = Number(post.diagnostics['divergences'] ?? 0)
          out.push({ div: d, minTau: Math.min(...tau) })
          rows.push(`| ${name} | ${d} | ${Math.min(...tau).toPrecision(2)} | 2 x ${draws} |`)
        }
        const [centred, flat] = out
        const ok = (centred?.div ?? 0) > 0 && flat?.div === 0
        return {
          passed: ok,
          verdict: ok
            ? 'the centred model diverges, the non-centred one does not'
            : 'unexpected: the divergences did not separate the two parametrisations',
          markdown: rows.join('\n'),
          budget: { draws, chains: 2 },
        }
      },
    },
    {
      name: 'geweke/kernels',
      description:
        'Geweke joint-distribution tests of the handler-layer kernels on the normal-normal program (a kernel that never rejects must fail).',
      run: ({ key, scale }) => {
        const iterations = scaled(6000, scale, 1000)
        const program = (y: number): Model => normalProgram(y)
        const kernels: { name: string; kernel: ReturnType<typeof resimulationKernel>; mustFail: boolean }[] = [
          { name: 'resimulation', kernel: resimulationKernel(), mustFail: false },
          { name: 'random walk', kernel: randomWalkKernel({ scale: 1.2 }), mustFail: false },
        ]
        const lines = [
          '| kernel | statistic | marginal-conditional | successive-conditional | z |',
          '| --- | --- | --- | --- | --- |',
        ]
        let passed = true
        kernels.forEach(({ name, kernel, mustFail }, i) => {
          const r = gewekeTest<KernelState, number>({
            key: foldIn(key, i),
            numSamples: Math.round(iterations / 2),
            numIterations: iterations,
            simulate: (k) => {
              const bound = kernel.on(program(0), undefined)
              const state = bound.init(foldIn(k, 0))
              const m = item((state.trace.sites.get('m') as { value: Parameters<typeof item>[0] }).value)
              return { state, data: m + z(foldIn(k, 1)) }
            },
            resimulateData: (s, k) =>
              item((s.trace.sites.get('m') as { value: Parameters<typeof item>[0] }).value) + z(k),
            transition: (s, y, k) => {
              const bound = kernel.on(program(y), undefined)
              return bound.step(k, { trace: bound.target.update(s.trace, {}).trace, logDensity: 0 }).state
            },
            statistics: {
              m: (s) => item((s.trace.sites.get('m') as { value: Parameters<typeof item>[0] }).value),
              y: (_s, y) => y,
            },
          })
          if (r.passes === mustFail) passed = false
          for (const s of r.statistics)
            lines.push(
              `| ${name} | ${s.name} | ${s.meanMarginal.toFixed(3)} | ${s.meanSuccessive.toFixed(3)} | ${s.z.toFixed(2)} |`,
            )
        })
        return {
          passed,
          verdict: passed ? 'both kernels pass' : 'a kernel differs between the simulators',
          markdown: lines.join('\n'),
          budget: { iterations },
        }
      },
    },
  ]
}
