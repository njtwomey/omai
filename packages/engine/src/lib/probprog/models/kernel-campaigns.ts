/**
 * The periodic validation of the kernel methods (`omai-engine/probprog/inference`): Geweke joint-distribution tests of the
 * kernels of the sampling library on a two-parameter model (with a kernel that never rejects, which must fail), simulation-based
 * calibration of every kernel method end to end (the knobs, the warm-up, the thinning, the posterior reading), the agreement of
 * the methods with the truth of the benchmark models (the exact conjugate model, the logistic regression by quadrature, the
 * non-centred eight schools by a long reference), and parallel tempering on the two-mode target against NUTS. Each is a `Campaign`
 * with a budget that scales; `scripts/probprog-validate.ts` runs them with the other campaigns, and none of it is part of
 * `make verify` or `make check`.
 */

import { normal } from '../../dists/index.ts'
import { normal as drawNormal, foldIn, type Key } from '../../random/index.ts'
import {
  barkerKernel,
  ellipticalSlice,
  hmcKernel,
  type MCMCKernel,
  malaKernel,
  numericTarget,
  randomWalkMetropolis,
  sliceSampler,
  ulaKernel,
} from '../../sampling/index.ts'
import { item } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { type AnyInferenceMethod, blockKernels, hybridMethod, methodKinds } from '../inference/index.ts'
import { type Model, observe, sample } from '../primitives.ts'
import {
  type Campaign,
  type CampaignOutcome,
  gewekeTest,
  type SbcResult,
  sbc,
  sbcMarkdown,
} from '../validation/index.ts'
import { benchmarkAgreement, runPlan } from './agreement.ts'
import { type BenchmarkModel, type MethodPlan, methodVerdicts } from './benchmark.ts'
import { benchmarkMarkdown } from './campaigns.ts'
import { normalNormal } from './conjugate.ts'
import { bimodal } from './geometry.ts'
import { logisticRegression } from './glm.ts'
import { budget } from './plans.ts'
import { eightSchoolsNonCentred } from './schools.ts'

const scaled = (base: number, scale: number, min = 10): number => Math.max(min, Math.round(base * scale))
const z = (k: Key): number => item(drawNormal(k, []))

// ---- Geweke ----------------------------------------------------------------------------------------------------------------------

// a, b ~ N(0, 1) and y ~ N(a + b, 1): the log density of (a, b) given y, with its gradient.
const posteriorTarget = (y: number) =>
  numericTarget((x) => -0.5 * ((x[0] ?? 0) ** 2 + (x[1] ?? 0) ** 2 + (y - (x[0] ?? 0) - (x[1] ?? 0)) ** 2), 2, {
    grad: (x) => {
      const r = y - (x[0] ?? 0) - (x[1] ?? 0)
      return [-(x[0] ?? 0) + r, -(x[1] ?? 0) + r]
    },
  })

const gewekeKernels: { readonly name: string; readonly make: (y: number) => MCMCKernel; readonly mustFail: boolean }[] =
  [
    { name: 'random walk', make: (y) => randomWalkMetropolis(posteriorTarget(y), { scale: 1.1 }), mustFail: false },
    { name: 'MALA', make: (y) => malaKernel(posteriorTarget(y), { stepSize: 0.7 }), mustFail: false },
    { name: 'Barker', make: (y) => barkerKernel(posteriorTarget(y), { stepSize: 1.2 }), mustFail: false },
    {
      name: 'HMC',
      make: (y) => hmcKernel(posteriorTarget(y), { stepSize: 0.5, numIntegrationSteps: 4 }),
      mustFail: false,
    },
    { name: 'slice', make: (y) => sliceSampler(posteriorTarget(y), { width: 2 }), mustFail: false },
    {
      name: 'elliptical slice',
      make: (y) => ellipticalSlice((x) => -0.5 * (y - (x[0] ?? 0) - (x[1] ?? 0)) ** 2, { covariance: 1 }, 2),
      mustFail: false,
    },
    // The unadjusted Langevin step at a size where its bias shows: it must be caught.
    {
      name: 'ULA without the Metropolis step (must fail)',
      make: (y) => ulaKernel(posteriorTarget(y), { stepSize: 1.2 }),
      mustFail: true,
    },
  ]

function gewekeCampaign(): Campaign {
  return {
    name: 'kernels/geweke',
    description:
      'Geweke joint-distribution tests of the kernels of the sampling library on a two-parameter Gaussian model: every adjusted kernel must pass, the unadjusted Langevin step at a large step size must fail.',
    run: ({ key, scale }): CampaignOutcome => {
      const iterations = scaled(8000, scale, 1000)
      const lines = [
        '| kernel | statistic | marginal-conditional | successive-conditional | z |',
        '| --- | --- | --- | --- | --- |',
      ]
      let passed = true
      const verdicts: string[] = []
      gewekeKernels.forEach(({ name, make, mustFail }, i) => {
        const r = gewekeTest<Float64Array, number>({
          key: foldIn(key, i),
          numSamples: Math.round(iterations / 2),
          numIterations: iterations,
          simulate: (k) => {
            const a = z(foldIn(k, 0))
            const b = z(foldIn(k, 1))
            return { state: Float64Array.of(a, b), data: a + b + z(foldIn(k, 2)) }
          },
          resimulateData: (s, k) => (s[0] ?? 0) + (s[1] ?? 0) + z(k),
          transition: (s, y, k) => {
            const kernel = make(y)
            return kernel.step(k, kernel.init(s)).state.position
          },
          statistics: {
            a: (s) => s[0] ?? 0,
            b: (s) => s[1] ?? 0,
            y: (_s, y) => y,
            'a^2': (s) => (s[0] ?? 0) ** 2,
            'y^2': (_s, y) => y * y,
            'a b': (s) => (s[0] ?? 0) * (s[1] ?? 0),
          },
        })
        if (r.passes === mustFail) passed = false
        verdicts.push(`${name}: ${r.passes ? 'passes' : 'differs'}`)
        for (const s of r.statistics)
          lines.push(
            `| ${name} | ${s.name} | ${s.meanMarginal.toFixed(3)} | ${s.meanSuccessive.toFixed(3)} | ${s.z.toFixed(2)} |`,
          )
      })
      return {
        passed,
        verdict: passed ? `as expected: ${verdicts.join('; ')}` : `unexpected: ${verdicts.join('; ')}`,
        markdown: lines.join('\n'),
        budget: { iterations },
      }
    },
  }
}

// ---- SBC ----------------------------------------------------------------------------------------------------------------------

// m ~ N(0, 1), y ~ N(m, 1): the posterior of m is N(y / 2, 1 / 2).
const oneLatent =
  (y: number): Model =>
  () => {
    const m = sample('m', normal(0, 1))
    observe('y', normal(m, 1), y)
  }
// a ~ N(0, 1), b | a ~ N(a, 1), y | b ~ N(b, 0.5): two latents, the posterior Gaussian.
const twoLatents =
  (y: number): Model =>
  () => {
    const a = sample('a', normal(0, 1))
    const b = sample('b', normal(a, 1))
    observe('y', normal(b, 0.5), y)
  }

interface SbcCase {
  readonly name: string
  readonly description: string
  readonly latents: 1 | 2
  /** The method on the program with the data, and its knobs (one chain, 19 draws after thinning). */
  readonly make: (program: Model) => AnyInferenceMethod
  readonly config: Record<string, number | string | boolean>
  /** Biased by design (a step-size error that the rank test may see): reported, not required to pass. */
  readonly report?: boolean
}

const methodOf =
  (kind: string) =>
  (program: Model): AnyInferenceMethod =>
    (methodKinds[kind] as NonNullable<(typeof methodKinds)[string]>)(program, {})

const SBC_CASES: readonly SbcCase[] = [
  {
    name: 'rwm',
    description: 'random-walk Metropolis with an adapted covariance',
    latents: 1,
    make: methodOf('rwm'),
    config: { numChains: 1, numWarmup: 150, numDraws: 19, thin: 25 },
  },
  {
    name: 'mala',
    description: 'MALA with window adaptation',
    latents: 1,
    make: methodOf('mala'),
    config: { numChains: 1, numWarmup: 120, numDraws: 19, thin: 10 },
  },
  {
    name: 'barker',
    description: "Barker's proposal with window adaptation",
    latents: 1,
    make: methodOf('barker'),
    config: { numChains: 1, numWarmup: 120, numDraws: 19, thin: 12 },
  },
  {
    name: 'hmc',
    description: 'HMC with a fixed trajectory and window adaptation',
    latents: 1,
    make: methodOf('hmc'),
    config: { numChains: 1, numWarmup: 100, numDraws: 19, thin: 4, numSteps: 5 },
  },
  {
    name: 'slice',
    description: 'coordinate-wise slice sampling',
    latents: 1,
    make: methodOf('slice'),
    config: { numChains: 1, numWarmup: 20, numDraws: 19, thin: 4 },
  },
  {
    name: 'elliptical-slice',
    description: 'elliptical slice sampling',
    latents: 1,
    make: methodOf('elliptical-slice'),
    config: { numChains: 1, numWarmup: 20, numDraws: 19, thin: 3 },
  },
  {
    name: 'aies',
    description: 'the affine-invariant ensemble sampler: the walkers of one iteration are independent draws',
    latents: 1,
    make: methodOf('aies'),
    config: { numChains: 1, numWalkers: 19, numWarmup: 100, numDraws: 1, thin: 1 },
  },
  {
    name: 'ensemble-slice',
    description: 'ensemble slice sampling: the walkers of one iteration are independent draws',
    latents: 1,
    make: methodOf('ensemble-slice'),
    config: { numChains: 1, numWalkers: 19, numWarmup: 60, numDraws: 1, thin: 1 },
  },
  {
    name: 'parallel-tempering',
    description: 'parallel tempering over HMC',
    latents: 1,
    make: methodOf('parallel-tempering'),
    config: { numChains: 1, numWarmup: 100, numDraws: 19, thin: 6, numSteps: 4, numTemperatures: 3, minBeta: 0.4 },
  },
  {
    name: 'chees',
    description: 'ChEES-HMC: the chains of one ensemble are independent draws',
    latents: 1,
    make: methodOf('chees'),
    config: { numChains: 1, numWalkers: 19, numWarmup: 150, numDraws: 1, thin: 1 },
  },
  {
    name: 'mclmc',
    description: 'microcanonical Langevin Monte Carlo (unadjusted: a small step-size bias)',
    latents: 2,
    make: methodOf('mclmc'),
    config: { numChains: 1, numWarmup: 120, numDraws: 19, thin: 8 },
    report: true,
  },
  {
    name: 'sgld',
    description: 'stochastic-gradient Langevin dynamics with the exact gradient (a step-size bias)',
    latents: 2,
    make: methodOf('sgld'),
    config: { numChains: 1, numWarmup: 100, numDraws: 19, thin: 25, stepSize: 0.05, trackDensity: false },
    report: true,
  },
  {
    name: 'hybrid',
    description: 'the user plan: the exact conjugate conditional of b with NUTS on a',
    latents: 2,
    make: (program) =>
      hybridMethod(program, {
        blocks: [
          { sites: ['a'], kernel: blockKernels.nuts() },
          { sites: ['b'], kernel: blockKernels.gibbs() },
        ],
      }) as unknown as AnyInferenceMethod,
    config: { numChains: 1, numWarmup: 100, numDraws: 19, thin: 4 },
  },
]

function sbcCampaign(c: SbcCase): Campaign {
  return {
    name: `kernels/sbc-${c.name}`,
    description: `Simulation-based calibration of the ${c.description}, through its method (knobs, warm-up, thinning and posterior).`,
    run: ({ key, scale }): CampaignOutcome => {
      const reps = scaled(100, scale, 20)
      const result: SbcResult = sbc({
        key,
        replications: reps,
        numRanks: 19,
        simulate: (k) => {
          if (c.latents === 1) {
            const m = z(foldIn(k, 0))
            return { data: m + z(foldIn(k, 1)), truth: { m } }
          }
          const a = z(foldIn(k, 0))
          const b = a + z(foldIn(k, 1))
          return { data: b + 0.5 * z(foldIn(k, 2)), truth: { a, b } }
        },
        fit: (y, k) => {
          const method = c.make(c.latents === 1 ? oneLatent(y) : twoLatents(y))
          const s = session(method, { config: c.config, steps: 100_000, mode: 'eager', key: k })
          const post = method.posterior?.(s.state, s.config)
          const take = (site: string): number[] => Array.from(post?.draws?.[site]?.data ?? [])
          return c.latents === 1 ? { m: take('m') } : { a: take('a'), b: take('b') }
        },
      })
      const verdict =
        c.report === true
          ? result.passes
            ? 'calibrated'
            : 'not calibrated (reported: biased by design)'
          : result.passes
            ? 'calibrated'
            : 'NOT calibrated'
      return {
        passed: c.report === true ? true : result.passes,
        verdict: `${verdict} over ${reps} replications`,
        markdown: sbcMarkdown(result),
        budget: { replications: reps, ranks: 19 },
      }
    },
  }
}

// ---- agreement with the benchmark truths -----------------------------------------------------------------------------

const AGREE = { meanError: 0.15, sdRatio: [0.85, 1.15] as const, ks: 0.12 }

/** The plans of the kernel methods on a benchmark model, budgets scaled: those that are consistent must agree, those biased by design are reported. */
function kernelPlans(scale: number, own: readonly string[] = [], hard = false): MethodPlan[] {
  const chains = (c: number, w: number, d: number): Record<string, number> => ({
    numChains: c,
    numWarmup: budget(w, scale, 40),
    numDraws: budget(d, scale, 40),
  })
  const plan = (
    kind: string,
    config: Record<string, number | boolean | string>,
    expect: MethodPlan['expect'],
    note: string,
    tolerances = AGREE,
  ): MethodPlan => ({
    kind,
    config,
    steps: 1_000_000,
    // On a hierarchy with a funnel the gradient-free and ensemble samplers mix too slowly for this budget: reported, not required.
    expect: hard && ['rwm', 'aies', 'ensemble-slice'].includes(kind) ? 'report' : expect,
    note:
      hard && ['rwm', 'aies', 'ensemble-slice'].includes(kind)
        ? `${note}; on this hierarchy it mixes too slowly at this budget`
        : note,
    tolerances,
  })
  return [
    plan('rwm', chains(2, 400, 1500), 'agree', 'a consistent sampler, slower mixing: agrees within Monte Carlo error', {
      ...AGREE,
      ks: 0.15,
    }),
    plan('mala', chains(2, 300, 800), 'agree', 'Metropolis-adjusted, so exact up to Monte Carlo error'),
    plan('barker', chains(2, 300, 1000), 'agree', 'Metropolis-adjusted, so exact up to Monte Carlo error'),
    plan(
      'hmc',
      { ...chains(2, 400, 800), numSteps: 12, dense: true },
      'agree',
      'a fixed trajectory with a tuned step size: exact up to Monte Carlo error',
    ),
    plan('slice', chains(2, 100, 800), 'agree', 'coordinate-wise slice sampling: exact up to Monte Carlo error'),
    plan(
      'aies',
      { numChains: 1, numWalkers: 16, numWarmup: budget(200, scale, 40), numDraws: budget(500, scale, 40) },
      'agree',
      'an affine-invariant ensemble: exact up to Monte Carlo error',
    ),
    plan(
      'ensemble-slice',
      { numChains: 1, numWalkers: 16, numWarmup: budget(100, scale, 40), numDraws: budget(300, scale, 40) },
      'agree',
      'an ensemble of slice steps: exact up to Monte Carlo error',
    ),
    plan(
      'chees',
      { numChains: 1, numWalkers: 16, numWarmup: budget(300, scale, 60), numDraws: budget(300, scale, 40) },
      'agree',
      'jittered HMC with an adapted trajectory: exact up to Monte Carlo error',
    ),
    plan(
      'parallel-tempering',
      { ...chains(2, 300, 600), numSteps: 6, numTemperatures: 3, minBeta: 0.3 },
      'agree',
      'the cold replica of a tempered ladder: exact up to Monte Carlo error',
    ),
    plan('mclmc', chains(2, 600, 1000), 'report', 'unadjusted: a small bias that shrinks with the step size'),
    plan(
      'sgld',
      { ...chains(2, 200, 3000), stepSize: 0.01, trackDensity: false },
      'report',
      'unadjusted Langevin with a constant step: a step-size bias in the spread',
    ),
    plan(
      'sghmc',
      { ...chains(2, 100, 1200), stepSize: 0.03, numSteps: 8, friction: 0.5, trackDensity: false },
      'report',
      'friction-damped Hamiltonian dynamics with a constant step: a step-size bias',
    ),
    ...own.map((kind) => plan(kind, chains(2, 100, 800), 'report', 'a Gaussian-prior sampler, where it applies')),
  ]
}

const withPlans = (model: BenchmarkModel, plans: (scale: number) => readonly MethodPlan[]): BenchmarkModel => ({
  ...model,
  plans,
})

const AGREEMENT_MODELS: readonly {
  readonly model: BenchmarkModel
  readonly extra: readonly string[]
  readonly hard?: boolean
}[] = [
  { model: normalNormal, extra: ['elliptical-slice'] },
  { model: logisticRegression, extra: ['elliptical-slice'] },
  { model: eightSchoolsNonCentred, extra: [], hard: true },
]

function agreementCampaign({ model, extra, hard }: (typeof AGREEMENT_MODELS)[number]): Campaign {
  return {
    name: `kernels/agreement-${model.name}`,
    description: `${model.title}: the kernel methods against the ${model.name === 'normal-normal' ? 'exact' : model.name === 'logistic-regression' ? 'quadrature' : 'reference'} truth.`,
    run: ({ key, scale }): CampaignOutcome => {
      const derived = withPlans(model, (s) => kernelPlans(s, extra, hard === true))
      const a = benchmarkAgreement(derived, { key, scale })
      const must = a.runs.filter((r) => r.plan.expect === 'agree').length
      return {
        passed: a.passed,
        verdict: a.passed
          ? `${must} kernel method(s) agree with the ${a.truth.kind} truth; ${a.runs.length - must} reported`
          : `${a.failures.length} row(s) of methods that must agree are outside their tolerances${a.surprises.length === 0 ? '' : `; ${a.surprises.length} surprise(s)`}`,
        markdown: benchmarkMarkdown(derived, a, methodVerdicts(derived, Object.keys(methodKinds))),
        budget: { plans: a.runs.length, rows: a.table.rows.length, scalePercent: Math.round(scale * 100) },
      }
    },
  }
}

// ---- parallel tempering on the two-mode target -----------------------------------------------------------------------

/** The fraction of each chain's draws in the upper mode (x[0] > 0). */
function modeFractions(draws: ArrayLike<number>, chains: number, dim: number): number[] {
  const n = draws.length / chains / dim
  return Array.from({ length: chains }, (_, c) => {
    let up = 0
    for (let i = 0; i < n; i++) if ((draws[(c * n + i) * dim] ?? 0) > 0) up += 1
    return up / n
  })
}

function bimodalCampaign(): Campaign {
  return {
    name: 'kernels/bimodal',
    description:
      'The equal two-mode target: parallel tempering over a random walk recovers the one-half weight of each mode in every chain; NUTS stays in one mode per chain.',
    run: ({ key, scale }): CampaignOutcome => {
      const chains = 4
      const nuts: MethodPlan = {
        kind: 'nuts',
        config: {
          numChains: chains,
          numWarmup: budget(200, scale, 40),
          numDraws: budget(400, scale, 40),
          inferDiscrete: false,
        },
        steps: 1_000_000,
        expect: 'report',
        note: 'one mode per chain',
      }
      const pt: MethodPlan = {
        kind: 'parallel-tempering',
        config: {
          kernel: 'rwm',
          numChains: chains,
          numWarmup: budget(300, scale, 60),
          numDraws: budget(2000, scale, 200),
          numTemperatures: 6,
          minBeta: 0.02,
          inferDiscrete: false,
        },
        steps: 1_000_000,
        expect: 'agree',
        note: 'tempered replicas cross the barrier between the modes',
      }
      const runN = runPlan(bimodal, nuts, { key: foldIn(key, 0) })
      const runP = runPlan(bimodal, pt, { key: foldIn(key, 1) })
      const fn = modeFractions(runN.posterior.draws?.['x']?.data ?? [], chains, 2)
      const fp = modeFractions(runP.posterior.draws?.['x']?.data ?? [], chains, 2)
      const pooled = fp.reduce((s, v) => s + v, 0) / chains
      const nutsStuck = fn.every((f) => f < 0.05 || f > 0.95)
      const ptBoth = fp.every((f) => f > 0.2 && f < 0.8) && Math.abs(pooled - 0.5) < 0.1
      const fmt = (xs: readonly number[]): string => xs.map((v) => v.toFixed(2)).join(', ')
      const lines = [
        '| method | fraction of each chain in the upper mode (truth 0.5) |',
        '| --- | --- |',
        `| NUTS | ${fmt(fn)} |`,
        `| parallel tempering over a random walk (6 temperatures down to beta 0.02) | ${fmt(fp)} (pooled ${pooled.toFixed(2)}) |`,
      ]
      return {
        passed: nutsStuck && ptBoth,
        verdict:
          nutsStuck && ptBoth
            ? 'parallel tempering recovers the mode weights in every chain; every NUTS chain stays in one mode'
            : `unexpected: NUTS ${nutsStuck ? 'stuck' : 'not stuck'}, tempering ${ptBoth ? 'recovers the weights' : 'does not recover the weights'}`,
        markdown: lines.join('\n'),
        budget: { chains, drawsPerChain: pt.config['numDraws'] as number },
      }
    },
  }
}

/**
 * The campaigns of the kernel methods, in a fixed order (campaign `i` runs under the key the runner derives from its position,
 * so adding one at the end changes no other): Geweke tests of the kernels, calibration of every kernel method, the agreement of
 * the methods with the truth of three benchmark models, and parallel tempering on the two-mode target.
 *
 * @returns The campaigns, named `kernels/...`.
 * @concept posterior-validation
 * @index function The periodic validation of the kernel methods: Geweke, calibration, agreement and the two-mode target.
 * @example
 * ```ts
 * import { kernelCampaigns } from 'omai-engine/probprog/models'
 * kernelCampaigns()[0]?.name // => 'kernels/geweke'
 * ```
 */
export function kernelCampaigns(): Campaign[] {
  return [
    gewekeCampaign(),
    ...SBC_CASES.map(sbcCampaign),
    ...AGREEMENT_MODELS.map(agreementCampaign),
    bimodalCampaign(),
  ]
}
