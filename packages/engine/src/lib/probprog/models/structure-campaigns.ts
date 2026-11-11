/**
 * The structure of the benchmark suite as periodic campaigns: `structure/benchmarks` reads every model's structure report
 * (what is conjugate and what the collapse handler can take, the Gaussian blocks, the width of the discrete part, the
 * minibatch axes, the independent components) and checks it against the findings written down here, so the table in
 * `docs/probprog/results/structure-benchmarks.md` records what each model is expected to show and a change in the analysis
 * that moves a finding fails the campaign; `structure/collapse-efficiency` runs NUTS on eight schools (centred) with and
 * without the plan's collapsed site and reports effective sample size per gradient evaluation. A model written as a program
 * (the AR(1) state space) is lifted by `fromProgram` and read from its program analysis. Nothing here selects a method:
 * these are reports a person reads. None of it runs in `make verify` or `make check`.
 */

import { OmaiError } from '../../base/index.ts'
import { analyse } from '../analysis.ts'
import { mcmc } from '../engines/nuts.ts'
import { applyCollapse, collapsePlan, fromProgram, structureReport, toProgram } from '../model/index.ts'
import type { Campaign, CampaignOutcome } from '../validation/runner.ts'
import type { BenchmarkModel } from './benchmark.ts'
import { eightSchoolsCentred, eightSchoolsNonCentred } from './schools.ts'
import { benchmarkModels } from './suite.ts'

/** What a model's structure report shows, as short strings a table can hold and a test can compare. */
export interface StructureFindings {
  /** The conjugate latents, `site` for a pair and `site (kind, via)` otherwise; `not exact` marks a near miss. */
  readonly conjugate: readonly string[]
  /** The sites the collapse handler can integrate out (the plan's sites). */
  readonly collapsible: readonly string[]
  /** The Gaussian blocks, `sites | given`. */
  readonly gaussianBlocks: readonly string[]
  /** The discrete part, `sites: width w, states per element k`, or `none`. */
  readonly discrete: string
  /** The dimensions along which the observations are exchangeable. */
  readonly minibatch: readonly string[]
  /** The number of independent sub-problems among the latents. */
  readonly components: number
  /** The kinds of suggestion, in order. */
  readonly suggestions: readonly string[]
}

/** The findings of a spec model, read from its structure report. */
export function structureFindings(model: BenchmarkModel): StructureFindings {
  const bound = model.bound()
  const r = structureReport(bound)
  const plan = collapsePlan(bound)
  const d = r.discretePart
  return {
    conjugate: r.conjugateBlocks.map((c) => {
      const vias = [...new Set(c.links.map((l) => l.via))]
      const tag =
        c.kind === 'pair' && vias.length === 1 && vias[0] === 'direct'
          ? ''
          : ` (${c.kind === 'pair' ? '' : `${c.kind}, `}${vias.join('+')})`
      return `${c.sites.join('+')}${tag}${c.holds ? '' : ' not exact'}`
    }),
    collapsible: plan.sites,
    gaussianBlocks: r.linearGaussianBlocks.map((b) => `${b.sites.join(',')} | ${b.given.join(',') || 'nothing'}`),
    discrete:
      d.sites.length === 0
        ? 'none'
        : d.components
            .map(
              (c) =>
                `${c.sites.join(',')}: width ${c.width}${c.cliqueStates === undefined ? '' : `, ${c.cliqueStates} states`}`,
            )
            .join('; '),
    minibatch: r.minibatchAxes.map((m) => m.dim),
    components: r.components.filter((c) => c.latents.length > 0).length,
    suggestions: r.suggestions.map((s) => s.kind),
  }
}

const none: StructureFindings = {
  conjugate: [],
  collapsible: [],
  gaussianBlocks: [],
  discrete: 'none',
  minibatch: [],
  components: 1,
  suggestions: [],
}

/**
 * What each spec model is expected to show. The comments say why: the finding is a property of the model, not of the code
 * that found it, so a change here is a change in what the analysis claims.
 */
export const STRUCTURE_EXPECTED: Readonly<Record<string, StructureFindings>> = {
  // One conjugate site, the whole model closed form; the plate is exchangeable.
  'beta-bernoulli': {
    ...none,
    conjugate: ['p'],
    collapsible: ['p'],
    minibatch: ['flip'],
    suggestions: ['closed-form', 'collapse'],
  },
  // Conjugate and linear-Gaussian as a whole.
  'normal-normal': {
    ...none,
    conjugate: ['mu'],
    collapsible: ['mu'],
    gaussianBlocks: ['mu | nothing'],
    minibatch: ['obs'],
    suggestions: ['closed-form', 'collapse', 'gaussian-block'],
  },
  'gamma-poisson': {
    ...none,
    conjugate: ['rate'],
    collapsible: ['rate'],
    minibatch: ['obs'],
    suggestions: ['closed-form', 'collapse'],
  },
  // Not conjugate, not Gaussian: only the observations' plate is structure.
  'logistic-regression': { ...none, minibatch: ['row'] },
  'probit-regression': { ...none, minibatch: ['row'] },
  'poisson-regression': { ...none, minibatch: ['row'], suggestions: ['minibatch'] },
  'negative-binomial-regression': { ...none, minibatch: ['row'], suggestions: ['minibatch'] },
  // Eight schools, centred: the group effects are conjugate to the observations given mu and tau (the handler takes them);
  // mu is conjugate to the effects (a hierarchy); {mu, theta} is a Gaussian block given tau; theta is a funnel.
  'eight-schools-centred': {
    ...none,
    conjugate: ['mu (hierarchical, direct)', 'theta'],
    collapsible: ['theta'],
    gaussianBlocks: ['mu,theta | tau'],
    minibatch: ['school'],
    suggestions: ['conditional', 'collapse', 'gaussian-block', 'reparameterise'],
  },
  // Non-centred: the effects are not sites of a conjugate pair for the handler (an affine map), but {mu, z} is a Gaussian
  // block given tau, and mu and z are conjugate given the rest through the affine map.
  'eight-schools-non-centred': {
    ...none,
    conjugate: ['mu (affine)', 'z (affine)'],
    gaussianBlocks: ['mu,z | tau'],
    minibatch: ['school'],
    suggestions: ['conditional', 'conditional', 'gaussian-block'],
  },
  // Mixtures: the labels are a finite discrete part with width 0 (independent given the means), summed out by enumeration;
  // the means are conjugate and Gaussian given the labels, except that the ordered one has a constraint, which truncates
  // the posterior: a near miss with its reason, and no Gaussian block.
  'gaussian-mixture-ordered': {
    ...none,
    conjugate: ['mu (index) not exact'],
    discrete: 'z: width 0, 2 states',
    minibatch: ['row'],
    suggestions: ['enumerate'],
  },
  'gaussian-mixture': {
    ...none,
    conjugate: ['mu (index)'],
    gaussianBlocks: ['mu | z'],
    discrete: 'z: width 0, 2 states',
    minibatch: ['row'],
    suggestions: ['enumerate', 'conditional', 'gaussian-block'],
  },
  // The funnels: centred is the funnel the report names; non-centred splits into two independent blocks.
  'funnel-centred': { ...none, suggestions: ['reparameterise'] },
  'funnel-non-centred': { ...none, components: 2, suggestions: ['split'] },
  banana: none,
  bimodal: { ...none, discrete: 'z: width 0, 2 states', suggestions: ['enumerate'] },
  // Radon: the whole hierarchy {mu, beta, alpha} is a Gaussian block given the two scales; alpha and beta are conjugate through
  // an affine map and mu through a hierarchy; the minibatch axes are the counties and the houses.
  radon: {
    ...none,
    conjugate: ['mu (hierarchical, direct)', 'beta (affine)', 'alpha (affine)'],
    gaussianBlocks: ['mu,beta,alpha | tau,sigma'],
    minibatch: ['county', 'house'],
    suggestions: ['conditional', 'conditional', 'conditional', 'gaussian-block', 'reparameterise'],
  },
  'trueskill-three-players': { ...none, minibatch: ['game'] },
  // A Gaussian chain: the scan is one block over the series (a Kalman smoother's model), no minibatch axis over time.
  'ar1-state-space': { ...none, gaussianBlocks: ['x | nothing'], suggestions: ['gaussian-block'] },

  // The later models. Recurrences: the report names the method that follows the chain; the states have no minibatch axis.
  'stochastic-volatility': { ...none, suggestions: ['recurrence'] },
  'gaussian-hmm': { ...none, suggestions: ['recurrence'] },
  // The centred Gaussian process: the function is conjugate to the observations given the length scale (a Gaussian block over
  // its twelve values), which the collapse handler takes; the kernel is a custom function, so message passing does not see it.
  'gaussian-process-regression': {
    ...none,
    conjugate: ['f'],
    collapsible: ['f'],
    gaussianBlocks: ['f | ell'],
    suggestions: ['collapse', 'gaussian-block', 'limit'],
  },
  // Whitened: the Cholesky factor mixes the points, so the custom function leaves no exchangeable axis and no conjugacy.
  'gaussian-process-non-centred': { ...none, suggestions: ['limit'] },
  // LDA: the mixtures and topics are Dirichlet-categorical conjugate (given the assignments), the assignments a finite discrete
  // site of width 0 whose elements are independent given the mixtures and topics.
  'lda-lite': {
    ...none,
    conjugate: ['theta (hierarchical, direct)', 'phi (index)'],
    discrete: 'z: width 0',
    minibatch: ['doc', 'token'],
    suggestions: ['enumerate', 'conditional', 'conditional'],
  },
  // The network: the output layer is a Gaussian block given the hidden layer.
  'bayesian-neural-network': {
    ...none,
    conjugate: ['w2 (affine)', 'b2 (affine)'],
    gaussianBlocks: ['w2,b2 | w1,b1'],
    minibatch: ['point'],
    suggestions: ['conditional', 'conditional', 'gaussian-block'],
  },
  // Independent abilities with a logistic link: no conjugacy, two exchangeable axes, nothing specific to exploit.
  'item-response-scoring': { ...none, minibatch: ['person', 'item'] },
  'stick-breaking-mixture': {
    ...none,
    discrete: 'z: width 0',
    minibatch: ['row'],
    suggestions: ['enumerate'],
  },
  'clutter-problem': { ...none, discrete: 'z: width 0, 2 states', minibatch: ['row'], suggestions: ['enumerate'] },
}

const cell = (v: unknown): string => String(v).replaceAll('|', '\\|')
const list = (xs: readonly string[]): string => (xs.length === 0 ? '-' : xs.join('; '))

function mismatch(a: StructureFindings, b: StructureFindings): string[] {
  const out: string[] = []
  for (const k of Object.keys(a) as (keyof StructureFindings)[])
    if (JSON.stringify(a[k]) !== JSON.stringify(b[k]))
      out.push(`${k}: found ${JSON.stringify(a[k])}, expected ${JSON.stringify(b[k])}`)
  return out
}

/** The structure campaign's table and text for the models, and the findings that moved. */
export function structureMarkdown(models: readonly BenchmarkModel[]): {
  readonly markdown: string
  readonly moved: readonly string[]
} {
  const rows: string[] = []
  const sections: string[] = []
  const moved: string[] = []
  for (const m of models) {
    if (m.definition === undefined) {
      // A program: lifted to a spec of its site structure, and read from the program analysis for what the lift cannot see.
      const program = m.program()
      const lifted = fromProgram(program, undefined)
      const a = analyse(program, undefined)
      const markov = a.markov[0]
      rows.push(
        `| ${m.name} | program, lifted | none: ${lifted.opaque.length} opaque parameters (each mean is a function of the previous state) | - | ${a.linearGaussian.linear ? 'whole model (program analysis)' : '-'} | none | ${markov === undefined ? '-' : `markov ${markov.name} (${markov.length})`} | - |`,
      )
      sections.push(
        `### ${m.title} (${m.name})\n\nA program, not a spec: \`fromProgram\` recovers ${lifted.spec.nodes.length} sites with ${lifted.constants.length} constant parameters and ${lifted.opaque.length} opaque ones (every mean is a function of the previous state, which a trace cannot express). The program analysis says: linear-Gaussian ${a.linearGaussian.linear ? 'yes' : 'no'} (${a.linearGaussian.reason}), ${markov === undefined ? 'no markov loop' : `a markov loop \`${markov.name}\` of length ${markov.length}`}. The exact route is the Kalman smoother (\`gaussian\` with \`method: 'kalman'\`), not the structure report.`,
      )
      continue
    }
    const found = structureFindings(m)
    const want = STRUCTURE_EXPECTED[m.name]
    if (want === undefined) moved.push(`${m.name}: no expected findings written`)
    else for (const x of mismatch(found, want)) moved.push(`${m.name} ${x}`)
    rows.push(
      `| ${m.name} | ${m.target} | ${cell(list(found.conjugate))} | ${cell(list(found.collapsible))} | ${cell(list(found.gaussianBlocks))} | ${cell(found.discrete)} | ${cell(list(found.minibatch))} | ${found.components} |`,
    )
    sections.push(`### ${m.title} (${m.name})\n\n\`\`\`text\n${structureReport(m.bound()).explain()}\n\`\`\``)
  }
  const markdown = [
    'What `structureReport` finds in each benchmark model. The columns are the findings the campaign checks (`STRUCTURE_EXPECTED` in `probprog/models/structure-campaigns.ts`): the conjugate latents (a hierarchy or an affine map is marked), what the `collapse` handler can integrate out, the Gaussian blocks (`sites | given`), the discrete part, the exchangeable plate axes and the number of independent sub-problems. Below, the full report of each model as `explain()` prints it. The report is advice: nothing is applied.',
    '',
    '| model | target | conjugate latents | collapsible by the handler | Gaussian blocks | discrete part | minibatch axes | components |',
    '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ...rows,
    '',
    ...sections.flatMap((s) => [s, '']),
  ].join('\n')
  return { markdown, moved }
}

function structureBenchmarks(): Campaign {
  return {
    name: 'structure/benchmarks',
    description: 'The structure report of every benchmark model against the findings written down for it.',
    run: (): CampaignOutcome => {
      const { markdown, moved } = structureMarkdown(benchmarkModels())
      return {
        passed: moved.length === 0,
        verdict:
          moved.length === 0
            ? `${benchmarkModels().length} models: every finding as expected`
            : `${moved.length} finding(s) moved: ${moved[0]}`,
        markdown:
          moved.length === 0
            ? markdown
            : `${markdown}\n\nFindings that moved:\n\n${moved.map((x) => `- ${x}`).join('\n')}\n`,
        budget: { models: benchmarkModels().length },
      }
    },
  }
}

/** Effective sample size and gradient evaluations of a posterior from the sampler's own record. */
function essPerGradient(
  post: ReturnType<typeof mcmc>,
  sites: readonly string[],
): { readonly ess: Record<string, number>; readonly gradients: number; readonly divergences: number } {
  const rows = post.summary()
  const ess: Record<string, number> = {}
  for (const s of sites) ess[s] = rows.find((r) => r.name === s)?.ess ?? Number.NaN
  const steps = post.chains?.stats.numSteps
  const gradients = steps === undefined ? Number.NaN : Array.from(steps.data).reduce((a, b) => a + b, 0)
  const div = post.diagnostics['divergences']
  return { ess, gradients, divergences: typeof div === 'number' ? div : Number.NaN }
}

function collapseEfficiency(): Campaign {
  return {
    name: 'structure/collapse-efficiency',
    description: 'NUTS on eight schools with the group effects integrated out by collapse, against the full model.',
    run: ({ key, scale }): CampaignOutcome => {
      const draws = Math.max(100, Math.round(1000 * scale))
      const warmup = Math.max(100, Math.round(500 * scale))
      const bound = eightSchoolsCentred.bound()
      const plan = collapsePlan(bound)
      if (plan.sites.length === 0) throw new OmaiError('value', 'structure/collapse-efficiency: nothing to collapse')
      const collapsed = applyCollapse(bound, plan)
      const options = { key, numChains: 4, numWarmup: warmup, numDraws: draws, targetAcceptance: 0.9 }
      const rows = [
        ['full (centred)', essPerGradient(mcmc(toProgram(bound), undefined, options), ['mu', 'tau'])],
        ['collapsed (theta integrated out)', essPerGradient(mcmc(collapsed.model, undefined, options), ['mu', 'tau'])],
        [
          'non-centred, no collapse',
          essPerGradient(mcmc(toProgram(eightSchoolsNonCentred.bound()), undefined, options), ['mu', 'tau']),
        ],
      ] as const
      const per = (r: (typeof rows)[number][1], s: string): number => (r.ess[s] ?? Number.NaN) / r.gradients
      const full = rows[0][1]
      const coll = rows[1][1]
      const gain = per(coll, 'tau') / per(full, 'tau')
      const passed = gain > 3 && coll.divergences === 0
      const table = [
        '| run | ESS mu | ESS tau | gradient evaluations | ESS tau per gradient | ESS mu per gradient | divergences |',
        '| --- | --- | --- | --- | --- | --- | --- |',
        ...rows.map(
          ([name, r]) =>
            `| ${name} | ${r.ess['mu']?.toFixed(0)} | ${r.ess['tau']?.toFixed(0)} | ${r.gradients} | ${per(r, 'tau').toExponential(2)} | ${per(r, 'mu').toExponential(2)} | ${r.divergences} |`,
        ),
      ].join('\n')
      return {
        passed,
        verdict: `collapsed NUTS gets ${gain.toFixed(0)} times the ESS of tau per gradient evaluation of the full centred model${coll.divergences === 0 ? ', with no divergences' : ''}`,
        markdown: [
          `Eight schools (Rubin's data), four chains of ${warmup} warm-up and ${draws} draws, target acceptance 0.9. ${plan.text.split('\n')[0]}`,
          '',
          table,
          '',
          'Gradient evaluations are the leapfrog steps of the sampling draws (warm-up is not counted, as the sampler does not report it). The collapsed program is the marginal likelihood of `y` given `mu` and `tau`; the draws of `theta` come back from `recover` and are not part of the cost.',
        ].join('\n'),
        budget: { chains: 4, warmup, draws },
      }
    },
  }
}

/**
 * The structure campaigns, `structure/benchmarks` and `structure/collapse-efficiency`.
 *
 * @returns The campaigns.
 * @concept model-analysis
 * @index function The structure of every benchmark model, and collapsed against full NUTS, as periodic campaigns.
 * @example
 * ```ts
 * import { structureCampaigns } from 'omai-engine/probprog/models'
 * structureCampaigns().map((c) => c.name) // => ['structure/benchmarks', 'structure/collapse-efficiency']
 * ```
 */
export function structureCampaigns(): Campaign[] {
  return [structureBenchmarks(), collapseEfficiency()]
}
