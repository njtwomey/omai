/**
 * The distribution families a model spec can name, as a closed table: for each family its parameter names (those of
 * the library's constructors), defaults, alternative parameterisations (`probs` or `logits`), the rank of its event
 * (a Dirichlet draw is a vector), whether it is discrete or an exponential family, and `make`, which builds the
 * library's own distribution from evaluated parameters. Nothing here is a second distribution: every `make` calls the
 * constructor in `omai-engine/dists`. The table is what turns a JSON `DistSpec` back into a `Distribution`.
 * One family is a composition: `normalSpikeSlab`, the zero-mean two-component scale mixture of the spike and slab prior,
 * built as `mixture` of two normals.
 *
 * @index table The distribution families a model spec names, mapped to the library's constructors.
 */

import { OmaiError } from '../../../base/index.ts'
import { mixture } from '../../../dists/compose/index.ts'
import {
  bernoulli,
  beta,
  binomial,
  categorical,
  cauchy,
  type Distribution,
  dirichlet,
  exponential,
  gamma,
  geometric,
  halfCauchy,
  halfNormal,
  inverseGamma,
  laplace,
  logNormal,
  multivariateNormal,
  negativeBinomial,
  normal,
  poisson,
  studentT,
  truncatedNormal,
  uniform,
  weibull,
} from '../../../dists/index.ts'
import { stack, sub, type Value } from '../../../tensor/index.ts'

/** The evaluated parameters of a family, by name. */
export type FamilyArgs = Readonly<Record<string, Value>>

/** One family of the table. */
export interface FamilyDef {
  readonly name: string
  /** Every parameter the family accepts. */
  readonly params: readonly string[]
  /** Values taken for parameters left out (the constructors' own defaults). */
  readonly defaults: Readonly<Record<string, number>>
  /** Groups of alternative parameters of which exactly one must be given (`['probs', 'logits']`). */
  readonly oneOf: readonly (readonly string[])[]
  /** The number of event axes of a draw (1 for a Dirichlet or a multivariate normal). */
  readonly eventRank: number
  readonly discrete: boolean
  /** Whether a discrete family has finitely many values (Bernoulli, binomial, categorical), so a latent can be enumerated. */
  readonly finite: boolean
  /** Whether the family has an exponential-family form (so message passing and conjugacy can use it). */
  readonly exponentialFamily: boolean
  /** Example parameters of a valid member, used by tests and drawings. */
  readonly example: Readonly<Record<string, number | readonly number[] | readonly (readonly number[])[]>>
  make(args: FamilyArgs): Distribution
}

function need(family: string, args: FamilyArgs, name: string): Value {
  const v = args[name]
  if (v === undefined) throw new OmaiError('value', `${family}: the parameter '${name}' is missing`, { family, name })
  return v
}

function pick(family: string, args: FamilyArgs, names: readonly string[]): { name: string; value: Value } {
  for (const n of names) {
    const v = args[n]
    if (v !== undefined) return { name: n, value: v }
  }
  throw new OmaiError('value', `${family}: give one of ${names.join(' or ')}`, { family })
}

type Make = (a: FamilyArgs) => Distribution

function family(
  name: string,
  params: readonly string[],
  make: Make,
  example: FamilyDef['example'],
  options: {
    readonly defaults?: Readonly<Record<string, number>>
    readonly oneOf?: readonly (readonly string[])[]
    readonly eventRank?: number
    readonly discrete?: boolean
    readonly finite?: boolean
    readonly exponentialFamily?: boolean
  } = {},
): FamilyDef {
  return {
    name,
    params,
    defaults: options.defaults ?? {},
    oneOf: options.oneOf ?? [],
    eventRank: options.eventRank ?? 0,
    discrete: options.discrete ?? false,
    finite: options.finite ?? false,
    exponentialFamily: options.exponentialFamily ?? false,
    example,
    make,
  }
}

const covariance = ['covariance', 'precision', 'scaleTril'] as const

/**
 * The table of distribution families: normal, halfNormal, logNormal, truncatedNormal, studentT, cauchy, halfCauchy,
 * laplace, exponential, gamma, inverseGamma, weibull, beta, uniform, bernoulli, binomial, geometric, negativeBinomial,
 * poisson, categorical, dirichlet and multivariateNormal. Parameter names are those of the constructors in
 * `omai-engine/dists`.
 *
 * @returns The families by name (a fresh object each call).
 * @concept model-spec
 * @example
 * ```ts
 * import { distributionFamilies } from 'omai-engine/probprog/model/dists'
 * const f = distributionFamilies()['normal']
 * f?.make({ loc: 0, scale: 2 } as never).name // => 'normal'
 * ```
 */
export function distributionFamilies(): Record<string, FamilyDef> {
  const defs: FamilyDef[] = [
    family(
      'normal',
      ['loc', 'scale'],
      (a) => normal(need('normal', a, 'loc'), need('normal', a, 'scale')),
      { loc: 0, scale: 1 },
      {
        defaults: { loc: 0, scale: 1 },
        exponentialFamily: true,
      },
    ),
    family(
      'halfNormal',
      ['scale'],
      (a) => halfNormal(need('halfNormal', a, 'scale')),
      { scale: 1 },
      {
        defaults: { scale: 1 },
        exponentialFamily: true,
      },
    ),
    family(
      'logNormal',
      ['mu', 'sigma'],
      (a) => logNormal(need('logNormal', a, 'mu'), need('logNormal', a, 'sigma')),
      { mu: 0, sigma: 1 },
      {
        defaults: { mu: 0, sigma: 1 },
        exponentialFamily: true,
      },
    ),
    family(
      'truncatedNormal',
      ['loc', 'scale', 'low', 'high'],
      (a) =>
        truncatedNormal(
          need('truncatedNormal', a, 'loc'),
          need('truncatedNormal', a, 'scale'),
          need('truncatedNormal', a, 'low'),
          need('truncatedNormal', a, 'high'),
        ),
      { loc: 0, scale: 1, low: -1, high: 2 },
    ),
    family(
      'studentT',
      ['df', 'loc', 'scale'],
      (a) => studentT(need('studentT', a, 'df'), need('studentT', a, 'loc'), need('studentT', a, 'scale')),
      { df: 3, loc: 0, scale: 1 },
      {
        defaults: { loc: 0, scale: 1 },
      },
    ),
    family(
      'cauchy',
      ['loc', 'scale'],
      (a) => cauchy(need('cauchy', a, 'loc'), need('cauchy', a, 'scale')),
      { loc: 0, scale: 1 },
      {
        defaults: { loc: 0, scale: 1 },
      },
    ),
    family(
      'halfCauchy',
      ['scale'],
      (a) => halfCauchy(need('halfCauchy', a, 'scale')),
      { scale: 1 },
      { defaults: { scale: 1 } },
    ),
    family(
      'laplace',
      ['loc', 'scale'],
      (a) => laplace(need('laplace', a, 'loc'), need('laplace', a, 'scale')),
      { loc: 0, scale: 1 },
      {
        defaults: { loc: 0, scale: 1 },
        exponentialFamily: true,
      },
    ),
    family(
      'exponential',
      ['rate'],
      (a) => exponential(need('exponential', a, 'rate')),
      { rate: 1 },
      {
        defaults: { rate: 1 },
        exponentialFamily: true,
      },
    ),
    family(
      'gamma',
      ['concentration', 'rate'],
      (a) => gamma(need('gamma', a, 'concentration'), need('gamma', a, 'rate')),
      { concentration: 2, rate: 1 },
      {
        defaults: { rate: 1 },
        exponentialFamily: true,
      },
    ),
    family(
      'inverseGamma',
      ['concentration', 'scale'],
      (a) => inverseGamma(need('inverseGamma', a, 'concentration'), need('inverseGamma', a, 'scale')),
      { concentration: 3, scale: 1 },
      { defaults: { scale: 1 }, exponentialFamily: true },
    ),
    family(
      'weibull',
      ['concentration', 'scale'],
      (a) => weibull(need('weibull', a, 'concentration'), need('weibull', a, 'scale')),
      { concentration: 2, scale: 1 },
      {
        defaults: { scale: 1 },
        exponentialFamily: true,
      },
    ),
    family(
      'beta',
      ['a', 'b'],
      (x) => beta(need('beta', x, 'a'), need('beta', x, 'b')),
      { a: 2, b: 3 },
      { exponentialFamily: true },
    ),
    family(
      'uniform',
      ['low', 'high'],
      (a) => uniform(need('uniform', a, 'low'), need('uniform', a, 'high')),
      { low: 0, high: 1 },
      {
        defaults: { low: 0, high: 1 },
      },
    ),
    family(
      'bernoulli',
      ['probs', 'logits'],
      (a) => {
        const p = pick('bernoulli', a, ['probs', 'logits'])
        return p.name === 'probs' ? bernoulli({ probs: p.value }) : bernoulli({ logits: p.value })
      },
      { probs: 0.3 },
      { oneOf: [['probs', 'logits']], discrete: true, finite: true, exponentialFamily: true },
    ),
    family(
      'binomial',
      ['n', 'probs'],
      (a) => binomial(need('binomial', a, 'n'), need('binomial', a, 'probs')),
      { n: 5, probs: 0.4 },
      {
        discrete: true,
        finite: true,
        exponentialFamily: true,
      },
    ),
    family(
      'geometric',
      ['probs'],
      (a) => geometric(need('geometric', a, 'probs')),
      { probs: 0.4 },
      { discrete: true, exponentialFamily: true },
    ),
    family(
      'negativeBinomial',
      ['n', 'probs'],
      (a) => negativeBinomial(need('negativeBinomial', a, 'n'), need('negativeBinomial', a, 'probs')),
      { n: 3, probs: 0.4 },
      { discrete: true, exponentialFamily: true },
    ),
    family(
      'poisson',
      ['rate'],
      (a) => poisson(need('poisson', a, 'rate')),
      { rate: 3 },
      { discrete: true, exponentialFamily: true },
    ),
    family(
      'categorical',
      ['probs', 'logits'],
      (a) => {
        const p = pick('categorical', a, ['probs', 'logits'])
        return p.name === 'probs' ? categorical({ probs: p.value }) : categorical({ logits: p.value })
      },
      { probs: [0.2, 0.3, 0.5] },
      { oneOf: [['probs', 'logits']], discrete: true, finite: true, exponentialFamily: true },
    ),
    family(
      'dirichlet',
      ['concentration'],
      (a) => dirichlet(need('dirichlet', a, 'concentration')),
      { concentration: [1, 2, 3] },
      {
        eventRank: 1,
        exponentialFamily: true,
      },
    ),
    family(
      'normalSpikeSlab',
      ['inclusion', 'slab', 'spike'],
      (a) => {
        const inclusion = need('normalSpikeSlab', a, 'inclusion')
        const slab = need('normalSpikeSlab', a, 'slab')
        const spike = need('normalSpikeSlab', a, 'spike')
        return mixture({ probs: stack([inclusion, sub(1, inclusion)], -1) }, [normal(0, slab), normal(0, spike)])
      },
      { inclusion: 0.5, slab: 2, spike: 0.1 },
    ),
    family(
      'multivariateNormal',
      ['loc', ...covariance],
      (a) => {
        const c = pick('multivariateNormal', a, covariance)
        const loc = need('multivariateNormal', a, 'loc')
        if (c.name === 'covariance') return multivariateNormal(loc, { covariance: c.value })
        if (c.name === 'precision') return multivariateNormal(loc, { precision: c.value })
        return multivariateNormal(loc, { scaleTril: c.value })
      },
      {
        loc: [0, 0],
        covariance: [
          [1, 0.3],
          [0.3, 1],
        ],
      },
      { oneOf: [covariance], eventRank: 1, exponentialFamily: true },
    ),
  ]
  return Object.fromEntries(defs.map((d) => [d.name, d]))
}
