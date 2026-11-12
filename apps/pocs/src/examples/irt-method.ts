/**
 * Item response theory as an example method: a matrix of right and wrong answers from students on questions, generated from
 * a known truth (the numbers of students and questions, the spreads of the abilities and the difficulties, whether the
 * questions discriminate equally, a seed), the library's hierarchical model (`hierarchicalIrtDefinition`: the Rasch model or
 * the two-parameter model, with population priors on the abilities and the difficulties, centred or non-centred) and one
 * inference method of the library. The location of the scale is fixed by a mean ability of 0, the scale by the discrimination
 * (Rasch) or the ability sd (2PL) being fixed: the truth is shifted and scaled the same way before it is compared. The derived
 * paths read the run: `irt/recovery` (the abilities, difficulties and discriminations against the truth with their 90%
 * intervals, for the first `students` students and `questions` questions, the method refitted on them), `irt/pairs`,
 * `irt/diagnostics`, `irt/calibration` (the share of true values inside their intervals), `irt/summary` (the rows of the
 * hyperparameters and the abilities) and `irt/check` (the run against a reference run of the same answers).
 *
 * @index estimator Item response theory example: Rasch or 2PL, population priors, known truth.
 */

import type { Posterior } from 'omai-engine/probprog'
import type { InferenceMethod } from 'omai-engine/probprog/inference'
import { generateIrt, hierarchicalIrtDefinition, type IrtModelKind, type IrtSpreads } from 'omai-engine/probprog/models'
import { key } from 'omai-engine/random'
import { choice, type DictSpace, dict, int, real } from 'omai-engine/space'
import {
  add,
  asTensor,
  div,
  exp,
  greaterEqual,
  item,
  lessEqual,
  logicalAnd,
  mul,
  square,
  sub,
  sum,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
  zeros,
} from 'omai-engine/tensor'
import { type Config, derived } from 'omai-engine/trace'
import { defineExample, type Example, type ExampleProblem, exampleNumber, exampleString } from './example-method.ts'
import {
  chainsRecord,
  checkRows,
  flattenCheck,
  type PairCoordinate,
  type PosteriorRows,
  pairsRecord,
  posteriorRows,
  withDerivedSites,
} from './hierarchical-views.ts'

/** The true values of an item response problem, in the parameterisation of the fitted model. */
export interface IrtTruth {
  readonly theta: readonly number[]
  readonly b: readonly number[]
  /** The discriminations (the 2PL model's `a`); null for the Rasch model, where they are 1. */
  readonly a: readonly number[] | null
  /** The truth of the hyperparameters by site (`mu_b`, and per population `tau_*` or `prec_*`), where there is one. */
  readonly hyper: Readonly<Record<string, number>>
}

/** The problem an item response configuration stands for. */
export interface IrtProblem extends ExampleProblem {
  readonly model: IrtModelKind
  /** Whether the populations are written centred (always so with fixed spreads). */
  readonly centred: boolean
  readonly spreads: IrtSpreads
  readonly discrimination: 'log-normal' | 'normal'
  readonly students: number
  readonly questions: number
  readonly truth: IrtTruth
}

const mean = (v: TensorLike): number => item(div(sum(v), Math.max(1, asTensor(v).shape[0] ?? 0)))

/** The sd of the log-discriminations when the questions are said to discriminate unequally. */
const VARIED = 0.4

/**
 * The problem a configuration stands for: the answers from the seed, the model, and the truth in the model's own
 * parameterisation: the abilities and difficulties shifted by the mean ability (the model fixes the mean ability at 0) and,
 * for the 2PL model, divided by the sd of the abilities (the model fixes it at 1) with the discriminations multiplied by it.
 *
 * @param config - The knobs (`students`, `questions`, `abilitySd`, `difficultySd`, `meanDifficulty`, `discrimination`, `dataSeed`, `model`, `spreads`, `parameterisation`, `precisionShape`, `precisionRate`, `spreadScale`, `fixedSd`, `discriminationPrior`).
 * @returns The problem.
 * @example
 * ```ts
 * import { irtProblem } from './index.ts'
 * const p = irtProblem({ students: 20, questions: 6, abilitySd: 1, difficultySd: 1, meanDifficulty: 0.5, discrimination: 'equal', dataSeed: 1, model: 'rasch', parameterisation: 'non-centred', spreadScale: 1 })
 * [p.students, p.questions, Math.abs(p.truth.theta.reduce((a, t) => a + t, 0)) < 1e-9] // => [20, 6, true]
 * ```
 */
export function irtProblem(config: Config): IrtProblem {
  const students = Math.max(2, Math.round(exampleNumber(config, 'students') || 40))
  const questions = Math.max(2, Math.round(exampleNumber(config, 'questions') || 12))
  const abilitySd = exampleNumber(config, 'abilitySd') || 1
  const difficultySd = exampleNumber(config, 'difficultySd') || 1
  const meanDifficulty = exampleNumber(config, 'meanDifficulty')
  const varied = exampleString(config, 'discrimination') === 'varied'
  const model: IrtModelKind = exampleString(config, 'model') === '2pl' ? '2pl' : 'rasch'
  const spreads: IrtSpreads =
    exampleString(config, 'spreads') === 'half-normal'
      ? 'half-normal'
      : exampleString(config, 'spreads') === 'fixed'
        ? 'fixed'
        : 'precision'
  const centred = exampleString(config, 'parameterisation') !== 'non-centred' || spreads === 'fixed'
  const discrimination =
    model === '2pl' && exampleString(config, 'discriminationPrior') === 'normal' ? 'normal' : 'log-normal'
  const g = generateIrt(key(exampleNumber(config, 'dataSeed')), {
    students,
    questions,
    abilitySd,
    difficultySd,
    meanDifficulty,
    discriminationSd: varied ? VARIED : 0,
  })
  const abilities = tensor(g.theta)
  const centre = mean(abilities)
  const sd = Math.sqrt(item(div(sum(square(sub(abilities, centre))), g.theta.length))) || 1
  const scale = model === '2pl' ? sd : 1
  const theta = toFlat(div(sub(abilities, centre), scale))
  const b = toFlat(div(sub(tensor(g.b), centre), scale))
  const a = model === '2pl' ? toFlat(mul(tensor(g.a), scale)) : null
  // The truth of the spreads, as the model writes them (an sd, or a precision), where the model has them and the truth is known.
  const spread = (sd: number): number => (spreads === 'precision' ? 1 / sd ** 2 : sd)
  const names = spreads === 'precision' ? 'prec' : 'tau'
  const hyper: Record<string, number> =
    spreads === 'fixed'
      ? { mu_b: (meanDifficulty - centre) / scale }
      : model === '2pl'
        ? {
            mu_b: (meanDifficulty - centre) / scale,
            [`${names}_b`]: spread(difficultySd / scale),
            ...(discrimination === 'log-normal' ? { mu_a: Math.log(scale) } : {}),
            ...(varied && discrimination === 'log-normal' ? { [`${names}_a`]: spread(VARIED) } : {}),
          }
        : { [`${names}_t`]: spread(abilitySd), mu_b: meanDifficulty - centre, [`${names}_b`]: spread(difficultySd) }
  return {
    model,
    centred,
    spreads,
    discrimination,
    students,
    questions,
    truth: { theta, b, a, hyper },
    definition: hierarchicalIrtDefinition(model, centred ? 'centred' : 'non-centred', {
      spreads,
      scale: exampleNumber(config, 'spreadScale') || 1,
      shape: exampleNumber(config, 'precisionShape') || 2,
      rate: exampleNumber(config, 'precisionRate') || 2,
      fixedSd: exampleNumber(config, 'fixedSd') || 1,
      discrimination,
    }),
    data: { y: g.y },
  }
}

/** The knobs of the data: how many students and questions, how they differ, and the seed. */
export const irtDataKnobs: DictSpace = dict({
  students: int(5, 400, { default: 40, log: true, label: 'students' }),
  questions: int(3, 60, { default: 12, log: true, label: 'questions' }),
  abilitySd: real(0.3, 3, { default: 1, step: 0.1, label: 'true ability sd' }),
  difficultySd: real(0.1, 3, { default: 1, step: 0.1, label: 'true difficulty sd' }),
  meanDifficulty: real(-2, 2, {
    default: 0.5,
    step: 0.1,
    label: 'true average difficulty',
    description: 'above 0, the average student gets fewer than half the questions right',
  }),
  discrimination: choice(['equal', 'varied'], {
    default: 'equal',
    label: 'true discrimination',
    descriptions: {
      equal: 'every question separates strong from weak students equally (the Rasch model is true)',
      varied: 'the questions differ in how sharply they separate students (the log-discriminations have sd 0.4)',
    },
  }),
  dataSeed: int(0, 9999, { default: 1, label: 'seed' }),
})

/** The knobs of the model: Rasch or 2PL, how the population is written, and the prior on the spreads. */
export const irtModelKnobs: DictSpace = dict({
  model: choice(['rasch', '2pl'], {
    default: 'rasch',
    label: 'model',
    descriptions: {
      rasch: 'Rasch (1PL): every question has discrimination 1, the ability sd is learned',
      '2pl': '2PL: each question has a discrimination, learned log-normally; the ability sd is fixed at 1',
    },
  }),
  spreads: choice(['precision', 'half-normal', 'fixed'], {
    default: 'precision',
    label: 'population spreads',
    descriptions: {
      precision:
        'the spread of the abilities, difficulties and discriminations is learned as a Gamma precision: the form expectation propagation and variational message passing read',
      'half-normal':
        'the spreads are learned as half-normal standard deviations: samplers and variational methods only',
      fixed: 'every spread is known: a plain logistic regression with one Gaussian per student and question',
    },
  }),
  parameterisation: choice(['centred', 'non-centred'], {
    default: 'centred',
    label: 'parameterisation',
    descriptions: {
      centred:
        'theta ~ N(0, 1/precision), b ~ N(mu, 1/precision): a funnel between each spread and its values, but message passing reads it',
      'non-centred':
        'theta = z / sqrt(precision), b = mu + z / sqrt(precision): no funnel, samplers and variational methods only',
    },
  }),
  precisionShape: real(0.5, 20, {
    default: 2,
    log: true,
    label: 'precision prior shape',
    description: 'shape of the Gamma prior on each precision',
  }),
  precisionRate: real(0.5, 20, {
    default: 2,
    log: true,
    label: 'precision prior rate',
    description: 'rate of the Gamma prior on each precision (the mean precision is shape / rate)',
  }),
  spreadScale: real(0.2, 5, {
    default: 1,
    log: true,
    label: 'spread prior scale',
    description:
      'scale of the half-normal priors on the spreads of the abilities, difficulties and log-discriminations',
  }),
  fixedSd: real(0.2, 5, {
    default: 1,
    log: true,
    label: 'known spread',
    description: 'the standard deviation of the abilities and the difficulties (and discriminations), taken as known',
  }),
  discriminationPrior: choice(['log-normal', 'normal'], {
    default: 'log-normal',
    label: '2PL discrimination prior',
    descriptions: {
      'log-normal': 'a = exp(log a): positive; samplers and variational methods (message passing has no rule for exp)',
      normal:
        'a normal around a learned centre near 1 (it can go negative): a product of two random values, which expectation propagation reads and variational message passing does not',
    },
  }),
})

/** The knobs that make the problem. */
export const irtProblemKnobs: readonly string[] = [
  'students',
  'questions',
  'abilitySd',
  'difficultySd',
  'meanDifficulty',
  'discrimination',
  'dataSeed',
  'model',
  'spreads',
  'parameterisation',
  'precisionShape',
  'precisionRate',
  'spreadScale',
  'fixedSd',
  'discriminationPrior',
]

/** The knobs a configuration does not use: the other forms' priors, and the 2PL discrimination for the Rasch model. */
export function irtHiddenKnobs(config: Config): readonly string[] {
  const spreads = exampleString(config, 'spreads')
  return [
    ...(spreads === 'half-normal' ? [] : ['spreadScale']),
    ...(spreads === 'precision' || spreads === '' ? [] : ['precisionShape', 'precisionRate']),
    ...(spreads === 'fixed' ? [] : ['fixedSd']),
    ...(spreads === 'fixed' ? ['parameterisation'] : []),
    ...(exampleString(config, 'model') === '2pl' ? [] : ['discriminationPrior']),
  ]
}

/** The sites of the hyperparameters a model has. */
const HYPER = ['tau_t', 'prec_t', 'mu_b', 'tau_b', 'prec_b', 'mu_a', 'tau_a', 'prec_a']

/** One kind of parameter against its truth: the posterior mean and the 90% interval of each, and how well they agree. */
export interface Recovery {
  readonly label: string
  readonly truth: readonly number[]
  readonly mean: readonly number[]
  readonly lo: readonly number[]
  readonly hi: readonly number[]
  /** The share of true values inside their 90% intervals. */
  readonly coverage: number
  readonly rmse: number
  /** The mean width of the intervals. */
  readonly width: number
  readonly correlation: number
}

function recovery(label: string, site: string, truth: readonly number[], rows: PosteriorRows): Recovery | null {
  const byName = new Map(rows.rows.map((r) => [r.name, r]))
  const found = truth.map((_, i) => byName.get(`${site}[${i}]`))
  if (found.some((r) => r === undefined)) return null
  const got = found as NonNullable<(typeof found)[number]>[]
  const m = got.map((r) => r.mean)
  const t = tensor(truth as number[])
  const lo = tensor(got.map((r) => r.lo))
  const hi = tensor(got.map((r) => r.hi))
  const inside = item(sum(logicalAnd(greaterEqual(t, lo), lessEqual(t, hi))))
  const mt = mean(t)
  const mm = mean(m)
  const cov = item(sum(mul(sub(t, mt), sub(tensor(m), mm))))
  const vt = item(sum(square(sub(t, mt))))
  const vm = item(sum(square(sub(tensor(m), mm))))
  return {
    label,
    truth,
    mean: m,
    lo: got.map((r) => r.lo),
    hi: got.map((r) => r.hi),
    coverage: inside / truth.length,
    rmse: Math.sqrt(item(div(sum(square(sub(t, tensor(m)))), truth.length))),
    width: mean(sub(hi, lo)),
    correlation: vt > 0 && vm > 0 ? cov / Math.sqrt(vt * vm) : Number.NaN,
  }
}

/** The draws of the derived sites a model defines from its latents, for a posterior that has only the latents. */
function withItemSites(post: Posterior, problem: IrtProblem): Posterior {
  const at = (v: number[] | undefined, j: number): number => v?.[j] ?? 0
  // A site's draw as a vector of `n` (zeros when the site is missing).
  const over = (v: number[] | undefined, n: number): Value => (v === undefined ? zeros([n]) : tensor(v))
  const q = problem.questions
  const p = problem.students
  // The sd of a population from the sd or the precision a form has as a latent.
  const spreadOf = (d: Readonly<Record<string, number[]>>, label: string): number =>
    d[`tau_${label}`] !== undefined
      ? at(d[`tau_${label}`], 0)
      : 1 / Math.sqrt(Math.max(1e-12, at(d[`prec_${label}`], 0)))
  // The `n` values of a population: drawn directly, or the mean plus the spread times a standard normal.
  const population = (
    d: Readonly<Record<string, number[]>>,
    site: string,
    label: string,
    mean: number,
    n: number,
  ): Value => (problem.centred ? over(d[site], n) : add(mean, mul(spreadOf(d, label), over(d[`z_${label}`], n))))
  const items = (d: Readonly<Record<string, number[]>>): Record<string, number[]> => ({
    b: toFlat(population(d, 'b', 'b', at(d['mu_b'], 0), q)),
    a: toFlat(
      problem.discrimination === 'normal'
        ? population(d, 'a', 'a', at(d['mu_a'], 0), q)
        : exp(population(d, 'log_a', 'a', at(d['mu_a'], 0), q)),
    ),
    theta: toFlat(problem.model === 'rasch' ? population(d, 'theta', 't', 0, p) : over(d['theta'], p)),
  })
  const names = problem.model === '2pl' ? ['theta', 'b', 'a'] : ['theta', 'b']
  return withDerivedSites(post, names, (d) => {
    const all = items(d)
    return Object.fromEntries(names.map((n) => [n, all[n] as number[]]))
  })
}

/** What the reference run hands to the check, and what the table of hyperparameters reads: the rows and the truth of each hyperparameter. */
export interface IrtSummary extends PosteriorRows {
  readonly truth: Readonly<Record<string, number>>
}

const irtDeclared: Example<IrtProblem> = defineExample<IrtProblem>({
  name: 'irt',
  budgetMs: 10000,
  description:
    'Item response theory (Rasch or 2PL) with population priors on the answers of generated students, one inference method, checked against the truth and a long run',
  modelName: 'irt',
  dataKnobs: irtDataKnobs,
  modelKnobs: irtModelKnobs,
  problemKnobs: irtProblemKnobs,
  problem: irtProblem,
  derived: ({ problem, config, kind, posterior: fitted, fit }) => {
    const posterior = (state: unknown): Posterior => withItemSites(fitted(state), problem)
    const sitesOf = (p: IrtProblem): string[] => ['theta', 'b', ...(p.model === '2pl' ? ['a'] : []), ...HYPER]
    const rowsOf = (post: Posterior, p: IrtProblem): PosteriorRows | undefined => {
      try {
        return posteriorRows(withItemSites(post, p), sitesOf(p))
      } catch {
        return undefined
      }
    }
    // The recovery of the first `students` students and `questions` questions: the method refitted on them.
    const recoveryOf = (state: unknown, students: number, questions: number): Record<string, unknown> | null => {
      const full = students >= problem.students && questions >= problem.questions
      const p = full ? problem : irtProblem({ ...config, students, questions })
      let post: Posterior
      try {
        post = full ? fitted(state) : fit({ ...config, students, questions })
      } catch {
        return null
      }
      const rows = rowsOf(post, p)
      if (rows === undefined) return null
      const abilities = recovery('ability', 'theta', p.truth.theta, rows)
      const difficulties = recovery('difficulty', 'b', p.truth.b, rows)
      if (abilities === null || difficulties === null) return null
      return {
        students: p.students,
        questions: p.questions,
        totalStudents: problem.students,
        totalQuestions: problem.questions,
        abilities,
        difficulties,
        discriminations: p.truth.a === null ? null : recovery('discrimination', 'a', p.truth.a, rows),
      }
    }
    const truth = problem.truth.hyper
    const kindOf = problem.spreads === 'precision' ? 'precision' : 'sd'
    const site = (label: string): string => `${problem.spreads === 'precision' ? 'prec' : 'tau'}_${label}`
    const spreadCoordinates: PairCoordinate[] =
      problem.spreads === 'fixed'
        ? []
        : [
            { label: `log difficulty ${kindOf}`, from: site('b'), log: true },
            problem.model === 'rasch'
              ? { label: `log ability ${kindOf}`, from: site('t'), log: true }
              : { label: `log discrimination ${kindOf}`, from: site('a'), log: true },
          ]
    const pairs: readonly PairCoordinate[] = [
      { label: 'ability 1', from: 'theta[0]' },
      { label: 'ability 2', from: 'theta[1]' },
      { label: 'ability 3', from: 'theta[2]' },
      { label: 'average difficulty', from: 'mu_b' },
      ...spreadCoordinates,
    ]
    const pairTruth: Record<string, number> = {
      'ability 1': problem.truth.theta[0] ?? 0,
      'ability 2': problem.truth.theta[1] ?? 0,
      'ability 3': problem.truth.theta[2] ?? 0,
      ...(truth['mu_b'] === undefined ? {} : { 'average difficulty': truth['mu_b'] }),
      ...Object.fromEntries(
        spreadCoordinates.flatMap((c) => {
          const t = truth[c.from]
          return t === undefined ? [] : [[c.label, Math.log(t)] as [string, number]]
        }),
      ),
    }
    return {
      'irt/recovery': derived<unknown>({
        cost: 'dear',
        description:
          'the abilities, difficulties and discriminations against the truth with 90% intervals, for the first `students` students and `questions` questions (args `{ students, questions }`, default all): the method refitted on them, with the share of true values inside their intervals, the error and the mean interval width',
        compute: (state, args) => {
          const a = (args ?? {}) as { students?: number; questions?: number }
          const s = Math.max(2, Math.min(problem.students, Math.round(Number(a.students ?? problem.students))))
          const q = Math.max(2, Math.min(problem.questions, Math.round(Number(a.questions ?? problem.questions))))
          return recoveryOf(state, s, q) as never
        },
      }),
      'irt/calibration': derived<unknown>({
        cost: 'dear',
        description:
          'the calibration of the 90% intervals of the run on all the data: the share of true abilities, difficulties, discriminations and hyperparameters inside them',
        compute: (state) => {
          const rows = rowsOf(fitted(state), problem)
          if (rows === undefined) return null
          const abilities = recovery('ability', 'theta', problem.truth.theta, rows)
          const difficulties = recovery('difficulty', 'b', problem.truth.b, rows)
          if (abilities === null || difficulties === null) return null
          const discriminations =
            problem.truth.a === null ? null : recovery('discrimination', 'a', problem.truth.a, rows)
          const hyper = rows.rows.filter((r) => truth[r.name] !== undefined)
          return {
            abilities: abilities.coverage,
            difficulties: difficulties.coverage,
            discriminations: discriminations?.coverage ?? null,
            abilityCorrelation: abilities.correlation,
            difficultyCorrelation: difficulties.correlation,
            hyperparameters:
              hyper.length === 0
                ? null
                : hyper.filter((r) => (truth[r.name] as number) >= r.lo && (truth[r.name] as number) <= r.hi).length /
                  hyper.length,
            counts: { students: problem.students, questions: problem.questions },
          } as never
        },
      }),
      'irt/pairs': derived<unknown>({
        cost: 'dear',
        description:
          'the pair plot of three abilities, the average difficulty and the spreads of the population, the divergent transitions marked (a posteriorPairs record)',
        compute: (state) => pairsRecord(posterior(state), pairs, kind, pairTruth) ?? null,
      }),
      'irt/diagnostics': derived<unknown>({
        cost: 'dear',
        description:
          'trace, rank, R-hat and ESS of the chains of a sampler for the hyperparameters and two abilities (a chainDiagnostics record)',
        compute: (state) => chainsRecord(posterior(state), [...HYPER, 'theta[0]', 'theta[1]']) ?? null,
      }),
      'irt/summary': derived<unknown>({
        cost: 'dear',
        description:
          'the mean, spread, 90% interval, R-hat and ESS of the hyperparameters, abilities, difficulties and discriminations (a posterior rows record) with the truth of each hyperparameter',
        compute: (state) => {
          const rows = rowsOf(fitted(state), problem)
          return rows === undefined ? null : ({ ...rows, truth } as never)
        },
      }),
      'irt/check': derived<unknown>({
        cost: 'dear',
        description:
          "the run against a reference run of the same answers (args `{ summary }`, the reference's `irt/summary`): the largest difference of means in the reference's sds and in Monte Carlo errors, the ratio of the spreads, and the hyperparameters on their own",
        compute: (state, args) => {
          const ref = (args as { summary?: PosteriorRows } | null | undefined)?.summary
          const mine = rowsOf(fitted(state), problem)
          if (ref === undefined || mine === undefined) return null
          const check = checkRows(mine, ref, HYPER)
          return check === undefined ? null : (flattenCheck(check) as never)
        },
      }),
    }
  },
})

/**
 * Item response theory as a declared example (knobs, problem and derived paths: `irt/recovery`, `irt/calibration`,
 * `irt/pairs`, `irt/diagnostics`, `irt/summary`, `irt/check`). Its `reference` keeps the answers and the model but writes the
 * population priors non-centred, so a centred run is checked against a run with no funnel, and gives the long run a minute.
 *
 * @concept inference-engines
 * @example
 * ```ts
 * import { irtExample } from './index.ts'
 * irtExample.reference({ parameterisation: 'centred' }, { method: 'nuts' })['parameterisation'] // => 'non-centred'
 * ```
 */
export const irtExample: Example<IrtProblem> = {
  ...irtDeclared,
  reference: (config, inference, budgetMs = 60000) => ({
    ...irtDeclared.reference(config, inference, budgetMs),
    parameterisation: 'non-centred',
  }),
}

/**
 * The item response example as a method, run by name and knobs: see the file's summary.
 *
 * @concept item-response-theory
 * @index estimator Item response method: Rasch or 2PL, generated answers, one inference method.
 * @example
 * ```ts
 * import { irtMethod } from './index.ts'
 * irtMethod.name // => 'irt'
 * ```
 */
export const irtMethod: InferenceMethod<unknown, Config, string> = irtExample.method

/** The steps a configuration's run is expected to take (see `exampleSteps`). */
export const irtSteps: (config: Config) => number = irtExample.steps
