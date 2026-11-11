/**
 * Item response theory with population priors on the abilities and the item parameters: students answer questions right
 * or wrong, the chance of a right answer is `sigmoid(a_j (theta_p - b_j))` for the ability `theta_p` of student `p`, the
 * difficulty `b_j` and the discrimination `a_j` of question `j`. Two models are written, the Rasch model (every
 * discrimination is 1; the ability sd is learned) and the two-parameter model 2PL (discriminations are learned, log-normal
 * or normal around a learned centre; the ability sd is fixed at 1). The spreads of the populations are a half-normal sd, a
 * Gamma precision (the form message passing reads) or fixed. The location of the scale is fixed by the abilities having
 * population mean 0 (the average difficulty `mu_b` is then identified), and the scale of the abilities by fixing the
 * discrimination at 1 (Rasch) or the ability sd at 1 (2PL): without them `theta + c, b + c` and `(theta/c, a c)` give the
 * same likelihood. The abilities and the difficulties are written centred or non-centred. A generator makes the answers from a
 * known truth, as a prefix of a table of 400 students by 60 questions, so more students or more questions are the same
 * people and items plus new ones.
 */

import { normal as drawNormal, uniform as drawUniform, foldIn, type Key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, type ModelDefinition, ops } from '../model/index.ts'

/** The two item response models {@link hierarchicalIrtDefinition} offers. */
export type IrtModelKind = 'rasch' | '2pl'

/** The parameterisation of the abilities and the item parameters. */
export type IrtParameterisation = 'centred' | 'non-centred'

/**
 * How the spread of a population (of abilities, difficulties, discriminations) is written: a half-normal sd (`tau_*`), a
 * Gamma precision (`prec_*`, the idiom the message-passing methods read: the normal's scale is `1 / sqrt(prec)`), or a known sd.
 */
export type IrtSpreads = 'half-normal' | 'precision' | 'fixed'

/** The priors of {@link hierarchicalIrtDefinition}. */
export interface IrtPriors {
  /** How the spreads are written (default `half-normal`). */
  readonly spreads?: IrtSpreads
  /** The scale of the half-normal priors of the spreads (default 1). */
  readonly scale?: number
  /** The shape and the rate of the Gamma priors of the precisions (default 2 and 2: a mean precision of 1). */
  readonly shape?: number
  readonly rate?: number
  /** The known sd of every population when the spreads are fixed (default 1). */
  readonly fixedSd?: number
  /**
   * The prior of the 2PL discriminations: `log-normal` (a positive `a = exp(log a)`, the samplers and variational methods
   * only) or `normal` (`a` itself normal around a learned centre near 1, which can go negative but is a plain product that
   * expectation propagation reads). Default `log-normal`.
   */
  readonly discrimination?: 'log-normal' | 'normal'
}

type Value = ReturnType<typeof ops.clip>

/**
 * The hierarchical item response model as a spec over `y`, the answers (`[student][question]`, 1 right, 0 wrong). The
 * populations (the abilities of the Rasch model, the difficulties, the 2PL discriminations) have a normal prior with a spread
 * written as `priors.spreads` says, centred or non-centred (`z_*` standard normal and the value derived; fixed spreads have
 * only the centred form). The latents: `mu_b`; per population `tau_*` (half-normal sd) or `prec_*` (Gamma precision) unless
 * fixed, and the population itself (`theta`, `b`, `log_a` or `a`) or its standardised `z_*`; 2PL has `theta` standard normal
 * (the scale constraint) and `mu_a`. Centred, with a Gamma precision or fixed spreads, the Rasch model is a logistic regression
 * with one Gaussian per student and question, and expectation propagation and variational message passing run it.
 *
 * @param model - Rasch or 2PL.
 * @param parameterisation - Centred or non-centred.
 * @param priors - How the spreads are written and their priors.
 * @returns The model definition, named `irt-<model>`.
 * @concept item-response-theory
 * @example
 * ```ts
 * import { hierarchicalIrtDefinition } from 'omai-engine/probprog/models'
 * const latents = (m: 'rasch' | '2pl', spreads: 'half-normal' | 'precision') =>
 *   hierarchicalIrtDefinition(m, 'centred', { spreads }).spec.nodes.filter((n) => n.kind === 'latent').map((n) => n.name)
 * [latents('rasch', 'half-normal'), latents('rasch', 'precision')] // => [['mu_b', 'tau_b', 'b', 'tau_t', 'theta'], ['mu_b', 'prec_b', 'b', 'prec_t', 'theta']]
 * ```
 */
export function hierarchicalIrtDefinition(
  model: IrtModelKind,
  parameterisation: IrtParameterisation = 'non-centred',
  priors: IrtPriors = {},
): ModelDefinition {
  const { spreads = 'half-normal', scale = 1, shape = 2, rate = 2, fixedSd = 1, discrimination = 'log-normal' } = priors
  const centred = parameterisation === 'centred' || spreads === 'fixed'
  return defineModel(`irt-${model}`, (m) => {
    const student = m.range('student')
    const question = m.range('question')
    const y = m.data('y', { dims: [student, question], support: 'binary', role: 'response', doc: 'the answers' })
    const clip = (x: Value): Value => ops.clip(x, 1e-6, 1e6)
    // One population of values: normal around `mean`, its spread as the priors say, centred or not.
    const population = (
      label: 't' | 'b' | 'a',
      site: string,
      mean: number | Value,
      dim: ReturnType<typeof m.range>,
      doc: string,
    ): Value => {
      const around = (v: Value): Value => (typeof mean === 'number' ? (mean === 0 ? v : v.add(mean)) : v.add(mean))
      if (spreads === 'fixed') return m.latent(site, d.normal(mean, fixedSd), { dims: [dim], doc })
      const sd =
        spreads === 'precision'
          ? ops.div(1, ops.sqrt(m.latent(`prec_${label}`, d.gamma(shape, rate), { doc: `the precision of ${doc}` })))
          : clip(m.latent(`tau_${label}`, d.halfNormal(scale), { doc: `the sd of ${doc}` }))
      if (centred) return m.latent(site, d.normal(mean, sd), { dims: [dim], doc })
      const z = m.latent(`z_${label}`, d.normal(0, 1), { dims: [dim], doc: `the standardised ${doc}` })
      return m.derived(site, around(sd.mul(z)), { doc })
    }
    const muB = m.latent('mu_b', d.normal(0, 3), { doc: 'the average difficulty' })
    const difficulty = population('b', 'b', muB, question, 'the difficulties of the questions')
    if (model === 'rasch') {
      const ability = population('t', 'theta', 0, student, 'the abilities of the students')
      m.observe('y', d.bernoulli({ logits: ability.sub(difficulty) }), y)
      return
    }
    const ability = m.latent('theta', d.normal(0, 1), { dims: [student], doc: 'the ability of each student (sd 1)' })
    if (discrimination === 'normal') {
      const muA = m.latent('mu_a', d.normal(1, 1), { doc: 'the centre of the discriminations' })
      const a = population('a', 'a', muA, question, 'the discriminations of the questions')
      m.observe('y', d.bernoulli({ logits: a.mul(ability.sub(difficulty)) }), y)
      return
    }
    const muA = m.latent('mu_a', d.normal(0, 1), { doc: 'the centre of the log-discriminations' })
    const logA = population('a', 'log_a', muA, question, 'the log-discriminations of the questions')
    const a = m.derived('a', logA.exp(), { doc: 'the discrimination of each question' })
    m.observe('y', d.bernoulli({ logits: a.mul(ability.sub(difficulty)) }), y)
  })
}

/** The largest table the generator makes (the data are its leading rows and columns). */
export const IRT_MAX_STUDENTS = 400
export const IRT_MAX_QUESTIONS = 60

/** What {@link generateIrt} asks for. */
export interface IrtOptions {
  /** The students, up to {@link IRT_MAX_STUDENTS}. */
  readonly students: number
  /** The questions, up to {@link IRT_MAX_QUESTIONS}. */
  readonly questions: number
  /** The sd of the abilities (default 1). */
  readonly abilitySd?: number
  /** The sd of the difficulties (default 1). */
  readonly difficultySd?: number
  /** The average difficulty (default 0.5). */
  readonly meanDifficulty?: number
  /** The sd of the log-discriminations (default 0: every discrimination is 1, the Rasch model is true). */
  readonly discriminationSd?: number
}

/** Answers with the truth that made them. */
export interface IrtData {
  /** The answers, `[student][question]`. */
  readonly y: number[][]
  /** The abilities (not centred: the models fix the mean ability at 0, so a comparison shifts them and the difficulties by the mean). */
  readonly theta: number[]
  readonly b: number[]
  readonly a: number[]
}

/**
 * Answers of students to questions from a known truth: abilities `N(0, abilitySd^2)`, difficulties `N(meanDifficulty, difficultySd^2)`, discriminations
 * `exp(discriminationSd z)`. The answers are drawn from a table of {@link IRT_MAX_STUDENTS} by {@link IRT_MAX_QUESTIONS}, so
 * the first 20 students of a table of 40 are the first 20 of a table of 20, and so for questions.
 *
 * @param key - The random key.
 * @param options - The sizes and spreads.
 * @returns The answers and the truth.
 * @concept item-response-theory
 * @index dataset Item response answers from known abilities, difficulties and discriminations.
 * @example
 * ```ts
 * import { generateIrt } from 'omai-engine/probprog/models'
 * import { key } from 'omai-engine/random'
 * const g = generateIrt(key(1), { students: 30, questions: 8 })
 * [g.y.length, g.y[0]?.length, g.theta.length] // => [30, 8, 30]
 * ```
 */
export function generateIrt(key: Key, options: IrtOptions): IrtData {
  const { abilitySd = 1, difficultySd = 1, meanDifficulty = 0.5, discriminationSd = 0 } = options
  const P = Math.max(1, Math.min(IRT_MAX_STUDENTS, Math.round(options.students)))
  const J = Math.max(1, Math.min(IRT_MAX_QUESTIONS, Math.round(options.questions)))
  const zt = toFlat(drawNormal(foldIn(key, 0), [IRT_MAX_STUDENTS]))
  const zb = toFlat(drawNormal(foldIn(key, 1), [IRT_MAX_QUESTIONS]))
  const za = toFlat(drawNormal(foldIn(key, 2), [IRT_MAX_QUESTIONS]))
  const u = toFlat(drawUniform(foldIn(key, 3), [IRT_MAX_STUDENTS * IRT_MAX_QUESTIONS]))
  const theta = Array.from({ length: P }, (_, i) => abilitySd * (zt[i] as number))
  const b = Array.from({ length: J }, (_, j) => meanDifficulty + difficultySd * (zb[j] as number))
  const a = Array.from({ length: J }, (_, j) => Math.exp(discriminationSd * (za[j] as number)))
  const y = theta.map((t, i) =>
    b.map((bj, j) =>
      (u[i * IRT_MAX_QUESTIONS + j] as number) < 1 / (1 + Math.exp(-(a[j] as number) * (t - bj))) ? 1 : 0,
    ),
  )
  return { y, theta, b, a }
}
