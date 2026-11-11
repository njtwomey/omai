/**
 * Item response theory with population priors on the abilities and the item parameters: students answer questions right
 * or wrong, the chance of a right answer is `sigmoid(a_j (theta_p - b_j))` for the ability `theta_p` of student `p`, the
 * difficulty `b_j` and the discrimination `a_j` of question `j`. Two models are written, the Rasch model (every
 * discrimination is 1; the ability sd is learned) and the two-parameter model 2PL (discriminations are learned, log-normal
 * around a learned centre; the ability sd is fixed at 1). The location of the scale is fixed by the abilities having
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

/** The scales of the priors of {@link hierarchicalIrtDefinition}. */
export interface IrtPriors {
  /** The scale of the half-normal priors of the ability sd, the difficulty sd and the sd of the log-discriminations (default 1). */
  readonly scale?: number
}

/**
 * The hierarchical item response model as a spec over `y`, the answers (`[student][question]`, 1 right, 0 wrong). The latents:
 * Rasch has `tau_t` (the ability sd), `mu_b`, `tau_b` and, non-centred, `z_t` per student and `z_b` per question with `theta`
 * and `b` derived (centred: `theta` and `b` themselves are the latents); 2PL has `mu_b`, `tau_b`, `mu_a`, `tau_a`, the
 * abilities `theta` (standard normal, the scale constraint) and `z_b`, `z_a` per question with `b` and `a` derived (centred:
 * `b` and the log-discriminations `log_a` are latents, `a` derived).
 *
 * @param model - Rasch or 2PL.
 * @param parameterisation - Centred or non-centred.
 * @param priors - The scales of the priors.
 * @returns The model definition, named `irt-<model>`.
 * @concept item-response-theory
 * @example
 * ```ts
 * import { hierarchicalIrtDefinition } from 'omai-engine/probprog/models'
 * const latents = (m: 'rasch' | '2pl') =>
 *   hierarchicalIrtDefinition(m).spec.nodes.filter((n) => n.kind === 'latent').map((n) => n.name)
 * [latents('rasch'), latents('2pl')] // => [['mu_b', 'tau_b', 'z_b', 'tau_t', 'z_t'], ['mu_b', 'tau_b', 'z_b', 'theta', 'mu_a', 'tau_a', 'z_a']]
 * ```
 */
export function hierarchicalIrtDefinition(
  model: IrtModelKind,
  parameterisation: IrtParameterisation = 'non-centred',
  priors: IrtPriors = {},
): ModelDefinition {
  const { scale = 1 } = priors
  const centred = parameterisation === 'centred'
  return defineModel(`irt-${model}`, (m) => {
    const student = m.range('student')
    const question = m.range('question')
    const y = m.data('y', { dims: [student, question], support: 'binary', role: 'response', doc: 'the answers' })
    const muB = m.latent('mu_b', d.normal(0, 3), { doc: 'the average difficulty' })
    const tauB = m.latent('tau_b', d.halfNormal(scale), { doc: 'the sd of the difficulties' })
    const clip = (x: Parameters<typeof ops.clip>[0]): ReturnType<typeof ops.clip> => ops.clip(x, 1e-6, 1e6)
    const difficulty = centred
      ? m.latent('b', d.normal(muB, clip(tauB)), { dims: [question], doc: 'the difficulty of each question' })
      : m.derived(
          'b',
          muB.add(
            tauB.mul(m.latent('z_b', d.normal(0, 1), { dims: [question], doc: 'the standardised difficulties' })),
          ),
          { doc: 'the difficulty of each question' },
        )
    if (model === 'rasch') {
      const tauT = m.latent('tau_t', d.halfNormal(scale), { doc: 'the sd of the abilities' })
      const ability = centred
        ? m.latent('theta', d.normal(0, clip(tauT)), { dims: [student], doc: 'the ability of each student' })
        : m.derived(
            'theta',
            tauT.mul(m.latent('z_t', d.normal(0, 1), { dims: [student], doc: 'the standardised abilities' })),
            { doc: 'the ability of each student' },
          )
      m.observe('y', d.bernoulli({ logits: ability.sub(difficulty) }), y)
      return
    }
    const ability = m.latent('theta', d.normal(0, 1), { dims: [student], doc: 'the ability of each student (sd 1)' })
    const muA = m.latent('mu_a', d.normal(0, 1), { doc: 'the centre of the log-discriminations' })
    const tauA = m.latent('tau_a', d.halfNormal(scale), { doc: 'the sd of the log-discriminations' })
    const logA = centred
      ? m.latent('log_a', d.normal(muA, clip(tauA)), {
          dims: [question],
          doc: 'the log-discrimination of each question',
        })
      : muA.add(
          tauA.mul(m.latent('z_a', d.normal(0, 1), { dims: [question], doc: 'the standardised log-discriminations' })),
        )
    const discrimination = m.derived('a', logA.exp(), { doc: 'the discrimination of each question' })
    m.observe('y', d.bernoulli({ logits: discrimination.mul(ability.sub(difficulty)) }), y)
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
