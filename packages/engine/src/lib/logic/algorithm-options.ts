/**
 * The slots of the stepped logic algorithms that vary as option tables: how the prime implicants are covered
 * (Quine-McCluskey: none, exact or greedy), how SampleSAT mixes WalkSAT and annealing moves, which concept learner runs
 * over the examples (Find-S or candidate elimination), and whether unification checks that a variable does not occur in
 * its binding. Each table is one `union` space for controls and grids.
 */

import { dict, int, type Option, type OptionTable, optionTable, plainOption, real } from '../space/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { candidateEliminationAlgorithm, findSAlgorithm, type LabelledExample } from './ilp/version-spaces.ts'
import type { SampleSatOptions } from './markov-logic.ts'
import type { CoverMethod, QuineMcCluskeyOptions } from './propositional.ts'
import type { UnifyOptions } from './unify.ts'

const cover = (description: string, method: CoverMethod): Option<QuineMcCluskeyOptions, Record<string, never>> =>
  plainOption(description, dict({}), () => ({ cover: method }))

/**
 * How Quine-McCluskey covers the models once the prime implicants are found, as one slot: `none` (stop with the primes),
 * `exact` (fewest terms, then fewest literals, by branch and bound) and `greedy` (the implicant covering most uncovered
 * models each time: fast, not always minimal).
 *
 * @concept normal-forms
 * @example
 * ```ts
 * import { coverMethods } from 'mdml-engine/logic'
 * coverMethods.make({ kind: 'greedy' }) // => { cover: 'greedy' }
 * ```
 */
export const coverMethods: OptionTable<QuineMcCluskeyOptions, Record<string, never>> = optionTable(
  {
    none: cover('stop with the prime implicants', 'none'),
    exact: cover('the fewest terms, then literals, by branch and bound', 'exact'),
    greedy: cover('the implicant covering most uncovered models each time', 'greedy'),
  },
  { default: 'exact', label: 'cover', description: 'How the models are covered by prime implicants.' },
)

function sampling(
  description: string,
  annealing: number,
  noise: number,
  temperature: number,
): Option<SampleSatOptions, Record<string, never>> {
  return plainOption(
    description,
    dict({
      annealing: real(0, 1, { default: annealing, label: 'annealing probability' }),
      noise: real(0, 1, { default: noise, label: 'WalkSAT noise' }),
      temperature: real(0.05, 5, { default: temperature, log: true, label: 'temperature' }),
      mixing: int(0, 100, { default: 10, label: 'mixing flips per variable' }),
    }),
    (k) => ({ annealing: k('annealing'), noise: k('noise'), temperature: k('temperature'), mixing: k('mixing') }),
  )
}

/**
 * How SampleSAT mixes its moves, as one slot: `balanced` (half annealing, half WalkSAT, the paper's setting),
 * `walksat` (mostly WalkSAT moves: finds a model quickly, mixes less) and `annealing` (mostly annealing: spreads over
 * the solution space, slower to find the first model).
 *
 * @concept markov-logic
 * @example
 * ```ts
 * import { sampleSatSchemes } from 'mdml-engine/logic'
 * sampleSatSchemes.make({ kind: 'walksat' }).annealing // => 0.1
 * ```
 */
export const sampleSatSchemes: OptionTable<SampleSatOptions, Record<string, never>> = optionTable(
  {
    balanced: sampling('half annealing moves, half WalkSAT moves', 0.5, 0.5, 0.5),
    walksat: sampling('mostly WalkSAT moves', 0.1, 0.5, 0.5),
    annealing: sampling('mostly annealing moves', 0.9, 0.5, 0.5),
  },
  { default: 'balanced', label: 'SampleSAT moves', description: 'The mix of WalkSAT and annealing moves.' },
)

/** Builds the stepped learner of a concept learner for examples over attribute domains. */
export type VersionSpaceLearner = (
  examples: readonly LabelledExample[],
  domains: readonly (readonly string[])[],
) => Algorithm<unknown>

/**
 * The concept learners over conjunctive hypotheses as one slot: `find-s` (the most specific hypothesis covering the
 * positives) and `candidate-elimination` (both boundaries of the version space).
 *
 * @concept version-spaces
 * @example
 * ```ts
 * import { versionSpaceLearners } from 'mdml-engine/logic'
 * versionSpaceLearners.make({ kind: 'find-s' })([], [['a']]).name // => 'find-s'
 * ```
 */
export const versionSpaceLearners: OptionTable<VersionSpaceLearner, Record<string, never>> = optionTable(
  {
    'find-s': plainOption<VersionSpaceLearner>(
      'the most specific hypothesis covering the positives',
      dict({}),
      () => (examples) => findSAlgorithm(examples) as unknown as Algorithm<unknown>,
    ),
    'candidate-elimination': plainOption<VersionSpaceLearner>(
      'both boundaries of the version space',
      dict({}),
      () => (examples, domains) => candidateEliminationAlgorithm(examples, domains) as unknown as Algorithm<unknown>,
    ),
  },
  {
    default: 'candidate-elimination',
    label: 'concept learner',
    description: 'How the hypothesis space is narrowed by examples.',
  },
)

/**
 * Whether unification checks that a variable does not occur in the term it is bound to, as one slot: `prolog` (no check,
 * fast, can build cyclic terms) and `sound` (the occurs check: X = f(X) fails).
 *
 * @concept unification
 * @example
 * ```ts
 * import { unificationVariants } from 'mdml-engine/logic'
 * unificationVariants.make({ kind: 'sound' }) // => { occursCheck: true }
 * ```
 */
export const unificationVariants: OptionTable<UnifyOptions, Record<string, never>> = optionTable(
  {
    prolog: plainOption<UnifyOptions>('no occurs check, as in standard Prolog', dict({}), () => ({
      occursCheck: false,
    })),
    sound: plainOption<UnifyOptions>('the occurs check on', dict({}), () => ({ occursCheck: true })),
  },
  { default: 'sound', label: 'unification', description: 'Whether a variable may be bound to a term containing it.' },
)
