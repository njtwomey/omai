/**
 * The eigenvalue iterations as one slot: an option table over power iteration (the dominant eigenvalue), inverse
 * iteration with a fixed shift (the eigenvalue nearest the shift) and Rayleigh quotient iteration (the shift follows the
 * estimate), each with the knobs of its own. `eigenIteration` builds the steppable algorithm a configuration names.
 */

import {
  type DictSpace,
  dict,
  int,
  type Knobs,
  knobOption,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../../space/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import {
  type EigenIterationOptions,
  type EigenIterationPath,
  type EigenIterationState,
  inverseIterationAlgorithm,
  powerIterationAlgorithm,
  rayleighQuotientIterationAlgorithm,
} from './eig.ts'

/** What an eigenvalue iteration is given: the matrix, and the start (a vector or a key, as for the functions). */
export interface EigenProblem extends Pick<EigenIterationOptions, 'x0' | 'key'> {
  readonly a: TensorLike
}

/** An eigenvalue iteration built from a configuration: the algorithm for a problem. */
export type EigenIterationMethod = (problem: EigenProblem) => Algorithm<EigenIterationState, EigenIterationPath>

/** What an iteration finds and how it converges, for validity rules and selectors. */
export interface EigenIterationProperties {
  /** Which eigenvalue it converges to. */
  readonly finds: 'largest magnitude' | 'nearest the shift' | 'nearest the start'
  /** Whether it solves a linear system at every step. */
  readonly solves: boolean
  /** The order of convergence, as text. */
  readonly order: string
}

function method(
  description: string,
  properties: EigenIterationProperties,
  space: DictSpace,
  build: (k: Knobs) => EigenIterationMethod,
): Option<EigenIterationMethod, EigenIterationProperties> {
  return knobOption(description, properties, space, build)
}

const startOf = (p: EigenProblem): EigenIterationOptions => ({
  ...(p.x0 === undefined ? {} : { x0: p.x0 }),
  ...(p.key === undefined ? {} : { key: p.key }),
})

/**
 * The eigenvalue iterations as one slot: `power` (multiply by A: the eigenvalue of largest magnitude, linearly at the
 * ratio of the two largest), `inverse` (solve with A - shift I: the eigenvalue nearest the shift) and `rayleigh` (inverse
 * iteration whose shift is the current estimate: cubic for symmetric matrices, but it goes where the start leads).
 *
 * @concept eigenvalues
 * @example
 * ```ts
 * import { eigenIterations } from 'omai-engine/numerics/linalg'
 * eigenIterations.names // => ['power', 'inverse', 'rayleigh']
 * ```
 */
export const eigenIterations: OptionTable<EigenIterationMethod, EigenIterationProperties> = optionTable(
  {
    power: method(
      'multiply by A and normalise: the eigenvector of the eigenvalue of largest magnitude',
      { finds: 'largest magnitude', solves: false, order: 'linear' },
      dict({
        tol: real(1e-15, 1e-3, { log: true, default: 1e-12, label: 'tolerance on the change of the estimate' }),
        maxIterations: int(1, 100_000, { log: true, default: 1000, label: 'iteration cap' }),
      }),
      (k) => (p) => powerIterationAlgorithm(p.a, { ...startOf(p), tol: k('tol'), maxIterations: k('maxIterations') }),
    ),
    inverse: method(
      'solve with A - shift I and normalise: the eigenvector of the eigenvalue nearest the shift',
      { finds: 'nearest the shift', solves: true, order: 'linear' },
      dict({
        shift: real(-1e6, 1e6, { default: 0, label: 'shift' }),
        tol: real(1e-15, 1e-3, { log: true, default: 1e-12, label: 'tolerance on the change of the estimate' }),
        maxIterations: int(1, 100_000, { log: true, default: 200, label: 'iteration cap' }),
      }),
      (k) => (p) =>
        inverseIterationAlgorithm(p.a, k('shift'), { ...startOf(p), tol: k('tol'), maxIterations: k('maxIterations') }),
    ),
    rayleigh: method(
      'inverse iteration with the shift set to the current Rayleigh quotient',
      { finds: 'nearest the start', solves: true, order: 'cubic (symmetric)' },
      dict({
        tol: real(1e-15, 1e-3, { log: true, default: 1e-13, label: 'tolerance on the change of the estimate' }),
        maxIterations: int(1, 10_000, { log: true, default: 50, label: 'iteration cap' }),
      }),
      (k) => (p) =>
        rayleighQuotientIterationAlgorithm(p.a, { ...startOf(p), tol: k('tol'), maxIterations: k('maxIterations') }),
    ),
  },
  {
    default: 'power',
    label: 'eigenvalue iteration',
    description: 'How one eigenpair of a matrix is found by repeated multiplication or solving.',
  },
)

/**
 * The steppable algorithm a configuration of the `eigenIterations` slot builds for a matrix.
 *
 * @param config - The option and its knobs, such as `{ kind: 'inverse', shift: 1.2 }`.
 * @param problem - The matrix and the start (a vector `x0` or a `key`).
 * @returns The algorithm; its state is an {@link EigenIterationState}.
 * @throws OmaiError with code `shape` for a matrix that is not square.
 * @concept eigenvalues
 * @example
 * ```ts
 * import { eigenIteration } from 'omai-engine/numerics/linalg'
 * eigenIteration({ kind: 'rayleigh' }, { a: [[2, 1], [1, 2]], x0: [1, 0.9] }).name // => 'rayleigh-quotient-iteration'
 * ```
 */
export function eigenIteration(
  config: SpaceValue,
  problem: EigenProblem,
): Algorithm<EigenIterationState, EigenIterationPath> {
  return eigenIterations.make(config)(problem)
}
