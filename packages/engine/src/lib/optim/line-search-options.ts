/**
 * The slot of a stepped line search that varies as an option table: backtracking with cubic interpolation (scipy's
 * Armijo search, optionally safeguarded), backtracking by a fixed contraction, and the Hager-Zhang search, each with
 * the knobs of its own. The table is one `union` space for controls and grids, and builds the algorithm for a line
 * function and a start from a configuration.
 */

import {
  bool,
  dict,
  int,
  type Knobs,
  knobOption,
  type Option,
  type OptionTable,
  optionTable,
  real,
} from '../space/index.ts'
import type { Algorithm } from '../trace/index.ts'
import type { LineFunction, LineStart } from './line-search.ts'
import {
  backtrackingAlgorithm,
  hagerZhangAlgorithm,
  type LineSearchPath,
  type LineSearchState,
} from './line-search-algorithms.ts'

/** Builds the stepped line search of a rule for a line function and its start. */
export type LineSearchBuilder = (phi: LineFunction, start: LineStart) => Algorithm<LineSearchState, LineSearchPath>

/** What a rule guarantees, for validity rules and selectors. */
export interface LineSearchRuleProperties {
  /** Whether it uses the slope of the line function (Hager-Zhang) or only its value (backtracking). */
  readonly usesSlope: boolean
  /** Whether the step it accepts also satisfies a curvature condition. */
  readonly curvature: boolean
}

function rule(
  description: string,
  properties: LineSearchRuleProperties,
  space: ReturnType<typeof dict>,
  build: (k: Knobs<number | boolean>) => LineSearchBuilder,
): Option<LineSearchBuilder, LineSearchRuleProperties> {
  return knobOption(description, properties, space, build)
}

/**
 * The line-search rules as one slot: `backtracking` (shrink the step by interpolating the rejected values until
 * the Armijo condition holds, optionally clipped to [0.1, 0.5] of the previous step), `halving` (multiply the step by
 * a fixed factor) and `hager-zhang` (bracket, then secant steps with bisection, accepting the Wolfe or approximate
 * Wolfe conditions). Each builds a stepped {@link LineSearchState} algorithm.
 *
 * @concept line-search
 * @example
 * ```ts
 * import { lineSearchRules } from 'omai-engine/optim'
 * const phi = { at: (a: number) => ({ value: (a - 0.25) ** 2, slope: 2 * (a - 0.25) }) }
 * lineSearchRules.make({ kind: 'hager-zhang' })(phi, { value: 0.0625, slope: -0.5 }).name // => 'optim/hager-zhang'
 * ```
 */
export const lineSearchRules: OptionTable<LineSearchBuilder, LineSearchRuleProperties> = optionTable(
  {
    backtracking: rule(
      'shrink the step by quadratic and cubic interpolation of the rejected values until sufficient decrease',
      { usesSlope: false, curvature: false },
      dict({
        c1: real(1e-8, 0.5, { default: 1e-4, label: 'sufficient decrease c1', log: true }),
        safeguard: bool({ default: false, label: 'clip steps to 0.1 to 0.5 of the last' }),
      }),
      (k) => (phi, start) =>
        backtrackingAlgorithm(phi, start, {
          c1: k('c1') as number,
          ...(k('safeguard') === true ? { safeguard: [0.1, 0.5] as const } : {}),
        }),
    ),
    halving: rule(
      'multiply the step by a fixed factor until sufficient decrease',
      { usesSlope: false, curvature: false },
      dict({
        c1: real(1e-8, 0.5, { default: 1e-4, label: 'sufficient decrease c1', log: true }),
        rho: real(0.1, 0.9, { default: 0.5, label: 'contraction factor' }),
        maxEvaluations: int(1, 200, { default: 50, label: 'most evaluations' }),
      }),
      (k) => (phi, start) =>
        backtrackingAlgorithm(phi, start, {
          c1: k('c1') as number,
          interpolation: 'halving',
          rho: k('rho') as number,
          maxEvaluations: k('maxEvaluations') as number,
        }),
    ),
    'hager-zhang': rule(
      'bracket, then secant steps with bisection safeguards, accepting the Wolfe or approximate Wolfe conditions',
      { usesSlope: true, curvature: true },
      dict({
        delta: real(1e-4, 0.4, { default: 0.1, label: 'sufficient decrease delta' }),
        sigma: real(0.5, 0.99, { default: 0.9, label: 'curvature sigma' }),
        epsilon: real(1e-12, 1e-2, { default: 1e-6, label: 'approximate Wolfe tolerance', log: true }),
        rho: real(1.5, 10, { default: 5, label: 'expansion factor' }),
      }),
      (k) => (phi, start) =>
        hagerZhangAlgorithm(phi, start, {
          delta: k('delta') as number,
          sigma: k('sigma') as number,
          epsilon: k('epsilon') as number,
          rho: k('rho') as number,
        }),
    ),
  },
  {
    default: 'backtracking',
    label: 'line search',
    description: 'How far to step along a descent direction: by backtracking or by a Wolfe search.',
  },
)
