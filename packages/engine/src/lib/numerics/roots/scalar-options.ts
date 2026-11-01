/**
 * The scalar root finders as one slot: an option table over bisection, Brent's method, Ridders' method, TOMS 748, Newton,
 * Halley and the secant method, each with the knobs of its own (tolerances and the iteration cap), and what it needs of
 * the problem (a bracket, a start, derivatives). The table is one `union` space for controls, grids and tuners;
 * `rootAlgorithm` builds the steppable algorithm a configuration names.
 */

import { MdmlError } from '../../base/index.ts'
import {
  type DictSpace,
  defaults,
  dict,
  int,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../../space/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import {
  bisectAlgorithm,
  brentqAlgorithm,
  newtonAlgorithm,
  type RootPath,
  type RootState,
  ridderAlgorithm,
  toms748Algorithm,
} from './scalar-algorithms.ts'

/** What a root finder is given: the function, and a bracket, a start or derivatives as the method needs them. */
export interface RootProblem {
  readonly f: (x: number) => number
  /** A bracket [a, b] with a sign change (bracketing methods). */
  readonly bracket?: readonly [number, number]
  /** The start (Newton, Halley, secant). */
  readonly x0?: number
  /** A second start for the secant method (default a small step from `x0`). */
  readonly x1?: number
  /** The derivative (Newton, Halley). */
  readonly fprime?: (x: number) => number
  /** The second derivative (Halley). */
  readonly fprime2?: (x: number) => number
}

/** A root finder built from a configuration: the algorithm for a problem. */
export type RootFinder = (problem: RootProblem) => Algorithm<RootState, RootPath>

/** What a root finder needs and guarantees, for validity rules and selectors. */
export interface RootFinderProperties {
  /** Whether it needs a bracket with a sign change (and then always converges) rather than a start. */
  readonly bracketed: boolean
  /** How many derivatives of f it needs (0, 1 or 2). */
  readonly derivatives: 0 | 1 | 2
  /** The order of convergence near a simple root, as text. */
  readonly order: string
}

type Knobs = (name: string) => number

const knobsOf = (space: DictSpace, config: Readonly<Record<string, SpaceValue>>): Knobs => {
  const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }
  return (name) => full[name] as number
}

function finder(
  description: string,
  properties: RootFinderProperties,
  space: DictSpace,
  build: (k: Knobs) => RootFinder,
): Option<RootFinder, RootFinderProperties> {
  return { description, properties, space, make: (config) => build(knobsOf(space, config)) }
}

const bracketKnobs = dict({
  xtol: real(1e-16, 1e-3, { log: true, default: 2e-12, label: 'absolute tolerance on x' }),
  rtol: real(1e-16, 1e-3, { log: true, default: 4 * 2.220446049250313e-16, label: 'relative tolerance on x' }),
  maxIterations: int(1, 10_000, { log: true, default: 100, label: 'iteration cap' }),
})

const openKnobs = dict({
  tol: real(1e-16, 1e-2, { log: true, default: 1.48e-8, label: 'absolute tolerance on the step' }),
  rtol: real(0, 1e-2, { default: 0, label: 'relative tolerance on the step' }),
  maxIterations: int(1, 10_000, { log: true, default: 50, label: 'iteration cap' }),
})

const needs = (what: string, has: boolean, name: string): void => {
  if (!has) throw new MdmlError('value', `${name}: the problem needs ${what}`)
}

function bracketed(
  name: string,
  make: (
    f: (x: number) => number,
    a: number,
    b: number,
    o: { xtol: number; rtol: number; maxIterations: number },
  ) => Algorithm<RootState, RootPath>,
): (k: Knobs) => RootFinder {
  return (k) => (p) => {
    needs('a bracket', p.bracket !== undefined, name)
    const [a, b] = p.bracket as readonly [number, number]
    return make(p.f, a, b, { xtol: k('xtol'), rtol: k('rtol'), maxIterations: k('maxIterations') })
  }
}

/**
 * The scalar root finders as one slot: `bisect` (halve a bracket), `brentq` (interpolation with a bisection fallback),
 * `ridder` (exponential interpolation), `toms748` (inverse cubic interpolation), `newton` (needs the derivative),
 * `halley` (needs two derivatives) and `secant` (needs a start only). The bracketing ones always converge; the open ones
 * are faster near the root but may leave it.
 *
 * @concept root-finding
 * @example
 * ```ts
 * import { rootFinders } from 'mdml-engine/numerics/roots'
 * rootFinders.names // => ['bisect', 'brentq', 'ridder', 'toms748', 'newton', 'halley', 'secant']
 * ```
 */
export const rootFinders: OptionTable<RootFinder, RootFinderProperties> = optionTable(
  {
    bisect: finder(
      'halve the bracket, keeping the half with the sign change',
      { bracketed: true, derivatives: 0, order: 'linear' },
      bracketKnobs,
      bracketed('bisect', bisectAlgorithm),
    ),
    brentq: finder(
      'inverse quadratic interpolation and secant steps, with bisection when they are not trusted',
      { bracketed: true, derivatives: 0, order: 'superlinear' },
      bracketKnobs,
      bracketed('brentq', brentqAlgorithm),
    ),
    ridder: finder(
      'exponential interpolation through the bracket and its midpoint',
      { bracketed: true, derivatives: 0, order: 'quadratic per two evaluations' },
      bracketKnobs,
      bracketed('ridder', ridderAlgorithm),
    ),
    toms748: finder(
      'inverse cubic interpolation, a double-length secant step and bisection',
      { bracketed: true, derivatives: 0, order: '2.7 per iteration' },
      bracketKnobs,
      bracketed('toms748', toms748Algorithm),
    ),
    newton: finder(
      'Newton steps from a start with the derivative',
      { bracketed: false, derivatives: 1, order: 'quadratic' },
      openKnobs,
      (k) => (p) => {
        needs('a start x0', p.x0 !== undefined, 'newton')
        needs('the derivative fprime', p.fprime !== undefined, 'newton')
        return newtonAlgorithm(p.f, p.x0 as number, {
          fprime: p.fprime as (x: number) => number,
          tol: k('tol'),
          rtol: k('rtol'),
          maxIterations: k('maxIterations'),
        })
      },
    ),
    halley: finder(
      'Newton steps with Halley’s correction from the second derivative',
      { bracketed: false, derivatives: 2, order: 'cubic' },
      openKnobs,
      (k) => (p) => {
        needs('a start x0', p.x0 !== undefined, 'halley')
        needs('the derivatives fprime and fprime2', p.fprime !== undefined && p.fprime2 !== undefined, 'halley')
        return newtonAlgorithm(p.f, p.x0 as number, {
          fprime: p.fprime as (x: number) => number,
          fprime2: p.fprime2 as (x: number) => number,
          tol: k('tol'),
          rtol: k('rtol'),
          maxIterations: k('maxIterations'),
        })
      },
    ),
    secant: finder(
      'secant steps through the last two iterates, no derivative',
      { bracketed: false, derivatives: 0, order: 'superlinear (1.618)' },
      openKnobs,
      (k) => (p) => {
        needs('a start x0', p.x0 !== undefined, 'secant')
        return newtonAlgorithm(p.f, p.x0 as number, {
          ...(p.x1 === undefined ? {} : { x1: p.x1 }),
          tol: k('tol'),
          rtol: k('rtol'),
          maxIterations: k('maxIterations'),
        })
      },
    ),
  },
  {
    default: 'brentq',
    label: 'root finder',
    description: 'How the root of a scalar function is found: by shrinking a bracket or by following the function.',
  },
)

/**
 * The steppable algorithm a configuration of the `rootFinders` slot builds for a problem.
 *
 * @param config - The option and its knobs, such as `{ kind: 'bisect', xtol: 1e-6 }` (knobs not given take defaults).
 * @param problem - The function and what the method needs (a bracket, a start, derivatives).
 * @returns The algorithm; its state is a {@link RootState}.
 * @throws MdmlError with code `value` when the problem lacks what the method needs.
 * @concept root-finding
 * @example
 * ```ts
 * import { rootAlgorithm } from 'mdml-engine/numerics/roots'
 * rootAlgorithm({ kind: 'newton' }, { f: (x) => x * x - 2, x0: 1, fprime: (x) => 2 * x }).name // => 'newton'
 * ```
 */
export function rootAlgorithm(config: SpaceValue, problem: RootProblem): Algorithm<RootState, RootPath> {
  return rootFinders.make(config)(problem)
}

/**
 * A complete value of the `rootFinders` slot: the option's name and its knobs, those not given at their defaults.
 *
 * @param kind - The option: `bisect`, `brentq`, `ridder`, `toms748`, `newton`, `halley` or `secant`.
 * @param knobs - Values for some of the option's own knobs.
 * @returns The configuration, `{ kind, ...knobs }` with every knob of the option.
 * @throws MdmlError with code `value` for an option the table lacks or a knob it lacks.
 * @example
 * ```ts
 * import { rootFinderValue } from 'mdml-engine/numerics/roots'
 * rootFinderValue('newton', { tol: 1e-10 }) // => { kind: 'newton', tol: 1e-10, rtol: 0, maxIterations: 50 }
 * ```
 */
export function rootFinderValue(kind: string, knobs: Readonly<Record<string, number>> = {}): SpaceValue {
  const space = rootFinders.space
  const option = space.options[kind]
  if (option === undefined) throw new MdmlError('value', `'${kind}' is not one of ${rootFinders.names.join(', ')}`)
  for (const name of Object.keys(knobs))
    if (!(name in option.spaces)) throw new MdmlError('value', `'${kind}' has no knob '${name}'`)
  return { [space.tag ?? 'kind']: kind, ...(defaults(option) as Readonly<Record<string, SpaceValue>>), ...knobs }
}
