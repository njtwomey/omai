/**
 * The slots of the finite-difference time steppers as option tables: the time scheme of the heat equation (explicit,
 * implicit, Crank-Nicolson, or any theta in between), the advection stencil (upwind, Lax-Friedrichs, Lax-Wendroff), and
 * the initial profile of a field (a sine mode, a Gaussian bump, a square pulse, a triangle). Each table is one `union`
 * space for controls, grids and tuners; the schemes carry their order and stability as properties, so a page can offer
 * only what suits a step size.
 */

import {
  type DictSpace,
  defaults,
  dict,
  int,
  type Option,
  type OptionTable,
  optionTable,
  type RealSpace,
  real,
  type SpaceValue,
} from '../../space/index.ts'

/** What a time scheme does, for validity rules and selectors. */
export interface HeatSchemeProperties {
  /** Whether each step solves a linear system. */
  readonly implicit: boolean
  /** The order of accuracy in time. */
  readonly order: 1 | 2
  /** Whether every step size is stable (`conditional`: dt must stay below dx^2 / (2 alpha) for the explicit scheme). */
  readonly stability: 'conditional' | 'unconditional' | 'depends on theta'
}

/** The weight of the new level in the theta method (0 explicit, 1 implicit, 1/2 Crank-Nicolson). */
export type Theta = number

function scheme(
  description: string,
  properties: HeatSchemeProperties,
  theta: Theta,
): Option<Theta, HeatSchemeProperties> {
  return { description, properties, space: dict({}), make: () => theta }
}

/**
 * The time schemes of the heat equation as one slot: `explicit` (forward Euler in time, cheap, stable only for
 * alpha dt / dx^2 <= 1/2), `implicit` (backward Euler, unconditionally stable, first order), `crank-nicolson` (the
 * trapezoidal rule, second order and unconditionally stable) and `theta` (any weight in between). Each value is the
 * weight theta of the new time level.
 *
 * @concept heat-equation
 * @example
 * ```ts
 * import { heatSchemes } from 'mdml-engine/dynamics/pde'
 * heatSchemes.make({ kind: 'crank-nicolson' }) // => 0.5
 * ```
 */
export const heatSchemes: OptionTable<Theta, HeatSchemeProperties> = optionTable(
  {
    explicit: scheme('forward Euler in time (FTCS)', { implicit: false, order: 1, stability: 'conditional' }, 0),
    implicit: scheme('backward Euler in time (BTCS)', { implicit: true, order: 1, stability: 'unconditional' }, 1),
    'crank-nicolson': scheme(
      'the trapezoidal rule in time, an average of the old and new levels',
      { implicit: true, order: 2, stability: 'unconditional' },
      0.5,
    ),
    theta: {
      description: 'a weighted average of the old and new levels, theta between 0 (explicit) and 1 (implicit)',
      properties: { implicit: true, order: 1, stability: 'depends on theta' },
      space: dict({ theta: real(0, 1, { default: 0.75, label: 'weight of the new level θ' }) }),
      make: (config) => Number(config['theta'] ?? 0.75),
    },
  },
  { default: 'crank-nicolson', label: 'time scheme', description: 'How the heat equation is advanced in time.' },
)

/** One cell of an advection stencil: the new value from the left, centre and right values and nu = a dt / dx. */
export type AdvectionStencil = (l: number, c: number, r: number, nu: number, a: number) => number

/** What an advection scheme does, for validity rules and selectors. */
export interface AdvectionSchemeProperties {
  /** The order of accuracy. */
  readonly order: 1 | 2
  /** Whether it smears a sharp front (first-order, dissipative) rather than ringing around it. */
  readonly dissipative: boolean
  /** Whether it creates new extrema near a front (oscillates). */
  readonly oscillates: boolean
}

function stencil(
  description: string,
  properties: AdvectionSchemeProperties,
  make: AdvectionStencil,
): Option<AdvectionStencil, AdvectionSchemeProperties> {
  return { description, properties, space: dict({}), make: () => make }
}

/**
 * The advection stencils as one slot: `upwind` (first order, monotone, smears a front), `lax-friedrichs` (first order,
 * more dissipative) and `lax-wendroff` (second order, sharper, rings behind a front). Each is stable for
 * |a| dt / dx <= 1.
 *
 * @concept advection-equation
 * @example
 * ```ts
 * import { advectionSchemes } from 'mdml-engine/dynamics/pde'
 * advectionSchemes.make({ kind: 'upwind' })(0, 1, 1, 0.5, 1) // => 0.5
 * ```
 */
export const advectionSchemes: OptionTable<AdvectionStencil, AdvectionSchemeProperties> = optionTable(
  {
    upwind: stencil(
      'difference against the direction the wave comes from',
      { order: 1, dissipative: true, oscillates: false },
      (l, c, r, nu, a) => (a >= 0 ? c - nu * (c - l) : c - nu * (r - c)),
    ),
    'lax-friedrichs': stencil(
      'the average of the neighbours, corrected by the central difference',
      { order: 1, dissipative: true, oscillates: false },
      (l, _c, r, nu) => 0.5 * (l + r) - 0.5 * nu * (r - l),
    ),
    'lax-wendroff': stencil(
      'a second-order Taylor step in time, with the central difference and a diffusion term',
      { order: 2, dissipative: false, oscillates: true },
      (l, c, r, nu) => c - 0.5 * nu * (r - l) + 0.5 * nu * nu * (r - 2 * c + l),
    ),
  },
  { default: 'upwind', label: 'advection scheme', description: 'How the transport equation is differenced.' },
)

/** What an initial profile is, for validity rules and selectors. */
export interface ProfileProperties {
  /** Whether it has a jump (a front the schemes will smear or ring around). */
  readonly discontinuous: boolean
  /** Whether it is smooth and periodic on the unit interval. */
  readonly periodic: boolean
}

const unit = (label: string, def: number): RealSpace => real(0, 1, { default: def, label })

function profile(
  description: string,
  properties: ProfileProperties,
  space: DictSpace,
  build: (k: (name: string) => number) => (x: number) => number,
): Option<(x: number) => number, ProfileProperties> {
  return {
    description,
    properties,
    space,
    make: (config) => {
      const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }
      return build((name) => Number(full[name]))
    },
  }
}

/**
 * Initial profiles on the unit interval as one slot: `sine` (a mode with a number of half-waves), `gaussian` (a bump
 * with a centre and a width), `pulse` (a square pulse between two points) and `triangle` (a tent with a centre and a
 * half-width).
 *
 * @example
 * ```ts
 * import { initialProfiles } from 'mdml-engine/dynamics/pde'
 * initialProfiles.make({ kind: 'pulse', left: 0.25, right: 0.5 })(0.3) // => 1
 * ```
 */
export const initialProfiles: OptionTable<(x: number) => number, ProfileProperties> = optionTable(
  {
    sine: profile(
      'a sine mode with a whole number of half-waves on [0, 1]',
      { discontinuous: false, periodic: true },
      dict({ modes: real(1, 20, { default: 1, label: 'half-waves' }) }),
      (k) => (x) => Math.sin(k('modes') * Math.PI * x),
    ),
    gaussian: profile(
      'a Gaussian bump',
      { discontinuous: false, periodic: false },
      dict({ centre: unit('centre', 0.3), width: real(0.005, 0.5, { log: true, default: 0.05, label: 'width' }) }),
      (k) => (x) => Math.exp(-0.5 * ((x - k('centre')) / k('width')) ** 2),
    ),
    pulse: profile(
      'a square pulse of height one',
      { discontinuous: true, periodic: false },
      dict({ left: unit('left edge', 0.2), right: unit('right edge', 0.4) }),
      (k) => (x) => (x >= k('left') && x <= k('right') ? 1 : 0),
    ),
    triangle: profile(
      'a tent of height one',
      { discontinuous: false, periodic: false },
      dict({ centre: unit('centre', 0.5), halfWidth: real(0.01, 0.5, { default: 0.25, label: 'half-width' }) }),
      (k) => (x) => Math.max(0, 1 - Math.abs(x - k('centre')) / k('halfWidth')),
    ),
  },
  { default: 'sine', label: 'initial profile', description: 'The field at time zero, as a function of position.' },
)

/** What a Poisson iteration does, for validity rules and selectors. */
export interface PoissonMethodProperties {
  /** Whether it needs a grid of 2^k + 1 nodes per side with Dirichlet conditions. */
  readonly needsNestedGrids: boolean
  /** Whether it updates nodes in place with the newest values (Gauss-Seidel, SOR). */
  readonly inPlace: boolean
  /** The rough number of sweeps it needs to gain a digit as the grid is refined: grows like n^2, n or stays constant. */
  readonly iterationsGrow: 'n squared' | 'n' | 'constant'
}

/** A Poisson iteration and its knobs: the fields of the solver's options that choose and tune it. */
export interface PoissonIteration {
  readonly method: 'jacobi' | 'gauss-seidel' | 'sor' | 'multigrid'
  readonly omega?: number
  readonly sweeps?: number
  readonly coarseSweeps?: number
}

function iteration(
  description: string,
  properties: PoissonMethodProperties,
  space: DictSpace,
  make: (k: (name: string) => number) => PoissonIteration,
): Option<PoissonIteration, PoissonMethodProperties> {
  return {
    description,
    properties,
    space,
    make: (config) => {
      const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }
      return make((name) => Number(full[name]))
    },
  }
}

/**
 * The Poisson iterations as one slot: `jacobi` (simultaneous updates), `gauss-seidel` (in-place updates), `sor`
 * (over-relaxed Gauss-Seidel with a relaxation factor) and `multigrid` (V-cycles with red-black smoothing; the number
 * of sweeps are knobs). The value spreads into the options of `poisson` and `poissonAlgorithm`.
 *
 * @concept poisson-equation
 * @example
 * ```ts
 * import { poissonMethods } from 'mdml-engine/dynamics/pde'
 * poissonMethods.make({ kind: 'sor', omega: 1.5 }) // => { method: 'sor', omega: 1.5 }
 * ```
 */
export const poissonMethods: OptionTable<PoissonIteration, PoissonMethodProperties> = optionTable(
  {
    jacobi: iteration(
      'replace every node by the average of its neighbours, all at once',
      { needsNestedGrids: false, inPlace: false, iterationsGrow: 'n squared' },
      dict({}),
      () => ({ method: 'jacobi' }),
    ),
    'gauss-seidel': iteration(
      'update the nodes one after another, each seeing the newest neighbours',
      { needsNestedGrids: false, inPlace: true, iterationsGrow: 'n squared' },
      dict({}),
      () => ({ method: 'gauss-seidel' }),
    ),
    sor: iteration(
      'Gauss-Seidel over-relaxed by a factor between 1 and 2',
      { needsNestedGrids: false, inPlace: true, iterationsGrow: 'n' },
      dict({ omega: real(0.5, 1.99, { default: 1.5, label: 'relaxation factor ω' }) }),
      (k) => ({ method: 'sor', omega: k('omega') }),
    ),
    multigrid: iteration(
      'V-cycles that smooth on each grid and correct on the next coarser one',
      { needsNestedGrids: true, inPlace: true, iterationsGrow: 'constant' },
      dict({
        sweeps: int(1, 20, { default: 2, label: 'smoothing sweeps per level' }),
        coarseSweeps: int(1, 500, { log: true, default: 50, label: 'sweeps on the coarsest grid' }),
      }),
      (k) => ({ method: 'multigrid', sweeps: k('sweeps'), coarseSweeps: k('coarseSweeps') }),
    ),
  },
  {
    default: 'gauss-seidel',
    label: 'iteration',
    description: 'How the Poisson system is relaxed towards its solution.',
  },
)
