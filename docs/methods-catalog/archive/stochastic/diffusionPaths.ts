/**
 * Diffusion processes watched, as a method of nothing but its configuration: a bundle of independent paths of an
 * Ornstein-Uhlenbeck, geometric Brownian, Brownian or Cox-Ingersoll-Ross process advanced by one Euler-Maruyama step
 * per step through the library's SDE stepper, with the bundle's mean and variance against the process's exact moments
 * on the tape, and the bundle's histogram at the viewed time against the exact law at that time as a probe.
 */

import { type CurveSet, curveSet } from '../../lib/exposition/index.ts'
import { normal } from '../../lib/random/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import {
  cir,
  type Diffusion,
  geometricBrownianMotion,
  ornsteinUhlenbeck,
  type Sde,
  sdeStep,
} from '../../lib/stochastic/index.ts'
import { add, mul, tensor, toFlat } from '../../lib/tensor/index.ts'
import {
  type Config,
  type Contract,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  tensor as tensorSource,
} from '../../lib/trace/index.ts'

/** The processes on offer. */
export type DiffusionName = 'ornstein-uhlenbeck' | 'geometric-brownian' | 'brownian' | 'cir'

/** The bundle: the time reached and every path's value. */
export interface DiffusionPathsState {
  readonly k: number
  readonly t: number
  readonly x: readonly number[]
}

/** A process to simulate: its equation, its starting point, and its exact moments and law when known. */
interface ProcessSpec {
  readonly sde: Sde
  readonly x0: number
  /** Paths are kept at or above this (the square-root process cannot go negative). */
  readonly floor?: number
  readonly mean?: (t: number) => number
  readonly variance?: (t: number) => number
  /** The density of X_t over a grid of values. */
  readonly density?: (t: number, grid: readonly number[]) => number[]
}

const first = (v: unknown): number => toFlat(v as never)[0] ?? Number.NaN

function fromDiffusion(d: Diffusion, x0: number, floor?: number): ProcessSpec {
  return {
    sde: d,
    x0,
    ...(floor === undefined ? {} : { floor }),
    ...(d.mean === undefined ? {} : { mean: (t: number) => first(d.mean?.(x0, t)) }),
    ...(d.variance === undefined ? {} : { variance: (t: number) => first(d.variance?.(x0, t)) }),
    ...(d.transition === undefined
      ? {}
      : {
          density: (t: number, grid: readonly number[]) => toFlat(d.transition?.(tensor(x0), t)?.prob(grid) as never),
        }),
  }
}

function processOf(name: DiffusionName, c: Config): ProcessSpec {
  const theta = c['theta'] as number
  const sigma = c['sigma'] as number
  const x0 = c['x0'] as number
  switch (name) {
    case 'ornstein-uhlenbeck':
      return fromDiffusion(ornsteinUhlenbeck({ theta, mu: 0, sigma }), x0)
    case 'geometric-brownian':
      return fromDiffusion(geometricBrownianMotion({ mu: 0.1, sigma }), Math.max(x0, 0.1))
    case 'cir':
      return fromDiffusion(cir({ kappa: theta, theta: 1, sigma }), Math.max(x0, 0.1), 0)
    case 'brownian': {
      const variance = (t: number): number => sigma * sigma * t
      return {
        sde: { drift: (_t, y) => mul(y, 0), diffusion: (_t, y) => add(mul(y, 0), tensor(sigma)) },
        x0,
        mean: () => x0,
        variance,
        density: (t, grid) => {
          const v = variance(t)
          return grid.map((g) => (v > 0 ? Math.exp((-0.5 * (g - x0) ** 2) / v) / Math.sqrt(2 * Math.PI * v) : 0))
        },
      }
    }
  }
}

function distributionRecord(p: ProcessSpec, state: DiffusionPathsState): CurveSet {
  const xs = state.x
  const lo = Math.min(...xs)
  const hi = Math.max(...xs)
  const pad = 0.25 * Math.max(hi - lo, 1e-3)
  const [a, b] = [lo - pad, hi + pad]
  const bins = 20
  const width = (b - a) / bins
  const counts = new Array<number>(bins).fill(0)
  for (const x of xs) {
    const i = Math.min(bins - 1, Math.floor((x - a) / width))
    counts[i] = (counts[i] ?? 0) + 1
  }
  const along: number[] = []
  const up: number[] = []
  counts.forEach((n, i) => {
    along.push(a + i * width, a + (i + 1) * width)
    const h = n / (xs.length * width)
    up.push(h, h)
  })
  const series: { name: string; x: number[]; y: number[] }[] = [{ name: 'the bundle', x: along, y: up }]
  if (p.density !== undefined && state.t > 0) {
    const grid = Array.from({ length: 121 }, (_, i) => a + ((b - a) * i) / 120)
    const d = p.density(state.t, grid).map((v) => (Number.isFinite(v) ? v : 0))
    series.push({ name: `the exact law at t = ${state.t.toFixed(2)}`, x: grid, y: d })
  }
  return curveSet({ series, xLabel: 'x', yLabel: 'density' })
}

/**
 * The paths of {@link diffusionPaths}: the bundle’s values, the time, the mean and variance over the bundle and their exact values, and the derived `paths/distribution`.
 *
 * @example
 * ```ts
 * import type { DiffusionPathsPath } from 'omai-engine/methods/stochastic'
 * const path: DiffusionPathsPath = 'paths/mean'
 * path // => 'paths/mean'
 * ```
 */
export type DiffusionPathsPath =
  | 'paths/x'
  | 'paths/t'
  | 'paths/mean'
  | 'paths/variance'
  | 'paths/exactMean'
  | 'paths/exactVariance'
  | 'paths/distribution'

/**
 * A bundle of paths of a diffusion, one Euler-Maruyama step per step.
 *
 * Configuration: `process` (ornstein-uhlenbeck, geometric-brownian, brownian, cir), `paths`, `dt`, `theta` (the rate
 * of mean reversion, where there is one), `sigma` (the volatility) and `x0` (where every path starts; the positive
 * processes start at least at 0.1). Paths: `paths/x` (every path's value, full), `paths/t`, `paths/mean`,
 * `paths/variance`, `paths/exactMean`, `paths/exactVariance`, and the probe `paths/distribution` (the bundle's
 * histogram at this time with the exact law over it).
 */
export const diffusionPaths: Method<DiffusionPathsState, Config, DiffusionPathsPath> = defineMethod<
  DiffusionPathsState,
  Config,
  DiffusionPathsPath
>({
  name: 'stochastic/diffusion-paths',
  config: dict({
    process: choice(['ornstein-uhlenbeck', 'geometric-brownian', 'brownian', 'cir'] as const, {
      default: 'ornstein-uhlenbeck',
    }),
    paths: int(1, 64, { default: 16 }),
    dt: real(0.001, 0.1, { default: 0.01, log: true }),
    theta: real(0.1, 5, { default: 1, label: 'mean reversion θ' }),
    sigma: real(0.05, 2, { default: 0.5, label: 'volatility σ' }),
    x0: real(-3, 3, { default: 2, label: 'start x₀' }),
  }),
  make: (c) => {
    const p = processOf(c['process'] as DiffusionName, c)
    const n = c['paths'] as number
    const dt = c['dt'] as number
    return {
      name: 'diffusion-paths',
      config: { process: c['process'] as string, paths: n, dt },
      init: () => ({ k: 0, t: 0, x: new Array<number>(n).fill(p.x0) }),
      step: (s, ctx) => {
        const dw = mul(normal(ctx.key, [n]), Math.sqrt(dt))
        let x = toFlat(sdeStep(p.sde, 'euler', s.t, tensor([...s.x]), dt, dw))
        if (p.floor !== undefined) x = x.map((v) => Math.max(p.floor ?? 0, v))
        const t = s.t + dt
        const mean = x.reduce((a, b) => a + b, 0) / n
        const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, n - 1)
        ctx.log('paths/x', x)
        ctx.log('paths/t', t)
        ctx.log('paths/mean', mean)
        ctx.log('paths/variance', variance)
        if (p.mean !== undefined) ctx.log('paths/exactMean', p.mean(t))
        if (p.variance !== undefined) ctx.log('paths/exactVariance', p.variance(t))
        return { k: s.k + 1, t, x }
      },
      serialise: (s) => ({ k: s.k, t: s.t, x: [...s.x] }),
      deserialise: (j) => j as unknown as DiffusionPathsState,
    }
  },
  contract: (c): Contract<DiffusionPathsState, DiffusionPathsPath> => {
    const p = processOf(c['process'] as DiffusionName, c)
    return {
      'paths/x': tensorSource({ role: 'state', shape: [c['paths'] as number], description: 'every path’s value' }),
      'paths/t': scalar({ description: 'the time reached' }),
      'paths/mean': scalar({ description: 'the mean over the bundle' }),
      'paths/variance': scalar({ description: 'the variance over the bundle' }),
      'paths/exactMean': scalar({ description: 'E[X_t], when the process has it in closed form' }),
      'paths/exactVariance': scalar({ description: 'Var[X_t], when the process has it in closed form' }),
      'paths/distribution': derived<DiffusionPathsState>({
        description: 'the bundle’s histogram at this time, with the exact law at this time over it',
        compute: (s) => distributionRecord(p, s),
      }),
    }
  },
  defaults: {
    'paths/x': Log({ detail: 'full' }),
    'paths/t': Log(),
    'paths/mean': Log(),
    'paths/variance': Log(),
    'paths/exactMean': Log(),
    'paths/exactVariance': Log(),
  },
})
