/**
 * Sampling by transformation, watched: uniform or standard normal samples pushed through a monotone map become
 * samples of a harder distribution (the exponential and the Cauchy by their inverse distribution functions, the
 * logistic, the log-normal, a bimodal law), a batch per step; the change of variables, p_Y(y) = p_U(g⁻¹(y)) / g′(g⁻¹(y)),
 * gives the density those samples pile up under. The picture is a probe: the map, the base density and the base
 * samples' histogram along the bottom, the target density and the pushed samples' histogram along the side, and the
 * latest samples' journeys up to the map and across.
 */

import { type CurveSet, curveSet } from '../../lib/exposition/index.ts'
import { ndtr } from '../../lib/numerics/special/index.ts'
import { type Key, normal, uniform } from '../../lib/random/index.ts'
import { choice, dict, int } from '../../lib/space/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import {
  type Config,
  type Contract,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  tensor,
} from '../../lib/trace/index.ts'

/** The target laws on offer. */
export type TransformTarget = 'exponential' | 'cauchy' | 'logistic' | 'lognormal' | 'bimodal'

/** The samples so far: the base draws and their images. */
export interface TransformSamplingState {
  readonly u: readonly number[]
  readonly y: readonly number[]
}

type Pair = readonly [number, number]

/** A monotone increasing map of a base variable, with what the change of variables needs. */
interface Transform {
  readonly base: 'uniform' | 'normal'
  readonly baseLabel: string
  readonly baseRange: Pair
  /** The window of the target drawn. */
  readonly range: Pair
  readonly g: (u: number) => number
  readonly gInverse: (y: number) => number
  readonly gDerivative: (u: number) => number
  readonly mean: number | undefined
}

const phi = (z: number): number => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)
const Phi = (z: number): number => toFlat(ndtr(z))[0] ?? Number.NaN

// The inverse of a monotone increasing map by bisection over [lo, hi].
function bisect(g: (u: number) => number, y: number, lo: number, hi: number): number {
  let a = lo
  let b = hi
  for (let i = 0; i < 60; i++) {
    const m = 0.5 * (a + b)
    if (g(m) < y) a = m
    else b = m
  }
  return 0.5 * (a + b)
}

function transformOf(name: TransformTarget): Transform {
  switch (name) {
    case 'exponential':
      return {
        base: 'uniform',
        baseLabel: 'u ~ Uniform(0, 1)',
        baseRange: [0, 1],
        range: [0, 6],
        g: (u) => -Math.log(1 - u),
        gInverse: (y) => 1 - Math.exp(-y),
        gDerivative: (u) => 1 / (1 - u),
        mean: 1,
      }
    case 'cauchy':
      return {
        base: 'uniform',
        baseLabel: 'u ~ Uniform(0, 1)',
        baseRange: [0, 1],
        range: [-8, 8],
        g: (u) => Math.tan(Math.PI * (u - 0.5)),
        gInverse: (y) => 0.5 + Math.atan(y) / Math.PI,
        gDerivative: (u) => Math.PI / Math.cos(Math.PI * (u - 0.5)) ** 2,
        mean: undefined,
      }
    case 'logistic':
      return {
        base: 'uniform',
        baseLabel: 'u ~ Uniform(0, 1)',
        baseRange: [0, 1],
        range: [-6, 6],
        g: (u) => Math.log(u / (1 - u)),
        gInverse: (y) => 1 / (1 + Math.exp(-y)),
        gDerivative: (u) => 1 / (u * (1 - u)),
        mean: 0,
      }
    case 'lognormal':
      return {
        base: 'normal',
        baseLabel: 'z ~ Normal(0, 1)',
        baseRange: [-4, 4],
        range: [0, 8],
        g: Math.exp,
        gInverse: Math.log,
        gDerivative: Math.exp,
        mean: Math.exp(0.5),
      }
    case 'bimodal': {
      // z + 2 tanh(2 z): mass is pushed away from the origin into two modes.
      const g = (z: number): number => z + 2 * Math.tanh(2 * z)
      return {
        base: 'normal',
        baseLabel: 'z ~ Normal(0, 1)',
        baseRange: [-4, 4],
        range: [-6, 6],
        g,
        gInverse: (y) => bisect(g, y, -12, 12),
        gDerivative: (z) => 1 + 4 / Math.cosh(2 * z) ** 2,
        mean: 0,
      }
    }
  }
}

const baseDensity = (t: Transform, u: number): number => (t.base === 'uniform' ? (u >= 0 && u <= 1 ? 1 : 0) : phi(u))
const baseCdf = (t: Transform, u: number): number => (t.base === 'uniform' ? Math.min(1, Math.max(0, u)) : Phi(u))
function targetDensity(t: Transform, y: number): number {
  const u = t.gInverse(y)
  if (!Number.isFinite(u)) return 0
  const d = baseDensity(t, u) / t.gDerivative(u)
  return Number.isFinite(d) ? d : 0
}
const targetCdf = (t: Transform, y: number): number => baseCdf(t, t.gInverse(y))

/** The Kolmogorov-Smirnov distance between the samples' empirical distribution and the exact one. */
function ksDistance(t: Transform, y: readonly number[]): number {
  const sorted = [...y].sort((a, b) => a - b)
  const n = sorted.length
  let d = 0
  sorted.forEach((v, i) => {
    const f = targetCdf(t, v)
    d = Math.max(d, Math.abs(f - i / n), Math.abs((i + 1) / n - f))
  })
  return d
}

// A histogram as a step curve: density per bin over `range`, scaled by `scale`, as doubled edges.
function histogram(values: readonly number[], range: Pair, bins: number): { edges: number[]; density: number[] } {
  const [lo, hi] = range
  const width = (hi - lo) / bins
  const counts = new Array<number>(bins).fill(0)
  let inside = 0
  for (const v of values) {
    const b = Math.floor((v - lo) / width)
    if (b >= 0 && b < bins) {
      counts[b] = (counts[b] ?? 0) + 1
      inside++
    }
  }
  const total = Math.max(1, values.length)
  const density = counts.map((c) => c / (total * width))
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + i * width)
  return { edges, density: inside === 0 ? density.map(() => 0) : density }
}

function stepCurve(edges: readonly number[], heights: readonly number[]): { along: number[]; up: number[] } {
  const along: number[] = []
  const up: number[] = []
  heights.forEach((h, i) => {
    along.push(edges[i] ?? 0, edges[i + 1] ?? 0)
    up.push(h, h)
  })
  return { along, up }
}

function picture(t: Transform, state: TransformSamplingState, batch: number): CurveSet {
  const [x0, x1] = t.baseRange
  const [y0, y1] = t.range
  const strip = 0.2
  const grid = (lo: number, hi: number, n: number): number[] =>
    Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
  // The map, blank where it leaves the window.
  const us = grid(x0 + 1e-6, x1 - 1e-6, 241)
  const map = us.map((u) => {
    const y = t.g(u)
    return y >= y0 - 0.05 * (y1 - y0) && y <= y1 + 0.05 * (y1 - y0) ? y : Number.NaN
  })
  // The base density and histogram along the bottom, in a strip of the target axis.
  const pBase = us.map((u) => baseDensity(t, u))
  const baseScale = (strip * (y1 - y0)) / Math.max(...pBase, 1e-9)
  const baseHist = histogram(state.u, t.baseRange, 30)
  const baseSteps = stepCurve(baseHist.edges, baseHist.density)
  // The target density and histogram along the side, in a strip of the base axis.
  const ys = grid(y0, y1, 241)
  const pTarget = ys.map((y) => targetDensity(t, y))
  const targetScale = (strip * (x1 - x0)) / Math.max(...pTarget, 1e-9)
  const targetHist = histogram(state.y, t.range, 30)
  const targetSteps = stepCurve(targetHist.edges, targetHist.density)
  // The latest batch's journeys: up from the base axis to the map, across to the target axis; gaps between them.
  const latest = state.u.slice(-batch)
  const px: number[] = []
  const py: number[] = []
  for (const u of latest) {
    const y = t.g(u)
    if (y < y0 || y > y1) continue
    px.push(u, u, x0, x0)
    py.push(y0, y, y, Number.NaN)
  }
  return curveSet({
    xLabel: t.baseLabel,
    yLabel: 'y = g(u)',
    series: [
      { name: 'the map g', x: us, y: map },
      { name: 'base density', x: us, y: pBase.map((p) => y0 + baseScale * p) },
      { name: 'base samples', x: baseSteps.along, y: baseSteps.up.map((h) => y0 + baseScale * h) },
      { name: 'target density', x: pTarget.map((p) => x0 + targetScale * p), y: ys },
      { name: 'pushed samples', x: targetSteps.up.map((h) => x0 + targetScale * h), y: targetSteps.along },
      { name: 'latest journeys', x: px, y: py },
    ],
  })
}

/**
 * The paths of {@link transformSampling}: the base draws and their images of each step, the samples so far, their mean and Kolmogorov-Smirnov distance, and the derived `transform/picture`.
 *
 * @example
 * ```ts
 * import type { TransformSamplingPath } from 'mdml-engine/methods/probability'
 * const path: TransformSamplingPath = 'samples/ks'
 * path // => 'samples/ks'
 * ```
 */
export type TransformSamplingPath =
  | 'samples/u'
  | 'samples/y'
  | 'samples/count'
  | 'samples/mean'
  | 'samples/ks'
  | 'transform/picture'

/**
 * Base samples pushed through a monotone map, a batch per step: sampling by transformation.
 *
 * Configuration: `target` (exponential, cauchy, logistic, lognormal, bimodal) and `batch` (samples per step). Paths:
 * `samples/u` and `samples/y` (the batch's base draws and their images, full), `samples/count`, `samples/mean` (of
 * the pushed samples so far), `samples/ks` (the Kolmogorov-Smirnov distance to the exact law), and the probe
 * `transform/picture` (the map, the densities and histograms on both axes, the latest journeys).
 */
export const transformSampling: Method<TransformSamplingState, Config, TransformSamplingPath> = defineMethod<
  TransformSamplingState,
  Config,
  TransformSamplingPath
>({
  name: 'probability/transform-sampling',
  config: dict({
    target: choice(['exponential', 'cauchy', 'logistic', 'lognormal', 'bimodal'] as const, { default: 'exponential' }),
    batch: int(1, 50, { default: 5, label: 'samples per step' }),
  }),
  make: (c, _key: Key) => {
    const t = transformOf(c['target'] as TransformTarget)
    const batch = c['batch'] as number
    return {
      name: 'transform-sampling',
      config: { target: c['target'] as string, batch },
      init: () => ({ u: [], y: [] }),
      step: (state, ctx) => {
        const draws = Array.from((t.base === 'uniform' ? uniform(ctx.key, [batch]) : normal(ctx.key, [batch])).data)
        const images = draws.map(t.g)
        const u = [...state.u, ...draws]
        const y = [...state.y, ...images]
        ctx.log('samples/u', draws)
        ctx.log('samples/y', images)
        ctx.log('samples/count', y.length)
        ctx.log('samples/mean', y.reduce((a, b) => a + b, 0) / y.length)
        ctx.log('samples/ks', ksDistance(t, y))
        return { u, y }
      },
      serialise: (s) => ({ u: [...s.u], y: [...s.y] }),
      deserialise: (j) => j as unknown as TransformSamplingState,
    }
  },
  contract: (c): Contract<TransformSamplingState, TransformSamplingPath> => {
    const t = transformOf(c['target'] as TransformTarget)
    const batch = c['batch'] as number
    return {
      'samples/u': tensor({ role: 'output', shape: [batch], description: 'the base draws of this step' }),
      'samples/y': tensor({ role: 'output', shape: [batch], description: 'their images under the map' }),
      'samples/count': scalar({ description: 'the pushed samples so far' }),
      'samples/mean': scalar({
        description: `the mean of the pushed samples so far${t.mean === undefined ? ' (the target has none)' : ` (exact ${t.mean.toFixed(3)})`}`,
      }),
      'samples/ks': scalar({
        description: 'the Kolmogorov-Smirnov distance between the pushed samples and the exact law',
        direction: 'lower',
      }),
      'transform/picture': derived<TransformSamplingState>({
        description: 'the map, the base and target densities with the samples’ histograms, and the latest journeys',
        compute: (s) => picture(t, s, batch),
      }),
    }
  },
  defaults: {
    'samples/u': Log({ detail: 'full' }),
    'samples/y': Log({ detail: 'full' }),
    'samples/count': Log(),
    'samples/mean': Log(),
    'samples/ks': Log(),
  },
})
