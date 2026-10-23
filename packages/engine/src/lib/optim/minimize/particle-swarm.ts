import { WhiteboxError } from '../../base/index.ts'
import { metricFunction } from '../../numerics/geometry/index.ts'
import { Generator } from '../../random/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { clipInto, finiteBounds } from './_bounds.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import type { Bounds } from './derivative-free.ts'
import {
  type FlatObjective,
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

const EMPTY = new Float64Array(0)

// ---------------------------------------------------------------------------------------------------------------
// Particle swarm

/** Options of {@link particleSwarm}. */
export interface ParticleSwarmOptions extends VectorMethodOptions {
  /** Box bounds (required, finite): the swarm starts uniformly in the box and stays in it. */
  readonly bounds: Bounds
  /** Number of particles (default 40, as SPSO 2011). */
  readonly swarmSize?: number
  /** Inertia weight w (default 0.7298, Clerc's constriction), or [start, end] for a linear decrease over the run. */
  readonly inertia?: number | readonly [number, number]
  /** Cognitive and social coefficients c1, c2 (default 1.49618 each). */
  readonly cognitive?: number
  readonly social?: number
  /** `global` best (default) or `ring` neighbourhoods of three particles. */
  readonly topology?: 'global' | 'ring'
  /** The largest velocity per coordinate, as a fraction of the bound range (default 0.5). */
  readonly velocityClamp?: number
  /** Stop when the best value improves by less than ftol over `patience` iterations (defaults 1e-10 and 50). */
  readonly ftol?: number
  readonly patience?: number
}

/** State of a particle swarm run. */
export interface ParticleSwarmState extends VectorState {
  readonly positions: readonly Float64Array[]
  readonly velocities: readonly Float64Array[]
  readonly personal: readonly Float64Array[]
  readonly personalValues: Float64Array
  /** Best value recorded `patience` iterations ago, and the iteration counter. */
  readonly history: readonly number[]
  readonly iteration: number
}

/**
 * Particle swarm optimisation (Kennedy and Eberhart 1995; Shi and Eberhart's inertia weight; Clerc's constriction
 * coefficients) as a traced algorithm: velocities pulled towards each particle's best and its neighbourhood's best,
 * clamped, with positions kept in the box (absorbing walls). Each step moves the whole swarm once.
 *
 * @param f - The objective.
 * @param x0 - A point fixing the parameter structure (it becomes the first particle).
 * @param options - Bounds, swarm size, coefficients, topology, velocity clamp and stagnation test.
 * @returns The algorithm.
 * @concept particle-swarm
 * @example
 * ```ts
 * import { particleSwarmAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * particleSwarmAlgorithm((x) => sum(square(x as Value)), [0, 0], { bounds: [[-1, 1], [-1, 1]] }).name // => 'particleSwarm'
 * ```
 */
export function particleSwarmAlgorithm(
  f: Objective,
  x0: Parameters,
  options: ParticleSwarmOptions,
): Algorithm<ParticleSwarmState> {
  const obj: FlatObjective = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new WhiteboxError('value', 'particleSwarm: no parameters')
  const euclid = metricFunction('euclidean', n)
  const box = finiteBounds(options.bounds, n, 'particleSwarm')
  const size = options.swarmSize ?? 40
  if (size < 2) throw new WhiteboxError('value', 'particleSwarm: swarmSize must be at least 2')
  const inertia = options.inertia ?? 0.7298
  const c1 = options.cognitive ?? 1.49618
  const c2 = options.social ?? 1.49618
  const ring = options.topology === 'ring'
  const vmax = Float64Array.from(box.upper, (u, i) => (options.velocityClamp ?? 0.5) * (u - (box.lower[i] ?? 0)))
  const ftol = options.ftol ?? 1e-10
  const patience = options.patience ?? 50
  const maxIterations = options.maxIterations ?? 1000
  const weightAt = (t: number): number =>
    typeof inertia === 'number'
      ? inertia
      : inertia[0] + ((inertia[1] - inertia[0]) * t) / Math.max(1, maxIterations - 1)

  const bestOf = (values: Float64Array): number => {
    let b = 0
    for (let i = 1; i < values.length; i++) if ((values[i] ?? 0) < (values[b] ?? 0)) b = i
    return b
  }

  return {
    name: 'particleSwarm',
    config: configOf({
      size,
      cognitive: c1,
      social: c2,
      topology: ring ? 'ring' : 'global',
      ftol,
      patience,
      inertia: typeof inertia === 'number' ? inertia : [...inertia],
    }),
    schema: vectorSchema({
      metrics: {
        meanValue: { description: 'mean objective over the particles' },
        diversity: { description: 'mean distance of the particles from their centroid' },
      },
      tensors: { positions: { role: 'state', description: 'particle positions' } },
    }),
    init: (key) => {
      const rng = new Generator(key)
      const u = rng.uniform([size, n]).data
      const w = rng.uniform([size, n]).data
      const range = (i: number): number => (box.upper[i] ?? 0) - (box.lower[i] ?? 0)
      const positions = Array.from({ length: size }, (_, k) =>
        k === 0
          ? clipInto(obj.x0, box)
          : Float64Array.from({ length: n }, (_, i) => (box.lower[i] ?? 0) + (u[k * n + i] ?? 0) * range(i)),
      )
      const velocities = Array.from({ length: size }, (_, k) =>
        Float64Array.from({ length: n }, (_, i) => 0.1 * range(i) * (2 * (w[k * n + i] ?? 0) - 1)),
      )
      const values = Float64Array.from(positions, (p) => obj.value(p))
      const b = bestOf(values)
      return {
        x: positions[b] as Float64Array,
        value: values[b] ?? Number.NaN,
        grad: EMPTY,
        nfev: size,
        ngev: 0,
        nhev: 0,
        step: 0,
        positions,
        velocities,
        personal: positions,
        personalValues: values,
        history: [values[b] ?? Number.NaN],
        iteration: 0,
      }
    },
    step: (s, ctx) => {
      const rng = new Generator(ctx.key)
      const r1 = rng.uniform([size, n]).data
      const r2 = rng.uniform([size, n]).data
      const w = weightAt(s.iteration)
      const globalBest = bestOf(s.personalValues)
      const positions: Float64Array[] = []
      const velocities: Float64Array[] = []
      for (let k = 0; k < size; k++) {
        let leader = globalBest
        if (ring) {
          const left = (k + size - 1) % size
          const right = (k + 1) % size
          leader = k
          for (const j of [left, right]) if ((s.personalValues[j] ?? 0) < (s.personalValues[leader] ?? 0)) leader = j
        }
        const x = s.positions[k] as Float64Array
        const v = s.velocities[k] as Float64Array
        const p = s.personal[k] as Float64Array
        const g = s.personal[leader] as Float64Array
        const nv = new Float64Array(n)
        const nx = new Float64Array(n)
        for (let i = 0; i < n; i++) {
          let vi =
            w * (v[i] ?? 0) +
            c1 * (r1[k * n + i] ?? 0) * ((p[i] ?? 0) - (x[i] ?? 0)) +
            c2 * (r2[k * n + i] ?? 0) * ((g[i] ?? 0) - (x[i] ?? 0))
          vi = Math.max(-(vmax[i] ?? 0), Math.min(vmax[i] ?? 0, vi))
          let xi = (x[i] ?? 0) + vi
          if (xi < (box.lower[i] ?? 0) || xi > (box.upper[i] ?? 0)) {
            xi = Math.min(Math.max(xi, box.lower[i] ?? 0), box.upper[i] ?? 0)
            vi = 0
          }
          nv[i] = vi
          nx[i] = xi
        }
        positions.push(nx)
        velocities.push(nv)
      }
      const values = Float64Array.from(positions, (p) => obj.value(p))
      const personal = s.personal.map((p, k) =>
        (values[k] ?? 0) < (s.personalValues[k] ?? 0) ? (positions[k] as Float64Array) : p,
      )
      const personalValues = Float64Array.from(s.personalValues, (pv, k) =>
        Math.min(pv, values[k] ?? Number.POSITIVE_INFINITY),
      )
      const b = bestOf(personalValues)
      const value = personalValues[b] ?? Number.NaN
      const history = [...s.history, value].slice(-(patience + 1))
      const stagnant = history.length > patience && (history[0] ?? 0) - value < ftol * Math.max(1, Math.abs(value))
      return {
        x: personal[b] as Float64Array,
        value,
        grad: EMPTY,
        nfev: s.nfev + size,
        ngev: 0,
        nhev: 0,
        step: euclid(personal[b] as Float64Array, 0, s.x, 0),
        positions,
        velocities,
        personal,
        personalValues,
        history,
        iteration: s.iteration + 1,
        ...(stagnant
          ? {
              stop: {
                status: 'converged' as const,
                message: `best value improved by less than ftol in ${patience} iterations`,
              },
            }
          : {}),
      }
    },
    done: vectorDone,
    observe: (s) => {
      const centroid = new Float64Array(n)
      for (const p of s.positions) for (let i = 0; i < n; i++) centroid[i] = (centroid[i] ?? 0) + (p[i] ?? 0) / size
      let diversity = 0
      let mean = 0
      for (const [k, p] of s.positions.entries()) {
        diversity += euclid(p, 0, centroid, 0) / size
        mean += (s.personalValues[k] ?? 0) / size
      }
      const flat = new Float64Array(size * n)
      for (const [k, p] of s.positions.entries()) flat.set(p, k * n)
      return observeVector(s, {
        metrics: { meanValue: mean, diversity },
        tensors: { positions: { role: 'state', value: flat, shape: [size, n] } },
      })
    },
  }
}

/**
 * Find a global minimum by particle swarm optimisation. `maxIterations` counts swarm moves (default 1000).
 *
 * @param f - The objective.
 * @param x0 - A point fixing the parameter structure (the first particle).
 * @param options - Bounds (required), swarm size, coefficients, topology, velocity clamp, stagnation test, key or
 *   seed and run options.
 * @returns The best position found.
 * @concept particle-swarm
 * @example
 * ```ts
 * import { particleSwarm } from 'whitebox-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * const r = particleSwarm((x) => sum(square(sub(x as Value, [1, -2]))), [0, 0], { bounds: [[-5, 5], [-5, 5]], seed: 0 })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e4) / 1e4) // => [1, -2]
 * ```
 */
export function particleSwarm(f: Objective, x0: Parameters, options: ParticleSwarmOptions): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(particleSwarmAlgorithm(f, x0, options), obj, options, 1000)
}
