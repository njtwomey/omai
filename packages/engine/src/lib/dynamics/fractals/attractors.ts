/**
 * Strange attractors of maps, drawn as the cloud of points an orbit leaves: the cloud accumulates batch by batch as a
 * traced algorithm over any planar map (Henon, de Jong, Clifford, Ikeda from `dynamics/maps`), to be coloured by the
 * density of points per pixel (`stats.histogram2d`); and the lobe-switching count that reads a flow's two wings off a
 * coordinate (Lorenz's x). The flows themselves (Lorenz, Rossler, Thomas) are `named-systems` integrated by the ODE
 * solvers.
 */

import { OmaiError } from '../../base/index.ts'
import { tensor, type Value } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../../trace/index.ts'
import { positiveInt } from '../automata/_core.ts'
import type { IteratedMap } from '../maps.ts'
import { numbersOf } from '../ode/_vec.ts'

/** Options of {@link mapCloud}. */
export interface MapCloudOptions {
  /** The start, a state of the map (default `[0.1, 0.1]`). */
  readonly start?: readonly number[]
  /** Iterates added per step (default 100). */
  readonly batch?: number
  /** Iterates thrown away first, so the cloud is on the attractor (default 100). */
  readonly transient?: number
  /** The most points kept (default 200000). */
  readonly keep?: number
}

/** The state of {@link mapCloud}. */
export interface MapCloudState {
  /** The current iterate. */
  readonly point: Float64Array
  /** The kept iterates so far, flat, `dimension` numbers each. */
  readonly cloud: Float64Array
  readonly generated: number
  /** Whether the orbit left the finite numbers (the cloud then stops growing). */
  readonly diverged: boolean
}

/** The paths of {@link mapCloud}. */
export type MapCloudPath = 'points/cloud' | 'points/count'

/** The log contract of {@link mapCloud}. */
export const mapCloudContract: Contract<MapCloudState, MapCloudPath> = defineContract({
  'points/cloud': tensorSource({ role: 'state', shape: ['n', 'dimension'], description: 'the iterates so far' }),
  'points/count': scalar({ description: 'iterates generated' }),
})

/**
 * The orbit of a map as an accumulating cloud, as a traced algorithm: each step iterates the map `batch` times and
 * keeps the iterates. Deterministic: the key is not used. Stops with status `diverged` if the orbit leaves the finite
 * numbers. The state serialises.
 *
 * @param map - A map of states of length `start.length`.
 * @param options - The start, batch, transient and the most points kept.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a non-positive batch or an empty start.
 * @concept strange-attractors
 * @example
 * ```ts
 * import { henonMap } from 'omai-engine/dynamics'
 * import { fractals } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(fractals.mapCloud(henonMap()), { seed: 0, maxSteps: 3 }).state.cloud.length // => 600
 * ```
 */
export function mapCloud(map: IteratedMap, options: MapCloudOptions = {}): Algorithm<MapCloudState, MapCloudPath> {
  const start = options.start ?? [0.1, 0.1]
  const batch = options.batch ?? 100
  const transient = options.transient ?? 100
  const keep = options.keep ?? 200_000
  positiveInt('mapCloud', 'batch', batch)
  positiveInt('mapCloud', 'keep', keep)
  if (start.length === 0) throw new OmaiError('value', 'mapCloud: the start must have at least one component')
  const d = start.length
  const next = (x: Float64Array): Float64Array => Float64Array.from(numbersOf(map(tensor(Array.from(x)) as Value)))
  return {
    name: 'dynamics/fractals/map-cloud',
    config: { start: [...start], batch, transient, keep },
    schema: contractSchema(mapCloudContract),
    init: () => {
      let x: Float64Array = Float64Array.from(start)
      for (let i = 0; i < transient; i++) x = next(x)
      return { point: x, cloud: new Float64Array(0), generated: 0, diverged: !x.every(Number.isFinite) }
    },
    step: (s) => {
      const room = Math.max(0, Math.min(batch, keep - s.cloud.length / d))
      const cloud = new Float64Array(s.cloud.length + d * room)
      cloud.set(s.cloud)
      let x: Float64Array = s.point
      let diverged = s.diverged
      for (let i = 0; i < batch && !diverged; i++) {
        x = next(x)
        diverged = !x.every(Number.isFinite)
        if (!diverged && i < room) cloud.set(x, s.cloud.length + d * i)
      }
      return { point: x, cloud, generated: s.generated + batch, diverged }
    },
    done: (s) => (s.diverged ? 'diverged' : undefined),
    observe: (s) => ({
      metrics: { 'points/count': s.generated },
      tensors: { 'points/cloud': { role: 'state', value: s.cloud, shape: [s.cloud.length / d, d] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MapCloudState,
  }
}

/**
 * How many times a series changes side of a threshold: the number of switches between the two wings of the Lorenz
 * butterfly when applied to its x coordinate.
 *
 * @param series - The coordinate along a trajectory.
 * @param threshold - The dividing value (default 0).
 * @returns The number of changes of side (values equal to the threshold count as the upper side).
 * @concept strange-attractors
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * fractals.lobeSwitches([1, 2, -1, -3, 4, 5, -2]) // => 3
 * ```
 */
export function lobeSwitches(series: ArrayLike<number>, threshold = 0): number {
  let switches = 0
  for (let i = 1; i < series.length; i++)
    if ((series[i] as number) >= threshold !== (series[i - 1] as number) >= threshold) switches++
  return switches
}
