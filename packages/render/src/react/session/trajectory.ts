/**
 * The tape's tensors as drawn paths in the plane: the latest full tensor of a path at a step, and the layers of a
 * trajectory panel (each path up to the viewed step ending in its current point, optionally with a covariance ellipse),
 * which the field panel also lays over a vector field.
 */

import { covarianceEllipse } from 'mdml-engine/numerics/geometry'
import { toFlat } from 'mdml-engine/tensor'
import { type Run, type TensorEvent, tensorSeries } from 'mdml-engine/trace'
import { lines } from '../../charts/lines.ts'
import type { Layer } from '../../charts/plot.ts'
import { points } from '../../charts/points.ts'
import type { ColourMode } from '../../theme/tokens.ts'
import { leaf } from './compare.ts'

/** The tensor events of a path with full values (subsampled or summarised ones cannot be drawn element by element). */
export function fullTensors(run: Run, path: string): TensorEvent[] {
  return tensorSeries(run, path).filter((e) => e.values !== undefined && e.indices === undefined)
}

/** The latest tensor event of a path at or before step `t`, with full values. */
export function tensorAt(run: Run, path: string, t: number): TensorEvent | undefined {
  return fullTensors(run, path)
    .filter((e) => e.step <= t)
    .at(-1)
}

/** What a trajectory draws: the paths, the two elements of each that are the plane's axes, and how much of the past. */
export interface TrajectoryOptions {
  readonly paths: readonly string[]
  /** The two elements (default the first two). */
  readonly dims?: readonly [number, number]
  /** Keep only the last so many points. */
  readonly trail?: number
  /** Draw points instead of lines. */
  readonly points?: boolean
  /** A covariance tensor path drawn as the 1σ ellipse at the first path's current point. */
  readonly ellipse?: string
  readonly mode: ColourMode
}

/**
 * The layers of tensor paths drawn in the plane up to step `t`: each path as a line (or points) in its own colour,
 * a marker at each current point and, when asked, the ellipse of a covariance tensor at the first one.
 *
 * @param run - The tape.
 * @param t - The viewed step.
 * @param options - Paths, the two elements, the trail, points or lines, the ellipse and the colour mode.
 * @returns The layers (none when no path has a point yet).
 */
export function trajectoryLayers(run: Run, t: number, options: TrajectoryOptions): Layer[] {
  const { mode } = options
  const [i, j] = options.dims ?? [0, 1]
  const layers: Layer[] = []
  const heads: { x: number; y: number }[] = []
  options.paths.forEach((path, k) => {
    const events = fullTensors(run, path).filter((e) => e.step <= t)
    const trail = options.trail === undefined ? events : events.slice(-options.trail)
    const x = trail.map((e) => e.values?.[i] ?? Number.NaN)
    const y = trail.map((e) => e.values?.[j] ?? Number.NaN)
    const name = leaf(path)
    if (x.length === 0) return
    layers.push(
      options.points === true
        ? points([{ x, y, name, opacity: 0.6, size: 5 }], { mode, colourOffset: k })
        : lines([{ x, y, name, width: 1.5, opacity: 0.9 }], { mode, colourOffset: k }),
    )
    heads.push({ x: x.at(-1) ?? Number.NaN, y: y.at(-1) ?? Number.NaN })
  })
  if (heads.length > 0)
    layers.push(
      points(
        heads.map((h, k) => ({ x: [h.x], y: [h.y], name: `__head${k}`, size: 12, symbol: 'circle' })),
        { mode },
      ),
    )
  if (options.ellipse !== undefined && heads[0] !== undefined) {
    const cov = tensorAt(run, options.ellipse, t)
    const d = cov?.shape[0] ?? 0
    if (cov?.values !== undefined && d > Math.max(i, j)) {
      const sigma = [
        [cov.values[i * d + i] ?? 0, cov.values[i * d + j] ?? 0],
        [cov.values[j * d + i] ?? 0, cov.values[j * d + j] ?? 0],
      ]
      const flat = toFlat(covarianceEllipse([heads[0].x, heads[0].y], sigma, { nStd: 1, points: 72 }).points)
      layers.push(
        lines(
          [
            {
              x: Array.from({ length: 73 }, (_, k) => flat[(2 * k) % flat.length] ?? 0),
              y: Array.from({ length: 73 }, (_, k) => flat[(2 * k + 1) % flat.length] ?? 0),
              name: '__ellipse',
              colour: 'muted',
              width: 1,
              dash: 'dashed',
            },
          ],
          { mode },
        ),
      )
    }
  }
  return layers
}
