/**
 * A classifier's decision surface as one quantity: the probability of each class, shaded in the class colours (blue for
 * class 0, orange for class 1, as the points are), with the decision boundary the 0.5 contour of that same probability.
 * The axes are the extent of the grid, which the shading covers exactly, so nothing is drawn outside the plot.
 */

import type { DecisionSurface } from 'omai-engine/exposition'
import { cartesian, contours, heatmap, points } from 'omai-render/charts'
import { type Mode, mix, paperOf, tab } from './colours.ts'

/** The two class colours (tab10 blue and orange), used by the surface and by the points. */
export const CLASS_COLOURS = [tab(0), tab(1)] as const
/** Their pale tints, for the shading, so the points stay visible on it. */
const tints = (mode: Mode): readonly [string, string] => [
  mix(tab(0), paperOf(mode), 0.45),
  mix(tab(1), paperOf(mode), 0.45),
]

/** The chart of a binary classifier's probability surface with its training rows (from a `decisionSurface` record). */
export function probabilitySurface(
  surface: DecisionSurface,
  options: { readonly legend?: boolean; readonly mode?: Mode } = {},
): unknown {
  const mode = options.mode ?? 'light'
  const TINTS = tints(mode)
  const probabilities = surface.probabilities
  if (probabilities === undefined || probabilities.length !== 2)
    throw new Error('probabilitySurface: needs two class probabilities')
  const x: [number, number] = [surface.x[0] ?? 0, surface.x[surface.x.length - 1] ?? 1]
  const y: [number, number] = [surface.y[0] ?? 0, surface.y[surface.y.length - 1] ?? 1]
  const p1 = probabilities[1] ?? []
  const layers = [
    // The probability of class 1: blue (class 0) at 0, white at 0.5, orange (class 1) at 1.
    heatmap(p1, {
      x,
      y,
      ramp: [TINTS[0], paperOf(mode), TINTS[1]],
      mode,
      range: [0, 1],
      centre: 0.5,
      colourBar: false,
    }),
    // The decision boundary is the 0.5 contour of that same probability.
    contours(p1, { x, y, levels: [0.5], labels: false, colour: 'fg', width: 1.5, mode }),
  ]
  const rows = surface.rows
  if (rows !== undefined) {
    const classes = rows.classes ?? ['0', '1']
    layers.push(
      points(
        [
          {
            x: rows.points.map((p) => p[0] ?? 0),
            y: rows.points.map((p) => p[1] ?? 0),
            groups: rows.labels ?? [],
            groupNames: Object.fromEntries(classes.map((c, i) => [String(i), `class ${c}`])),
            size: 6,
          },
        ],
        { mode },
      ),
    )
  }
  return cartesian(layers, {
    mode,
    x: { label: surface.features[0], limits: [x[0], x[1]] },
    y: { label: surface.features[1], limits: [y[0], y[1]] },
    legend: options.legend ?? false,
  })
}
