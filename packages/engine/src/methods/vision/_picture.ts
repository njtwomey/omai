/**
 * The picture a vision method probes: an image as a decision surface's field (pixel columns along x, rows along y, row
 * 0 at the top) with optional points over it (a contour, tracked corners, superpixel centres), so the surface view
 * draws it like any other field with rows and the step cursor moves it.
 */

import { type DecisionSurface, decisionSurface } from '../../lib/exposition/index.ts'

/** What is drawn over the image. */
export interface PictureOptions {
  /** The field's name on the colour bar (default 'intensity'). */
  readonly name?: string
  /** Points (x, y) in pixel coordinates. */
  readonly points?: readonly (readonly [number, number])[]
  /** A class index per point, and the class names. */
  readonly labels?: readonly number[]
  readonly classes?: readonly string[]
}

/**
 * An image as a surface record, drawn as cells.
 *
 * @param pixels - The image, row-major, `height` times `width` values.
 * @param height - Rows.
 * @param width - Columns.
 * @param options - The field's name and the points over it.
 * @returns The record.
 */
export function picture(
  pixels: ArrayLike<number>,
  height: number,
  width: number,
  options: PictureOptions = {},
): DecisionSurface {
  const x = Array.from({ length: width }, (_, i) => i)
  const y = Array.from({ length: height }, (_, j) => j)
  // A chart's y grows upwards and an image's rows grow downwards: row r is drawn at y = height - 1 - r.
  const values = y.map((j) => {
    const r = height - 1 - j
    return x.map((i) => pixels[r * width + i] ?? 0)
  })
  const points = options.points
  return decisionSurface({
    features: ['x', 'y'],
    x,
    y,
    values,
    field: { name: options.name ?? 'intensity', smooth: false },
    ...(options.classes === undefined ? {} : { classes: options.classes }),
    ...(points === undefined
      ? {}
      : {
          rows: {
            points: points.map(([px, py]) => [px, height - 1 - py]),
            features: ['x', 'y'] as const,
            ...(options.labels === undefined ? {} : { labels: options.labels }),
            ...(options.classes === undefined ? {} : { classes: options.classes }),
          },
        }),
  })
}
