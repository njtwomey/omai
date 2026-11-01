/**
 * The synthetic pictures the vision methods work on, from the datasets' generators: a geometric scene (overlapping
 * polygons at a few grey levels over a dark background, with pixel noise) as flat grey levels with its clean version,
 * its region labels and its visible corners; a single shape (a circle, square or triangle) with its mask and box; and
 * the grid helper that hands flat pixels to the library as an `[H, W]` image. Each knows the answer its method is after.
 */

import { generate } from '../../lib/data/index.ts'
import { geometricScenes, shapes } from '../../lib/datasets/index.ts'
import type { Key } from '../../lib/random/index.ts'
import { toFlat, type Value } from '../../lib/tensor/index.ts'

/** A scene: square, grey, row-major. */
export interface Scene {
  readonly size: number
  /** The pixels with noise, in [0, 1]. */
  readonly noisy: number[]
  /** The pixels without noise. */
  readonly clean: number[]
  /** The region of each pixel: 0 the background, k the k-th polygon drawn (later polygons lie on top). */
  readonly regions: number[]
  /** The polygons' corners that are not covered by a later polygon, as (x, y). */
  readonly corners: readonly (readonly [number, number])[]
}

/**
 * One geometric scene.
 *
 * @param key - The key the scene is drawn with.
 * @param options - The side in pixels, the pixel noise and the number of polygons (default 4).
 * @returns The scene.
 */
export function scene(
  key: Key,
  options: { readonly size: number; readonly noise: number; readonly polygons?: number },
): Scene {
  const { size, noise } = options
  const g = generate(geometricScenes, { knobs: { n: 1, size, polygons: options.polygons ?? 4, noise }, key })
  const corners = (g.truth.latents?.['corners'] as readonly (readonly (readonly number[])[])[] | undefined)?.[0] ?? []
  return {
    size,
    noisy: toFlat(g.data.images),
    clean: toFlat(g.truth.latents?.['clean'] as Value),
    regions: toFlat(g.data.masks as Value),
    corners: corners.map(([x, y]) => [x ?? 0, y ?? 0] as const),
  }
}

/** One shape on a dark background: its pixels, its mask (1 inside), its box and its name. */
export interface ShapeScene {
  readonly size: number
  readonly pixels: number[]
  readonly mask: number[]
  /** `[x0, y0, x1, y1]` in pixels. */
  readonly box: readonly [number, number, number, number]
  readonly name: string
}

/**
 * One shape (a circle, square or triangle at a random position, size and angle).
 *
 * @param key - The key the shape is drawn with.
 * @param options - The side in pixels and the pixel noise.
 * @returns The shape's picture.
 */
export function shape(key: Key, options: { readonly size: number; readonly noise: number }): ShapeScene {
  const { size, noise } = options
  const g = generate(shapes, { knobs: { n: 1, size, perImage: 1, noise }, key })
  const box = toFlat(g.data.boxes?.[0] as Value)
  const label = toFlat(g.data.labels as Value)[0] ?? 0
  return {
    size,
    pixels: toFlat(g.data.images),
    mask: toFlat(g.data.masks as Value).map((v) => (v > 0 ? 1 : 0)),
    box: [box[0] ?? 0, box[1] ?? 0, box[2] ?? 0, box[3] ?? 0],
    name: g.data.classes?.[label] ?? 'shape',
  }
}

/**
 * Flat pixels as rows.
 *
 * @param pixels - Row-major values.
 * @param width - The row length.
 * @returns The rows.
 */
export function grid(pixels: ArrayLike<number>, width: number): number[][] {
  const rows: number[][] = []
  for (let r = 0; r * width < pixels.length; r++)
    rows.push(Array.from({ length: width }, (_, i) => pixels[r * width + i] ?? 0))
  return rows
}
