/**
 * Filmstrips of an environment's rendered frames: every few frames of an episode side by side in one scene.
 */

import { grid, type Scene } from 'mdml-engine/scene'

/**
 * A filmstrip: every `every`-th frame (and the last) side by side in rows of `columns`, each in a cell of the given size.
 *
 * @param frames - The scenes.
 * @param options - Columns (default 6), the cell size (default 120 by 90) and the stride (default 1).
 * @returns One scene holding the strip.
 */
export function filmstrip(
  frames: readonly Scene[],
  options: { readonly columns?: number; readonly cell?: readonly [number, number]; readonly every?: number } = {},
): Scene {
  const every = Math.max(1, options.every ?? 1)
  const picked = frames.filter((_, i) => i % every === 0 || i === frames.length - 1)
  return grid(picked, options.columns ?? 6, options.cell ?? [120, 90])
}
