/**
 * Linked brushing and hover across figures: a small store of shared state (the hovered item and position, the selected
 * items, a brushed region) that figures read and update, with helpers for toggling a selection and finding the points
 * inside a brush.
 */

import type { Range } from './aspect.ts'

/** What linked figures share: the hovered item and data position, the selected items, and a brushed region. */
export interface LinkState {
  /** The hovered item (a point index or id), if any. */
  readonly hover?: string | number
  /** The hovered data position, if any. */
  readonly pointer?: { readonly x?: number; readonly y?: number }
  /** The selected items. */
  readonly selection: readonly (string | number)[]
  /** The brushed rectangle in data coordinates, if any. */
  readonly brush?: { readonly x?: Range; readonly y?: Range }
  /** The zoomed x range shared by linked axes, if any. */
  readonly zoom?: Range
}

/**
 * A shared state for linked brushing and hover across figures: get, set (with a function of the old state) and
 * subscribe. Each call makes an independent store; the library keeps no global link state.
 */
export interface LinkStore {
  get(): LinkState
  set(update: Partial<LinkState> | ((state: LinkState) => LinkState)): void
  subscribe(listener: (state: LinkState) => void): () => void
}

/**
 * Make a link store. Figures that read and write the same store hover, select and zoom together; React views bind to
 * it with `useSyncExternalStore`.
 *
 * @param initial - The initial state (default: nothing hovered or selected).
 * @returns The store.
 * @example
 * ```ts
 * import { linkStore } from 'whitebox-render/figure'
 * const link = linkStore()
 * link.set({ hover: 3 })
 * link.get().hover // => 3
 * ```
 */
export function linkStore(initial: Partial<LinkState> = {}): LinkStore {
  let state: LinkState = { selection: [], ...initial }
  const listeners = new Set<(state: LinkState) => void>()
  return {
    get: () => state,
    set: (update) => {
      const next = typeof update === 'function' ? update(state) : { ...state, ...update }
      if (next === state) return
      state = next
      for (const l of [...listeners]) l(state)
    },
    subscribe: (listener) => {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
  }
}

/**
 * A selection with an item added if absent or removed if present (a click on a point).
 *
 * @param selection - The current selection.
 * @param item - The item.
 * @returns The new selection.
 * @example
 * ```ts
 * import { toggleSelection } from 'whitebox-render/figure'
 * toggleSelection([1, 2], 2) // => [1]
 * toggleSelection([1], 3) // => [1, 3]
 * ```
 */
export function toggleSelection(selection: readonly (string | number)[], item: string | number): (string | number)[] {
  return selection.includes(item) ? selection.filter((s) => s !== item) : [...selection, item]
}

/**
 * The indices of points inside a brushed rectangle (an omitted side is unbounded).
 *
 * @param points - Points as `[x, y]`.
 * @param brush - The brushed x and y ranges.
 * @returns The indices of the points inside.
 * @example
 * ```ts
 * import { brushSelection } from 'whitebox-render/figure'
 * brushSelection([[0, 0], [1, 1], [2, 2]], { x: [0.5, 3] }) // => [1, 2]
 * ```
 */
export function brushSelection(
  points: readonly (readonly [number, number])[],
  brush: { readonly x?: Range; readonly y?: Range },
): number[] {
  const inside = (v: number, r: Range | undefined): boolean => r === undefined || (v >= r[0] && v <= r[1])
  const out: number[] = []
  points.forEach(([x, y], i) => {
    if (inside(x, brush.x) && inside(y, brush.y)) out.push(i)
  })
  return out
}
