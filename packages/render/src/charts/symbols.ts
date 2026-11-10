/**
 * Named marker symbols beyond ECharts' built-ins, as filled path symbols: `plane` (an aircraft seen from above, nose
 * up) and `cross` (a filled x). A point set's `symbol` takes a name from here or any ECharts symbol; with
 * `symbolRotate` a marker that has a direction (a plane's heading) is turned to it.
 */

/**
 * Marker paths by name, as ECharts `path://` symbols. Each is a filled shape (a path symbol is filled, so strokes alone
 * would not show) and is fitted to the marker size. `plane` points up (north) at rotation 0, so `symbolRotate` is a
 * compass heading: degrees clockwise from up.
 *
 * @index table Named marker symbols as ECharts path symbols: `SYMBOLS.plane` (nose up), `SYMBOLS.cross`.
 * @example
 * ```ts
 * import { SYMBOLS } from 'omai-render/charts'
 * SYMBOLS.plane.startsWith('path://') // => true
 * Object.keys(SYMBOLS) // => ['plane', 'cross']
 * ```
 */
export const SYMBOLS: Readonly<Record<'plane' | 'cross', string>> = {
  plane:
    'path://M12,1C12.8,1 13.4,2.5 13.4,4.5L13.4,9L22,14.5L22,16.5L13.4,13.8L13.4,19L16,21L16,22.5L12,21.5L8,22.5L8,21L10.6,19L10.6,13.8L2,16.5L2,14.5L10.6,9L10.6,4.5C10.6,2.5 11.2,1 12,1Z',
  cross: 'path://M-3,-2L-2,-3L0,-1L2,-3L3,-2L1,0L3,2L2,3L0,1L-2,3L-3,2L-1,0Z',
}

/**
 * The ECharts symbol for a name: a named symbol of {@link SYMBOLS}, else the name itself (an ECharts built-in such as
 * `circle`, a `path://` or an `image://`).
 *
 * @param name - A symbol name.
 * @returns What to give ECharts as `symbol`.
 * @example
 * ```ts
 * import { SYMBOLS, symbolOf } from 'omai-render/charts'
 * symbolOf('plane') // => SYMBOLS.plane
 * symbolOf('diamond') // => 'diamond'
 * ```
 */
export function symbolOf(name: string): string {
  return Object.hasOwn(SYMBOLS, name) ? SYMBOLS[name as keyof typeof SYMBOLS] : name
}
