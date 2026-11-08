/**
 * Where a popup goes when it is drawn inside a container instead of on the page: the container's own coordinates,
 * corrected for the scale the container is drawn at (a CSS transform on it or above it), kept inside the container and
 * flipped above the anchor when there is no room below.
 */

/** A rectangle as the browser measures it: edges and size in screen pixels. */
interface Measured {
  readonly left: number
  readonly top: number
  readonly right: number
  readonly bottom: number
  readonly width: number
  readonly height: number
}

/**
 * The left and top of a popup inside a positioned container.
 *
 * @param o - The anchor's and the container's measured boxes, the container's own (unscaled) size, the popup's measured
 *   size and which edge of the anchor it lines up with.
 * @returns The popup's `left` and `top` in the container's coordinates.
 */
export function placeInContainer(o: {
  readonly anchor: Measured
  readonly container: Measured
  readonly size: { readonly width: number; readonly height: number }
  readonly menu: { readonly width: number; readonly height: number }
  readonly align: 'start' | 'end'
}): { readonly left: number; readonly top: number } {
  const scale = o.container.width / Math.max(1, o.size.width) || 1
  const width = o.menu.width / scale
  const height = o.menu.height / scale
  const below = (o.anchor.bottom - o.container.top) / scale + 4
  const above = (o.anchor.top - o.container.top) / scale - height - 4
  const top = below + height > o.size.height - 8 && above > 8 ? above : below
  const left =
    o.align === 'end' ? (o.anchor.right - o.container.left) / scale - width : (o.anchor.left - o.container.left) / scale
  return { left: Math.max(8, Math.min(left, o.size.width - width - 8)), top }
}
