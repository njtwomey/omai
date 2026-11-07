// Internal: whether the reader has asked the system for less motion, which playback that would start by itself respects.

/**
 * Whether the page's reader prefers reduced motion (`prefers-reduced-motion: reduce`); false where there is no
 * `matchMedia` (server rendering, tests without a browser).
 *
 * @returns True when animation that starts by itself should not.
 */
export function prefersReducedMotion(): boolean {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches
}
