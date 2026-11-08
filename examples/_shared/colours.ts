/**
 * The colours of the figures, all taken from the render package's light theme (`tokens('light')`) so that every example
 * uses the same slots: `fg` for ink, `muted`, `grid`, `bg` for the page and `tab0` to `tab9` for categories. Chart layers
 * name a slot (`colour: 'tab0'`); the few places that need a concrete colour (SVG text, a colour ramp) read it from here.
 */
import { tokens } from 'mdml-render/theme'

const light = tokens('light')

/** The page, the text, the quiet text and the grid lines of the light theme. */
export const PAPER: string = light.bg
export const INK: string = light.fg
export const MUTED: string = light.muted
export const GRID: string = light.grid

/** The colour of a categorical slot (`tab0` to `tab9`). */
export const tab = (index: number): string => light[`tab${((index % 10) + 10) % 10}` as 'tab0']

/** A colour moved towards another by a share (0 keeps `from`, 1 gives `to`), as `#rrggbb`. */
export function mix(from: string, to: string, share: number): `#${string}` {
  const channels = (hex: string): number[] => [1, 3, 5].map((i) => Number.parseInt(hex.slice(i, i + 2), 16))
  const a = channels(from)
  const b = channels(to)
  return `#${a
    .map((v, i) =>
      Math.round(v + ((b[i] ?? v) - v) * share)
        .toString(16)
        .padStart(2, '0'),
    )
    .join('')}`
}
