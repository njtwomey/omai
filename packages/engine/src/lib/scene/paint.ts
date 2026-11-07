/**
 * Colours for scenes: the light and dark palettes that give theme tokens (`fg`, `bg`, `accent`, `tab0` to `tab9` and
 * the rest) their values, and `parsePaint`, which resolves a token or CSS colour string to RGBA.
 */

import { clamp, MdmlError } from '../base/index.ts'
import type { Paint, ThemeToken } from './types.ts'

/**
 * A colour with alpha, `[r, g, b, a]`. From `parsePaint` red, green and blue run from 0 to 255 and alpha from 0 to 1;
 * the texture samplers of `graphics` return linear values from 0 to 1.
 */
export type Rgba = readonly [r: number, g: number, b: number, a: number]

const TAB10 = {
  tab0: '#1f77b4',
  tab1: '#ff7f0e',
  tab2: '#2ca02c',
  tab3: '#d62728',
  tab4: '#9467bd',
  tab5: '#8c564b',
  tab6: '#e377c2',
  tab7: '#7f7f7f',
  tab8: '#bcbd22',
  tab9: '#17becf',
} as const

/**
 * The colours the rasteriser gives theme tokens in light and dark mode (tab10 for `tab0` to `tab9`). mdml-render's
 * themes resolve the same tokens for SVG and charts.
 *
 * @returns The two palettes.
 * @example
 * ```ts
 * import { palettes } from 'mdml-engine/scene'
 * palettes.light.tab0 // => '#1f77b4'
 * ```
 */
export const palettes: {
  readonly light: Readonly<Record<ThemeToken, string>>
  readonly dark: Readonly<Record<ThemeToken, string>>
} = {
  light: {
    fg: '#1f2328',
    bg: '#ffffff',
    surface: '#f6f8fa',
    muted: '#6e7781',
    grid: '#d0d7de',
    accent: '#0969da',
    ...TAB10,
  },
  dark: {
    fg: '#e6edf3',
    bg: '#0d1117',
    surface: '#161b22',
    muted: '#8b949e',
    grid: '#30363d',
    accent: '#2f81f7',
    ...TAB10,
  },
}

function hex(digits: string): number {
  const v = Number.parseInt(digits, 16)
  if (!Number.isFinite(v)) throw new MdmlError('value', `parsePaint: bad hex digits '${digits}'`)
  return v
}

/**
 * Resolve a paint to a colour: theme tokens through a palette, then `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`,
 * `rgb(r, g, b)` and `rgba(r, g, b, a)`.
 *
 * @param paint - The paint.
 * @param palette - Token colours (default the light palette).
 * @returns The colour, or `undefined` for `none`.
 * @throws MdmlError with code `value` for text that is not a colour.
 * @example
 * ```ts
 * import { parsePaint } from 'mdml-engine/scene'
 * parsePaint('#f80') // => [255, 136, 0, 1]
 * parsePaint('rgba(10, 20, 30, 0.5)') // => [10, 20, 30, 0.5]
 * parsePaint('tab3') // => [214, 39, 40, 1]
 * ```
 */
export function parsePaint(
  paint: Paint,
  palette: Readonly<Record<ThemeToken, string>> = palettes.light,
): Rgba | undefined {
  if (paint === 'none') return undefined
  const token = (palette as Readonly<Record<string, string>>)[paint]
  if (token !== undefined) return parsePaint(token as Paint, palette)
  const s = paint.trim()
  if (s.startsWith('#')) {
    const d = s.slice(1)
    if (d.length === 3 || d.length === 4) {
      const [r, g, b, a = 'f'] = d
      return [hex(r ?? '') * 17, hex(g ?? '') * 17, hex(b ?? '') * 17, (hex(a) * 17) / 255]
    }
    if (d.length === 6 || d.length === 8) {
      const a = d.length === 8 ? hex(d.slice(6, 8)) / 255 : 1
      return [hex(d.slice(0, 2)), hex(d.slice(2, 4)), hex(d.slice(4, 6)), a]
    }
  }
  const m = /^rgba?\(([^)]*)\)$/.exec(s)
  if (m !== null) {
    const parts = (m[1] ?? '').split(',').map((x) => Number(x.trim()))
    if ((parts.length === 3 || parts.length === 4) && parts.every(Number.isFinite)) {
      const [r = 0, g = 0, b = 0, a = 1] = parts
      return [clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255), clamp(a, 0, 1)]
    }
  }
  throw new MdmlError('value', `parsePaint: '${paint}' is not a colour`)
}
