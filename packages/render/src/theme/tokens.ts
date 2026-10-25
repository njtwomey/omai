/**
 * Design tokens of the light and dark themes (foreground, background, surface, muted, grid, accent, semantic colours
 * and `tab0` to `tab9`), as values, as `--wb-*` CSS custom properties, and as the resolution of a scene's paint to a
 * concrete colour.
 */

import { type Paint, palettes, parsePaint, type ThemeToken } from 'whitebox-engine/scene'

/** A colour mode: the light or the dark theme. */
export type ColourMode = 'light' | 'dark'

/**
 * The design tokens of one theme. The scene tokens (`fg`, `bg`, `surface`, `muted`, `grid`, `accent`, `tab0` to
 * `tab9`) are exactly whitebox-engine's `palettes`, so a scene rasterised in Node and the same scene drawn as SVG or in a
 * chart agree. The rest are for chrome and semantics: `fgSecondary` (axis names, legends), `axis` (axis lines),
 * `tooltip` (tooltip background), `success`, `warning`, `danger` and `info`, and the font stack.
 */
export interface Tokens extends Readonly<Record<ThemeToken, string>> {
  readonly mode: ColourMode
  readonly fgSecondary: string
  readonly axis: string
  readonly tooltip: string
  readonly success: string
  readonly warning: string
  readonly danger: string
  readonly info: string
  readonly font: string
  readonly monoFont: string
}

const FONT = "'Inter Variable', 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif"
const MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

const LIGHT: Tokens = {
  ...palettes.light,
  mode: 'light',
  fgSecondary: '#424a53',
  axis: '#afb8c1',
  tooltip: '#ffffff',
  success: '#1a7f37',
  warning: '#9a6700',
  danger: '#cf222e',
  info: '#0969da',
  font: FONT,
  monoFont: MONO,
}

const DARK: Tokens = {
  ...palettes.dark,
  mode: 'dark',
  fgSecondary: '#c9d1d9',
  axis: '#484f58',
  tooltip: '#1c2128',
  success: '#3fb950',
  warning: '#d29922',
  danger: '#f85149',
  info: '#58a6ff',
  font: FONT,
  monoFont: MONO,
}

/**
 * The design tokens of a theme.
 *
 * @param mode - `light` (default) or `dark`.
 * @returns The tokens.
 * @example
 * ```ts
 * import { tokens } from 'whitebox-render/theme'
 * tokens('light').tab0 // => '#1f77b4'
 * tokens('dark').bg // => '#0d1117'
 * ```
 */
export function tokens(mode: ColourMode = 'light'): Tokens {
  return mode === 'dark' ? DARK : LIGHT
}

const CSS_NAMES: readonly (keyof Tokens)[] = [
  'fg',
  'bg',
  'surface',
  'muted',
  'grid',
  'accent',
  'fgSecondary',
  'axis',
  'tooltip',
  'success',
  'warning',
  'danger',
  'info',
  'tab0',
  'tab1',
  'tab2',
  'tab3',
  'tab4',
  'tab5',
  'tab6',
  'tab7',
  'tab8',
  'tab9',
  'font',
  'monoFont',
]

function kebab(name: string): string {
  return name.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)
}

/**
 * The tokens of a theme as CSS custom properties (`--wb-fg`, `--wb-fg-secondary`, `--wb-tab0`, ...), for a style
 * attribute or a stylesheet rule.
 *
 * @param mode - The colour mode.
 * @param prefix - The variable prefix (default `wb`).
 * @returns Property names mapped to values.
 * @example
 * ```ts
 * import { cssVariables } from 'whitebox-render/theme'
 * cssVariables('light')['--wb-accent'] // => '#0969da'
 * ```
 */
export function cssVariables(mode: ColourMode = 'light', prefix = 'wb'): Record<string, string> {
  const t = tokens(mode)
  const out: Record<string, string> = {}
  for (const name of CSS_NAMES) out[`--${prefix}-${kebab(name)}`] = String(t[name])
  return out
}

/**
 * The tokens of a theme as a CSS rule body (`--wb-fg: #1f2328; ...`), for a `<style>` block.
 *
 * @param mode - The colour mode.
 * @param selector - The rule's selector (default `:root`).
 * @returns The CSS text.
 * @example
 * ```ts
 * import { cssText } from 'whitebox-render/theme'
 * cssText('dark', '.dark').startsWith('.dark {') // => true
 * ```
 */
export function cssText(mode: ColourMode = 'light', selector = ':root'): string {
  const body = Object.entries(cssVariables(mode))
    .map(([k, v]) => `  ${k}: ${v};`)
    .join('\n')
  return `${selector} {\n${body}\n}`
}

/**
 * Resolve a scene {@link Paint} to a CSS colour for a theme: tokens become their colour, literals pass through, and
 * `none` stays `none`.
 *
 * @param paint - The paint.
 * @param mode - The colour mode.
 * @returns A CSS colour or `none`.
 * @example
 * ```ts
 * import { resolvePaint } from 'whitebox-render/theme'
 * resolvePaint('tab1', 'light') // => '#ff7f0e'
 * resolvePaint('#123456', 'dark') // => '#123456'
 * ```
 */
export function resolvePaint(paint: Paint, mode: ColourMode = 'light'): string {
  if (paint === 'none') return 'none'
  const t = tokens(mode) as unknown as Readonly<Record<string, string>>
  return t[paint] ?? paint
}

/**
 * Whether a string is a theme token name.
 *
 * @param name - Any string.
 * @returns `true` for `fg`, `bg`, `surface`, `muted`, `grid`, `accent` and `tab0` to `tab9`.
 * @example
 * ```ts
 * import { isThemeToken } from 'whitebox-render/theme'
 * isThemeToken('tab9') // => true
 * isThemeToken('#fff') // => false
 * ```
 */
export function isThemeToken(name: string): name is ThemeToken {
  return Object.hasOwn(palettes.light, name)
}

/**
 * The colour of a paint as `[r, g, b, a]` for a theme (channels 0 to 255, alpha 0 to 1), or `undefined` for `none`.
 *
 * @param paint - The paint.
 * @param mode - The colour mode.
 * @returns The channels.
 * @example
 * ```ts
 * import { paintRgba } from 'whitebox-render/theme'
 * paintRgba('fg', 'dark') // => [230, 237, 243, 1]
 * ```
 */
export function paintRgba(
  paint: Paint,
  mode: ColourMode = 'light',
): readonly [number, number, number, number] | undefined {
  return parsePaint(paint, mode === 'dark' ? palettes.dark : palettes.light)
}
