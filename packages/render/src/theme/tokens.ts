/**
 * Design tokens of the light and dark themes (foreground, background, surface, muted, grid, accent, semantic colours
 * and `tab0` to `tab9`), as values, as `--wb-*` CSS custom properties, and as the resolution of a scene's paint to a
 * concrete colour.
 */

import { type Paint, palettes, parsePaint, type ThemeToken } from 'omai-engine/scene'

/** A colour mode: the light or the dark theme. */
export type ColourMode = 'light' | 'dark'

/**
 * The design tokens of one theme. The scene tokens (`fg`, `bg`, `surface`, `muted`, `grid`, `accent`, `tab0` to
 * `tab9`) are exactly omai-engine's `palettes`, so a scene rasterised in Node and the same scene drawn as SVG or in a
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
  fgSecondary: '#525252',
  axis: '#d4d4d4',
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
  fgSecondary: '#d4d4d4',
  axis: '#404040',
  tooltip: '#171717',
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
 * @index function Light and dark design tokens: `tokens('dark').bg`.
 * @example
 * ```ts
 * import { tokens } from 'omai-render/theme'
 * tokens('light').tab0 // => '#1f77b4'
 * tokens('dark').bg // => '#0a0a0a'
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
 * import { cssVariables } from 'omai-render/theme'
 * cssVariables('light')['--wb-accent'] // => '#171717'
 * ```
 */
export function cssVariables(mode: ColourMode = 'light', prefix = 'wb'): Record<string, string> {
  const t = tokens(mode)
  const out: Record<string, string> = {}
  for (const name of CSS_NAMES) out[`--${prefix}-${kebab(name)}`] = String(t[name])
  return out
}

/** A partial set of design tokens: the ones a host restyles (`{ accent: '#7c3aed' }`). The mode cannot be changed. */
export type TokenPatch = Readonly<Partial<Omit<Tokens, 'mode'>>>

/**
 * A host's restyling of the tokens: a patch for every mode, a patch for each mode (`{ light: {...}, dark: {...} }`), or
 * a function of the mode.
 */
export type TokenOverride =
  | TokenPatch
  | { readonly light?: TokenPatch; readonly dark?: TokenPatch }
  | ((mode: ColourMode) => TokenPatch)

/**
 * The patch a {@link TokenOverride} makes in a mode.
 *
 * @param override - The override.
 * @param mode - The colour mode.
 * @returns The patch (empty for a per-mode override with nothing for this mode).
 * @example
 * ```ts
 * import { resolveTokenOverride } from 'omai-render/theme'
 * resolveTokenOverride({ light: { accent: '#00f' }, dark: { accent: '#0ff' } }, 'dark') // => { accent: '#0ff' }
 * resolveTokenOverride((mode) => ({ fg: mode === 'dark' ? '#fff' : '#000' }), 'light') // => { fg: '#000' }
 * ```
 */
export function resolveTokenOverride(override: TokenOverride, mode: ColourMode): TokenPatch {
  if (typeof override === 'function') return override(mode)
  if ('light' in override || 'dark' in override)
    return (override as { light?: TokenPatch; dark?: TokenPatch })[mode] ?? {}
  return override as TokenPatch
}

/**
 * The tokens of a theme with a host's overrides merged over them, later overrides over earlier ones.
 *
 * @param mode - The colour mode.
 * @param overrides - The overrides, outermost first; `undefined` entries are skipped.
 * @returns The tokens (the very object of `tokens(mode)` when nothing overrides it).
 * @example
 * ```ts
 * import { tokensWith } from 'omai-render/theme'
 * tokensWith('light', { accent: '#7c3aed' }).accent // => '#7c3aed'
 * tokensWith('light', { accent: '#7c3aed' }).fg // => '#0a0a0a'
 * ```
 */
export function tokensWith(mode: ColourMode, ...overrides: readonly (TokenOverride | undefined)[]): Tokens {
  const base = tokens(mode)
  let out: Tokens = base
  for (const o of overrides) {
    if (o === undefined) continue
    const patch = resolveTokenOverride(o, mode)
    const defined = Object.entries(patch).filter(([, v]) => v !== undefined)
    if (defined.length > 0) out = { ...out, ...Object.fromEntries(defined), mode }
  }
  return out
}

/**
 * A short key naming how a token object differs from the standard tokens of its mode: `undefined` for the standard
 * tokens, otherwise a string that is the same for the same differences (used to name a registered chart theme).
 *
 * @param t - The tokens, from {@link tokensWith}.
 * @returns The key, or `undefined`.
 * @example
 * ```ts
 * import { tokenVariant, tokens, tokensWith } from 'omai-render/theme'
 * tokenVariant(tokens('dark')) // => undefined
 * tokenVariant(tokensWith('dark', { accent: '#f0f' })) === tokenVariant(tokensWith('dark', { accent: '#f0f' })) // => true
 * ```
 */
export function tokenVariant(t: Tokens): string | undefined {
  const base = tokens(t.mode) as unknown as Readonly<Record<string, string>>
  const changed = Object.entries(t as unknown as Readonly<Record<string, string>>).filter(([k, v]) => base[k] !== v)
  if (changed.length === 0) return undefined
  let h = 5381
  for (const ch of JSON.stringify(changed)) h = ((h << 5) + h + ch.charCodeAt(0)) | 0
  return (h >>> 0).toString(36)
}

/** The names of the tokens a host can map to its own CSS variables: every token but the mode. */
export type TokenName = Exclude<keyof Tokens, 'mode'>

/** A map from a token name to the host's CSS variable (`{ fg: '--foreground' }`) or to any CSS value. */
export type CssVariableMap = Readonly<Partial<Record<TokenName, string>>>

/**
 * The `--wb-*` custom properties that read a host's own variables: each token name mapped to a host variable (`--name`)
 * becomes `--wb-<token>: var(--name)`, and a value that is not a bare variable name is used as written
 * (`'hsl(var(--foreground))'`).
 *
 * @param map - Token names to the host's variables.
 * @param prefix - The variable prefix (default `wb`).
 * @returns Property names mapped to values, for a style attribute.
 * @example
 * ```ts
 * import { mappedVariables } from 'omai-render/theme'
 * mappedVariables({ fg: '--foreground', fgSecondary: 'hsl(var(--muted-foreground))' })
 * // => { '--wb-fg': 'var(--foreground)', '--wb-fg-secondary': 'hsl(var(--muted-foreground))' }
 * ```
 */
export function mappedVariables(map: CssVariableMap, prefix = 'wb'): Record<string, string> {
  const out: Record<string, string> = {}
  for (const [name, host] of Object.entries(map)) {
    if (host === undefined || host === '') continue
    out[`--${prefix}-${kebab(name)}`] = /^--[\w-]+$/.test(host) ? `var(${host})` : host
  }
  return out
}

/**
 * The tokens as `--wb-*` custom properties, taken from a token object (the tokens of a mode with overrides) rather than
 * from a mode.
 *
 * @param t - The tokens.
 * @param only - Write only these names (default all).
 * @param prefix - The variable prefix (default `wb`).
 * @returns Property names mapped to values.
 * @example
 * ```ts
 * import { tokenVariables, tokensWith } from 'omai-render/theme'
 * tokenVariables(tokensWith('light', { accent: '#7c3aed' }), ['accent']) // => { '--wb-accent': '#7c3aed' }
 * ```
 */
export function tokenVariables(t: Tokens, only?: readonly TokenName[], prefix = 'wb'): Record<string, string> {
  const out: Record<string, string> = {}
  for (const name of only ?? CSS_NAMES) if (name !== 'mode') out[`--${prefix}-${kebab(name)}`] = String(t[name])
  return out
}

/**
 * The tokens of a theme as a CSS rule body (`--wb-fg: #0a0a0a; ...`), for a `<style>` block.
 *
 * @param mode - The colour mode.
 * @param selector - The rule's selector (default `:root`).
 * @returns The CSS text.
 * @index function The tokens as `--wb-*` CSS custom properties: `cssText('dark', '.dark')`.
 * @example
 * ```ts
 * import { cssText } from 'omai-render/theme'
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
 * import { resolvePaint } from 'omai-render/theme'
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
 * import { isThemeToken } from 'omai-render/theme'
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
 * import { paintRgba } from 'omai-render/theme'
 * paintRgba('fg', 'dark') // => [230, 237, 243, 1]
 * ```
 */
export function paintRgba(
  paint: Paint,
  mode: ColourMode = 'light',
): readonly [number, number, number, number] | undefined {
  return parsePaint(paint, mode === 'dark' ? palettes.dark : palettes.light)
}
