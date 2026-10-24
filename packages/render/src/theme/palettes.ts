import { WhiteboxError } from 'whitebox-engine/base'
import { type Paint, parsePaint, type Rgba } from 'whitebox-engine/scene'
import type { ColourMode } from './tokens.ts'

/** The tab10 categorical palette (matplotlib's default cycle), the colours of the `tab0` to `tab9` tokens. */
export const tab10: readonly string[] = [
  '#1f77b4',
  '#ff7f0e',
  '#2ca02c',
  '#d62728',
  '#9467bd',
  '#8c564b',
  '#e377c2',
  '#7f7f7f',
  '#bcbd22',
  '#17becf',
]

/**
 * The categorical colour of a series or class index: tab10, cycling after ten.
 *
 * @param index - The index (0 is `tab0`).
 * @returns The colour.
 * @example
 * ```ts
 * import { categoricalColour } from 'whitebox-render/theme'
 * categoricalColour(1) // => '#ff7f0e'
 * categoricalColour(11) // => '#ff7f0e'
 * ```
 */
export function categoricalColour(index: number): string {
  const n = tab10.length
  return tab10[((Math.trunc(index) % n) + n) % n] ?? '#1f77b4'
}

/** Names of the sequential ramps. */
export type SequentialName = 'viridis' | 'magma' | 'blues' | 'greys'
/** Names of the diverging ramps. */
export type DivergingName = 'rdbu' | 'coolwarm'

const VIRIDIS = ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725']
const MAGMA = ['#000004', '#1c1044', '#4f127b', '#812581', '#b5367a', '#e55064', '#fb8761', '#fec287', '#fcfdbf']
const BLUES_LIGHT = ['#f6f8fa', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
const BLUES_DARK = ['#161b22', '#182232', '#172d4d', '#163a6d', '#174a94', '#1c5bbb', '#2a6fde', '#3f87f5', '#62a2ff']
const GREYS_LIGHT = ['#f6f8fa', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525']
const GREYS_DARK = ['#161b22', '#30363d', '#484f58', '#6e7681', '#8b949e', '#c9d1d9', '#f0f6fc']
const RDBU = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
const COOLWARM = ['#3b4cc0', '#7b9ff9', '#c0d4f5', '#f2cbb7', '#ee8468', '#b40426']

/**
 * The stops of a sequential ramp, from low to high. `viridis` and `magma` are the same in both modes; `blues` and
 * `greys` start from the theme's surface colour, so the lowest values are the quietest part of a figure in light and
 * dark mode alike.
 *
 * @param name - The ramp (default `viridis`).
 * @param mode - The colour mode.
 * @returns The stops as hex colours.
 * @example
 * ```ts
 * import { sequentialRamp } from 'whitebox-render/theme'
 * sequentialRamp('viridis')[0] // => '#440154'
 * sequentialRamp('blues', 'dark')[0] // => '#161b22'
 * ```
 */
export function sequentialRamp(name: SequentialName = 'viridis', mode: ColourMode = 'light'): readonly string[] {
  switch (name) {
    case 'viridis':
      return VIRIDIS
    case 'magma':
      return MAGMA
    case 'blues':
      return mode === 'dark' ? BLUES_DARK : BLUES_LIGHT
    case 'greys':
      return mode === 'dark' ? GREYS_DARK : GREYS_LIGHT
  }
}

/**
 * The stops of a diverging ramp, from strong blue (low) through a pale neutral (the midpoint) to strong red (high). The
 * midpoint stays pale in dark mode too, so zero never reads as a dark band.
 *
 * @param name - The ramp (default `rdbu`).
 * @returns The stops as hex colours (an odd number, the middle one the midpoint).
 * @example
 * ```ts
 * import { divergingRamp } from 'whitebox-render/theme'
 * divergingRamp('rdbu')[4] // => '#f7f7f7'
 * ```
 */
export function divergingRamp(name: DivergingName = 'rdbu'): readonly string[] {
  return name === 'coolwarm' ? COOLWARM : RDBU
}

/**
 * Parse a CSS colour (`#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`, `rgb(...)`, `rgba(...)`, or a theme token in light
 * mode) to channels.
 *
 * @param colour - The colour.
 * @returns `[r, g, b, a]`.
 * @throws WhiteboxError with code `value` for text that is not a colour, or `none`.
 * @example
 * ```ts
 * import { parseColour } from 'whitebox-render/theme'
 * parseColour('#ff8000') // => [255, 128, 0, 1]
 * ```
 */
export function parseColour(colour: string): Rgba {
  const c = parsePaint(colour as Paint)
  if (c === undefined) throw new WhiteboxError('value', `parseColour: '${colour}' has no colour`)
  return c
}

function hex2(v: number): string {
  return Math.round(Math.min(255, Math.max(0, v)))
    .toString(16)
    .padStart(2, '0')
}

/**
 * Write channels as `#rrggbb`, or `#rrggbbaa` when alpha is below one.
 *
 * @param rgba - The channels.
 * @returns The hex colour.
 * @example
 * ```ts
 * import { toHex } from 'whitebox-render/theme'
 * toHex([255, 128, 0, 1]) // => '#ff8000'
 * toHex([0, 0, 0, 0.5]) // => '#00000080'
 * ```
 */
export function toHex(rgba: Rgba): string {
  const [r, g, b, a] = rgba
  return `#${hex2(r)}${hex2(g)}${hex2(b)}${a < 1 ? hex2(a * 255) : ''}`
}

/**
 * Mix two colours: the fraction `t` of the way from `a` to `b`, channel by channel in sRGB.
 *
 * @param a - The first colour.
 * @param b - The second colour.
 * @param t - The fraction (0 gives `a`, 1 gives `b`).
 * @returns The mixed colour as hex.
 * @example
 * ```ts
 * import { mixColours } from 'whitebox-render/theme'
 * mixColours('#000000', '#ffffff', 0.5) // => '#808080'
 * ```
 */
export function mixColours(a: string, b: string, t: number): string {
  const p = parseColour(a)
  const q = parseColour(b)
  const u = Math.min(1, Math.max(0, t))
  return toHex([0, 1, 2, 3].map((i) => (p[i] ?? 0) + ((q[i] ?? 0) - (p[i] ?? 0)) * u) as unknown as Rgba)
}

/**
 * The colour at fraction `t` along evenly spaced stops, interpolated linearly in sRGB as ECharts' continuous visual map
 * does, so a mark coloured here matches a heatmap cell of the same value. `t` is clamped to [0, 1]; NaN gives the first
 * stop.
 *
 * @param stops - At least one colour.
 * @param t - The fraction.
 * @returns The colour as hex.
 * @throws WhiteboxError with code `value` when there are no stops.
 * @example
 * ```ts
 * import { interpolateColours } from 'whitebox-render/theme'
 * interpolateColours(['#000000', '#ffffff'], 0.25) // => '#404040'
 * ```
 */
export function interpolateColours(stops: readonly string[], t: number): string {
  return toHex(interpolateRgba(stops.map(parseColour), t))
}

/**
 * {@link interpolateColours} on parsed stops, returning channels (the fast path for images).
 *
 * @param stops - Parsed colours.
 * @param t - The fraction.
 * @returns The channels.
 * @throws WhiteboxError with code `value` when there are no stops.
 * @example
 * ```ts
 * import { interpolateRgba } from 'whitebox-render/theme'
 * interpolateRgba([[0, 0, 0, 1], [255, 255, 255, 1]], 1) // => [255, 255, 255, 1]
 * ```
 */
export function interpolateRgba(stops: readonly Rgba[], t: number): Rgba {
  const n = stops.length
  if (n === 0) throw new WhiteboxError('value', 'interpolateRgba: give at least one stop')
  const first = stops[0] as Rgba
  if (n === 1) return first
  const u = Math.min(Math.max(Number.isFinite(t) ? t : 0, 0), 1) * (n - 1)
  const i = Math.min(Math.floor(u), n - 2)
  const a = stops[i] as Rgba
  const b = stops[i + 1] as Rgba
  const f = u - i
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f]
}

/**
 * `n` evenly spaced colours along a ramp (for discrete levels, contour bands or categorical use of a ramp).
 *
 * @param stops - The ramp's stops.
 * @param n - How many colours.
 * @returns The colours, first and last at the ramp's ends.
 * @example
 * ```ts
 * import { sampleRamp } from 'whitebox-render/theme'
 * sampleRamp(['#000000', '#ffffff'], 3) // => ['#000000', '#808080', '#ffffff']
 * ```
 */
export function sampleRamp(stops: readonly string[], n: number): string[] {
  if (n <= 1) return n === 1 ? [interpolateColours(stops, 0.5)] : []
  return Array.from({ length: n }, (_, i) => interpolateColours(stops, i / (n - 1)))
}

/**
 * A colour scale: a function from values to colours along a ramp over a domain. With three domain values the middle one
 * maps to the ramp's midpoint (a diverging scale centred on zero, say).
 *
 * @param stops - The ramp.
 * @param domain - `[low, high]` or `[low, mid, high]`.
 * @returns The scale.
 * @example
 * ```ts
 * import { colourScale } from 'whitebox-render/theme'
 * const s = colourScale(['#000000', '#808080', '#ffffff'], [-1, 0, 4])
 * s(0) // => '#808080'
 * s(4) // => '#ffffff'
 * ```
 */
export function colourScale(
  stops: readonly string[],
  domain: readonly [number, number] | readonly [number, number, number],
): (value: number) => string {
  const parsed = stops.map(parseColour)
  return (value: number): string => toHex(interpolateRgba(parsed, scaleFraction(value, domain)))
}

/**
 * Where a value sits on a domain, as a fraction in [0, 1]; with a midpoint, each half maps linearly to half the range.
 *
 * @param value - The value.
 * @param domain - `[low, high]` or `[low, mid, high]`.
 * @returns The fraction, clamped.
 * @example
 * ```ts
 * import { scaleFraction } from 'whitebox-render/theme'
 * scaleFraction(5, [0, 10]) // => 0.5
 * scaleFraction(1, [-1, 0, 4]) // => 0.625
 * ```
 */
export function scaleFraction(
  value: number,
  domain: readonly [number, number] | readonly [number, number, number],
): number {
  const clamp01 = (x: number): number => (Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : 0)
  if (domain.length === 3) {
    const [lo, mid, hi] = domain
    if (value <= mid) return mid > lo ? clamp01(0.5 * ((value - lo) / (mid - lo))) : 0.5
    return hi > mid ? clamp01(0.5 + 0.5 * ((value - mid) / (hi - mid))) : 0.5
  }
  const [lo, hi] = domain
  return hi > lo ? clamp01((value - lo) / (hi - lo)) : 0.5
}

function linear(c: number): number {
  const s = c / 255
  return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
}

function gamma(l: number): number {
  const v = l <= 0.0031308 ? 12.92 * l : 1.055 * l ** (1 / 2.4) - 0.055
  return Math.min(255, Math.max(0, v * 255))
}

/**
 * The WCAG relative luminance of a colour (0 for black, 1 for white).
 *
 * @param colour - The colour.
 * @returns The luminance.
 * @example
 * ```ts
 * import { relativeLuminance } from 'whitebox-render/theme'
 * relativeLuminance('#ffffff') // => 1
 * ```
 */
export function relativeLuminance(colour: string): number {
  const [r, g, b] = parseColour(colour)
  return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)
}

/**
 * The WCAG contrast ratio of two colours, from 1 (identical) to 21 (black on white). Text wants at least 4.5.
 *
 * @param a - One colour.
 * @param b - The other.
 * @returns The ratio.
 * @example
 * ```ts
 * import { contrastRatio } from 'whitebox-render/theme'
 * contrastRatio('#000000', '#ffffff') // => 21
 * ```
 */
export function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(a)
  const lb = relativeLuminance(b)
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)
}

/** Kinds of colour-vision deficiency simulated by {@link simulateColourVision}. */
export type ColourVision = 'protanopia' | 'deuteranopia' | 'tritanopia'

// Machado, Oliveira and Fernandes (2009), "A physiologically-based model for simulation of color vision deficiency",
// IEEE TVCG 15(6): the severity-1 matrices, applied to linear RGB.
const MACHADO: Readonly<Record<ColourVision, readonly number[]>> = {
  protanopia: [0.152286, 1.052583, -0.204868, 0.114503, 0.786281, 0.099216, -0.003882, -0.048116, 1.051998],
  deuteranopia: [0.367322, 0.860646, -0.227968, 0.280085, 0.672501, 0.047413, -0.01182, 0.04294, 0.968881],
  tritanopia: [1.255528, -0.076749, -0.178779, -0.078411, 0.930809, 0.147602, 0.004733, 0.691367, 0.3039],
}

/**
 * How a colour looks with a colour-vision deficiency (Machado et al., 2009, full severity).
 *
 * @param colour - The colour.
 * @param kind - `protanopia`, `deuteranopia` or `tritanopia`.
 * @returns The simulated colour as hex.
 * @example
 * ```ts
 * import { simulateColourVision } from 'whitebox-render/theme'
 * simulateColourVision('#ffffff', 'deuteranopia') // => '#ffffff'
 * ```
 */
export function simulateColourVision(colour: string, kind: ColourVision): string {
  const [r, g, b, a] = parseColour(colour)
  const m = MACHADO[kind]
  const l = [linear(r), linear(g), linear(b)]
  const row = (i: number): number =>
    (m[3 * i] ?? 0) * (l[0] ?? 0) + (m[3 * i + 1] ?? 0) * (l[1] ?? 0) + (m[3 * i + 2] ?? 0) * (l[2] ?? 0)
  return toHex([gamma(row(0)), gamma(row(1)), gamma(row(2)), a])
}

function lab(colour: string): readonly [number, number, number] {
  const [r, g, b] = parseColour(colour)
  const [lr, lg, lb] = [linear(r), linear(g), linear(b)]
  const x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / 0.95047
  const y = 0.2126729 * lr + 0.7151522 * lg + 0.072175 * lb
  const z = (0.0193339 * lr + 0.119192 * lg + 0.9503041 * lb) / 1.08883
  const f = (t: number): number => (t > 216 / 24389 ? Math.cbrt(t) : ((24389 / 27) * t + 16) / 116)
  return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))]
}

/**
 * The perceptual distance between two colours (CIE76 delta E in CIELAB): about 2.3 is just noticeable, above 10 the
 * colours read as different.
 *
 * @param a - One colour.
 * @param b - The other.
 * @returns Delta E.
 * @example
 * ```ts
 * import { colourDistance } from 'whitebox-render/theme'
 * Math.round(colourDistance('#000000', '#ffffff')) // => 100
 * ```
 */
export function colourDistance(a: string, b: string): number {
  const p = lab(a)
  const q = lab(b)
  return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2])
}

/** The result of {@link colourBlindCheck}: the least pairwise distance under each kind of vision. */
export interface ColourBlindReport {
  readonly normal: number
  readonly protanopia: number
  readonly deuteranopia: number
  readonly tritanopia: number
  /** The worst pair under any simulation, as indices into the colours. */
  readonly worstPair: readonly [number, number]
  /** Whether every pair stays at least `threshold` apart under every simulation. */
  readonly safe: boolean
}

/**
 * Check that categorical colours stay distinguishable under colour-vision deficiencies: the least pairwise delta E of
 * the colours as seen normally and under each simulation.
 *
 * @param colours - At least two colours.
 * @param threshold - The least acceptable delta E (default 10).
 * @returns The report.
 * @example
 * ```ts
 * import { colourBlindCheck } from 'whitebox-render/theme'
 * colourBlindCheck(['#1f77b4', '#ff7f0e']).safe // => true
 * colourBlindCheck(['#d62728', '#2ca02c']).deuteranopia < colourBlindCheck(['#d62728', '#2ca02c']).normal // => true
 * ```
 */
export function colourBlindCheck(colours: readonly string[], threshold = 10): ColourBlindReport {
  const kinds = ['protanopia', 'deuteranopia', 'tritanopia'] as const
  const views: Record<'normal' | ColourVision, string[]> = {
    normal: [...colours],
    protanopia: colours.map((c) => simulateColourVision(c, 'protanopia')),
    deuteranopia: colours.map((c) => simulateColourVision(c, 'deuteranopia')),
    tritanopia: colours.map((c) => simulateColourVision(c, 'tritanopia')),
  }
  let worst = Number.POSITIVE_INFINITY
  let worstPair: [number, number] = [0, 0]
  const least = (cs: readonly string[]): number => {
    let m = Number.POSITIVE_INFINITY
    for (let i = 0; i < cs.length; i++)
      for (let j = i + 1; j < cs.length; j++) {
        const d = colourDistance(cs[i] ?? '#000', cs[j] ?? '#000')
        m = Math.min(m, d)
        if (d < worst) {
          worst = d
          worstPair = [i, j]
        }
      }
    return m
  }
  const normal = least(views.normal)
  const [protanopia, deuteranopia, tritanopia] = kinds.map((k) => least(views[k])) as [number, number, number]
  return {
    normal,
    protanopia,
    deuteranopia,
    tritanopia,
    worstPair,
    safe: Math.min(normal, protanopia, deuteranopia, tritanopia) >= threshold,
  }
}
