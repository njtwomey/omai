/**
 * The figure model: a `Figure` is a fixed area with a width, a box aspect, padding and type size, as a matplotlib
 * figure is, with defaults per medium (page, blog, slide), scaling to another width and the plotting area left inside
 * the padding.
 */

import { MdmlError } from 'mdml-engine/base'
import type { ColourMode } from '../theme/tokens.ts'

/** Where a figure will be shown; each medium has a default width, aspect and type size. */
export type Medium = 'page' | 'blog' | 'slide' | 'thumbnail'

/** Space around the plotting area, in pixels. */
export interface Padding {
  readonly top: number
  readonly right: number
  readonly bottom: number
  readonly left: number
}

/**
 * A figure: a fixed area with a width and a box aspect (width over height), as a matplotlib figure is. Content never
 * resizes it; it changes shape only when these properties say so.
 */
export interface Figure {
  readonly kind: 'figure'
  readonly medium: Medium
  /** Width in pixels. */
  readonly width: number
  /** Height in pixels (`width / aspect`). */
  readonly height: number
  /** The box aspect, width over height. */
  readonly aspect: number
  /** Room for axes, labels and the title, in pixels. */
  readonly padding: Padding
  /** Base font size in pixels. */
  readonly fontSize: number
  readonly mode: ColourMode
  readonly title?: string
}

/** Options of {@link figure}. */
export interface FigureOptions {
  readonly medium?: Medium
  readonly width?: number
  /** Width over height, or a ratio such as `'16:9'`. */
  readonly aspect?: number | `${number}:${number}`
  /** A height in pixels; sets the aspect from the width when given. */
  readonly height?: number
  readonly padding?: Partial<Padding>
  readonly fontSize?: number
  readonly mode?: ColourMode
  readonly title?: string
}

/** The defaults of each medium: width, aspect, font size and padding. */
export const MEDIA: Readonly<
  Record<
    Medium,
    { readonly width: number; readonly aspect: number; readonly fontSize: number; readonly padding: Padding }
  >
> = {
  page: { width: 640, aspect: 1.6, fontSize: 11, padding: { top: 28, right: 20, bottom: 44, left: 56 } },
  blog: { width: 720, aspect: 1.5, fontSize: 12, padding: { top: 32, right: 24, bottom: 48, left: 60 } },
  slide: { width: 960, aspect: 16 / 9, fontSize: 16, padding: { top: 40, right: 32, bottom: 60, left: 80 } },
  thumbnail: { width: 320, aspect: 1.6, fontSize: 9, padding: { top: 12, right: 8, bottom: 24, left: 32 } },
}

/**
 * Parse a box aspect: a number (width over height) or a ratio such as `'16:9'`.
 *
 * @param aspect - The aspect.
 * @returns Width over height.
 * @throws MdmlError with code `value` for a non-positive or malformed aspect.
 * @example
 * ```ts
 * import { parseAspect } from 'mdml-render/figure'
 * parseAspect('2:1') // => 2
 * parseAspect(1.5) // => 1.5
 * ```
 */
export function parseAspect(aspect: number | string): number {
  const value =
    typeof aspect === 'number'
      ? aspect
      : (() => {
          const [w, h] = aspect.split(':').map(Number)
          return w !== undefined && h !== undefined ? w / h : Number.NaN
        })()
  if (!(value > 0 && Number.isFinite(value))) throw new MdmlError('value', `figure: bad aspect '${aspect}'`)
  return value
}

/**
 * Make a figure: the medium's defaults, overridden by the options. The height follows from the width and aspect unless
 * given.
 *
 * @param options - Medium, width, aspect or height, padding, font size, colour mode and title.
 * @returns The figure.
 * @example
 * ```ts
 * import { figure } from 'mdml-render/figure'
 * figure().height // => 400
 * figure({ medium: 'slide' }).height // => 540
 * figure({ width: 400, aspect: '2:1' }).height // => 200
 * ```
 */
export function figure(options: FigureOptions = {}): Figure {
  const medium = options.medium ?? 'page'
  const d = MEDIA[medium]
  const width = options.width ?? d.width
  if (!(width > 0)) throw new MdmlError('value', 'figure: width must be positive')
  const aspect =
    options.height !== undefined
      ? parseAspect(width / options.height)
      : options.aspect !== undefined
        ? parseAspect(options.aspect)
        : d.aspect
  const padding = { ...d.padding, ...options.padding }
  return {
    kind: 'figure',
    medium,
    width,
    height: Math.round(width / aspect),
    aspect,
    padding,
    fontSize: options.fontSize ?? d.fontSize,
    mode: options.mode ?? 'light',
    ...(options.title === undefined ? {} : { title: options.title }),
  }
}

/**
 * A copy of a figure at another width, keeping its aspect and scaling padding and type with it (for a slide version of
 * a page figure, say).
 *
 * @param fig - The figure.
 * @param width - The new width.
 * @returns The scaled figure.
 * @example
 * ```ts
 * import { figure, scaleFigure } from 'mdml-render/figure'
 * scaleFigure(figure(), 1280).height // => 800
 * ```
 */
export function scaleFigure(fig: Figure, width: number): Figure {
  const s = width / fig.width
  const p = fig.padding
  return {
    ...fig,
    width,
    height: Math.round(width / fig.aspect),
    fontSize: Math.round(fig.fontSize * s * 10) / 10,
    padding: { top: p.top * s, right: p.right * s, bottom: p.bottom * s, left: p.left * s },
  }
}

/** An axis-aligned rectangle in pixels. */
export interface Rect {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

/**
 * The plotting area of a figure: its box less the padding.
 *
 * @param fig - The figure.
 * @returns The rectangle.
 * @example
 * ```ts
 * import { figure, plotArea } from 'mdml-render/figure'
 * plotArea(figure()).width // => 564
 * ```
 */
export function plotArea(fig: Figure): Rect {
  const p = fig.padding
  return {
    x: p.left,
    y: p.top,
    width: Math.max(0, fig.width - p.left - p.right),
    height: Math.max(0, fig.height - p.top - p.bottom),
  }
}
