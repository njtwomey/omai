/**
 * Whole-figure export: ECharts options drawn to SVG text by ECharts' server-side renderer, charts and HTML parts
 * (labels, typeset maths) composed into one SVG document, SVG rasterised to a canvas, a PNG or RGBA pixels in the
 * browser, and export sizes for the page and for 1920 by 1080 slides.
 */

import type { EChartsCoreOption } from 'echarts/core'
import { OmaiError } from 'omai-engine/base'
import type { Pixels } from 'omai-engine/scene'
import { echartsCore } from '../charts/echarts-core.ts'
import type { Figure } from '../figure/figure.ts'
import { themeName } from '../theme/echarts.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { escapeXml } from './svg.ts'

/**
 * An ECharts option drawn to SVG text by ECharts' server-side renderer: the same chart as on the page, in Node or the
 * browser, for export, slides and blog posts.
 *
 * @param option - The option.
 * @param size - Width and height in pixels, and the colour mode (theme).
 * @returns The SVG text.
 * @index function A chart to SVG text: `chartToSvg(option, { width, height })`.
 * @example
 * ```ts
 * import { lineChart } from 'omai-render/charts'
 * import { chartToSvg } from 'omai-render/svg'
 * chartToSvg(lineChart([{ y: [1, 2] }]), { width: 320, height: 200 }).startsWith('<svg') // => true
 * ```
 */
export function chartToSvg(
  option: EChartsCoreOption,
  size: { readonly width: number; readonly height: number; readonly mode?: ColourMode },
): string {
  const echarts = echartsCore()
  const chart = echarts.init(null, themeName(size.mode ?? 'light'), {
    renderer: 'svg',
    ssr: true,
    width: size.width,
    height: size.height,
  })
  try {
    chart.setOption({ ...option, animation: false })
    return chart.renderToSVGString()
  } finally {
    chart.dispose()
  }
}

/** One part of a composed figure: SVG text (a chart) or HTML (a label, typeset maths) placed in a box. */
export interface FigurePart {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
  readonly svg?: string
  readonly html?: string
}

/**
 * Compose parts into one SVG document of a given size: SVG parts are nested at their positions, HTML parts go into
 * foreign objects (centred in their boxes). The whole figure, not one panel, is what exports.
 *
 * @param width - The figure's width.
 * @param height - The figure's height.
 * @param parts - The parts.
 * @param options - Background colour.
 * @returns The SVG text.
 * @index function Charts and labels in one SVG document: `composeSvg(width, height, parts)`.
 * @example
 * ```ts
 * import { composeSvg } from 'omai-render/svg'
 * composeSvg(100, 50, [{ x: 0, y: 0, width: 50, height: 50, html: '<b>a</b>' }], { background: '#fff' }).includes('foreignObject') // => true
 * ```
 */
export function composeSvg(
  width: number,
  height: number,
  parts: readonly FigurePart[],
  options: { readonly background?: string } = {},
): string {
  const body = parts
    .map((p) => {
      if (p.svg !== undefined) {
        // Nest the part's own <svg> at its position, dropping any XML declaration.
        const inner = p.svg.replace(/^<\?xml[^>]*>\s*/, '')
        return `<svg x="${p.x}" y="${p.y}" width="${p.width}" height="${p.height}" overflow="visible">${inner}</svg>`
      }
      if (p.html !== undefined)
        return `<foreignObject x="${p.x}" y="${p.y}" width="${p.width}" height="${p.height}"><div xmlns="http://www.w3.org/1999/xhtml" style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;text-align:center">${p.html}</div></foreignObject>`
      return ''
    })
    .join('')
  const back =
    options.background === undefined ? '' : `<rect width="100%" height="100%" fill="${escapeXml(options.background)}"/>`
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${back}${body}</svg>`
}

/**
 * SVG text as a `data:image/svg+xml` URL.
 *
 * @param svg - The SVG text.
 * @returns The data URL.
 * @example
 * ```ts
 * import { svgDataUrl } from 'omai-render/svg'
 * svgDataUrl('<svg/>') // => 'data:image/svg+xml;charset=utf-8,%3Csvg%2F%3E'
 * ```
 */
export function svgDataUrl(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

/**
 * Draw SVG text onto a new canvas in the browser (through an image), at a pixel ratio, on a background.
 *
 * @param svg - The SVG text.
 * @param width - Width in CSS pixels.
 * @param height - Height in CSS pixels.
 * @param options - Pixel ratio (default 2) and background colour.
 * @returns A promise of the canvas.
 * @throws OmaiError with code `not-implemented` outside a browser, and `value` when the SVG does not load.
 */
export function svgToCanvas(
  svg: string,
  width: number,
  height: number,
  options: { readonly scale?: number; readonly background?: string } = {},
): Promise<HTMLCanvasElement> {
  if (typeof document === 'undefined' || typeof Image === 'undefined')
    return Promise.reject(new OmaiError('not-implemented', 'svgToCanvas: needs a browser (canvas and Image)'))
  const scale = options.scale ?? 2
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = Math.max(1, Math.round(width * scale))
      canvas.height = Math.max(1, Math.round(height * scale))
      const ctx = canvas.getContext('2d')
      if (ctx === null) {
        reject(new OmaiError('not-implemented', 'svgToCanvas: no 2-D canvas'))
        return
      }
      if (options.background !== undefined) {
        ctx.fillStyle = options.background
        ctx.fillRect(0, 0, canvas.width, canvas.height)
      }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      resolve(canvas)
    }
    img.onerror = () => reject(new OmaiError('value', 'svgToCanvas: the SVG did not load as an image'))
    img.src = svgDataUrl(svg)
  })
}

/**
 * Rasterise SVG text to a PNG data URL in the browser (through an image and a canvas), at a pixel ratio for sharp
 * slides.
 *
 * @param svg - The SVG text.
 * @param width - Width in CSS pixels.
 * @param height - Height in CSS pixels.
 * @param options - Pixel ratio (default 2) and background colour.
 * @returns A promise of the PNG data URL.
 * @throws OmaiError with code `not-implemented` outside a browser.
 * @index function SVG text to a PNG data URL in a browser: `svgToPng(svg, width, height)`.
 */
export async function svgToPng(
  svg: string,
  width: number,
  height: number,
  options: { readonly scale?: number; readonly background?: string } = {},
): Promise<string> {
  const canvas = await svgToCanvas(svg, width, height, options)
  return canvas.toDataURL('image/png')
}

/**
 * Rasterise SVG text to RGBA pixels in the browser (for animation frames, GIF and video export).
 *
 * @param svg - The SVG text.
 * @param width - Width in CSS pixels.
 * @param height - Height in CSS pixels.
 * @param options - Pixel ratio (default 1) and background colour.
 * @returns A promise of the pixels.
 * @throws OmaiError with code `not-implemented` outside a browser; a `SecurityError` where the browser taints
 * canvases drawn from SVG with foreign objects.
 *
 * @index function SVG text to RGBA pixels in a browser: `svgToPixels(svg, width, height)`.
 */
export async function svgToPixels(
  svg: string,
  width: number,
  height: number,
  options: { readonly scale?: number; readonly background?: string } = {},
): Promise<Pixels> {
  const canvas = await svgToCanvas(svg, width, height, { scale: 1, ...options })
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height)
  return { width: canvas.width, height: canvas.height, rgba: data.data }
}

/**
 * Export sizes for a figure per destination: its own size for the page, and for slides the full width of a 1920 by
 * 1080 slide at the figure's aspect (capped at the slide's height).
 *
 * @param fig - The figure.
 * @param target - `page` (default), `blog` (1440 wide) or `slide`.
 * @returns The width, height and the pixel ratio to render at from the figure's size.
 * @index function Export size of a figure for a page, blog or slide: `exportSize(fig, 'slide')`.
 * @example
 * ```ts
 * import { exportSize } from 'omai-render/svg'
 * import { figure } from 'omai-render/figure'
 * exportSize(figure(), 'slide') // => { width: 1728, height: 1080, scale: 2.7 }
 * ```
 */
export function exportSize(
  fig: Figure,
  target: 'page' | 'blog' | 'slide' = 'page',
): { readonly width: number; readonly height: number; readonly scale: number } {
  if (target === 'page') return { width: fig.width, height: fig.height, scale: 1 }
  const maxW = target === 'slide' ? 1920 : 1440
  const maxH = target === 'slide' ? 1080 : Number.POSITIVE_INFINITY
  const width = Math.min(maxW, maxH * fig.aspect)
  return {
    width: Math.round(width),
    height: Math.round(width / fig.aspect),
    scale: Math.round((width / fig.width) * 1000) / 1000,
  }
}
