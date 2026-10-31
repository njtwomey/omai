/**
 * The pure part of exporting a method page: which derived paths a layout reads (so a tape file can store them), the
 * steps of a recording from step a to b, and the frame that puts a page's figures under its title and over its caption
 * as one SVG document (the PNG, GIF and WebM are rasterisations of it).
 */

import type { Json } from 'mdml-engine/base'
import type { ProbeRequest } from '../../producer/tape.ts'
import { escapeXml } from '../../svg/svg.ts'
import { type ColourMode, tokens } from '../../theme/tokens.ts'
import type { Narrative, Panel } from './layout.ts'

const OVERLAY_PATHS: Readonly<Record<string, readonly string[]>> = {
  assignments: ['model/assignments', 'model/centres'],
  components: ['model/params'],
  fit: ['model/predictions'],
  band: ['model/predictive'],
  surface: ['model/decision-surface'],
  experts: ['model/assignments', 'model/predictions', 'model/experts'],
}

function requestsOfPanel(panel: Panel): ProbeRequest[] {
  const at = (path: string | undefined, args?: Json): ProbeRequest[] =>
    path === undefined ? [] : [args === undefined ? { path } : { path, args }]
  switch (panel.kind) {
    case 'rows':
      return [
        ...(panel.overlay === undefined ? [] : (OVERLAY_PATHS[panel.overlay] ?? []).map((path) => ({ path }))),
        ...[panel.values, panel.size, panel.marks, panel.regions, panel.focus].flatMap((s) => at(s?.path, s?.args)),
      ]
    case 'probe':
    case 'field':
    case 'strip':
    case 'table':
    case 'images':
    case 'graph':
    case 'points':
    case 'scene':
      return at(panel.path, panel.args)
    case 'trajectory':
      return at(panel.over?.path, panel.over?.args)
    case 'tensor':
      return at(panel.marksPath, panel.marksArgs)
    default:
      return []
  }
}

/**
 * The derived paths a layout (or a narrative's pages) reads, with their arguments, each once: what a tape file must
 * store for the page to be reproduced from it. Paths of the tape need no storing (the events hold them); a request for
 * one that is not derived is dropped when the file is written.
 *
 * @param layout - The panels, or a narrative.
 * @returns The requests.
 */
export function probeRequestsOf(layout: readonly Panel[] | Narrative): ProbeRequest[] {
  const panels = Array.isArray(layout)
    ? (layout as readonly Panel[])
    : (layout as Narrative).pages.flatMap((p) => p.layout)
  const seen = new Set<string>()
  const out: ProbeRequest[] = []
  for (const panel of panels)
    for (const r of requestsOfPanel(panel)) {
      const key = `${r.path}|${JSON.stringify(r.args ?? null)}`
      if (seen.has(key)) continue
      seen.add(key)
      out.push(r)
    }
  return out
}

/**
 * The steps of a recording from step `from` to step `to`: every step when few, else an even spread of at most `max`
 * that holds both ends. A range given backwards is read forwards.
 *
 * @param from - The first step.
 * @param to - The last step.
 * @param max - The most frames.
 * @returns The steps, ascending.
 */
export function framesBetween(from: number, to: number, max: number): number[] {
  const a = Math.round(Math.min(from, to))
  const b = Math.round(Math.max(from, to))
  if (b - a + 1 <= max) return Array.from({ length: b - a + 1 }, (_, i) => a + i)
  const n = Math.max(2, Math.floor(max))
  const steps = new Set<number>()
  for (let i = 0; i < n; i++) steps.add(a + Math.round((i * (b - a)) / (n - 1)))
  return [...steps]
}

/** The words of a caption wrapped to lines of at most `columns` characters. */
export function wrapText(text: string, columns: number): string[] {
  const lines: string[] = []
  let line = ''
  for (const word of text.split(/\s+/).filter((w) => w !== '')) {
    if (line !== '' && line.length + 1 + word.length > columns) {
      lines.push(line)
      line = word
    } else line = line === '' ? word : `${line} ${word}`
  }
  if (line !== '') lines.push(line)
  return lines
}

/** What the frame around a page's figures holds. */
export interface PageFrame {
  readonly title?: string
  readonly caption?: string
  readonly mode: ColourMode
}

/**
 * A page as one SVG document: the figures' SVG (as `figureSvg` composes them) under the title, over the caption, on
 * the page's background.
 *
 * @param inner - The figures' SVG document.
 * @param width - Its width in pixels.
 * @param height - Its height in pixels.
 * @param frame - The title, caption and colour mode.
 * @returns The page's SVG and its size.
 */
export function pageSvg(
  inner: string,
  width: number,
  height: number,
  frame: PageFrame,
): { readonly svg: string; readonly width: number; readonly height: number } {
  const t = tokens(frame.mode)
  const head = frame.title === undefined ? 0 : 44
  const lines = frame.caption === undefined ? [] : wrapText(frame.caption, Math.max(20, Math.floor(width / 7)))
  const foot = lines.length === 0 ? 0 : 14 + lines.length * 17
  const total = height + head + foot
  const title =
    frame.title === undefined
      ? ''
      : `<text x="16" y="29" font-family="${escapeXml(t.font)}" font-size="18" font-weight="600" fill="${t.fg}">${escapeXml(frame.title)}</text>`
  const caption = lines
    .map(
      (l, i) =>
        `<text x="16" y="${head + height + 22 + i * 17}" font-family="${escapeXml(t.font)}" font-size="13" fill="${t.muted}">${escapeXml(l)}</text>`,
    )
    .join('')
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${total}" viewBox="0 0 ${width} ${total}">` +
    `<rect width="${width}" height="${total}" fill="${t.bg}"/>${title}` +
    `<g transform="translate(0 ${head})">${inner}</g>${caption}</svg>`
  return { svg, width, height: total }
}
