/**
 * Capital letters of 66 real font instances (23 families across six design classes, at several weights, widths and
 * postures) as outlines in dense correspondence: sample k of a glyph's contour sits at the same place of the letter in
 * every font (a stem foot, a serif tip, the apex of an A), so the outlines can be averaged, interpolated and modelled
 * (after Campbell and Kautz, 2014, Learning a Manifold of Fonts), or drawn as one closed path. Fonts from Google Fonts
 * under the SIL Open Font License 1.1 or the Apache License 2.0 (each font's licence is in its `FontInfo`); the
 * outlines are vendored in `_fonts.data.ts`.
 */

import { MdmlError } from '../../base/index.ts'
import { type ClassificationData, xyClasses } from '../../data/index.ts'
import { normaliseOutline, type Point2, resampleClosed, stitchStrokes } from '../../graphics/index.ts'
import { fromStorage } from '../../tensor/index.ts'
import { CAP_HEIGHT, FONTS, GLYPHS, SOURCE, VECTORS, WIDTH } from './_fonts.data.ts'

/** The six design classes, in the order of a font dataset's class labels 0 to 5. */
export const FONT_CLASSES: readonly ['sans', 'humanist', 'serif', 'didone', 'slab', 'mono'] = [
  'sans',
  'humanist',
  'serif',
  'didone',
  'slab',
  'mono',
]

/** A design class: sans (grotesque and geometric), humanist sans, serif, didone, slab serif or monospaced. */
export type FontClass = (typeof FONT_CLASSES)[number]

/** One font instance: a family at one weight, width and posture, with its design class and licence. */
export interface FontInfo {
  readonly family: string
  readonly style: string
  readonly cls: FontClass
  /** CSS weight, 100 to 900. */
  readonly weight: number
  /** Width as a percentage of normal. */
  readonly width: number
  readonly italic: boolean
  /** SPDX identifier of the font's licence (OFL-1.1 or Apache-2.0). */
  readonly licence: string
}

/** Where one character sits in a font's vector: its samples per contour (outer first, then counters) and offset. */
export interface GlyphLayout {
  readonly char: string
  readonly contours: readonly number[]
  /** Index of the first value (x of sample 0 of contour 0); x and y alternate. */
  readonly offset: number
}

let decoded: Int16Array | null = null

// Every font's vector, decoded once from base64 into little-endian int16 values.
function table(): Int16Array {
  if (decoded !== null) return decoded
  const binary = atob(VECTORS)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  const view = new DataView(bytes.buffer)
  const values = new Int16Array(bytes.length / 2)
  for (let i = 0; i < values.length; i++) values[i] = view.getInt16(2 * i, true)
  decoded = values
  return values
}

/**
 * The 66 font instances, in the order of a font dataset's rows.
 *
 * @returns The fonts: family, style, design class, weight, width, posture and licence.
 * @example
 * ```ts
 * import { fontInstances } from 'mdml-engine/datasets/outlines'
 * const f = fontInstances()
 * [f.length, `${f[0]?.family} ${f[0]?.style}`] // => [66, 'Roboto Light']
 * ```
 */
export function fontInstances(): FontInfo[] {
  return FONTS.map((f) => ({ ...f, cls: f.cls as FontClass }))
}

/**
 * The characters with outlines, and where each sits in a font's vector. Only the 23 capitals
 * "ABCDEFGHIJLMNOPRSTUVXYZ" are available (K, Q and W vary in their contours between designs).
 *
 * @returns The layouts, one per character.
 * @example
 * ```ts
 * import { fontGlyphs } from 'mdml-engine/datasets/outlines'
 * fontGlyphs().map((g) => g.char).join('') // => 'ABCDEFGHIJLMNOPRSTUVXYZ'
 * ```
 */
export function fontGlyphs(): GlyphLayout[] {
  return GLYPHS.map(({ char, contours, offset }) => ({ char, contours, offset }))
}

/**
 * Fonts as a classification dataset: one row per font, the features the x and y of every sample of every contour of
 * the chosen characters (font units, cap height 700), the label the design class.
 *
 * @param options - The characters (default all 23 capitals).
 * @returns The dataset.
 * @throws MdmlError with code `value` for a character without outlines.
 * @example
 * ```ts
 * import { fonts } from 'mdml-engine/datasets/outlines'
 * const d = fonts({ chars: 'O' })
 * [d.x.shape, d.classes.length] // => [[66, 132], 6]
 * ```
 */
export function fonts(options: { readonly chars?: string } = {}): ClassificationData {
  const glyphs = fontGlyphs()
  const byChar = new Map(glyphs.map((g) => [g.char, g]))
  const chars = [...(options.chars ?? glyphs.map((g) => g.char).join(''))]
  const columns: number[] = []
  const names: string[] = []
  for (const c of chars) {
    const g = byChar.get(c)
    if (g === undefined)
      throw new MdmlError('value', `fonts: no outlines for '${c}'; available: ${[...byChar.keys()].join('')}`)
    const samples = g.contours.reduce((a, b) => a + b, 0)
    for (let k = 0; k < 2 * samples; k++) columns.push(g.offset + k)
    for (const [contour, m] of g.contours.entries())
      for (let j = 0; j < m; j++) names.push(`${c}${contour}.${j}.x`, `${c}${contour}.${j}.y`)
  }
  const values = table()
  const rows = FONTS.length
  const d = columns.length
  const x = new Float64Array(rows * d)
  for (let r = 0; r < rows; r++) for (let j = 0; j < d; j++) x[r * d + j] = values[r * WIDTH + (columns[j] ?? 0)] ?? 0
  const labels = Int32Array.from(fontInstances(), (f) => FONT_CLASSES.indexOf(f.cls))
  return xyClasses(fromStorage(x, [rows, d], 'float64'), fromStorage(labels, [rows], 'int32'), {
    name: 'fonts',
    description: `Outlines of ${chars.join('')} in ${rows} font instances, sampled in dense correspondence.`,
    source: `Google Fonts (${SOURCE}), outlines put in correspondence by the AI Field Notes font builder`,
    citation: 'Campbell and Kautz (2014), Learning a Manifold of Fonts, ACM Transactions on Graphics 33(4)',
    licence: 'SIL OFL 1.1 and Apache 2.0 (per font)',
    featureNames: names,
    classes: [...FONT_CLASSES],
  })
}

/**
 * One glyph's contours in one font, in font units (y up, cap height 700): the outer contour first, then the counters.
 *
 * @param font - The font's index (see {@link fontInstances}).
 * @param char - The character.
 * @returns The contours' points.
 * @throws MdmlError with code `value` for an unknown font or character.
 * @example
 * ```ts
 * import { glyphContours } from 'mdml-engine/datasets/outlines'
 * glyphContours(0, 'O').map((c) => c.length) // => [42, 24]
 * ```
 */
export function glyphContours(font: number, char: string): Point2[][] {
  if (!(Number.isInteger(font) && font >= 0 && font < FONTS.length))
    throw new MdmlError('value', `glyphContours: no font ${font}`)
  const g = GLYPHS.find((x) => x.char === char)
  if (g === undefined) throw new MdmlError('value', `glyphContours: no outlines for '${char}'`)
  const values = table()
  let at = font * WIDTH + g.offset
  return g.contours.map((m) => {
    const pts: Point2[] = []
    for (let j = 0; j < m; j++) pts.push([values[at + 2 * j] ?? 0, values[at + 2 * j + 1] ?? 0])
    at += 2 * m
    return pts
  })
}

/**
 * One glyph in one font as a single closed path of `n` points (its contours stitched together), uniform in arc
 * length, centred and of unit radius.
 *
 * @param font - The font's index.
 * @param char - The character.
 * @param n - How many points.
 * @returns The outline.
 * @example
 * ```ts
 * import { glyphOutline } from 'mdml-engine/datasets/outlines'
 * glyphOutline(5, 'A', 256).length // => 256
 * ```
 */
export function glyphOutline(font: number, char: string, n: number): Point2[] {
  const strokes = glyphContours(font, char).map((points) => ({ points, closed: true }))
  return normaliseOutline(resampleClosed(stitchStrokes(strokes), n), false)
}

/** The cap height of the outlines, in font units. */
export const FONT_CAP_HEIGHT: number = CAP_HEIGHT
