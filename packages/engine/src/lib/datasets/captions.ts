/**
 * Scenes with captions and questions: small RGB images of coloured shapes in the cells of a grid, each with a sentence
 * from a template grammar describing it and questions with answers, the attributes of every object as the truth. For
 * contrastive image-text alignment, captioning and visual question answering at toy scale.
 */

import { type DatasetGenerator, type ImageData, images } from '../data/index.ts'
import { foldIn, permutation, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { fromStorage } from '../tensor/index.ts'
import { stamp } from './_util.ts'

const SHAPES = ['circle', 'square', 'triangle'] as const
const COLOURS = ['red', 'green', 'blue', 'yellow'] as const
const RGB: Readonly<Record<(typeof COLOURS)[number], readonly [number, number, number]>> = {
  red: [1, 0.1, 0.1],
  green: [0.1, 0.8, 0.2],
  blue: [0.15, 0.25, 1],
  yellow: [1, 0.9, 0.1],
}
const CELL_NAMES = ['top left', 'top right', 'bottom left', 'bottom right'] as const

/** One object of a scene: its shape, colour and cell (0 top left, 1 top right, 2 bottom left, 3 bottom right). */
export interface SceneObject {
  readonly shape: (typeof SHAPES)[number]
  readonly colour: (typeof COLOURS)[number]
  readonly cell: number
}

/** A question about a scene and its answer. */
export interface SceneQuestion {
  readonly question: string
  readonly answer: string
}

function inside(shape: (typeof SHAPES)[number], dx: number, dy: number, r: number): boolean {
  if (shape === 'circle') return dx * dx + dy * dy <= r * r
  if (shape === 'square') return Math.abs(dx) <= 0.85 * r && Math.abs(dy) <= 0.85 * r
  return dy >= -r && dy <= r && Math.abs(dx) <= ((dy + r) / (2 * r)) * r
}

/** The sentence describing a scene: its objects in reading order of their cells. */
function caption(objects: readonly SceneObject[]): string {
  const parts = [...objects]
    .sort((a, b) => a.cell - b.cell)
    .map((o) => `a ${o.colour} ${o.shape} at ${CELL_NAMES[o.cell]}`)
  return parts.length === 1 ? (parts[0] ?? '') : `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`
}

/** Questions that every scene can answer: counts, existence of a pair, the colour of a uniquely named shape, what is in a cell. */
function questions(objects: readonly SceneObject[]): SceneQuestion[] {
  const out: SceneQuestion[] = [{ question: 'how many shapes are there', answer: String(objects.length) }]
  for (const s of SHAPES) {
    const of = objects.filter((o) => o.shape === s)
    if (of.length === 1) out.push({ question: `what colour is the ${s}`, answer: of[0]?.colour ?? '' })
  }
  for (const [cell, name] of CELL_NAMES.entries()) {
    const o = objects.find((x) => x.cell === cell)
    out.push({ question: `what is at ${name}`, answer: o === undefined ? 'nothing' : `${o.colour} ${o.shape}` })
  }
  const first = objects[0]
  if (first !== undefined) out.push({ question: `is there a ${first.colour} ${first.shape}`, answer: 'yes' })
  const absent = COLOURS.flatMap((c) => SHAPES.map((s) => ({ c, s }))).find(
    ({ c, s }) => !objects.some((o) => o.colour === c && o.shape === s),
  )
  if (absent !== undefined) out.push({ question: `is there a ${absent.c} ${absent.s}`, answer: 'no' })
  return out
}

/**
 * Captioned scenes: RGB images of one to `maxObjects` coloured shapes (circle, square, triangle; red, green, blue,
 * yellow) in distinct cells of a two by two grid, with a caption from a template grammar and questions with answers. The
 * images carry boxes and shape classes for each object; the truth has every scene's objects, captions and questions
 * and the vocabulary of the grammar, so a model's alignment of image and sentence can be scored exactly.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { captionedShapes } from 'mdml-engine/datasets'
 * (generate(captionedShapes, { knobs: { n: 3 } }).truth.extra?.['captions'] as string[]).length // => 3
 * ```
 */
export const captionedShapes: DatasetGenerator<ImageData> = {
  name: 'captionedShapes',
  kind: 'image',
  description: 'Scenes of coloured shapes with template captions and questions, the attributes as truth.',
  knobs: dict({
    n: int(1, 1_000_000, { label: 'scenes', description: 'The number of scenes.', default: 200 }),
    size: int(16, 256, { label: 'image side', description: 'The side of each square image in pixels.', default: 32 }),
    maxObjects: int(1, 4, {
      label: 'most objects',
      description: 'Each scene has between one and this many objects.',
      default: 2,
    }),
    noise: real(0, 1, {
      label: 'pixel noise',
      description: 'The standard deviation of the pixel noise.',
      default: 0.02,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const s = knobs['size'] as number
    const maxObjects = knobs['maxObjects'] as number
    const noise = knobs['noise'] as number
    const pixels = new Float64Array(n * s * s * 3)
    const scenes: SceneObject[][] = []
    const boxes: number[][][] = []
    const boxLabels: number[][] = []
    const half = s / 2
    const radius = 0.3 * s
    for (let i = 0; i < n; i++) {
      const u = uniform(foldIn(foldIn(k, 0), i), [1 + 3 * maxObjects]).data
      const count = 1 + Math.floor((u[0] ?? 0) * maxObjects)
      const cells = Array.from(permutation(foldIn(foldIn(k, 1), i), 4).data.subarray(0, count))
      const objects: SceneObject[] = cells.map((cell, j) => ({
        shape: SHAPES[Math.floor((u[1 + 3 * j] ?? 0) * SHAPES.length)] ?? 'circle',
        colour: COLOURS[Math.floor((u[2 + 3 * j] ?? 0) * COLOURS.length)] ?? 'red',
        cell,
      }))
      scenes.push(objects)
      const eps = uniform(foldIn(foldIn(k, 2), i), [s * s * 3]).data
      for (let y = 0; y < s; y++)
        for (let x = 0; x < s; x++) {
          let rgb: readonly [number, number, number] = [0.05, 0.05, 0.05]
          for (const o of objects) {
            const cx = (o.cell % 2) * half + half / 2
            const cy = Math.floor(o.cell / 2) * half + half / 2
            if (inside(o.shape, x + 0.5 - cx, y + 0.5 - cy, radius)) rgb = RGB[o.colour]
          }
          for (let c = 0; c < 3; c++)
            pixels[((i * s + y) * s + x) * 3 + c] = Math.min(
              1,
              Math.max(0, (rgb[c] ?? 0) + noise * ((eps[(y * s + x) * 3 + c] ?? 0.5) - 0.5) * 3.46),
            )
        }
      boxes.push(
        objects.map((o) => {
          const cx = (o.cell % 2) * half + half / 2
          const cy = Math.floor(o.cell / 2) * half + half / 2
          return [
            Math.max(0, cx - radius),
            Math.max(0, cy - radius),
            Math.min(s, cx + radius),
            Math.min(s, cy + radius),
          ]
        }),
      )
      boxLabels.push(objects.map((o) => SHAPES.indexOf(o.shape)))
    }
    const captions = scenes.map(caption)
    const vocabulary = [
      ...new Set([...COLOURS, ...SHAPES, ...CELL_NAMES.flatMap((c) => c.split(' ')), 'a', 'at', 'and']),
    ].sort()
    return {
      data: images(fromStorage(pixels, [n, s, s, 3], 'float64'), {
        boxes,
        boxLabels,
        classes: [...SHAPES],
        meta: stamp('captionedShapes', captionedShapes.description, knobs),
      }),
      truth: {
        extra: {
          scenes,
          captions,
          questions: scenes.map(questions),
          vocabulary,
          shapes: [...SHAPES],
          colours: [...COLOURS],
          cells: [...CELL_NAMES],
        },
      },
    }
  },
}
