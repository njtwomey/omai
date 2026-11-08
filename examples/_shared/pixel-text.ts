/**
 * Text for animations, drawn as shapes. The scene rasteriser that makes the frames of a GIF draws no text, so labels
 * are built from a 5 by 7 pixel font: each lit pixel run becomes a rectangle. Capitals, digits and a few marks are
 * enough for axis labels, tick labels and a legend.
 */
import { type Paint, type RectNode, rect } from 'mdml-engine/scene'

const ROWS = 7
const COLS = 5

/** Glyphs as seven rows of five pixels, `#` lit. */
const FONT: Readonly<Record<string, readonly string[]>> = {
  ' ': ['.....', '.....', '.....', '.....', '.....', '.....', '.....'],
  '0': ['.###.', '#...#', '#..##', '#.#.#', '##..#', '#...#', '.###.'],
  '1': ['..#..', '.##..', '..#..', '..#..', '..#..', '..#..', '.###.'],
  '2': ['.###.', '#...#', '....#', '...#.', '..#..', '.#...', '#####'],
  '3': ['####.', '....#', '....#', '.###.', '....#', '....#', '####.'],
  '4': ['...#.', '..##.', '.#.#.', '#..#.', '#####', '...#.', '...#.'],
  '5': ['#####', '#....', '####.', '....#', '....#', '#...#', '.###.'],
  '6': ['..##.', '.#...', '#....', '####.', '#...#', '#...#', '.###.'],
  '7': ['#####', '....#', '...#.', '..#..', '.#...', '.#...', '.#...'],
  '8': ['.###.', '#...#', '#...#', '.###.', '#...#', '#...#', '.###.'],
  '9': ['.###.', '#...#', '#...#', '.####', '....#', '...#.', '.##..'],
  '.': ['.....', '.....', '.....', '.....', '.....', '.##..', '.##..'],
  ',': ['.....', '.....', '.....', '.....', '.##..', '..#..', '.#...'],
  '-': ['.....', '.....', '.....', '#####', '.....', '.....', '.....'],
  '+': ['.....', '..#..', '..#..', '#####', '..#..', '..#..', '.....'],
  '=': ['.....', '.....', '#####', '.....', '#####', '.....', '.....'],
  ':': ['.....', '.##..', '.##..', '.....', '.##..', '.##..', '.....'],
  '(': ['...#.', '..#..', '.#...', '.#...', '.#...', '..#..', '...#.'],
  ')': ['.#...', '..#..', '...#.', '...#.', '...#.', '..#..', '.#...'],
  '/': ['....#', '...#.', '...#.', '..#..', '.#...', '.#...', '#....'],
  '%': ['##..#', '##.#.', '...#.', '..#..', '.#...', '.#.##', '#..##'],
  A: ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
  B: ['####.', '#...#', '#...#', '####.', '#...#', '#...#', '####.'],
  C: ['.###.', '#...#', '#....', '#....', '#....', '#...#', '.###.'],
  D: ['####.', '#...#', '#...#', '#...#', '#...#', '#...#', '####.'],
  E: ['#####', '#....', '#....', '####.', '#....', '#....', '#####'],
  F: ['#####', '#....', '#....', '####.', '#....', '#....', '#....'],
  G: ['.###.', '#...#', '#....', '#.###', '#...#', '#...#', '.####'],
  H: ['#...#', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
  I: ['.###.', '..#..', '..#..', '..#..', '..#..', '..#..', '.###.'],
  J: ['..###', '...#.', '...#.', '...#.', '...#.', '#..#.', '.##..'],
  K: ['#...#', '#..#.', '#.#..', '##...', '#.#..', '#..#.', '#...#'],
  L: ['#....', '#....', '#....', '#....', '#....', '#....', '#####'],
  M: ['#...#', '##.##', '#.#.#', '#.#.#', '#...#', '#...#', '#...#'],
  N: ['#...#', '##..#', '#.#.#', '#..##', '#...#', '#...#', '#...#'],
  O: ['.###.', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
  P: ['####.', '#...#', '#...#', '####.', '#....', '#....', '#....'],
  Q: ['.###.', '#...#', '#...#', '#...#', '#.#.#', '#..#.', '.##.#'],
  R: ['####.', '#...#', '#...#', '####.', '#.#..', '#..#.', '#...#'],
  S: ['.####', '#....', '#....', '.###.', '....#', '....#', '####.'],
  T: ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'],
  U: ['#...#', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
  V: ['#...#', '#...#', '#...#', '#...#', '#...#', '.#.#.', '..#..'],
  W: ['#...#', '#...#', '#...#', '#.#.#', '#.#.#', '##.##', '#...#'],
  X: ['#...#', '#...#', '.#.#.', '..#..', '.#.#.', '#...#', '#...#'],
  Y: ['#...#', '#...#', '.#.#.', '..#..', '..#..', '..#..', '..#..'],
  Z: ['#####', '....#', '...#.', '..#..', '.#...', '#....', '#####'],
}

/** Options of {@link pixelText}. */
export interface PixelTextOptions {
  /** The pixel size of one font pixel (default 1: glyphs 5 by 7 pixels, 6 pixels apart). */
  readonly scale?: number
  readonly fill?: Paint
  /** Where `x` is: the left of the text (default), its middle or its right end. */
  readonly anchor?: 'start' | 'middle' | 'end'
}

/** The width in pixels of a text at a scale. */
export const pixelTextWidth = (text: string, scale = 1): number => Math.max(0, text.length * (COLS + 1) - 1) * scale

/**
 * A text as rectangles, its top left or middle or right at `(x, y)`; letters are capitals (lower case is raised).
 *
 * @param text - The text; characters the font lacks are left blank.
 * @param x - The horizontal position.
 * @param y - The top of the text.
 * @param options - Scale, colour and anchor.
 * @returns The rectangles.
 */
export function pixelText(text: string, x: number, y: number, options: PixelTextOptions = {}): RectNode[] {
  const { scale = 1, fill = 'fg', anchor = 'start' } = options
  const width = pixelTextWidth(text, scale)
  const left = anchor === 'start' ? x : anchor === 'middle' ? x - width / 2 : x - width
  const nodes: RectNode[] = []
  for (const [k, character] of [...text.toUpperCase()].entries()) {
    const glyph = FONT[character] ?? FONT[' ']
    if (glyph === undefined) continue
    for (let row = 0; row < ROWS; row++) {
      const bits = glyph[row] ?? ''
      let col = 0
      while (col < COLS) {
        if (bits[col] !== '#') {
          col++
          continue
        }
        let end = col
        while (end < COLS && bits[end] === '#') end++
        nodes.push(rect(left + (k * (COLS + 1) + col) * scale, y + row * scale, (end - col) * scale, scale, { fill }))
        col = end
      }
    }
  }
  return nodes
}
