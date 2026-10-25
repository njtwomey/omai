/**
 * Cellular automata played step by step: Wolfram's elementary automata, where each cell's next state is a rule of
 * itself and its two neighbours, drawn as a space-time diagram filling row by row, and Conway's Game of Life generation
 * by generation from classic starting patterns.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { type BoardSteps, boardSteps, type Explained } from '../../lib/exposition/index.ts'
import { type Key, key as makeKey, uniform } from '../../lib/random/index.ts'

/** Options of {@link elementaryAutomaton}. */
export interface ElementaryOptions {
  /** Cells per row (default 63). */
  readonly width?: number
  /** Rows after the first (default 40). */
  readonly steps?: number
  /** One live cell in the middle (default) or random cells (half alive). */
  readonly start?: 'single' | 'random'
  readonly key?: Key
}

/**
 * Wolfram's elementary cellular automaton: a row of cells, each next state a function (the rule's bits) of the cell
 * and its two neighbours, with the row wrapping round. Recorded as a space-time diagram (time running down) that
 * fills one row per step.
 *
 * @param rule - The rule number, 0 to 255 (30 chaotic, 90 Sierpinski, 110 universal, 184 traffic).
 * @param options - Width, steps, start and key.
 * @returns The rows, with a `boardSteps` record.
 * @throws MdmlError `value` for a rule outside 0 to 255.
 */
export function elementaryAutomaton(
  rule: number,
  options: ElementaryOptions = {},
): Explained<{ readonly rows: readonly (readonly number[])[] }> {
  if (!(Number.isInteger(rule) && rule >= 0 && rule <= 255))
    throw new MdmlError('value', `elementaryAutomaton: rule ${rule} is not 0 to 255`)
  const width = options.width ?? 63
  const steps = options.steps ?? 40
  const first =
    options.start === 'random'
      ? Array.from(uniform(options.key ?? makeKey(0), [width]).data, (u) => (u < 0.5 ? 1 : 0))
      : Array.from({ length: width }, (_, i) => (i === Math.floor(width / 2) ? 1 : 0))
  const rows: number[][] = [first]
  for (let t = 0; t < steps; t++) {
    const r = rows[rows.length - 1] as number[]
    rows.push(
      r.map((c, i) => {
        const pattern = 4 * (r[(i - 1 + width) % width] ?? 0) + 2 * c + (r[(i + 1) % width] ?? 0)
        return (rule >> pattern) & 1
      }),
    )
  }
  const height = rows.length
  const record: BoardSteps = boardSteps({
    method: `rule ${rule}`,
    layout: { type: 'grid', rows: height, columns: width },
    categories: ['dead', 'alive'],
    steps: rows.map((_, t) => ({
      cells: rows.flatMap((r, i) => (i <= t ? r : new Array<number>(width).fill(0))),
      caption: `row ${t}: ${rows[t]?.reduce((a, b) => a + b, 0)} alive`,
    })),
    metrics: [{ name: 'alive', values: rows.map((r) => r.reduce((a, b) => a + b, 0)) }],
    stepName: 'step',
  })
  return { rows, exposition: [record] }
}

/** Starting patterns for the Game of Life. */
export type LifePattern = 'glider' | 'r-pentomino' | 'gosper gun' | 'pulsar' | 'random'

const PATTERNS: Readonly<Record<Exclude<LifePattern, 'random'>, readonly string[]>> = {
  glider: ['.#.', '..#', '###'],
  'r-pentomino': ['.##', '##.', '.#.'],
  'gosper gun': [
    '........................#...........',
    '......................#.#...........',
    '............##......##............##',
    '...........#...#....##............##',
    '##........#.....#...##..............',
    '##........#...#.##....#.#...........',
    '..........#.....#.......#...........',
    '...........#...#....................',
    '............##......................',
  ],
  pulsar: [
    '..###...###..',
    '.............',
    '#....#.#....#',
    '#....#.#....#',
    '#....#.#....#',
    '..###...###..',
    '.............',
    '..###...###..',
    '#....#.#....#',
    '#....#.#....#',
    '#....#.#....#',
    '.............',
    '..###...###..',
  ],
}

/** Options of {@link gameOfLife}. */
export interface LifeOptions {
  /** The board (default 32 by 48). */
  readonly rows?: number
  readonly cols?: number
  /** The start (default the glider gun). */
  readonly pattern?: LifePattern
  /** Generations (default 80). */
  readonly generations?: number
  /** Wrap the edges into a torus (default false: cells beyond the edge are dead). */
  readonly wrap?: boolean
  /** Random starts: the share of live cells (default 0.3). */
  readonly density?: number
  readonly key?: Key
}

/**
 * Conway's Game of Life: a live cell with two or three live neighbours survives, a dead cell with exactly three is
 * born, every other cell is dead next generation. Recorded generation by generation with the population.
 *
 * @param options - Board size, pattern, generations, wrapping, density and key.
 * @returns The final board and the population per generation, with a `boardSteps` record.
 */
export function gameOfLife(
  options: LifeOptions = {},
): Explained<{ readonly board: readonly number[]; readonly population: readonly number[] }> {
  const rows = options.rows ?? 32
  const cols = options.cols ?? 48
  const pattern = options.pattern ?? 'gosper gun'
  const wrap = options.wrap ?? false
  let board = new Array<number>(rows * cols).fill(0)
  if (pattern === 'random') {
    const u = uniform(options.key ?? makeKey(0), [rows * cols]).data
    board = Array.from(u, (x) => (x < (options.density ?? 0.3) ? 1 : 0))
  } else {
    const shape = PATTERNS[pattern]
    const r0 = pattern === 'gosper gun' ? 1 : Math.floor((rows - shape.length) / 2)
    const c0 = pattern === 'gosper gun' ? 1 : Math.floor((cols - (shape[0]?.length ?? 0)) / 2)
    if (shape.length > rows || (shape[0]?.length ?? 0) > cols)
      throw new MdmlError('value', `gameOfLife: the ${pattern} does not fit a ${rows} by ${cols} board`)
    shape.forEach((line, r) => {
      for (const [c, ch] of [...line].entries()) if (ch === '#') board[(r0 + r) * cols + c0 + c] = 1
    })
  }
  const boards: number[][] = [board]
  for (let g = 0; g < (options.generations ?? 80); g++) {
    const b = boards[boards.length - 1] as number[]
    const next = b.map((alive, i) => {
      const r = Math.floor(i / cols)
      const c = i % cols
      let n = 0
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue
          let rr = r + dr
          let cc = c + dc
          if (wrap) {
            rr = (rr + rows) % rows
            cc = (cc + cols) % cols
          } else if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue
          n += b[rr * cols + cc] ?? 0
        }
      return n === 3 || (alive === 1 && n === 2) ? 1 : 0
    })
    boards.push(next)
  }
  const population = boards.map((b) => b.reduce((a, x) => a + x, 0))
  const record: BoardSteps = boardSteps({
    method: 'game of life',
    layout: { type: 'grid', rows, columns: cols },
    categories: ['dead', 'alive'],
    steps: boards.map((b, g) => ({ cells: b, caption: `generation ${g}: ${population[g]} alive` })),
    metrics: [{ name: 'population', values: population }],
    stepName: 'generation',
  })
  return { board: boards[boards.length - 1] as number[], population, exposition: [record] }
}
