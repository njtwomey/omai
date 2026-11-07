/**
 * Conway's Game of Life and its life-like cousins: totalistic rules on the eight neighbours written as a B/S string
 * (`B3/S23` is Life, `B36/S23` HighLife, `B2/S` Seeds), named patterns from run-length encoding (blinker, glider,
 * lightweight spaceship, R-pentomino, Gosper's gun and others, each with the period and drift it is known for), the
 * detection of a period from the boards seen so far, and the automaton as a traced algorithm that logs the board, the
 * population, births and deaths per generation, and the period once the board has settled into a cycle.
 */

import { MdmlError, maxOf } from '../../base/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  event,
  scalar,
  tensor,
} from '../../trace/index.ts'
import { codec, draws, hashBoard, mod, positiveInt, unit } from './_core.ts'

/** A life-like rule: a dead cell with a neighbour count in `birth` comes alive, a live one in `survive` stays. */
export interface LifeRule {
  readonly birth: readonly number[]
  readonly survive: readonly number[]
}

/**
 * Parse a rule written `B3/S23` (or `b3/s23`, or the old `23/3` survive-slash-birth form).
 *
 * @param text - The rule.
 * @returns Its birth and survival counts.
 * @throws MdmlError with code `value` for text that is not a rule or counts above 8.
 * @concept cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * automata.parseLifeRule('B36/S23') // => { birth: [3, 6], survive: [2, 3] }
 * automata.parseLifeRule('23/3') // => { birth: [3], survive: [2, 3] }
 * ```
 */
export function parseLifeRule(text: string): LifeRule {
  const bs = /^\s*B([0-8]*)\s*\/\s*S([0-8]*)\s*$/i.exec(text)
  const sb = /^\s*([0-8]*)\s*\/\s*([0-8]*)\s*$/.exec(text)
  const digits = (d: string | undefined): number[] => Array.from(d ?? '', Number)
  if (bs !== null) return { birth: digits(bs[1]), survive: digits(bs[2]) }
  if (sb !== null) return { birth: digits(sb[2]), survive: digits(sb[1]) }
  throw new MdmlError('value', `parseLifeRule: cannot read '${text}' (expected B3/S23)`)
}

/**
 * A rule as text.
 *
 * @param rule - Birth and survival counts.
 * @returns The `B../S..` string.
 * @concept cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * automata.formatLifeRule({ birth: [3], survive: [2, 3] }) // => 'B3/S23'
 * ```
 */
export function formatLifeRule(rule: LifeRule): string {
  return `B${rule.birth.join('')}/S${rule.survive.join('')}`
}

/** A pattern: its live cells as `[row, column]`, and what is known of how it behaves. */
export interface LifePattern {
  readonly name: string
  /** The live cells, `[row, column]`, with the top left of the bounding box at the origin. */
  readonly cells: readonly (readonly [number, number])[]
  /** The period in generations (1 for a still life, absent for a pattern that grows or has no cycle). */
  readonly period?: number
  /** How far one period moves it, `[rows, columns]` (a spaceship; absent or `[0, 0]` for an oscillator). */
  readonly displacement?: readonly [number, number]
}

/**
 * Read a pattern from run-length encoding (`o` live, `b` dead, `$` end of row, `!` end, runs as digits).
 *
 * @param rle - The encoding, without the header line.
 * @returns The live cells `[row, column]`.
 * @throws MdmlError with code `value` for a character the format does not have.
 * @concept cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * automata.parseRle('bob$2bo$3o!') // => [[0, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
 * ```
 */
export function parseRle(rle: string): [number, number][] {
  const cells: [number, number][] = []
  let r = 0
  let c = 0
  let run = ''
  for (const ch of rle.replace(/\s+/g, '')) {
    if (ch >= '0' && ch <= '9') run += ch
    else {
      const k = run === '' ? 1 : Number(run)
      run = ''
      if (ch === 'o') for (let i = 0; i < k; i++) cells.push([r, c++])
      else if (ch === 'b') c += k
      else if (ch === '$') {
        r += k
        c = 0
      } else if (ch === '!') break
      else throw new MdmlError('value', `parseRle: unexpected '${ch}'`)
    }
  }
  return cells
}

function pattern(
  name: string,
  rle: string,
  extra: { period?: number; displacement?: readonly [number, number] } = {},
): LifePattern {
  return { name, cells: parseRle(rle), ...extra }
}

/**
 * Named patterns of Life, with the periods and drifts they are known for (the glider moves one cell down and right
 * every 4 generations, the lightweight spaceship two cells left every 4, Gosper's gun emits a glider every 30).
 */
export const LIFE_PATTERNS: Readonly<Record<string, LifePattern>> = {
  block: pattern('block', '2o$2o!', { period: 1 }),
  beehive: pattern('beehive', 'b2o$o2bo$b2o!', { period: 1 }),
  blinker: pattern('blinker', '3o!', { period: 2 }),
  toad: pattern('toad', 'b3o$3o!', { period: 2 }),
  beacon: pattern('beacon', '2o$2o$2b2o$2b2o!', { period: 2 }),
  pentadecathlon: pattern('pentadecathlon', '2bo4bo$2ob4ob2o$2bo4bo!', { period: 15 }),
  glider: pattern('glider', 'bob$2bo$3o!', { period: 4, displacement: [1, 1] }),
  lwss: pattern('lwss', 'bo2bo$o$o3bo$4o!', { period: 4, displacement: [0, -2] }),
  'r-pentomino': pattern('r-pentomino', 'b2o$2ob$bo!'),
  acorn: pattern('acorn', 'bo5b$3bo3b$2o2b3o!'),
  diehard: pattern('diehard', '6bob$2o6b$bo3b3o!'),
  'gosper-gun': pattern(
    'gosper-gun',
    '24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$10bo5bo7bo$11bo3bo$12b2o!',
    { period: 30 },
  ),
}

/**
 * The period of a board sequence: how many generations ago the newest board last occurred, from their hashes.
 *
 * @param hashes - The hashes of the recent boards, oldest first, newest last.
 * @returns The distance to the previous occurrence of the newest hash, or 0 if it has not occurred before.
 * @concept cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * [automata.detectPeriod([5, 7, 9, 7]), automata.detectPeriod([5, 7, 9, 3]), automata.detectPeriod([4, 4])] // => [2, 0, 1]
 * ```
 */
export function detectPeriod(hashes: readonly number[]): number {
  const n = hashes.length
  const last = hashes[n - 1]
  for (let d = 1; d < n; d++) if (hashes[n - 1 - d] === last) return d
  return 0
}

/** How to start a Life board: a random soup, a named or given pattern placed in the middle, or the cells themselves. */
export type LifeStart = 'soup' | string | LifePattern | Uint8Array

/** Options of {@link lifeAutomaton}. */
export interface LifeOptions {
  /** The rule, as text or counts (default `B3/S23`). */
  readonly rule?: string | LifeRule
  /** Board size (default 64 by 64). */
  readonly width?: number
  readonly height?: number
  /** Whether the board is a torus (default true) or has dead cells beyond its edges. */
  readonly wrap?: boolean
  /** The start (default `'soup'`). */
  readonly start?: LifeStart
  /** The chance a cell of a soup is live (default 0.3). */
  readonly density?: number
  /** How many recent boards to remember for spotting a cycle (default 64, so periods up to 63 are found). */
  readonly window?: number
}

/** The state of {@link lifeAutomaton}. */
export interface LifeState {
  readonly cells: Uint8Array
  readonly generation: number
  readonly births: number
  readonly deaths: number
  /** The hashes of the most recent boards, newest last. */
  readonly recent: readonly number[]
  /** The period once the board has come round again (0 until then). */
  readonly period: number
}

/** The paths of {@link lifeAutomaton}. */
export type LifePath =
  | 'life/board'
  | 'life/population'
  | 'life/births'
  | 'life/deaths'
  | 'life/period'
  | 'life/stabilised'

/** The log contract of {@link lifeAutomaton}. */
export const lifeContract: Contract<LifeState, LifePath> = defineContract({
  'life/board': tensor({ role: 'state', shape: ['height', 'width'], description: 'the cells, 1 live' }),
  'life/population': scalar({ description: 'live cells' }),
  'life/births': scalar({ description: 'cells that came alive this generation' }),
  'life/deaths': scalar({ description: 'cells that died this generation' }),
  'life/period': scalar({ description: 'the period of the cycle the board is in (0 until it repeats)' }),
  'life/stabilised': event({ context: { period: 'the cycle length', generation: 'when it was found' } }),
})

/**
 * One generation of a life-like rule.
 *
 * @param cells - The board, row-major, 1 live.
 * @param width - Columns.
 * @param height - Rows.
 * @param rule - Birth and survival counts.
 * @param wrap - Whether the board is a torus (default) or bounded by dead cells.
 * @returns The next board and the births and deaths it took.
 * @concept cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * const rule = automata.parseLifeRule('B3/S23')
 * const blinker = Uint8Array.of(0, 0, 0, 1, 1, 1, 0, 0, 0)
 * Array.from(automata.lifeStep(blinker, 3, 3, rule, false).cells) // => [0, 1, 0, 0, 1, 0, 0, 1, 0]
 * ```
 */
export function lifeStep(
  cells: Uint8Array,
  width: number,
  height: number,
  rule: LifeRule,
  wrap = true,
): { cells: Uint8Array; births: number; deaths: number } {
  const born = new Uint8Array(9)
  const stay = new Uint8Array(9)
  for (const b of rule.birth) born[b] = 1
  for (const s of rule.survive) stay[s] = 1
  const out = new Uint8Array(cells.length)
  let births = 0
  let deaths = 0
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      let n = 0
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue
          let rr = r + dr
          let cc = c + dc
          if (wrap) {
            rr = mod(rr, height)
            cc = mod(cc, width)
          } else if (rr < 0 || rr >= height || cc < 0 || cc >= width) continue
          n += cells[rr * width + cc] ?? 0
        }
      }
      const alive = cells[r * width + c] === 1
      const next = alive ? stay[n] : born[n]
      out[r * width + c] = next ?? 0
      if (next === 1 && !alive) births++
      if (next === 0 && alive) deaths++
    }
  }
  return { cells: out, births, deaths }
}

/**
 * Lay a pattern in the middle of an empty board.
 *
 * @param pattern - The pattern.
 * @param width - Columns.
 * @param height - Rows.
 * @returns The board.
 * @throws MdmlError with code `shape` if the pattern does not fit.
 * @concept cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * automata.placePattern(automata.LIFE_PATTERNS.block!, 4, 4).join('') // => '0000011001100000'
 * ```
 */
export function placePattern(pattern: LifePattern, width: number, height: number): Uint8Array {
  const rows = maxOf(pattern.cells.map(([r]) => r)) + 1
  const cols = maxOf(pattern.cells.map(([, c]) => c)) + 1
  if (rows > height || cols > width) throw new MdmlError('shape', `placePattern: ${pattern.name} does not fit`)
  const top = Math.floor((height - rows) / 2)
  const left = Math.floor((width - cols) / 2)
  const out = new Uint8Array(width * height)
  for (const [r, c] of pattern.cells) out[(top + r) * width + left + c] = 1
  return out
}

function population(cells: Uint8Array): number {
  let n = 0
  for (const c of cells) n += c
  return n
}

/**
 * A life-like cellular automaton as a traced algorithm: one generation per step, the board, population, births and
 * deaths on the tape, and `life/stabilised` fired (with the period) the first time a board comes round again within the
 * window. The state serialises, so a run can be scrubbed and resumed.
 *
 * @param options - The rule, size, wrap, start, soup density and the window for finding cycles.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a bad size, density or unknown pattern name, `shape` if a pattern or cells do not fit.
 * @concept cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(automata.lifeAutomaton({ start: 'blinker', width: 8, height: 8 }), { seed: 0, maxSteps: 6 })
 * [run.state.generation, run.state.period] // => [6, 2]
 * ```
 */
export function lifeAutomaton(options: LifeOptions = {}): Algorithm<LifeState, LifePath> {
  const rule = typeof options.rule === 'object' ? options.rule : parseLifeRule(options.rule ?? 'B3/S23')
  const width = options.width ?? 64
  const height = options.height ?? 64
  const wrap = options.wrap ?? true
  const start = options.start ?? 'soup'
  const density = options.density ?? 0.3
  const window = options.window ?? 64
  positiveInt('lifeAutomaton', 'width', width)
  positiveInt('lifeAutomaton', 'height', height)
  positiveInt('lifeAutomaton', 'window', window)
  unit('lifeAutomaton', 'density', density)
  let placed: Uint8Array | undefined
  if (start instanceof Uint8Array) {
    if (start.length !== width * height)
      throw new MdmlError('shape', 'lifeAutomaton: start cells must be width x height')
    placed = start
  } else if (typeof start === 'object') placed = placePattern(start, width, height)
  else if (start !== 'soup') {
    const named = LIFE_PATTERNS[start]
    if (named === undefined) throw new MdmlError('value', `lifeAutomaton: no pattern named '${start}'`)
    placed = placePattern(named, width, height)
  }
  return {
    name: 'dynamics/automata/life',
    config: {
      rule: formatLifeRule(rule),
      width,
      height,
      wrap,
      start: typeof start === 'string' ? start : typeof start === 'object' && 'name' in start ? start.name : 'cells',
      density,
      window,
    },
    schema: contractSchema(lifeContract),
    init: (key) => {
      let cells: Uint8Array
      if (placed !== undefined) cells = Uint8Array.from(placed)
      else {
        const u = draws(key, width * height)
        cells = Uint8Array.from({ length: width * height }, (_, i) => ((u[i] ?? 1) < density ? 1 : 0))
      }
      return { cells, generation: 0, births: 0, deaths: 0, recent: [hashBoard(cells)], period: 0 }
    },
    step: (s, ctx) => {
      const next = lifeStep(s.cells, width, height, rule, wrap)
      const recent = [...s.recent, hashBoard(next.cells)].slice(-window)
      const found = s.period === 0 ? detectPeriod(recent) : s.period
      if (s.period === 0 && found > 0) ctx.emit('life/stabilised', { period: found, generation: s.generation + 1 })
      return {
        cells: next.cells,
        generation: s.generation + 1,
        births: next.births,
        deaths: next.deaths,
        recent,
        period: found,
      }
    },
    observe: (s) => ({
      metrics: {
        'life/population': population(s.cells),
        'life/births': s.births,
        'life/deaths': s.deaths,
        'life/period': s.period,
      },
      tensors: { 'life/board': { role: 'state', value: s.cells, shape: [height, width] } },
    }),
    ...codec<LifeState>(),
  }
}
