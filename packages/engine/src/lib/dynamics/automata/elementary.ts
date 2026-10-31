/**
 * Elementary cellular automata, Wolfram's 256 rules: a row of cells, each 0 or 1, replaced by a rule of its own state
 * and its two neighbours' (the rule number's bits are the table). The rule table, a step and a whole space-time board,
 * the symmetries of the rule space (mirror and complement), a rule's class by simulation, and the automaton as a traced
 * algorithm that emits one row per step with its density and entropy. Rule 90 (Pascal's triangle mod 2), rule 30 (a
 * random bit stream in its centre column), rule 110 (universal) and rule 184 (traffic) are the famous ones.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, key as makeKey } from '../../random/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  derived,
  scalar,
  tensor,
} from '../../trace/index.ts'
import { codec, draws, mod, positiveInt, unit } from './_core.ts'

/** What lies beyond the ends of the row: the ring (`periodic`) or a wall of dead cells (`zero`). */
export type ElementaryBoundary = 'periodic' | 'zero'

/** Options of {@link elementaryAutomaton}. */
export interface ElementaryOptions {
  /** The rule number, 0 to 255 (default 30). */
  readonly rule?: number
  /** The width of the row (default 101). */
  readonly width?: number
  /**
   * The first row: `'single'` (one live cell in the middle), `'random'` (each cell live with chance `density`), or the
   * cells themselves (default `'single'`).
   */
  readonly start?: 'single' | 'random' | readonly number[]
  /** The chance a cell is live in a `'random'` start (default 0.5). */
  readonly density?: number
  /** The ends (default `'periodic'`). */
  readonly boundary?: ElementaryBoundary
}

/** The state of {@link elementaryAutomaton}: the current row and how many generations have passed. */
export interface ElementaryState {
  readonly row: Uint8Array
  readonly generation: number
}

/** The paths of {@link elementaryAutomaton}. */
export type ElementaryPath = 'automaton/row' | 'automaton/density' | 'automaton/entropy' | 'rule/icon'

function checkRule(name: string, rule: number): void {
  if (!(Number.isInteger(rule) && rule >= 0 && rule <= 255))
    throw new MdmlError('value', `${name}: the rule must be an integer in 0..255, got ${rule}`)
}

/**
 * The table of an elementary rule: entry `4 l + 2 c + r` is the new state of a cell in state `c` with left neighbour
 * `l` and right neighbour `r`, which is bit `4 l + 2 c + r` of the rule number.
 *
 * @param rule - The rule number, 0 to 255.
 * @returns Eight entries, each 0 or 1.
 * @throws MdmlError with code `value` for a rule outside 0..255.
 * @concept elementary-cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * Array.from(automata.ruleTable(30)) // => [0, 1, 1, 1, 1, 0, 0, 0]
 * ```
 */
export function ruleTable(rule: number): Uint8Array {
  checkRule('ruleTable', rule)
  const t = new Uint8Array(8)
  for (let i = 0; i < 8; i++) t[i] = (rule >> i) & 1
  return t
}

/**
 * One generation of an elementary automaton.
 *
 * @param row - The cells, each 0 or 1.
 * @param rule - The rule number, 0 to 255.
 * @param boundary - The ends of the row (default periodic).
 * @returns The next row.
 * @throws MdmlError with code `value` for a rule outside 0..255.
 * @concept elementary-cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * Array.from(automata.elementaryStep(Uint8Array.of(0, 0, 1, 0, 0), 90)) // => [0, 1, 0, 1, 0]
 * ```
 */
export function elementaryStep(row: Uint8Array, rule: number, boundary: ElementaryBoundary = 'periodic'): Uint8Array {
  const table = ruleTable(rule)
  const n = row.length
  const out = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    const l = boundary === 'periodic' ? (row[mod(i - 1, n)] ?? 0) : i > 0 ? (row[i - 1] ?? 0) : 0
    const r = boundary === 'periodic' ? (row[mod(i + 1, n)] ?? 0) : i < n - 1 ? (row[i + 1] ?? 0) : 0
    out[i] = table[4 * l + 2 * (row[i] ?? 0) + r] ?? 0
  }
  return out
}

/**
 * The space-time board of an elementary automaton from a single live cell in the middle of a row wide enough that the
 * pattern never reaches the ends (the width is `2 generations + 1`): row `g` is generation `g`, so the board is the
 * picture of the rule.
 *
 * @param rule - The rule number, 0 to 255.
 * @param generations - How many generations after the first row.
 * @returns A flat `(generations + 1) x (2 generations + 1)` board, row-major, and its width.
 * @throws MdmlError with code `value` for a rule outside 0..255 or a negative generation count.
 * @concept elementary-cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * const { board, width } = automata.elementaryBoard(90, 3)
 * Array.from(board.slice(3 * width, 4 * width)) // => [1, 0, 1, 0, 1, 0, 1]
 * ```
 */
export function elementaryBoard(rule: number, generations: number): { board: Uint8Array; width: number } {
  checkRule('elementaryBoard', rule)
  if (!(Number.isInteger(generations) && generations >= 0))
    throw new MdmlError('value', `elementaryBoard: generations must be a non-negative integer, got ${generations}`)
  const width = 2 * generations + 1
  const board = new Uint8Array((generations + 1) * width)
  let row: Uint8Array = new Uint8Array(width)
  row[generations] = 1
  for (let g = 0; g <= generations; g++) {
    board.set(row, g * width)
    row = elementaryStep(row, rule, 'zero')
  }
  return { board, width }
}

/**
 * The rule that does what `rule` does on the mirror image of every row (left and right neighbours swapped).
 *
 * @param rule - The rule number.
 * @returns The mirrored rule number (30 and 86 are mirrors; 90 is its own).
 * @concept elementary-cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * [automata.mirrorRule(30), automata.mirrorRule(110), automata.mirrorRule(90)] // => [86, 124, 90]
 * ```
 */
export function mirrorRule(rule: number): number {
  const t = ruleTable(rule)
  let out = 0
  for (let l = 0; l < 2; l++)
    for (let c = 0; c < 2; c++) for (let r = 0; r < 2; r++) out |= (t[4 * l + 2 * c + r] ?? 0) << (4 * r + 2 * c + l)
  return out
}

/**
 * The rule that does what `rule` does with live and dead swapped (the picture in negative).
 *
 * @param rule - The rule number.
 * @returns The complemented rule number (30 and 135 are complements).
 * @concept elementary-cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * [automata.complementRule(30), automata.complementRule(110), automata.complementRule(0)] // => [135, 137, 255]
 * ```
 */
export function complementRule(rule: number): number {
  const t = ruleTable(rule)
  let out = 0
  for (let i = 0; i < 8; i++) out |= (1 - (t[7 - i] ?? 0)) << i
  return out
}

/**
 * The number of live cells of rule 90 from a single live cell after `n` generations: `2^popcount(n)`, since the rows
 * are Pascal's triangle modulo 2 (Sierpinski's triangle).
 *
 * @param n - The generation.
 * @returns The live cells, exactly.
 * @concept elementary-cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * [0, 1, 2, 3, 4, 7, 8].map(automata.rule90Cells) // => [1, 2, 2, 4, 2, 8, 2]
 * ```
 */
export function rule90Cells(n: number): number {
  let bits = 0
  for (let m = n; m > 0; m = Math.floor(m / 2)) bits += m % 2
  return 2 ** bits
}

/** Wolfram's classes: 1 dies out or fills in, 2 settles into cycles, 3 is chaotic, 4 mixes order and chaos. */
export type WolframClass = 1 | 2 | 3 | 4

function isShifted(a: Uint8Array, b: Uint8Array): boolean {
  const n = a.length
  for (let s = 0; s < n; s++) {
    let same = true
    for (let i = 0; i < n && same; i++) same = a[i] === b[(i + s) % n]
    if (same) return true
  }
  return false
}

/**
 * Wolfram's class of a rule, found by running it: from a random ring (width 101, 300 generations), class 1 if the row
 * ends uniform, class 2 if it ends as a cyclic shift of one of the last 48 rows (a cycle, possibly a moving one),
 * class 3 otherwise. Class 4 is told by name, not by running: rule 110 and its mirror and complement images (124, 137,
 * 193), the one universal family; the test cannot tell class 4 from 3 on a finite ring.
 *
 * @param rule - The rule number.
 * @param seed - The seed of the random ring (default 0), so the answer is reproducible.
 * @returns The class, 1 to 4.
 * @throws MdmlError with code `value` for a rule outside 0..255.
 * @concept elementary-cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * [0, 204, 30, 110].map((r) => automata.wolframClass(r)) // => [1, 2, 3, 4]
 * ```
 */
export function wolframClass(rule: number, seed = 0): WolframClass {
  checkRule('wolframClass', rule)
  if ([110, 124, 137, 193].includes(rule)) return 4
  const width = 101
  const u = draws(foldIn(makeKey(seed), rule), width)
  let row: Uint8Array = Uint8Array.from(u.subarray(0, width), (x) => (x < 0.5 ? 1 : 0))
  const recent: Uint8Array[] = []
  for (let g = 0; g < 300; g++) {
    row = elementaryStep(row, rule)
    recent.push(row)
    if (recent.length > 49) recent.shift()
  }
  if (row.every((c) => c === row[0])) return 1
  for (let k = 0; k < recent.length - 1; k++) if (isShifted(row, recent[k] as Uint8Array)) return 2
  return 3
}

/** The rule as its eight neighbourhoods, from 111 down to 000, and what each becomes (Wolfram's icon). */
function ruleIcon(rule: number): { neighbourhood: number[]; becomes: number }[] {
  const table = ruleTable(rule)
  return Array.from({ length: 8 }, (_, k) => {
    const i = 7 - k
    return { neighbourhood: [(i >> 2) & 1, (i >> 1) & 1, i & 1], becomes: table[i] ?? 0 }
  })
}

/**
 * The log contract of {@link elementaryAutomaton} for a rule: the row, its density and entropy, and the rule's icon.
 *
 * @param rule - The rule number, 0 to 255.
 * @returns The contract.
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * Object.keys(automata.elementaryContract(30)) // => ['automaton/row', 'automaton/density', 'automaton/entropy', 'rule/icon']
 * ```
 */
export function elementaryContract(rule: number): Contract<ElementaryState, ElementaryPath> {
  return defineContract({
    'automaton/row': tensor({ role: 'state', shape: ['width'], description: 'the cells of the current row' }),
    'automaton/density': scalar({ description: 'the share of live cells in the row' }),
    'automaton/entropy': scalar({ unit: 'bits', description: "the binary entropy of the row's density" }),
    'rule/icon': derived<ElementaryState>({
      description: 'the rule as its eight neighbourhoods and what each becomes',
      compute: () => ruleIcon(rule),
    }),
  })
}

function density(row: Uint8Array): number {
  let live = 0
  for (const c of row) live += c
  return row.length === 0 ? 0 : live / row.length
}

/**
 * The binary entropy, in bits, of a share.
 *
 * @param p - A share in [0, 1].
 * @returns -p log2 p - (1 - p) log2 (1 - p), with 0 log 0 = 0.
 * @concept elementary-cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * [automata.bitEntropy(0), automata.bitEntropy(0.5), automata.bitEntropy(1)] // => [0, 1, 0]
 * ```
 */
export function bitEntropy(p: number): number {
  const h = (x: number): number => (x > 0 ? -x * Math.log2(x) : 0)
  return h(p) + h(1 - p)
}

/**
 * An elementary cellular automaton as a traced algorithm: one generation per step. The state is the row and the
 * generation (it serialises), and each step is a pure function of it, so a run can be scrubbed and replayed. It logs
 * the row (`automaton/row`), its density and its entropy, and the rule's table (`rule/icon`) as a derived path.
 *
 * @param options - The rule, width, start, density and boundary.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a rule outside 0..255, a width below 1 or a density outside [0, 1].
 * @concept elementary-cellular-automata
 * @example
 * ```ts
 * import { automata } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(automata.elementaryAutomaton({ rule: 90, width: 9 }), { seed: 0, maxSteps: 3 })
 * Array.from(run.state.row) // => [0, 1, 0, 1, 0, 1, 0, 1, 0]
 * ```
 */
export function elementaryAutomaton(options: ElementaryOptions = {}): Algorithm<ElementaryState, ElementaryPath> {
  const rule = options.rule ?? 30
  const width = options.width ?? 101
  const boundary = options.boundary ?? 'periodic'
  const start = options.start ?? 'single'
  const p = options.density ?? 0.5
  checkRule('elementaryAutomaton', rule)
  positiveInt('elementaryAutomaton', 'width', width)
  unit('elementaryAutomaton', 'density', p)
  if (Array.isArray(start) && start.length !== width)
    throw new MdmlError('shape', 'elementaryAutomaton: an explicit start must have `width` cells')
  return {
    name: 'dynamics/automata/elementary',
    config: { rule, width, boundary, start: typeof start === 'string' ? start : 'explicit', density: p },
    schema: contractSchema(elementaryContract(rule)),
    init: (key) => {
      const row = new Uint8Array(width)
      if (start === 'single') row[Math.floor(width / 2)] = 1
      else if (start === 'random') {
        const u = draws(key, width)
        for (let i = 0; i < width; i++) row[i] = (u[i] ?? 1) < p ? 1 : 0
      } else for (let i = 0; i < width; i++) row[i] = (start[i] ?? 0) ? 1 : 0
      return { row, generation: 0 }
    },
    step: (s) => ({ row: elementaryStep(s.row, rule, boundary), generation: s.generation + 1 }),
    observe: (s) => ({
      metrics: { 'automaton/density': density(s.row), 'automaton/entropy': bitEntropy(density(s.row)) },
      tensors: { 'automaton/row': { role: 'state', value: s.row, shape: [width] } },
    }),
    ...codec<ElementaryState>(),
  }
}
