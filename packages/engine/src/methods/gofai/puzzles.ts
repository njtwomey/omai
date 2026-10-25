/**
 * Named puzzles as constraint satisfaction problems (AIMA chapter 6): n-queens, Sudoku (an easy grid and one that needs
 * search) and map colouring, with Australia's states and random planar maps.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { type Key, uniform } from '../../lib/random/index.ts'
import { allDifferent, binaryConstraint, type Constraint, type Csp, csp } from '../../lib/search/index.ts'

/**
 * The n-queens puzzle as a CSP: one variable per column (`'0'` to `'n-1'`), valued by the row of its queen, with no two
 * queens sharing a row or a diagonal.
 *
 * @param n - The board size (at least 1).
 * @returns The CSP.
 */
export function nQueens(n: number): Csp<number> {
  if (!(Number.isInteger(n) && n >= 1)) throw new WhiteboxError('value', `nQueens: n = ${n} must be a positive integer`)
  const cols = Array.from({ length: n }, (_, i) => String(i))
  const rows = Array.from({ length: n }, (_, i) => i)
  const constraints = cols.flatMap((a, i) =>
    cols.slice(i + 1).map((b, k) => binaryConstraint<number>(a, b, (r, s) => r !== s && Math.abs(r - s) !== k + 1)),
  )
  return csp(cols, Object.fromEntries(cols.map((c) => [c, rows])), constraints)
}

/** Sudoku puzzles, row by row with `.` for blanks: an easy one (AIMA's), and a harder one that needs search. */
export const SUDOKUS: Readonly<Record<'easy' | 'hard' | 'mini', string>> = {
  easy: '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..',
  hard: '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......',
  mini: '1..4..1..2..3..2',
}

/**
 * A Sudoku as a CSP: one variable per cell (`'0'` to `'80'`, row by row), digits 1 to 9 (1 to 4 for a 4 by 4 grid),
 * all different along rows, columns and boxes; givens have one value.
 *
 * @param grid - The puzzle, 81 (or 16) characters, `.` or `0` for a blank.
 * @returns The CSP.
 * @throws WhiteboxError `value` for a grid of the wrong length.
 */
export function sudoku(grid: string): Csp<number> {
  const side = grid.length === 16 ? 4 : grid.length === 81 ? 9 : 0
  if (side === 0) throw new WhiteboxError('value', `sudoku: a grid has 16 or 81 cells, not ${grid.length}`)
  const box = side === 4 ? 2 : 3
  const cells = Array.from({ length: side * side }, (_, i) => String(i))
  const digits = Array.from({ length: side }, (_, i) => i + 1)
  const domains = Object.fromEntries(
    cells.map((c, i) => {
      const d = Number(grid[i])
      return [c, d >= 1 && d <= side ? [d] : digits]
    }),
  )
  const units: string[][] = []
  for (let r = 0; r < side; r++) units.push(digits.map((_, c) => String(side * r + c)))
  for (let c = 0; c < side; c++) units.push(digits.map((_, r) => String(side * r + c)))
  for (let br = 0; br < box; br++)
    for (let bc = 0; bc < box; bc++) {
      const unit: string[] = []
      for (let r = 0; r < box; r++)
        for (let c = 0; c < box; c++) unit.push(String(side * (box * br + r) + box * bc + c))
      units.push(unit)
    }
  const seen = new Set<string>()
  const pairs: Constraint<number>[] = []
  for (const unit of units)
    for (const p of allDifferent<number>(unit)) {
      const id = [...p.scope].sort().join(',')
      if (seen.has(id)) continue
      seen.add(id)
      pairs.push(p)
    }
  return csp(cells, domains, pairs)
}

/** A map to colour: regions with drawing positions (y down) and the borders between them. */
export interface ColouringMap {
  readonly regions: readonly string[]
  readonly positions: readonly (readonly [number, number])[]
  readonly borders: readonly (readonly [string, string])[]
}

/** The states and territories of Australia (AIMA figure 6.1). */
export const AUSTRALIA: ColouringMap = {
  regions: ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T'],
  positions: [
    [1, 2.4],
    [3, 1.2],
    [3.3, 3],
    [5, 1.5],
    [5.4, 3.3],
    [4.8, 4.4],
    [5.1, 5.6],
  ],
  borders: [
    ['WA', 'NT'],
    ['WA', 'SA'],
    ['NT', 'SA'],
    ['NT', 'Q'],
    ['SA', 'Q'],
    ['SA', 'NSW'],
    ['SA', 'V'],
    ['Q', 'NSW'],
    ['NSW', 'V'],
  ],
}

/**
 * A random map: regions at random points of the unit square, each bordering its nearest neighbours (borders kept
 * when they do not cross one already drawn, so the map stays planar and four colours always suffice).
 *
 * @param key - The key.
 * @param n - The number of regions (default 12).
 * @param nearest - Neighbours each region tries to border (default 4).
 * @returns The map, regions named `R1` to `Rn`.
 */
export function randomMap(key: Key, n = 12, nearest = 4): ColouringMap {
  const u = Array.from(uniform(key, [n, 2]).data)
  const at = (i: number): [number, number] => [u[2 * i] ?? 0, u[2 * i + 1] ?? 0]
  const regions = Array.from({ length: n }, (_, i) => `R${i + 1}`)
  const candidates: [number, number, number][] = []
  for (let i = 0; i < n; i++) {
    const [xi, yi] = at(i)
    const near = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .map((j) => [j, Math.hypot(xi - at(j)[0], yi - at(j)[1])] as const)
      .sort((p, q) => p[1] - q[1])
      .slice(0, nearest)
    for (const [j, d] of near) candidates.push([Math.min(i, j), Math.max(i, j), d])
  }
  candidates.sort((p, q) => p[2] - q[2])
  const crosses = (a: number, b: number, c: number, d: number): boolean => {
    if (a === c || a === d || b === c || b === d) return false
    const o = (p: number, q: number, r: number): number => {
      const [px, py] = at(p)
      const [qx, qy] = at(q)
      const [rx, ry] = at(r)
      return Math.sign((qx - px) * (ry - py) - (qy - py) * (rx - px))
    }
    return o(a, b, c) !== o(a, b, d) && o(c, d, a) !== o(c, d, b)
  }
  const kept: [number, number][] = []
  const seen = new Set<string>()
  for (const [a, b] of candidates) {
    const id = `${a},${b}`
    if (seen.has(id)) continue
    seen.add(id)
    if (kept.some(([c, d]) => crosses(a, b, c, d))) continue
    kept.push([a, b])
  }
  return {
    regions,
    positions: regions.map((_, i) => at(i)),
    borders: kept.map(([a, b]) => [regions[a] ?? '', regions[b] ?? ''] as const),
  }
}

/**
 * Map colouring as a CSP: one variable per region, neighbouring regions in different colours.
 *
 * @param map - The regions and borders.
 * @param colours - The colours (default red, green and blue).
 * @returns The CSP.
 */
export function mapColouring(map: ColouringMap, colours: readonly string[] = ['red', 'green', 'blue']): Csp<string> {
  return csp(
    map.regions,
    Object.fromEntries(map.regions.map((r) => [r, [...colours]])),
    map.borders.map(([a, b]) => binaryConstraint<string>(a, b, (x, y) => x !== y)),
  )
}
