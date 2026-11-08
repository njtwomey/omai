/**
 * Tours: travelling-salesman instances whose optimal tours are known, and Hamiltonian-cycle puzzles with a planted cycle.
 * The optimum of a small instance is computed exactly by the Held-Karp dynamic programme (`2^n n^2`); the circle and the
 * grid have optima known in closed form, so a heuristic's tour can be scored against the truth at any size.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'

/** A travelling-salesman instance: city coordinates, the distance matrix and the best tour known with its length. */
export interface TspInstance {
  readonly name: string
  /** City coordinates, `[n][2]`. */
  readonly cities: readonly (readonly [number, number])[]
  /** The Euclidean distance matrix, row-major `n x n`. */
  readonly distances: Float64Array
  /** A tour of optimal length: a permutation of the cities. */
  readonly optimalTour: readonly number[]
  /** Its length. */
  readonly optimalLength: number
}

/**
 * The length of a closed tour.
 *
 * @param distances - The row-major `n x n` distance matrix.
 * @param tour - The order the cities are visited in.
 * @returns The length, returning to the start.
 * @example
 * ```ts
 * import { tourLength } from 'omai-engine/problems'
 * tourLength(Float64Array.of(0, 1, 2, 1, 0, 1, 2, 1, 0), [0, 1, 2]) // => 4
 * ```
 */
export function tourLength(distances: ArrayLike<number>, tour: readonly number[]): number {
  const n = tour.length
  let total = 0
  for (let i = 0; i < n; i++) total += distances[(tour[i] ?? 0) * n + (tour[(i + 1) % n] ?? 0)] ?? 0
  return total
}

/**
 * The exact optimal tour by the Held-Karp dynamic programme over subsets: `O(2^n n^2)` time, so up to about 15 cities.
 *
 * @param distances - The row-major `n x n` distance matrix.
 * @param n - The number of cities.
 * @returns An optimal tour starting at city 0, and its length.
 * @throws OmaiError with code `value` for more than 16 cities.
 * @example
 * ```ts
 * import { heldKarp } from 'omai-engine/problems'
 * heldKarp(Float64Array.of(0, 1, 5, 1, 0, 1, 5, 1, 0), 3).length // => 7
 * ```
 */
export function heldKarp(
  distances: ArrayLike<number>,
  n: number,
): { readonly tour: number[]; readonly length: number } {
  if (n > 16) throw new OmaiError('value', `heldKarp: ${n} cities is too many for the exact method`)
  if (n === 1) return { tour: [0], length: 0 }
  const m = n - 1
  const full = 1 << m
  const best = new Float64Array(full * m).fill(Number.POSITIVE_INFINITY)
  const from = new Int8Array(full * m).fill(-1)
  const d = (a: number, b: number): number => distances[a * n + b] ?? 0
  for (let j = 0; j < m; j++) best[(1 << j) * m + j] = d(0, j + 1)
  for (let set = 1; set < full; set++) {
    for (let j = 0; j < m; j++) {
      if ((set & (1 << j)) === 0) continue
      const here = best[set * m + j] ?? Number.POSITIVE_INFINITY
      if (!Number.isFinite(here)) continue
      for (let k = 0; k < m; k++) {
        if ((set & (1 << k)) !== 0) continue
        const next = set | (1 << k)
        const cost = here + d(j + 1, k + 1)
        if (cost < (best[next * m + k] ?? Number.POSITIVE_INFINITY)) {
          best[next * m + k] = cost
          from[next * m + k] = j
        }
      }
    }
  }
  let length = Number.POSITIVE_INFINITY
  let last = 0
  for (let j = 0; j < m; j++) {
    const cost = (best[(full - 1) * m + j] ?? Number.POSITIVE_INFINITY) + d(j + 1, 0)
    if (cost < length) {
      length = cost
      last = j
    }
  }
  const tour: number[] = []
  let set = full - 1
  for (let j = last; j >= 0; ) {
    tour.push(j + 1)
    const previous = from[set * m + j] ?? -1
    set &= ~(1 << j)
    j = previous
  }
  return { tour: [0, ...tour.reverse()], length }
}

function instanceOf(
  name: string,
  cities: readonly (readonly [number, number])[],
  optimal?: readonly number[],
): TspInstance {
  const n = cities.length
  const distances = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      distances[i * n + j] = Math.hypot(
        (cities[i]?.[0] ?? 0) - (cities[j]?.[0] ?? 0),
        (cities[i]?.[1] ?? 0) - (cities[j]?.[1] ?? 0),
      )
  if (optimal !== undefined)
    return { name, cities, distances, optimalTour: optimal, optimalLength: tourLength(distances, optimal) }
  const exact = heldKarp(distances, n)
  return { name, cities, distances, optimalTour: exact.tour, optimalLength: exact.length }
}

/**
 * `n` cities evenly spaced on a circle of radius 1: the optimal tour visits them in order, and its length is the
 * perimeter of the regular polygon, `2 n sin(pi / n)`, at any size.
 *
 * @param n - The number of cities (at least 3).
 * @returns The instance.
 * @example
 * ```ts
 * import { circleTour } from 'omai-engine/problems'
 * circleTour(4).optimalLength // => 5.656854249492381
 * ```
 */
export function circleTour(n: number): TspInstance {
  if (!(Number.isInteger(n) && n >= 3)) throw new OmaiError('value', `circleTour: ${n} cities`)
  const cities = Array.from(
    { length: n },
    (_, i) => [Math.cos((2 * Math.PI * i) / n), Math.sin((2 * Math.PI * i) / n)] as const,
  )
  return instanceOf(
    `circle-${n}`,
    cities,
    Array.from({ length: n }, (_, i) => i),
  )
}

/**
 * The cities of a `rows x cols` lattice with unit spacing, `rows` even or `cols` even: a Hamiltonian cycle of unit steps
 * exists, so the optimal tour has length `rows cols`.
 *
 * @param rows - The rows (at least 2).
 * @param cols - The columns (at least 2).
 * @returns The instance.
 * @throws OmaiError with code `value` when both sides are odd (no tour of unit steps exists).
 * @example
 * ```ts
 * import { gridTour } from 'omai-engine/problems'
 * gridTour(4, 5).optimalLength // => 20
 * ```
 */
export function gridTour(rows: number, cols: number): TspInstance {
  if (rows < 2 || cols < 2 || (rows % 2 === 1 && cols % 2 === 1))
    throw new OmaiError('value', `gridTour: a ${rows} x ${cols} lattice has no tour of unit steps`)
  const cities: (readonly [number, number])[] = []
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) cities.push([c, r])
  const at = (r: number, c: number): number => r * cols + c
  // Down the first column's neighbour and snake along the other columns, returning along column 0 (rows even), or the transpose.
  const tour: number[] = []
  if (rows % 2 === 0) {
    for (let c = 0; c < cols; c++) tour.push(at(0, c))
    for (let r = 1; r < rows; r++) {
      const row =
        r % 2 === 1
          ? Array.from({ length: cols - 1 }, (_, i) => cols - 1 - i)
          : Array.from({ length: cols - 1 }, (_, i) => i + 1)
      for (const c of row) tour.push(at(r, c))
    }
    for (let r = rows - 1; r >= 1; r--) tour.push(at(r, 0))
  } else {
    const t = gridTour(cols, rows)
    return instanceOf(
      `grid-${rows}x${cols}`,
      cities,
      t.optimalTour.map((i) => at(i % rows, Math.floor(i / rows))),
    )
  }
  return instanceOf(`grid-${rows}x${cols}`, cities, tour)
}

/**
 * `n` cities drawn uniformly in the unit square, with the exact optimum by Held-Karp.
 *
 * @param key - The key.
 * @param n - The number of cities (3 to 13 keeps it fast).
 * @returns The instance.
 * @example
 * ```ts
 * import { randomTour } from 'omai-engine/problems'
 * import { key } from 'omai-engine/random'
 * randomTour(key(0), 8).optimalTour.length // => 8
 * ```
 */
export function randomTour(key: Key, n = 10): TspInstance {
  const u = uniform(foldIn(key, 0), [n, 2]).data
  const cities = Array.from({ length: n }, (_, i) => [u[2 * i] ?? 0, u[2 * i + 1] ?? 0] as const)
  return instanceOf(`random-${n}`, cities)
}

/** A Hamiltonian-cycle puzzle: a graph, the cycle planted in it and the verdict. */
export interface HamiltonianPuzzle {
  readonly n: number
  /** Adjacency lists. */
  readonly adjacency: readonly (readonly number[])[]
  /** The cycle planted, as a permutation of the vertices (every consecutive pair is an edge). */
  readonly planted: readonly number[]
}

/**
 * Whether an order of the vertices is a Hamiltonian cycle of a graph.
 *
 * @param adjacency - Adjacency lists.
 * @param order - The vertices in cycle order.
 * @returns Whether every vertex appears once and consecutive vertices (the last and first too) are adjacent.
 * @example
 * ```ts
 * import { isHamiltonianCycle } from 'omai-engine/problems'
 * isHamiltonianCycle([[1, 2], [0, 2], [0, 1]], [0, 1, 2]) // => true
 * ```
 */
export function isHamiltonianCycle(adjacency: readonly (readonly number[])[], order: readonly number[]): boolean {
  const n = adjacency.length
  if (order.length !== n || new Set(order).size !== n) return false
  return order.every((v, i) => adjacency[v]?.includes(order[(i + 1) % n] ?? -1) === true)
}

/**
 * A graph with a Hamiltonian cycle planted in it: a random cyclic order of the vertices joined as a ring, plus `extra`
 * further random edges as decoys.
 *
 * @param key - The key.
 * @param n - The number of vertices (at least 3).
 * @param extra - The number of decoy edges (default `n`).
 * @returns The graph and the planted cycle.
 * @example
 * ```ts
 * import { isHamiltonianCycle, plantedHamiltonian } from 'omai-engine/problems'
 * import { key } from 'omai-engine/random'
 * const p = plantedHamiltonian(key(0), 8)
 * isHamiltonianCycle(p.adjacency, p.planted) // => true
 * ```
 */
export function plantedHamiltonian(key: Key, n: number, extra: number = n): HamiltonianPuzzle {
  if (!(Number.isInteger(n) && n >= 3)) throw new OmaiError('value', `plantedHamiltonian: ${n} vertices`)
  const u = uniform(foldIn(key, 0), [n]).data
  const planted = Array.from({ length: n }, (_, i) => i).sort((a, b) => (u[a] ?? 0) - (u[b] ?? 0))
  const edges = new Set<string>()
  const add = (a: number, b: number): void => {
    if (a !== b) edges.add(a < b ? `${a},${b}` : `${b},${a}`)
  }
  for (let i = 0; i < n; i++) add(planted[i] ?? 0, planted[(i + 1) % n] ?? 0)
  const v = uniform(foldIn(key, 1), [2 * extra]).data
  for (let e = 0; e < extra; e++) add(Math.floor((v[2 * e] ?? 0) * n), Math.floor((v[2 * e + 1] ?? 0) * n))
  const adjacency: number[][] = Array.from({ length: n }, () => [])
  for (const edge of edges) {
    const [a, b] = edge.split(',').map(Number) as [number, number]
    adjacency[a]?.push(b)
    adjacency[b]?.push(a)
  }
  return { n, adjacency, planted }
}

/**
 * A Hamiltonian cycle of a graph by backtracking, or `undefined` when there is none (exact, for graphs up to a few dozen
 * vertices).
 *
 * @param adjacency - Adjacency lists.
 * @returns A cycle starting at vertex 0.
 * @example
 * ```ts
 * import { findHamiltonianCycle } from 'omai-engine/problems'
 * findHamiltonianCycle([[1], [0, 2], [1]]) // => undefined
 * ```
 */
export function findHamiltonianCycle(adjacency: readonly (readonly number[])[]): number[] | undefined {
  const n = adjacency.length
  const path = [0]
  const used = new Uint8Array(n)
  used[0] = 1
  const extend = (): boolean => {
    if (path.length === n) return adjacency[path[n - 1] ?? 0]?.includes(0) === true
    for (const next of adjacency[path[path.length - 1] ?? 0] ?? []) {
      if (used[next] === 1) continue
      used[next] = 1
      path.push(next)
      if (extend()) return true
      path.pop()
      used[next] = 0
    }
    return false
  }
  return extend() ? path : undefined
}
