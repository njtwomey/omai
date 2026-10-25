/**
 * The variation and selection operators of evolutionary algorithms: tournament, roulette, stochastic universal, rank
 * and truncation selection; crossovers for fixed-length genomes (one-point, two-point, uniform), permutations (order,
 * PMX, cycle) and real vectors (blend, simulated binary, arithmetic); and bit-flip, swap, inversion, scramble, Gaussian
 * and polynomial mutations, all driven by explicit keys.
 */

import { MdmlError } from '../../base/index.ts'
import type { Key } from '../../random/index.ts'
import { draws } from '../_draw.ts'

// ------------------------------------------------------------------------------------------------ selection

function checkCount(n: number, name: string): void {
  if (!(Number.isInteger(n) && n >= 0)) throw new MdmlError('value', `${name}: n = ${n} must be a non-negative integer`)
}

/**
 * Tournament selection: each pick is the fittest of `size` individuals drawn uniformly with replacement (ties to the
 * first drawn).
 *
 * @param key - The random key.
 * @param fitness - The fitness of each individual (higher is better).
 * @param n - The number of picks.
 * @param size - The tournament size (default 3).
 * @returns The indices picked.
 * @throws MdmlError with code `value` when the population is empty or a count is invalid.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { tournamentSelection } from 'mdml-engine/search'
 * tournamentSelection(key(0), [1, 5, 2], 4, 3).length // => 4
 * tournamentSelection(key(0), [1, 5, 2], 2, 50) // => [1, 1]
 * ```
 */
export function tournamentSelection(key: Key, fitness: readonly number[], n: number, size = 3): number[] {
  checkCount(n, 'tournamentSelection')
  if (fitness.length === 0 || !(size >= 1)) throw new MdmlError('value', 'tournamentSelection: empty population')
  const d = draws(key)
  return Array.from({ length: n }, () => {
    let best = d.int(fitness.length)
    for (let i = 1; i < size; i++) {
      const c = d.int(fitness.length)
      if ((fitness[c] ?? 0) > (fitness[best] ?? 0)) best = c
    }
    return best
  })
}

function wheel(key: Key, weights: readonly number[], n: number, universal: boolean): number[] {
  const total = weights.reduce((a, b) => a + b, 0)
  if (!(total > 0)) return Array.from({ length: n }, (_, i) => i % weights.length)
  const d = draws(key)
  const pointers = universal
    ? (() => {
        const start = d.uniform() * (total / n)
        return Array.from({ length: n }, (_, i) => start + (i * total) / n)
      })()
    : Array.from({ length: n }, () => d.uniform() * total)
  return pointers.map((p) => {
    let acc = 0
    for (let i = 0; i < weights.length; i++) {
      acc += weights[i] ?? 0
      if (p < acc) return i
    }
    return weights.length - 1
  })
}

/**
 * Fitness-proportionate (roulette-wheel) selection: each pick is individual i with probability f_i / sum f.
 *
 * @param key - The random key.
 * @param fitness - Non-negative fitnesses.
 * @param n - The number of picks.
 * @returns The indices picked (uniform when every fitness is 0).
 * @throws MdmlError with code `value` when a fitness is negative or the population is empty.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { rouletteSelection } from 'mdml-engine/search'
 * rouletteSelection(key(1), [0, 1, 0], 3) // => [1, 1, 1]
 * ```
 */
export function rouletteSelection(key: Key, fitness: readonly number[], n: number): number[] {
  checkCount(n, 'rouletteSelection')
  if (fitness.length === 0 || fitness.some((f) => !(f >= 0))) {
    throw new MdmlError('value', 'rouletteSelection: needs a non-empty population of non-negative fitnesses')
  }
  return wheel(key, fitness, n, false)
}

/**
 * Stochastic universal sampling (Baker 1987): n equally spaced pointers on one spin of the roulette wheel, so each
 * individual is picked within one of its expected count.
 *
 * @param key - The random key.
 * @param fitness - Non-negative fitnesses.
 * @param n - The number of picks.
 * @returns The indices picked, in wheel order.
 * @throws MdmlError with code `value` when a fitness is negative or the population is empty.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { stochasticUniversalSampling } from 'mdml-engine/search'
 * stochasticUniversalSampling(key(2), [1, 1, 2], 4) // => [0, 1, 2, 2]
 * ```
 */
export function stochasticUniversalSampling(key: Key, fitness: readonly number[], n: number): number[] {
  checkCount(n, 'stochasticUniversalSampling')
  if (fitness.length === 0 || fitness.some((f) => !(f >= 0))) {
    throw new MdmlError('value', 'stochasticUniversalSampling: needs non-negative fitnesses')
  }
  return wheel(key, fitness, n, true)
}

/**
 * Linear ranking selection (Baker 1985): individuals are ranked worst to best and picked with probability
 * (2 - s) / N + 2 r (s - 1) / (N (N - 1)) for rank r, with selection pressure s in [1, 2].
 *
 * @param key - The random key.
 * @param fitness - The fitnesses (any sign).
 * @param n - The number of picks.
 * @param pressure - The selection pressure s (default 1.5).
 * @returns The indices picked.
 * @throws MdmlError with code `value` when the pressure is outside [1, 2] or the population is empty.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { rankSelection } from 'mdml-engine/search'
 * rankSelection(key(0), [-5, 3], 3, 2) // => [1, 1, 1]
 * ```
 */
export function rankSelection(key: Key, fitness: readonly number[], n: number, pressure = 1.5): number[] {
  checkCount(n, 'rankSelection')
  if (fitness.length === 0 || !(pressure >= 1 && pressure <= 2)) {
    throw new MdmlError('value', 'rankSelection: needs a population and a pressure in [1, 2]')
  }
  const N = fitness.length
  const order = fitness.map((f, i) => ({ f, i })).sort((a, b) => a.f - b.f || b.i - a.i)
  const weights = new Array<number>(N).fill(0)
  order.forEach(({ i }, r) => {
    weights[i] = N === 1 ? 1 : (2 - pressure) / N + (2 * r * (pressure - 1)) / (N * (N - 1))
  })
  return wheel(key, weights, n, false)
}

/**
 * Truncation selection: the n fittest individuals (ties to the lower index), as in (mu, lambda) strategies.
 *
 * @param fitness - The fitnesses.
 * @param n - The number to keep.
 * @returns Their indices, fittest first.
 * @throws MdmlError with code `value` when n exceeds the population.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { truncationSelection } from 'mdml-engine/search'
 * truncationSelection([3, 9, 1, 9], 2) // => [1, 3]
 * ```
 */
export function truncationSelection(fitness: readonly number[], n: number): number[] {
  checkCount(n, 'truncationSelection')
  if (n > fitness.length) throw new MdmlError('value', 'truncationSelection: n exceeds the population')
  return fitness
    .map((f, i) => ({ f, i }))
    .sort((a, b) => b.f - a.f || a.i - b.i)
    .slice(0, n)
    .map((x) => x.i)
}

// ------------------------------------------------------------------------------------------------ crossover

function sameLength(a: readonly unknown[], b: readonly unknown[], name: string): void {
  if (a.length !== b.length) throw new MdmlError('shape', `${name}: parents have lengths ${a.length} and ${b.length}`)
}

/**
 * One-point crossover: swap the tails after a random cut (bit strings, or any fixed-length genome).
 *
 * @param key - The random key.
 * @param a - A parent.
 * @param b - Another parent of the same length.
 * @returns Two children.
 * @throws MdmlError with code `shape` when the lengths differ.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { onePointCrossover } from 'mdml-engine/search'
 * const [c, d] = onePointCrossover(key(0), [0, 0, 0, 0], [1, 1, 1, 1])
 * c.reduce((s, x) => s + x, 0) + d.reduce((s, x) => s + x, 0) // => 4
 * ```
 */
export function onePointCrossover<T>(key: Key, a: readonly T[], b: readonly T[]): [T[], T[]] {
  sameLength(a, b, 'onePointCrossover')
  const cut = a.length < 2 ? 0 : 1 + draws(key).int(a.length - 1)
  return [
    [...a.slice(0, cut), ...b.slice(cut)],
    [...b.slice(0, cut), ...a.slice(cut)],
  ]
}

/**
 * Two-point crossover: swap the segment between two random cuts.
 *
 * @param key - The random key.
 * @param a - A parent.
 * @param b - Another parent of the same length.
 * @returns Two children.
 * @throws MdmlError with code `shape` when the lengths differ.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { twoPointCrossover } from 'mdml-engine/search'
 * twoPointCrossover(key(3), [0, 0, 0, 0, 0], [1, 1, 1, 1, 1])[0].length // => 5
 * ```
 */
export function twoPointCrossover<T>(key: Key, a: readonly T[], b: readonly T[]): [T[], T[]] {
  sameLength(a, b, 'twoPointCrossover')
  const d = draws(key)
  let i = d.int(a.length + 1)
  let j = d.int(a.length + 1)
  if (i > j) [i, j] = [j, i]
  return [
    [...a.slice(0, i), ...b.slice(i, j), ...a.slice(j)],
    [...b.slice(0, i), ...a.slice(i, j), ...b.slice(j)],
  ]
}

/**
 * Uniform crossover: each position comes from either parent with probability `p` (and the other child gets the
 * other).
 *
 * @param key - The random key.
 * @param a - A parent.
 * @param b - Another parent of the same length.
 * @param p - The probability of swapping a position (default 0.5).
 * @returns Two children.
 * @throws MdmlError with code `shape` when the lengths differ.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { uniformCrossover } from 'mdml-engine/search'
 * uniformCrossover(key(0), [0, 0, 0], [1, 1, 1], 1) // => [[1, 1, 1], [0, 0, 0]]
 * ```
 */
export function uniformCrossover<T>(key: Key, a: readonly T[], b: readonly T[], p = 0.5): [T[], T[]] {
  sameLength(a, b, 'uniformCrossover')
  const d = draws(key)
  const c: T[] = []
  const e: T[] = []
  a.forEach((x, i) => {
    const y = b[i] as T
    if (d.uniform() < p) {
      c.push(y)
      e.push(x)
    } else {
      c.push(x)
      e.push(y)
    }
  })
  return [c, e]
}

function checkPermutation(p: readonly number[], name: string): void {
  const seen = new Uint8Array(p.length)
  for (const x of p) {
    if (!(Number.isInteger(x) && x >= 0 && x < p.length) || seen[x] === 1) {
      throw new MdmlError('value', `${name}: parents must be permutations of 0 .. n - 1`)
    }
    seen[x] = 1
  }
}

function cuts(key: Key, n: number): [number, number] {
  const d = draws(key)
  let i = d.int(n)
  let j = d.int(n)
  if (i > j) [i, j] = [j, i]
  return [i, j]
}

function ox(a: readonly number[], b: readonly number[], i: number, j: number): number[] {
  const n = a.length
  const child = new Array<number>(n).fill(-1)
  const used = new Uint8Array(n)
  for (let k = i; k <= j; k++) {
    child[k] = a[k] as number
    used[a[k] as number] = 1
  }
  let pos = (j + 1) % n
  for (let t = 0; t < n; t++) {
    const x = b[(j + 1 + t) % n] as number
    if (used[x] === 1) continue
    child[pos] = x
    used[x] = 1
    pos = (pos + 1) % n
  }
  return child
}

/**
 * Order crossover (OX1, Davis 1985) for permutations: each child keeps a slice of one parent and fills the rest in the
 * order the other parent visits the remaining elements, starting after the slice.
 *
 * @param key - The random key.
 * @param a - A permutation of 0 .. n - 1.
 * @param b - Another.
 * @returns Two child permutations.
 * @throws MdmlError with code `value` when a parent is not a permutation.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { orderCrossover } from 'mdml-engine/search'
 * orderCrossover(key(0), [0, 1, 2, 3, 4], [4, 3, 2, 1, 0]).every((c) => [...c].sort().join() === '0,1,2,3,4') // => true
 * ```
 */
export function orderCrossover(key: Key, a: readonly number[], b: readonly number[]): [number[], number[]] {
  sameLength(a, b, 'orderCrossover')
  checkPermutation(a, 'orderCrossover')
  checkPermutation(b, 'orderCrossover')
  if (a.length === 0) return [[], []]
  const [i, j] = cuts(key, a.length)
  return [ox(a, b, i, j), ox(b, a, i, j)]
}

function pmx(a: readonly number[], b: readonly number[], i: number, j: number): number[] {
  const n = a.length
  const child = new Array<number>(n).fill(-1)
  const at = new Int32Array(n).fill(-1)
  for (let k = i; k <= j; k++) {
    child[k] = a[k] as number
    at[a[k] as number] = k
  }
  for (let k = i; k <= j; k++) {
    const x = b[k] as number
    if (at[x] !== -1) continue
    // Follow the mapping from b's segment value through a's segment until a free position outside it.
    let pos = k
    for (;;) {
      const v = a[pos] as number
      pos = b.indexOf(v)
      if (pos < i || pos > j) break
    }
    child[pos] = x
    at[x] = pos
  }
  for (let k = 0; k < n; k++) if (child[k] === -1) child[k] = b[k] as number
  return child
}

/**
 * Partially mapped crossover (PMX, Goldberg and Lingle 1985) for permutations: copy a segment of one parent, then place
 * the other parent's conflicting segment values by following the segment mapping, and copy the rest.
 *
 * @param key - The random key.
 * @param a - A permutation of 0 .. n - 1.
 * @param b - Another.
 * @returns Two child permutations.
 * @throws MdmlError with code `value` when a parent is not a permutation.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { pmxCrossover } from 'mdml-engine/search'
 * pmxCrossover(key(1), [0, 1, 2, 3, 4], [4, 3, 2, 1, 0]).every((c) => [...c].sort().join() === '0,1,2,3,4') // => true
 * ```
 */
export function pmxCrossover(key: Key, a: readonly number[], b: readonly number[]): [number[], number[]] {
  sameLength(a, b, 'pmxCrossover')
  checkPermutation(a, 'pmxCrossover')
  checkPermutation(b, 'pmxCrossover')
  if (a.length === 0) return [[], []]
  const [i, j] = cuts(key, a.length)
  return [pmx(a, b, i, j), pmx(b, a, i, j)]
}

/**
 * Cycle crossover (CX, Oliver, Smith and Holland 1987) for permutations: the positions split into cycles of the two
 * parents; children take alternate cycles from each parent, so every element keeps a parent's position. Deterministic.
 *
 * @param a - A permutation of 0 .. n - 1.
 * @param b - Another.
 * @returns Two child permutations.
 * @throws MdmlError with code `value` when a parent is not a permutation.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { cycleCrossover } from 'mdml-engine/search'
 * cycleCrossover([0, 1, 2, 3, 4, 5, 6, 7], [7, 4, 1, 0, 2, 5, 3, 6]) // => [[0, 4, 1, 3, 2, 5, 6, 7], [7, 1, 2, 0, 4, 5, 3, 6]]
 * ```
 */
export function cycleCrossover(a: readonly number[], b: readonly number[]): [number[], number[]] {
  sameLength(a, b, 'cycleCrossover')
  checkPermutation(a, 'cycleCrossover')
  checkPermutation(b, 'cycleCrossover')
  const n = a.length
  const cycle = new Int32Array(n).fill(-1)
  const where = new Int32Array(n)
  a.forEach((x, i) => {
    where[x] = i
  })
  let c = 0
  for (let start = 0; start < n; start++) {
    if (cycle[start] !== -1) continue
    for (let i = start; cycle[i] === -1; i = where[b[i] as number] ?? 0) cycle[i] = c
    c += 1
  }
  const odd = (i: number): boolean => (cycle[i] ?? 0) % 2 === 1
  return [a.map((x, i) => (odd(i) ? (b[i] as number) : x)), b.map((x, i) => (odd(i) ? (a[i] as number) : x))]
}

/**
 * Blend crossover BLX-alpha (Eshelman and Schaffer 1993) for real vectors: each child coordinate is uniform on the
 * parents' interval widened by alpha times its length on each side.
 *
 * @param key - The random key.
 * @param a - A parent.
 * @param b - Another of the same length.
 * @param alpha - The widening (default 0.5).
 * @returns Two children.
 * @throws MdmlError with code `shape` when the lengths differ.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { blendCrossover } from 'mdml-engine/search'
 * blendCrossover(key(0), [0, 0], [1, 1], 0)[0].every((x) => x >= 0 && x <= 1) // => true
 * ```
 */
export function blendCrossover(
  key: Key,
  a: readonly number[],
  b: readonly number[],
  alpha = 0.5,
): [number[], number[]] {
  sameLength(a, b, 'blendCrossover')
  const d = draws(key)
  const child = (): number[] =>
    a.map((x, i) => {
      const y = b[i] as number
      const lo = Math.min(x, y)
      const span = Math.abs(x - y)
      return lo - alpha * span + d.uniform() * (1 + 2 * alpha) * span
    })
  return [child(), child()]
}

/**
 * Simulated binary crossover (SBX, Deb and Agrawal 1995) for real vectors: children spread around the parents like
 * one-point crossover of binary codes, with distribution index eta (large eta keeps children near the parents),
 * clipped to optional bounds.
 *
 * @param key - The random key.
 * @param a - A parent.
 * @param b - Another of the same length.
 * @param options - The distribution index eta (default 15) and bounds.
 * @returns Two children whose coordinate means equal the parents'.
 * @throws MdmlError with code `shape` when the lengths differ.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { simulatedBinaryCrossover } from 'mdml-engine/search'
 * const [c, d] = simulatedBinaryCrossover(key(0), [1, 2], [3, 2])
 * Math.abs((c[0] ?? 0) + (d[0] ?? 0) - 4) < 1e-12 // => true
 * ```
 */
export function simulatedBinaryCrossover(
  key: Key,
  a: readonly number[],
  b: readonly number[],
  options: { readonly eta?: number; readonly lower?: readonly number[]; readonly upper?: readonly number[] } = {},
): [number[], number[]] {
  sameLength(a, b, 'simulatedBinaryCrossover')
  const eta = options.eta ?? 15
  const d = draws(key)
  const c: number[] = []
  const e: number[] = []
  a.forEach((x, i) => {
    const y = b[i] as number
    const u = d.uniform()
    const beta = u <= 0.5 ? (2 * u) ** (1 / (eta + 1)) : (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    const clip = (v: number): number =>
      Math.min(
        options.upper?.[i] ?? Number.POSITIVE_INFINITY,
        Math.max(options.lower?.[i] ?? Number.NEGATIVE_INFINITY, v),
      )
    c.push(clip(0.5 * ((1 + beta) * x + (1 - beta) * y)))
    e.push(clip(0.5 * ((1 - beta) * x + (1 + beta) * y)))
  })
  return [c, e]
}

/**
 * Arithmetic (whole) crossover for real vectors: children are the convex combinations lambda a + (1 - lambda) b and
 * (1 - lambda) a + lambda b for one uniform lambda.
 *
 * @param key - The random key.
 * @param a - A parent.
 * @param b - Another of the same length.
 * @returns Two children.
 * @throws MdmlError with code `shape` when the lengths differ.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { arithmeticCrossover } from 'mdml-engine/search'
 * const [c, d] = arithmeticCrossover(key(0), [0, 10], [10, 0])
 * (c[0] ?? 0) + (d[0] ?? 0) // => 10
 * ```
 */
export function arithmeticCrossover(key: Key, a: readonly number[], b: readonly number[]): [number[], number[]] {
  sameLength(a, b, 'arithmeticCrossover')
  const t = draws(key).uniform()
  return [a.map((x, i) => t * x + (1 - t) * (b[i] as number)), a.map((x, i) => (1 - t) * x + t * (b[i] as number))]
}

// ------------------------------------------------------------------------------------------------ mutation

/**
 * Bit-flip mutation: flip each bit with probability `rate`.
 *
 * @param key - The random key.
 * @param bits - The bits (0 or 1).
 * @param rate - The flip probability (default 1 / length).
 * @returns The mutated bits.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { bitFlipMutation } from 'mdml-engine/search'
 * bitFlipMutation(key(0), [0, 1, 0], 1) // => [1, 0, 1]
 * ```
 */
export function bitFlipMutation(
  key: Key,
  bits: readonly number[],
  rate: number = 1 / Math.max(1, bits.length),
): number[] {
  const d = draws(key)
  return bits.map((b) => (d.uniform() < rate ? 1 - b : b))
}

/**
 * Swap mutation for permutations: exchange two random positions.
 *
 * @param key - The random key.
 * @param perm - The permutation.
 * @returns The mutated permutation.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { swapMutation } from 'mdml-engine/search'
 * [...swapMutation(key(0), [0, 1, 2, 3])].sort().join() // => '0,1,2,3'
 * ```
 */
export function swapMutation<T>(key: Key, perm: readonly T[]): T[] {
  const p = [...perm]
  if (p.length < 2) return p
  const d = draws(key)
  const i = d.int(p.length)
  let j = d.int(p.length - 1)
  if (j >= i) j += 1
  ;[p[i], p[j]] = [p[j] as T, p[i] as T]
  return p
}

/**
 * Inversion mutation for permutations: reverse a random segment (the 2-opt move).
 *
 * @param key - The random key.
 * @param perm - The permutation.
 * @returns The mutated permutation.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { inversionMutation } from 'mdml-engine/search'
 * inversionMutation(key(4), [0, 1, 2, 3, 4]).length // => 5
 * ```
 */
export function inversionMutation<T>(key: Key, perm: readonly T[]): T[] {
  if (perm.length < 2) return [...perm]
  const [i, j] = cuts(key, perm.length)
  return [...perm.slice(0, i), ...perm.slice(i, j + 1).reverse(), ...perm.slice(j + 1)]
}

/**
 * Scramble mutation for permutations: shuffle a random segment.
 *
 * @param key - The random key.
 * @param perm - The permutation.
 * @returns The mutated permutation.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { scrambleMutation } from 'mdml-engine/search'
 * [...scrambleMutation(key(2), [0, 1, 2, 3, 4])].sort().join() // => '0,1,2,3,4'
 * ```
 */
export function scrambleMutation<T>(key: Key, perm: readonly T[]): T[] {
  if (perm.length < 2) return [...perm]
  const [i, j] = cuts(key, perm.length)
  const d = draws(key)
  const p = [...perm]
  for (let k = j; k > i; k--) {
    const r = i + d.int(k - i + 1)
    ;[p[k], p[r]] = [p[r] as T, p[k] as T]
  }
  return p
}

/**
 * Gaussian mutation for real vectors: add N(0, sigma^2) noise to each coordinate with probability `rate`, clipped to
 * optional bounds.
 *
 * @param key - The random key.
 * @param x - The vector.
 * @param options - sigma (default 0.1), rate (default 1) and bounds.
 * @returns The mutated vector.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { gaussianMutation } from 'mdml-engine/search'
 * gaussianMutation(key(0), [0, 0], { sigma: 1, lower: [0, 0], upper: [0, 0] }) // => [0, 0]
 * ```
 */
export function gaussianMutation(
  key: Key,
  x: readonly number[],
  options: {
    readonly sigma?: number
    readonly rate?: number
    readonly lower?: readonly number[]
    readonly upper?: readonly number[]
  } = {},
): number[] {
  const { sigma = 0.1, rate = 1 } = options
  const d = draws(key)
  return x.map((v, i) => {
    const moved = d.uniform() < rate ? v + sigma * d.normal() : v
    return Math.min(
      options.upper?.[i] ?? Number.POSITIVE_INFINITY,
      Math.max(options.lower?.[i] ?? Number.NEGATIVE_INFINITY, moved),
    )
  })
}

/**
 * Polynomial mutation (Deb and Goyal 1996) for bounded real vectors: each coordinate moves with probability `rate` by
 * a polynomially distributed step scaled to its range, staying within the bounds.
 *
 * @param key - The random key.
 * @param x - The vector.
 * @param lower - Lower bounds.
 * @param upper - Upper bounds.
 * @param options - The distribution index eta (default 20) and rate (default 1 / length).
 * @returns The mutated vector.
 * @throws MdmlError with code `shape` when the bounds do not match the vector.
 * @concept evolutionary-algorithms
 * @example
 * ```ts
 * import { key } from 'mdml-engine/random'
 * import { polynomialMutation } from 'mdml-engine/search'
 * polynomialMutation(key(0), [0.5], [0], [1], { rate: 1 }).every((v) => v >= 0 && v <= 1) // => true
 * ```
 */
export function polynomialMutation(
  key: Key,
  x: readonly number[],
  lower: readonly number[],
  upper: readonly number[],
  options: { readonly eta?: number; readonly rate?: number } = {},
): number[] {
  if (lower.length !== x.length || upper.length !== x.length) {
    throw new MdmlError('shape', 'polynomialMutation: bounds must match the vector')
  }
  const eta = options.eta ?? 20
  const rate = options.rate ?? 1 / Math.max(1, x.length)
  const d = draws(key)
  return x.map((v, i) => {
    const lo = lower[i] as number
    const hi = upper[i] as number
    if (d.uniform() >= rate || !(hi > lo)) return v
    const u = d.uniform()
    const delta1 = (v - lo) / (hi - lo)
    const delta2 = (hi - v) / (hi - lo)
    const p = 1 / (eta + 1)
    const dq =
      u < 0.5
        ? (2 * u + (1 - 2 * u) * (1 - delta1) ** (eta + 1)) ** p - 1
        : 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta2) ** (eta + 1)) ** p
    return Math.min(hi, Math.max(lo, v + dq * (hi - lo)))
  })
}
