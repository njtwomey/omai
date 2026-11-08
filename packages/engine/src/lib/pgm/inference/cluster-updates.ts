/**
 * Cluster updates for the q-state Potts model (q = 2 is the Ising model) on a periodic square lattice with energy
 * -sum delta(s_i, s_j): the Swendsen-Wang update (every aligned bond is kept with probability 1 - exp(-beta), the
 * resulting clusters each get a fresh uniformly random spin) and the Wolff update (grow and recolour a single cluster
 * from a random seed), beside the single-spin Metropolis sweep. Near the critical coupling beta_c = ln(1 + sqrt q) the
 * single-spin chain slows down critically, with autocorrelation time growing like L^2, while the cluster updates
 * decorrelate in a handful of sweeps; the exact critical coupling, the critical energy of the Ising lattice and Onsager's
 * spontaneous magnetisation are given as truth. The samplers run as `Algorithm`s with the energy and magnetisation
 * per site on the tape.
 */

import { maxOf, OmaiError } from '../../base/index.ts'
import { uniformStream } from '../../random/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../../trace/index.ts'

/**
 * The critical coupling of the q-state Potts model on the square lattice (Potts 1952): ln(1 + sqrt q), so 0.8814 for
 * the Ising model with the energy -sum delta (twice the Ising coupling of the +-1 convention, 0.4407).
 *
 * @param q - The number of states (at most 4 for a continuous transition).
 * @returns beta_c.
 * @example
 * ```ts
 * import { pottsCriticalCoupling } from 'omai-engine/pgm'
 * pottsCriticalCoupling(2) // => 0.881373587019543
 * ```
 */
export function pottsCriticalCoupling(q: number): number {
  return Math.log(1 + Math.sqrt(q))
}

/**
 * Onsager's spontaneous magnetisation of the Ising model, (1 - sinh(beta)^-4)^(1/8) in the Potts coupling beta = 2K, for
 * beta above the critical coupling and zero below it.
 *
 * @param beta - The Potts coupling of the energy -sum delta (q = 2).
 * @returns The magnetisation per site of the infinite lattice.
 * @example
 * ```ts
 * import { onsagerMagnetisation } from 'omai-engine/pgm'
 * onsagerMagnetisation(0.5) // => 0
 * ```
 */
export function onsagerMagnetisation(beta: number): number {
  const s = Math.sinh(beta) ** -4
  return s >= 1 ? 0 : (1 - s) ** 0.125
}

/** The energy per site of the critical Ising lattice in the Potts convention: -(2 + sqrt 2) / 2, from Onsager's -sqrt 2 in +-1 spins. */
export const ISING_CRITICAL_ENERGY_PER_SITE: number = -(2 + Math.SQRT2) / 2

/** A lattice of Potts spins, row-major, side L. */
export interface PottsConfiguration {
  readonly L: number
  readonly q: number
  readonly spins: Int32Array
}

const right = (i: number, L: number): number => (i % L === L - 1 ? i - L + 1 : i + 1)
const down = (i: number, L: number): number => (i + L >= L * L ? i + L - L * L : i + L)
const left = (i: number, L: number): number => (i % L === 0 ? i + L - 1 : i - 1)
const up = (i: number, L: number): number => (i < L ? i + L * L - L : i - L)

/**
 * The energy per site, -(number of aligned bonds) / L^2.
 *
 * @param c - The configuration.
 * @returns The energy per site, between -2 and 0.
 * @example
 * ```ts
 * import { pottsEnergyPerSite } from 'omai-engine/pgm'
 * pottsEnergyPerSite({ L: 2, q: 2, spins: Int32Array.of(0, 0, 0, 0) }) // => -2
 * ```
 */
export function pottsEnergyPerSite(c: PottsConfiguration): number {
  const { L, spins } = c
  let aligned = 0
  for (let i = 0; i < L * L; i++) {
    if (spins[i] === spins[right(i, L)]) aligned++
    if (spins[i] === spins[down(i, L)]) aligned++
  }
  return -aligned / (L * L)
}

/**
 * The magnetisation per site, (q max_a n_a / N - 1) / (q - 1): one for an ordered lattice, near zero for a random one.
 *
 * @param c - The configuration.
 * @returns The magnetisation in [0, 1].
 * @example
 * ```ts
 * import { pottsMagnetisation } from 'omai-engine/pgm'
 * pottsMagnetisation({ L: 2, q: 2, spins: Int32Array.of(0, 1, 0, 1) }) // => 0
 * ```
 */
export function pottsMagnetisation(c: PottsConfiguration): number {
  const counts = new Array<number>(c.q).fill(0)
  for (const s of c.spins) counts[s] = (counts[s] ?? 0) + 1
  return ((c.q * maxOf(counts)) / (c.L * c.L) - 1) / (c.q - 1)
}

/**
 * One Swendsen-Wang update: aligned neighbours are bonded with probability 1 - exp(-beta), the connected clusters are
 * found by union-find, and every cluster takes a uniformly random spin, independently.
 *
 * @param c - The configuration, updated in place.
 * @param beta - The coupling.
 * @param u - A source of uniform draws.
 * @returns The number of clusters.
 * @example
 * ```ts
 * import { swendsenWangUpdate } from 'omai-engine/pgm'
 * // At beta = 0 no bond is kept, so every site is its own cluster.
 * swendsenWangUpdate({ L: 2, q: 2, spins: Int32Array.of(0, 0, 0, 0) }, 0, () => 0.5) // => 4
 * ```
 */
export function swendsenWangUpdate(c: PottsConfiguration, beta: number, u: () => number): number {
  const { L, q, spins } = c
  const n = L * L
  const parent = Int32Array.from({ length: n }, (_, i) => i)
  const find = (a: number): number => {
    let r = a
    while (parent[r] !== r) r = parent[r] as number
    let x = a
    while (parent[x] !== r) {
      const next = parent[x] as number
      parent[x] = r
      x = next
    }
    return r
  }
  const p = 1 - Math.exp(-beta)
  const bond = (i: number, j: number): void => {
    if (spins[i] === spins[j] && u() < p) {
      const a = find(i)
      const b = find(j)
      if (a !== b) parent[a] = b
    }
  }
  for (let i = 0; i < n; i++) {
    bond(i, right(i, L))
    bond(i, down(i, L))
  }
  const colour = new Int32Array(n).fill(-1)
  let clusters = 0
  for (let i = 0; i < n; i++) {
    const r = find(i)
    if (colour[r] === -1) {
      colour[r] = Math.min(Math.floor(u() * q), q - 1)
      clusters++
    }
    spins[i] = colour[r] as number
  }
  return clusters
}

/**
 * One Wolff update: a random seed site, a cluster grown through aligned neighbours each added with probability
 * 1 - exp(-beta), and the whole cluster recoloured to a different spin chosen uniformly.
 *
 * @param c - The configuration, updated in place.
 * @param beta - The coupling.
 * @param u - A source of uniform draws.
 * @returns The size of the flipped cluster.
 * @example
 * ```ts
 * import { wolffUpdate } from 'omai-engine/pgm'
 * // At very strong coupling the cluster is the whole ordered lattice.
 * wolffUpdate({ L: 3, q: 2, spins: new Int32Array(9) }, 50, () => 0.5) // => 9
 * ```
 */
export function wolffUpdate(c: PottsConfiguration, beta: number, u: () => number): number {
  const { L, q, spins } = c
  const n = L * L
  const seed = Math.min(Math.floor(u() * n), n - 1)
  const old = spins[seed] as number
  const shift = 1 + Math.min(Math.floor(u() * (q - 1)), q - 2)
  const fresh = (old + shift) % q
  const p = 1 - Math.exp(-beta)
  const stack = [seed]
  spins[seed] = fresh
  let size = 1
  while (stack.length > 0) {
    const i = stack.pop() as number
    for (const j of [right(i, L), down(i, L), left(i, L), up(i, L)]) {
      if (spins[j] === old && u() < p) {
        spins[j] = fresh
        stack.push(j)
        size++
      }
    }
  }
  return size
}

/**
 * One Metropolis sweep: every site in turn proposes a different uniformly chosen spin, accepted with probability
 * min(1, exp(beta (aligned neighbours after - before))).
 *
 * @param c - The configuration, updated in place.
 * @param beta - The coupling.
 * @param u - A source of uniform draws.
 * @returns The number of accepted flips.
 * @example
 * ```ts
 * import { metropolisSweep } from 'omai-engine/pgm'
 * // At beta = 0 every proposal is accepted.
 * metropolisSweep({ L: 2, q: 2, spins: new Int32Array(4) }, 0, () => 0.5) // => 4
 * ```
 */
export function metropolisSweep(c: PottsConfiguration, beta: number, u: () => number): number {
  const { L, q, spins } = c
  let accepted = 0
  for (let i = 0; i < L * L; i++) {
    const old = spins[i] as number
    const proposal = (old + 1 + Math.min(Math.floor(u() * (q - 1)), q - 2)) % q
    let before = 0
    let after = 0
    for (const j of [right(i, L), down(i, L), left(i, L), up(i, L)]) {
      if (spins[j] === old) before++
      if (spins[j] === proposal) after++
    }
    if (u() < Math.exp(beta * (after - before))) {
      spins[i] = proposal
      accepted++
    }
  }
  return accepted
}

/** Which update a sampler uses. */
export type PottsUpdate = 'swendsen-wang' | 'wolff' | 'metropolis'

/** Options of {@link pottsSamplerAlgorithm}. */
export interface PottsSamplerOptions {
  readonly L: number
  readonly q?: number
  readonly beta: number
  /** The start: all spins equal (default) or uniformly random. */
  readonly start?: 'ordered' | 'random'
  /**
   * Wolff cluster flips per step (default ceil(L^0.25): at the critical Ising coupling the mean cluster holds about
   * L^1.75 of the L^2 sites). It is fixed, not chosen from the cluster sizes met, which would bias the chain.
   */
  readonly wolffFlips?: number
}

/** The state of a Potts sampler: a sweep is one step. */
export interface PottsState {
  readonly t: number
  readonly spins: readonly number[]
  readonly energy: number
  readonly magnetisation: number
  /** Swendsen-Wang: clusters. Wolff: sites flipped in the sweep. Metropolis: accepted flips. */
  readonly work: number
}

/**
 * A Potts sampler as an algorithm, one update per step: a Metropolis sweep, a Swendsen-Wang update, or a fixed number
 * of Wolff cluster flips (about a sweep of work at criticality). It records the energy and magnetisation per site and its state serialises.
 *
 * @param update - The update rule.
 * @param options - Lattice side, states, coupling, start.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { pottsSamplerAlgorithm } from 'omai-engine/pgm'
 * pottsSamplerAlgorithm('swendsen-wang', { L: 8, beta: 0.4 }).name // => 'potts-swendsen-wang'
 * ```
 */
export function pottsSamplerAlgorithm(update: PottsUpdate, options: PottsSamplerOptions): Algorithm<PottsState> {
  const L = options.L
  const q = options.q ?? 2
  if (!(Number.isInteger(L) && L >= 2) || !(Number.isInteger(q) && q >= 2))
    throw new OmaiError('value', 'potts: L and q must be integers of at least 2')
  const view = (spins: Int32Array): { energy: number; magnetisation: number } => {
    const c = { L, q, spins }
    return { energy: pottsEnergyPerSite(c), magnetisation: pottsMagnetisation(c) }
  }
  return {
    name: `potts-${update}`,
    config: { update, L, q, beta: options.beta, start: options.start ?? 'ordered' },
    schema: channels({
      metrics: {
        energy: { description: 'energy per site, -aligned bonds / N', direction: 'lower' },
        magnetisation: { description: 'Potts magnetisation per site, 0 disordered to 1 ordered' },
        work: { description: 'clusters (Swendsen-Wang), flipped sites (Wolff) or accepted flips (Metropolis)' },
      },
      tensors: {},
    }),
    init: (key) => {
      const spins = new Int32Array(L * L)
      if ((options.start ?? 'ordered') === 'random') {
        const u = uniformStream(key)
        for (let i = 0; i < spins.length; i++) spins[i] = Math.min(Math.floor(u() * q), q - 1)
      }
      return { t: 0, spins: Array.from(spins), ...view(spins), work: 0 }
    },
    step: (s, ctx) => {
      const u = uniformStream(ctx.key)
      const c = { L, q, spins: Int32Array.from(s.spins) }
      let work = 0
      if (update === 'swendsen-wang') work = swendsenWangUpdate(c, options.beta, u)
      else if (update === 'metropolis') work = metropolisSweep(c, options.beta, u)
      else
        for (let k = 0; k < (options.wolffFlips ?? Math.ceil(L ** 0.25)); k++) work += wolffUpdate(c, options.beta, u)
      return { t: s.t + 1, spins: Array.from(c.spins), ...view(c.spins), work }
    },
    observe: (s) => ({ metrics: { energy: s.energy, magnetisation: s.magnetisation, work: s.work } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PottsState,
  }
}

/**
 * The integrated autocorrelation time of a series with Sokal's automatic window (the smallest M with M >= 6 tau(M)):
 * 1/2 + the sum of the autocorrelations up to the window. It counts how many sweeps make one independent sample.
 *
 * @param x - The series.
 * @returns The time in steps (about 1/2 for independent draws).
 * @example
 * ```ts
 * import { integratedAutocorrelationTime } from 'omai-engine/pgm'
 * integratedAutocorrelationTime([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]) < 1 // => true
 * ```
 */
export function integratedAutocorrelationTime(x: ArrayLike<number>): number {
  const n = x.length
  const mean = Array.from(x).reduce((a, b) => a + b, 0) / n
  const d = Array.from(x, (v) => v - mean)
  const c0 = d.reduce((a, v) => a + v * v, 0) / n
  if (c0 === 0) return 0.5
  let tau = 0.5
  for (let m = 1; m < n / 2; m++) {
    let c = 0
    for (let t = 0; t + m < n; t++) c += (d[t] ?? 0) * (d[t + m] ?? 0)
    tau += c / (n - m) / c0
    if (m >= 6 * tau) break
  }
  return tau
}
