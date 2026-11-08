/**
 * Sugarscape, Epstein and Axtell's artificial society: agents with a sight range and a metabolism wander a landscape
 * of sugar with two peaks, move to the richest visible unoccupied site, harvest it and burn sugar to live. Migration to
 * the peaks, a carrying capacity and an inequality of wealth (the Gini coefficient climbing from the equality of the
 * start) emerge from rules that mention none of them. The landscape, the model as a traced algorithm with the agents'
 * positions and wealth, and the Gini coefficient of the wealth on the tape.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { foldIn, permutation } from '../../random/index.ts'
import { gini } from '../../stats/index.ts'
import { type Algorithm, type Contract, contractSchema, defineContract, scalar, tensor } from '../../trace/index.ts'
import { codec, draws, positiveInt } from './_core.ts'

/**
 * The sugar capacity of every site of a square landscape with two peaks: 4 within a band of the peak, falling by one
 * every band out to 0 (a band is an eighth of the side). The peaks are in the north-east and south-west quadrants.
 *
 * @param size - The side of the square.
 * @returns The capacity per site, row-major, in 0..4.
 * @concept sugarscape
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * const c = automata.sugarLandscape(50)
 * [c[35 * 50 + 15], c[0], Math.max(...c)] // => [4, 0, 4]
 * ```
 */
export function sugarLandscape(size: number): Uint8Array {
  positiveInt('sugarLandscape', 'size', size)
  const band = size / 8
  const peaks = [
    [0.7 * size, 0.3 * size],
    [0.3 * size, 0.7 * size],
  ] as const
  const out = new Uint8Array(size * size)
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) {
      let best = 0
      for (const [pr, pc] of peaks) {
        const d = Math.hypot(r - pr, c - pc)
        best = Math.max(best, Math.max(0, 4 - Math.floor(d / band)))
      }
      out[r * size + c] = best
    }
  return out
}

/** Options of {@link sugarscape}. */
export interface SugarscapeOptions {
  /** The side of the square landscape (default 50); it does not wrap. */
  readonly size?: number
  /** How many agents (default 400). */
  readonly agents?: number
  /** Sugar regrown per site per step, up to its capacity (default 1). */
  readonly growth?: number
  /** Ranges (inclusive) of the agents' sight, metabolism and starting sugar: `[low, high]`. */
  readonly vision?: readonly [number, number]
  readonly metabolism?: readonly [number, number]
  readonly endowment?: readonly [number, number]
  /** Range of the age at which an agent dies of old age (default `[60, 100]`); `null` for immortal agents. */
  readonly lifespan?: readonly [number, number] | null
  /** Whether a dead agent is replaced by a newborn with fresh traits at a random free site (default true). */
  readonly replace?: boolean
}

/** The state of {@link sugarscape}. */
export interface SugarscapeState {
  /** Sugar on every site now, row-major. */
  readonly sugar: Float64Array
  /** Per agent: row, column, wealth, sight, metabolism, age, lifespan (Infinity encoded as 1e9). */
  readonly agents: Float64Array
  readonly count: number
  /** Deaths in the last step. */
  readonly deaths: number
}

/** The paths of {@link sugarscape}. */
export type SugarscapePath = 'scape/positions' | 'scape/wealth' | 'scape/sugar' | 'scape/gini' | 'scape/population'

/** The log contract of {@link sugarscape}. */
export const sugarscapeContract: Contract<SugarscapeState, SugarscapePath> = defineContract({
  'scape/positions': tensor({ role: 'state', shape: ['agents', 2], description: 'row and column of every agent' }),
  'scape/wealth': tensor({ role: 'state', shape: ['agents'], description: 'the sugar each agent holds' }),
  'scape/sugar': tensor({ role: 'state', shape: ['size', 'size'], description: 'sugar on every site' }),
  'scape/gini': scalar({ description: "the Gini coefficient of the agents' wealth" }),
  'scape/population': scalar({ description: 'agents alive' }),
})

const FIELDS = 7
const R = 0
const C = 1
const W = 2
const V = 3
const M = 4
const A = 5
const L = 6

/**
 * Sugarscape as a traced algorithm: one tick per step. In a random order each agent looks along the four directions as
 * far as its sight, picks the unoccupied site (its own included) with the most sugar, the nearest on ties, moves there
 * and harvests it; then it burns its metabolism and ages. An agent whose sugar runs out or who reaches its lifespan dies
 * (and is replaced if `replace`). Then every site regrows. Wealth is the sugar an agent holds, so a population that
 * starts equal ends unequal. The state serialises.
 *
 * @param options - The landscape, the agents' trait ranges, growth, lifespan and replacement.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a bad size, more agents than sites or a range that is backwards.
 * @concept sugarscape
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const s = runAlgorithm(automata.sugarscape({ size: 20, agents: 30 }), { seed: 0, maxSteps: 5 }).state
 * s.count // => 30
 * ```
 */
export function sugarscape(options: SugarscapeOptions = {}): Algorithm<SugarscapeState, SugarscapePath> {
  const size = options.size ?? 50
  const nAgents = options.agents ?? 400
  const growth = options.growth ?? 1
  const vision = options.vision ?? [1, 6]
  const metabolism = options.metabolism ?? [1, 4]
  const endowment = options.endowment ?? [5, 25]
  const lifespan = options.lifespan === undefined ? ([60, 100] as const) : options.lifespan
  const replace = options.replace ?? true
  positiveInt('sugarscape', 'size', size)
  positiveInt('sugarscape', 'agents', nAgents)
  if (nAgents > size * size) throw new OmaiError('value', 'sugarscape: more agents than sites')
  for (const [name, range] of [
    ['vision', vision],
    ['metabolism', metabolism],
    ['endowment', endowment],
  ] as const)
    if (!(range[0] <= range[1])) throw new OmaiError('value', `sugarscape: the ${name} range is backwards`)
  const capacity = sugarLandscape(size)
  const sites = size * size
  const range = (u: number, [lo, hi]: readonly [number, number]): number => lo + Math.floor(u * (hi - lo + 1))
  // Fresh traits for a newborn at a site, from five uniform draws.
  const newborn = (out: Float64Array, slot: number, site: number, u: ArrayLike<number>, at: number): void => {
    const o = slot * FIELDS
    out[o + R] = Math.floor(site / size)
    out[o + C] = site % size
    out[o + W] = range(u[at] ?? 0, endowment)
    out[o + V] = range(u[at + 1] ?? 0, vision)
    out[o + M] = range(u[at + 2] ?? 0, metabolism)
    out[o + A] = 0
    out[o + L] = lifespan === null ? 1e9 : range(u[at + 3] ?? 0, lifespan)
  }
  return {
    name: 'dynamics/automata/sugarscape',
    config: { size, agents: nAgents, growth, vision, metabolism, endowment, lifespan, replace } as Json,
    schema: contractSchema(sugarscapeContract),
    init: (key) => {
      const order = permutation(key, sites).data
      const u = draws(foldIn(key, 1), 4 * nAgents)
      const agents = new Float64Array(nAgents * FIELDS)
      for (let a = 0; a < nAgents; a++) newborn(agents, a, order[a] ?? 0, u, 4 * a)
      // The first agents sit anywhere; ages start spread so that not all die together.
      if (lifespan !== null)
        for (let a = 0; a < nAgents; a++) agents[a * FIELDS + A] = Math.floor((u[4 * a + 3] ?? 0) * (lifespan[0] / 2))
      return { sugar: Float64Array.from(capacity), agents, count: nAgents, deaths: 0 }
    },
    step: (s, ctx) => {
      const sugar = Float64Array.from(s.sugar)
      const agents = Float64Array.from(s.agents)
      const occupied = new Uint8Array(sites)
      for (let a = 0; a < s.count; a++)
        occupied[(agents[a * FIELDS + R] ?? 0) * size + (agents[a * FIELDS + C] ?? 0)] = 1
      const order = permutation(foldIn(ctx.key, 0), s.count).data
      const tie = draws(foldIn(ctx.key, 1), s.count * 4)
      const dead: number[] = []
      for (let k = 0; k < s.count; k++) {
        const a = order[k] ?? 0
        const o = a * FIELDS
        const r = agents[o + R] ?? 0
        const c = agents[o + C] ?? 0
        const sight = agents[o + V] ?? 1
        // Candidate sites: own, then the four directions out to the sight; most sugar, then nearest, then a random pick.
        let bestSugar = sugar[r * size + c] ?? 0
        let bestDist = 0
        let best: number[] = [r * size + c]
        const dirs = [
          [-1, 0],
          [1, 0],
          [0, -1],
          [0, 1],
        ] as const
        for (const [dr, dc] of dirs) {
          for (let d = 1; d <= sight; d++) {
            const rr = r + dr * d
            const cc = c + dc * d
            if (rr < 0 || rr >= size || cc < 0 || cc >= size) break
            const site = rr * size + cc
            if (occupied[site] === 1) continue
            const sg = sugar[site] ?? 0
            if (sg > bestSugar || (sg === bestSugar && d < bestDist)) {
              bestSugar = sg
              bestDist = d
              best = [site]
            } else if (sg === bestSugar && d === bestDist) best.push(site)
          }
        }
        const to = best[Math.min(best.length - 1, Math.floor((tie[k] ?? 0) * best.length))] as number
        occupied[r * size + c] = 0
        occupied[to] = 1
        agents[o + R] = Math.floor(to / size)
        agents[o + C] = to % size
        agents[o + W] = (agents[o + W] ?? 0) + (sugar[to] ?? 0) - (agents[o + M] ?? 0)
        sugar[to] = 0
        agents[o + A] = (agents[o + A] ?? 0) + 1
        if ((agents[o + W] ?? 0) <= 0 || (agents[o + A] ?? 0) >= (agents[o + L] ?? 1e9)) dead.push(a)
      }
      let count = s.count
      let out = agents
      if (dead.length > 0) {
        const free: number[] = []
        for (const a of dead) occupied[(agents[a * FIELDS + R] ?? 0) * size + (agents[a * FIELDS + C] ?? 0)] = 0
        if (replace) {
          for (let i = 0; i < sites; i++) if (occupied[i] === 0) free.push(i)
          const place = permutation(foldIn(ctx.key, 2), free.length).data
          const u = draws(foldIn(ctx.key, 3), 4 * dead.length)
          dead.forEach((a, j) => {
            newborn(out, a, free[place[j] ?? 0] as number, u, 4 * j)
          })
        } else {
          const gone = new Set(dead)
          const kept = new Float64Array((s.count - dead.length) * FIELDS)
          let w = 0
          for (let a = 0; a < s.count; a++)
            if (!gone.has(a)) kept.set(agents.subarray(a * FIELDS, (a + 1) * FIELDS), FIELDS * w++)
          out = kept
          count = s.count - dead.length
        }
      }
      for (let i = 0; i < sites; i++) sugar[i] = Math.min(capacity[i] ?? 0, (sugar[i] ?? 0) + growth)
      return { sugar, agents: out, count, deaths: dead.length }
    },
    observe: (s) => {
      const pos = new Float64Array(2 * s.count)
      const wealth = new Float64Array(s.count)
      for (let a = 0; a < s.count; a++) {
        pos[2 * a] = s.agents[a * FIELDS + R] ?? 0
        pos[2 * a + 1] = s.agents[a * FIELDS + C] ?? 0
        wealth[a] = s.agents[a * FIELDS + W] ?? 0
      }
      return {
        metrics: { 'scape/gini': s.count === 0 ? 0 : gini(wealth), 'scape/population': s.count },
        tensors: {
          'scape/positions': { role: 'state', value: pos, shape: [s.count, 2] },
          'scape/wealth': { role: 'state', value: wealth, shape: [s.count] },
          'scape/sugar': { role: 'state', value: s.sugar, shape: [size, size] },
        },
      }
    },
    ...codec<SugarscapeState>(),
  }
}
