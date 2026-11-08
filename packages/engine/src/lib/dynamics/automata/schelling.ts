/**
 * Schelling's model of segregation: two kinds of agent on a grid with some vacant sites, each agent content when at
 * least a share (the tolerance) of its occupied neighbours are its own kind, and a discontented agent moves to a vacant
 * site. A mild preference, one third say, is enough to segregate the city almost completely: the segregation index
 * climbs from the random baseline to a value no one chose. The model as a traced algorithm, with the index and the
 * count of unhappy agents on the tape.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, permutation } from '../../random/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  event,
  scalar,
  tensor,
} from '../../trace/index.ts'
import { codec, draws, mod, positiveInt, unit } from './_core.ts'

/** Where a discontented agent goes: to a random vacant site, or to a random vacant site where it would be content. */
export type SchellingDestination = 'random' | 'satisfying'

/** Options of {@link schelling}. */
export interface SchellingOptions {
  /** Columns and rows (default 40 by 40). */
  readonly width?: number
  readonly height?: number
  /** The share of sites left vacant (default 0.1). */
  readonly vacancy?: number
  /** The share of agents that are of the second kind (default 0.5). */
  readonly groupShare?: number
  /** The least share of occupied neighbours that must be of the agent's own kind (default 1/3). */
  readonly tolerance?: number
  /** How far an agent sees, in sites (Moore neighbourhood; default 1). */
  readonly radius?: number
  /** Whether the grid is a torus (default false: edges have fewer neighbours). */
  readonly wrap?: boolean
  /** Where movers go (default `'random'`). */
  readonly destination?: SchellingDestination
}

/** The state of {@link schelling}. */
export interface SchellingState {
  /** 0 vacant, 1 and 2 the two kinds, row-major. */
  readonly cells: Uint8Array
  /** Agents discontented at the end of the round. */
  readonly unhappy: number
  /** Agents that moved in the round. */
  readonly moved: number
  /** The round (0 is the random start). */
  readonly round: number
}

/** The paths of {@link schelling}. */
export type SchellingPath = 'city/board' | 'city/unhappy' | 'city/moved' | 'city/segregation' | 'city/settled'

/** The log contract of {@link schelling}. */
export const schellingContract: Contract<SchellingState, SchellingPath> = defineContract({
  'city/board': tensor({ role: 'state', shape: ['height', 'width'], description: '0 vacant, 1 and 2 the kinds' }),
  'city/unhappy': scalar({ description: 'agents discontented at the end of the round' }),
  'city/moved': scalar({ description: 'agents that moved in the round' }),
  'city/segregation': scalar({ description: "the mean share of neighbours of the agent's own kind" }),
  'city/settled': event({ context: { round: 'the round in which no one was left unhappy' } }),
})

/**
 * The segregation index of a city: the share of an agent's occupied neighbours that are of its own kind, averaged over
 * the agents that have a neighbour. Randomly placed, two kinds of equal numbers give about 1/2; complete segregation
 * approaches 1.
 *
 * @param cells - The board, 0 vacant, 1 and 2 the kinds.
 * @param width - Columns.
 * @param height - Rows.
 * @param radius - The neighbourhood radius (default 1).
 * @param wrap - Whether the board is a torus (default false).
 * @returns The mean own-kind share, in [0, 1].
 * @concept schelling-segregation
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * automata.segregationIndex(Uint8Array.of(1, 1, 0, 2, 2), 5, 1) // => 1
 * automata.segregationIndex(Uint8Array.of(1, 2, 1, 2), 4, 1) // => 0
 * ```
 */
export function segregationIndex(cells: Uint8Array, width: number, height: number, radius = 1, wrap = false): number {
  let total = 0
  let agents = 0
  for (let i = 0; i < cells.length; i++) {
    if (cells[i] === 0) continue
    const { same, occupied } = neighbourhood(cells, width, height, i, cells[i] ?? 0, radius, wrap)
    if (occupied === 0) continue
    total += same / occupied
    agents++
  }
  return agents === 0 ? 0 : total / agents
}

function neighbourhood(
  cells: Uint8Array,
  width: number,
  height: number,
  site: number,
  kind: number,
  radius: number,
  wrap: boolean,
): { same: number; occupied: number } {
  const r = Math.floor(site / width)
  const c = site % width
  let same = 0
  let occupied = 0
  for (let dr = -radius; dr <= radius; dr++) {
    for (let dc = -radius; dc <= radius; dc++) {
      if (dr === 0 && dc === 0) continue
      let rr = r + dr
      let cc = c + dc
      if (wrap) {
        rr = mod(rr, height)
        cc = mod(cc, width)
      } else if (rr < 0 || rr >= height || cc < 0 || cc >= width) continue
      const k = cells[rr * width + cc] ?? 0
      if (k === 0) continue
      occupied++
      if (k === kind) same++
    }
  }
  return { same, occupied }
}

/**
 * Schelling's segregation model as a traced algorithm: one round per step. At the start of each round the discontented
 * agents (those with fewer than `tolerance` of their occupied neighbours of their own kind) are listed; in random order
 * each moves to a vacant site (its own site becoming vacant), and the round ends with a count of who is still
 * discontented. `city/settled` fires, once, in the first round that ends with no one discontented. The state
 * serialises.
 *
 * @param options - Size, vacancy, group share, tolerance, radius, wrap and destination rule.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a bad size or a share outside [0, 1].
 * @concept schelling-segregation
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const s = runAlgorithm(automata.schelling({ width: 10, height: 10, tolerance: 0 }), { seed: 0, maxSteps: 3 }).state
 * [s.unhappy, s.moved] // => [0, 0]
 * ```
 */
export function schelling(options: SchellingOptions = {}): Algorithm<SchellingState, SchellingPath> {
  const width = options.width ?? 40
  const height = options.height ?? 40
  const vacancy = options.vacancy ?? 0.1
  const groupShare = options.groupShare ?? 0.5
  const tolerance = options.tolerance ?? 1 / 3
  const radius = options.radius ?? 1
  const wrap = options.wrap ?? false
  const destination = options.destination ?? 'random'
  positiveInt('schelling', 'width', width)
  positiveInt('schelling', 'height', height)
  positiveInt('schelling', 'radius', radius)
  unit('schelling', 'vacancy', vacancy)
  unit('schelling', 'groupShare', groupShare)
  unit('schelling', 'tolerance', tolerance)
  if (destination !== 'random' && destination !== 'satisfying')
    throw new OmaiError('value', `schelling: unknown destination '${destination}'`)
  const n = width * height
  const content = (cells: Uint8Array, site: number, kind: number): boolean => {
    const { same, occupied } = neighbourhood(cells, width, height, site, kind, radius, wrap)
    return occupied === 0 || same / occupied >= tolerance
  }
  const countUnhappy = (cells: Uint8Array): number => {
    let u = 0
    for (let i = 0; i < n; i++) if (cells[i] !== 0 && !content(cells, i, cells[i] ?? 0)) u++
    return u
  }
  return {
    name: 'dynamics/automata/schelling',
    config: { width, height, vacancy, groupShare, tolerance, radius, wrap, destination },
    schema: contractSchema(schellingContract),
    init: (key) => {
      // A random arrangement with exactly the stated numbers: shuffle a deck of vacant sites and the two kinds.
      const vacant = Math.round(vacancy * n)
      const second = Math.round(groupShare * (n - vacant))
      const deck = new Uint8Array(n)
      for (let i = vacant; i < n; i++) deck[i] = i < vacant + second ? 2 : 1
      const order = permutation(key, n).data
      const cells = new Uint8Array(n)
      for (let i = 0; i < n; i++) cells[order[i] ?? 0] = deck[i] ?? 0
      return { cells, unhappy: countUnhappy(cells), moved: 0, round: 0 }
    },
    step: (s, ctx) => {
      const cells = Uint8Array.from(s.cells)
      const movers: number[] = []
      for (let i = 0; i < n; i++) if (cells[i] !== 0 && !content(cells, i, cells[i] ?? 0)) movers.push(i)
      const order = permutation(foldIn(ctx.key, 0), movers.length).data
      const u = draws(foldIn(ctx.key, 1), movers.length)
      const vacancies: number[] = []
      for (let i = 0; i < n; i++) if (cells[i] === 0) vacancies.push(i)
      let moved = 0
      for (let k = 0; k < movers.length; k++) {
        const from = movers[order[k] ?? 0] as number
        if (vacancies.length === 0) break
        const kind = cells[from] ?? 0
        let pool = vacancies.map((_, j) => j)
        if (destination === 'satisfying') {
          cells[from] = 0
          const ok = pool.filter((j) => content(cells, vacancies[j] as number, kind))
          cells[from] = kind
          // Content is judged with the mover's old site vacated and its new site still empty (it does not count itself).
          if (ok.length > 0) pool = ok
        }
        const pick = pool[Math.min(pool.length - 1, Math.floor((u[k] ?? 0) * pool.length))] as number
        const to = vacancies[pick] as number
        cells[to] = kind
        cells[from] = 0
        vacancies[pick] = from
        moved++
      }
      const unhappy = countUnhappy(cells)
      if (unhappy === 0 && s.unhappy > 0) ctx.emit('city/settled', { round: s.round + 1 })
      return { cells, unhappy, moved, round: s.round + 1 }
    },
    observe: (s) => ({
      metrics: {
        'city/unhappy': s.unhappy,
        'city/moved': s.moved,
        'city/segregation': segregationIndex(s.cells, width, height, radius, wrap),
      },
      tensors: { 'city/board': { role: 'state', value: s.cells, shape: [height, width] } },
    }),
    ...codec<SchellingState>(),
  }
}
