/**
 * Simultaneous-move multi-agent environments (PettingZoo's parallel API): repeated two-player matrix games such as the
 * prisoner's dilemma, matching pennies, rock-paper-scissors and the stag hunt, pursuit-evasion on a grid, and
 * cooperative navigation from the multi-agent particle environments (MPE `simple_spread`), with the particle-world
 * physics step it runs on.
 */

import { WhiteboxError } from '../../../lib/base/index.ts'
import {
  matchingPennies,
  type NormalFormGame,
  payoffMatrices,
  prisonersDilemma,
  rockPaperScissors,
  stagHunt,
} from '../../../lib/games/index.ts'
import { foldIn, type Key, permutation, uniform } from '../../../lib/random/index.ts'
import type { ParallelEnv, PerAgent } from '../../../lib/rl/index.ts'
import { circle, line, rect, type SceneNode, scene, text } from '../../../lib/scene/index.ts'
import { bool, box, choice, type DictSpace, dict, discrete, int, real, type Space } from '../../../lib/space/index.ts'

/**
 * A named factory of a simultaneous-move multi-agent environment (the parallel counterpart of `EnvDefinition`):
 * `create(options)` returns a {@link ParallelEnv}.
 *
 * @concept multi-agent
 */
export interface ParallelDefinition<S, O, A> {
  readonly id: string
  readonly description: string
  /** Marks a multi-agent definition, so registries of single-agent environments skip it. */
  readonly multiAgent: 'parallel'
  readonly options?: DictSpace
  create(options?: Readonly<Record<string, unknown>>): ParallelEnv<S, O, A>
}

const num = (options: Readonly<Record<string, unknown>> | undefined, name: string, fallback: number): number => {
  const v = options?.[name]
  return typeof v === 'number' ? v : fallback
}
const u1 = (k: Key): number => uniform(k, []).data[0] ?? 0
const noInfos = (agents: readonly string[]): PerAgent<Readonly<Record<string, never>>> =>
  Object.fromEntries(agents.map((a) => [a, {}]))
const flags = (agents: readonly string[], value: boolean): PerAgent<boolean> =>
  Object.fromEntries(agents.map((a) => [a, value]))

// ---------------------------------------------------------------------------------------------------- matrix games

/** A bimatrix: `payoffs[i][j]` is `[reward of player_0, reward of player_1]` when they play actions `i` and `j`. */
export type Bimatrix = readonly (readonly (readonly [number, number])[])[]

/** The payoffs of a two-player normal-form game from `games` as a bimatrix (`+ 0` turns zero-sum `-0`s into `0`). */
function bimatrixOf(game: NormalFormGame): Bimatrix {
  const [a, b] = payoffMatrices(game)
  return a.map((row, i) => row.map((v, j): readonly [number, number] => [v + 0, (b[i]?.[j] ?? 0) + 0]))
}

/**
 * The classic two-player matrix games, with the payoffs of `games` (`prisonersDilemma`, `matchingPennies`,
 * `rockPaperScissors`, `stagHunt`): the prisoner's dilemma (0 cooperate, 1 defect), matching pennies (0 heads, 1
 * tails; `player_0` wins on a match), rock-paper-scissors (0 rock, 1 paper, 2 scissors) and the stag hunt (0 stag, 1
 * hare).
 */
export const MATRIX_GAMES: Readonly<Record<string, Bimatrix>> = {
  prisonersDilemma: bimatrixOf(prisonersDilemma()),
  matchingPennies: bimatrixOf(matchingPennies()),
  rockPaperScissors: bimatrixOf(rockPaperScissors()),
  stagHunt: bimatrixOf(stagHunt()),
}

/** The state of a repeated matrix game: rounds played and the previous joint action. */
export interface MatrixGameState {
  readonly t: number
  /** The previous joint action `[a0, a1]`, if any. */
  readonly last?: readonly [number, number]
}

/** Options of {@link matrixGame}. */
export interface MatrixGameOptions {
  /** One of the {@link MATRIX_GAMES} (default `'prisonersDilemma'`). */
  readonly game?: string
  /** A custom bimatrix, replacing `game`. */
  readonly payoffs?: Bimatrix
  /** Truncate after this many rounds (default never). */
  readonly rounds?: number
}

/**
 * A repeated two-player matrix game (the iterated prisoner's dilemma and friends): both agents choose an action at
 * once and receive the bimatrix payoffs. Each agent observes the previous joint action encoded as
 * `1 + a0 * n1 + a1` (0 before the first round), so tabular learners can condition on it. The game never terminates
 * by itself; `rounds` truncates it (or wrap it in `parallelTimeLimit`).
 *
 * @concept multi-agent
 * @example
 * ```ts
 * import { matrixGame } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = matrixGame.create({ game: 'stagHunt' })
 * env.step(env.reset(key(0)).state, { player_0: 0, player_1: 0 }, key(1)).rewards // => { player_0: 4, player_1: 4 }
 * ```
 */
export const matrixGame: ParallelDefinition<MatrixGameState, number, number> = {
  id: 'MatrixGame-v0',
  description: 'A repeated two-player matrix game.',
  multiAgent: 'parallel',
  options: dict({
    game: choice(['prisonersDilemma', 'matchingPennies', 'rockPaperScissors', 'stagHunt'], {
      default: 'prisonersDilemma',
    }),
    rounds: int(1, 100_000, { description: 'rounds before truncation (absent: never)' }),
  }),
  create: (options) => {
    const opts = (options ?? {}) as MatrixGameOptions
    const name = opts.game ?? 'prisonersDilemma'
    const payoffs = opts.payoffs ?? MATRIX_GAMES[name]
    if (payoffs === undefined) throw new WhiteboxError('value', `matrixGame: no game ${name}`)
    const n0 = payoffs.length
    const n1 = payoffs[0]?.length ?? 0
    if (n0 === 0 || n1 === 0 || payoffs.some((row) => row.length !== n1))
      throw new WhiteboxError('value', 'matrixGame: payoffs must be a non-empty n0 x n1 bimatrix')
    const agents = ['player_0', 'player_1']
    const rounds = opts.rounds
    const code = (s: MatrixGameState): number => (s.last === undefined ? 0 : 1 + s.last[0] * n1 + s.last[1])
    const obs = discrete(1 + n0 * n1, { label: 'previous joint action (0 at the start)' })
    const live = (s: MatrixGameState): readonly string[] => (rounds !== undefined && s.t >= rounds ? [] : agents)
    const id = 'MatrixGame-v0'
    const title = opts.payoffs === undefined ? name : 'custom bimatrix'
    return {
      spec: {
        id,
        description: 'A repeated two-player matrix game.',
        deterministic: true,
        ...(rounds === undefined ? {} : { maxEpisodeSteps: rounds }),
      },
      possibleAgents: agents,
      observationSpaces: { player_0: obs, player_1: obs },
      actionSpaces: { player_0: discrete(n0), player_1: discrete(n1) },
      agents: live,
      reset: () => ({ state: { t: 0 }, observations: { player_0: 0, player_1: 0 }, infos: noInfos(agents) }),
      step: (s, actions) => {
        const a0 = actions['player_0'] ?? 0
        const a1 = actions['player_1'] ?? 0
        if (!(a0 >= 0 && a0 < n0 && a1 >= 0 && a1 < n1 && Number.isInteger(a0) && Number.isInteger(a1)))
          throw new WhiteboxError('value', `matrixGame: actions out of range (${a0}, ${a1})`)
        const [r0, r1] = payoffs[a0]?.[a1] ?? [0, 0]
        const state: MatrixGameState = { t: s.t + 1, last: [a0, a1] }
        const o = code(state)
        const truncated = rounds !== undefined && state.t >= rounds
        return {
          state,
          observations: { player_0: o, player_1: o },
          rewards: { player_0: r0, player_1: r1 },
          terminations: flags(agents, false),
          truncations: flags(agents, truncated),
          infos: noInfos(agents),
        }
      },
      render: (s) => {
        const cell = 72
        const nodes: SceneNode[] = []
        for (let i = 0; i < n0; i++) {
          nodes.push(text(cell / 2, cell * (i + 1.5), `${i}`, { anchor: 'middle', baseline: 'middle', size: 16 }))
          for (let j = 0; j < n1; j++) {
            if (i === 0)
              nodes.push(text(cell * (j + 1.5), cell / 2, `${j}`, { anchor: 'middle', baseline: 'middle', size: 16 }))
            const chosen = s.last !== undefined && s.last[0] === i && s.last[1] === j
            const [r0, r1] = payoffs[i]?.[j] ?? [0, 0]
            nodes.push(
              rect(cell * (j + 1), cell * (i + 1), cell, cell, {
                fill: chosen ? 'tab1' : 'bg',
                stroke: 'grid',
                strokeWidth: 2,
                id: `cell-${i}-${j}`,
              }),
              text(cell * (j + 1.5), cell * (i + 1.5), `${r0}, ${r1}`, {
                anchor: 'middle',
                baseline: 'middle',
                size: 16,
              }),
            )
          }
        }
        nodes.push(text(8, cell * (n0 + 1) + 20, `round ${s.t}`, { size: 14 }))
        return scene(cell * (n1 + 1) + 8, cell * (n0 + 1) + 32, nodes, { title })
      },
    }
  },
}

// ------------------------------------------------------------------------------------------------- pursuit-evasion

/** The state of pursuit-evasion: positions as `[row, column]`, which evaders are still free, and the step count. */
export interface PursuitState {
  readonly pursuers: readonly (readonly [number, number])[]
  readonly evaders: readonly (readonly [number, number])[]
  readonly free: readonly boolean[]
  readonly t: number
}

/** Options of {@link pursuitEvasion}. */
export interface PursuitOptions {
  /** The grid side (default 8). */
  readonly size?: number
  readonly pursuers?: number
  readonly evaders?: number
  /** How evaders move: `'random'` (default, as in SISL pursuit), `'flee'` (greedily away) or `'agent'` (learners). */
  readonly evaderControl?: 'random' | 'flee' | 'agent'
  /** Manhattan distance at which a pursuer touches an evader (default 1: adjacent or on the same cell). */
  readonly captureRadius?: number
  /** Pursuers that must touch an evader at once to catch it (default 1; SISL pursuit uses 2). */
  readonly nCatch?: number
  /** Reward to each touching pursuer when an evader is caught (default 5). */
  readonly catchReward?: number
  /** Reward to each pursuer touching a free evader, per step (default 0.01). */
  readonly tagReward?: number
  /** Reward to every pursuer per step (default -0.1); free evader agents receive its negative. */
  readonly urgencyReward?: number
  /** `'vector'` relative positions (default) or a `'local'` window of `obsRange` cells (SISL style). */
  readonly observation?: 'vector' | 'local'
  readonly obsRange?: number
  /** Steps before truncation (default 100). */
  readonly maxCycles?: number
}

/** The moves of pursuit-evasion: stay, up, down, left, right. */
export const GRID_MOVES: readonly (readonly [number, number])[] = [
  [0, 0],
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
]

const manhattan = (a: readonly [number, number], b: readonly [number, number]): number =>
  Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1])

/**
 * Pursuit-evasion on a grid (in the spirit of PettingZoo's SISL pursuit): pursuers `pursuer_0..` chase evaders that
 * move at random, flee greedily, or are agents `evader_0..` themselves. Everyone moves at once (stay, up, down,
 * left, right, clipped to the grid). An evader is caught when `nCatch` pursuers are within `captureRadius`; each of
 * them gets `catchReward`, pursuers touching a free evader get `tagReward`, and every pursuer pays `urgencyReward`
 * per step (free evader agents receive its negative and `-catchReward` when caught). The episode terminates when
 * every evader is caught and is truncated after `maxCycles` steps. Observations are positions relative to the
 * observer, scaled by the grid size, or a SISL-style local window with channels for walls, pursuers and evaders.
 *
 * @concept multi-agent
 * @example
 * ```ts
 * import { pursuitEvasion } from 'whitebox-engine/methods/rl'
 * pursuitEvasion.create({ pursuers: 3, evaderControl: 'agent' }).possibleAgents // => ['pursuer_0', 'pursuer_1', 'pursuer_2', 'evader_0']
 * ```
 */
export const pursuitEvasion: ParallelDefinition<PursuitState, readonly number[], number> = {
  id: 'PursuitEvasion-v0',
  description: 'Pursuers chase evaders on a grid.',
  multiAgent: 'parallel',
  options: dict({
    size: int(3, 32, { default: 8 }),
    pursuers: int(1, 8, { default: 2 }),
    evaders: int(1, 8, { default: 1 }),
    evaderControl: choice(['random', 'flee', 'agent'], { default: 'random' }),
    captureRadius: int(0, 2, { default: 1 }),
    nCatch: int(1, 4, { default: 1 }),
    catchReward: real(0, 10, { default: 5 }),
    tagReward: real(0, 1, { default: 0.01 }),
    urgencyReward: real(-1, 0, { default: -0.1 }),
    observation: choice(['vector', 'local'], { default: 'vector' }),
    obsRange: int(3, 11, { default: 5, description: 'odd window side' }),
    maxCycles: int(1, 10_000, { default: 100 }),
  }),
  create: (options) => {
    const o = (options ?? {}) as PursuitOptions
    const size = o.size ?? 8
    const nP = o.pursuers ?? 2
    const nE = o.evaders ?? 1
    if (nP + nE > size * size) throw new WhiteboxError('value', 'pursuitEvasion: too many agents for the grid')
    const control = o.evaderControl ?? 'random'
    const radius = o.captureRadius ?? 1
    const nCatch = o.nCatch ?? 1
    const catchReward = o.catchReward ?? 5
    const tagReward = o.tagReward ?? 0.01
    const urgency = o.urgencyReward ?? -0.1
    const local = o.observation === 'local'
    const range = o.obsRange ?? 5
    if (range % 2 !== 1 || range < 1) throw new WhiteboxError('value', 'pursuitEvasion: obsRange must be odd')
    const maxCycles = o.maxCycles ?? 100
    const pursuerNames = Array.from({ length: nP }, (_, i) => `pursuer_${i}`)
    const evaderNames = control === 'agent' ? Array.from({ length: nE }, (_, i) => `evader_${i}`) : []
    const all = [...pursuerNames, ...evaderNames]
    const pursuerSize = local ? 3 * range * range : 2 + 2 * (nP - 1) + 3 * nE
    const evaderSize = local ? 3 * range * range : 2 + 2 * nP + 3 * (nE - 1)
    const observationSpaces: Record<string, Space> = {}
    const actionSpaces: Record<string, Space> = {}
    for (const a of pursuerNames) observationSpaces[a] = box(-1, 1, [pursuerSize])
    for (const a of evaderNames) observationSpaces[a] = box(-1, 1, [evaderSize])
    for (const a of all) actionSpaces[a] = discrete(5, { label: 'stay, up, down, left, right' })
    const move = (p: readonly [number, number], a: number): [number, number] => {
      const d = GRID_MOVES[a] ?? [0, 0]
      return [Math.min(size - 1, Math.max(0, p[0] + d[0])), Math.min(size - 1, Math.max(0, p[1] + d[1]))]
    }
    const observeOne = (s: PursuitState, me: readonly [number, number], self: number, isPursuer: boolean): number[] => {
      if (local) {
        const h = Math.floor(range / 2)
        const out = new Array<number>(3 * range * range).fill(0)
        for (let dr = -h; dr <= h; dr++)
          for (let dc = -h; dc <= h; dc++) {
            const r = me[0] + dr
            const c = me[1] + dc
            const i = (dr + h) * range + (dc + h)
            if (r < 0 || r >= size || c < 0 || c >= size) out[i] = 1
            else {
              let pc = 0
              for (const p of s.pursuers) if (p[0] === r && p[1] === c) pc += 1
              let ec = 0
              for (const [j, e] of s.evaders.entries()) if (s.free[j] === true && e[0] === r && e[1] === c) ec += 1
              out[range * range + i] = Math.min(1, pc / Math.max(1, nP))
              out[2 * range * range + i] = Math.min(1, ec / Math.max(1, nE))
            }
          }
        return out
      }
      const rel = (p: readonly [number, number]): number[] => [(p[0] - me[0]) / size, (p[1] - me[1]) / size]
      const out = [me[0] / size, me[1] / size]
      for (const [j, p] of s.pursuers.entries()) if (!(isPursuer && j === self)) out.push(...rel(p))
      for (const [j, e] of s.evaders.entries()) {
        if (!isPursuer && j === self) continue
        out.push(...(s.free[j] === true ? rel(e) : [0, 0]), s.free[j] === true ? 1 : 0)
      }
      return out
    }
    const live = (s: PursuitState): string[] =>
      s.t >= maxCycles || s.free.every((f) => !f) ? [] : [...pursuerNames, ...evaderNames.filter((_, j) => s.free[j])]
    const observeAll = (s: PursuitState): Record<string, number[]> => {
      const out: Record<string, number[]> = {}
      for (const [i, a] of pursuerNames.entries()) out[a] = observeOne(s, s.pursuers[i] ?? [0, 0], i, true)
      for (const [j, a] of evaderNames.entries()) out[a] = observeOne(s, s.evaders[j] ?? [0, 0], j, false)
      return out
    }
    const scripted = (s: PursuitState, j: number, k: Key): number => {
      if (control === 'random') return Math.min(4, Math.floor(u1(foldIn(k, j)) * 5))
      const e = s.evaders[j] ?? [0, 0]
      let best = 0
      let bestD = -1
      for (let a = 0; a < 5; a++) {
        const q = move(e, a)
        let d = Infinity
        for (const p of s.pursuers) d = Math.min(d, manhattan(p, q))
        if (d > bestD) {
          bestD = d
          best = a
        }
      }
      return best
    }
    return {
      spec: { id: 'PursuitEvasion-v0', description: 'Pursuers chase evaders on a grid.', maxEpisodeSteps: maxCycles },
      possibleAgents: all,
      observationSpaces,
      actionSpaces,
      agents: live,
      reset: (k) => {
        const cells = permutation(k, size * size).data
        const at = (i: number): [number, number] => {
          const c = cells[i] ?? 0
          return [Math.floor(c / size), c % size]
        }
        const state: PursuitState = {
          pursuers: Array.from({ length: nP }, (_, i) => at(i)),
          evaders: Array.from({ length: nE }, (_, j) => at(nP + j)),
          free: new Array<boolean>(nE).fill(true),
          t: 0,
        }
        return { state, observations: observeAll(state), infos: noInfos(all) }
      },
      step: (s, actions, k) => {
        const pursuers = s.pursuers.map((p, i) => move(p, actions[pursuerNames[i] ?? ''] ?? 0))
        const evaders = s.evaders.map((e, j) => {
          if (s.free[j] !== true) return e
          const a = control === 'agent' ? (actions[evaderNames[j] ?? ''] ?? 0) : scripted(s, j, k)
          return move(e, a)
        })
        const rewards: Record<string, number> = {}
        for (const a of pursuerNames) rewards[a] = urgency
        for (const [j, a] of evaderNames.entries()) if (s.free[j] === true) rewards[a] = -urgency
        const free = [...s.free]
        for (const [j, e] of evaders.entries()) {
          if (s.free[j] !== true) continue
          const touching = pursuers.flatMap((p, i) => (manhattan(p, e) <= radius ? [i] : []))
          for (const i of touching) {
            const name = pursuerNames[i] ?? ''
            rewards[name] = (rewards[name] ?? 0) + tagReward
          }
          if (touching.length >= nCatch) {
            free[j] = false
            for (const i of touching) {
              const name = pursuerNames[i] ?? ''
              rewards[name] = (rewards[name] ?? 0) + catchReward
            }
            const ev = evaderNames[j]
            if (ev !== undefined) rewards[ev] = (rewards[ev] ?? 0) - catchReward
          }
        }
        const state: PursuitState = { pursuers, evaders, free, t: s.t + 1 }
        const allCaught = free.every((f) => !f)
        const terminations: Record<string, boolean> = {}
        const truncations: Record<string, boolean> = {}
        for (const a of pursuerNames) {
          terminations[a] = allCaught
          truncations[a] = !allCaught && state.t >= maxCycles
        }
        for (const [j, a] of evaderNames.entries()) {
          if (s.free[j] !== true) continue
          terminations[a] = free[j] !== true
          truncations[a] = free[j] === true && state.t >= maxCycles
        }
        return { state, observations: observeAll(state), rewards, terminations, truncations, infos: noInfos(all) }
      },
      render: (s) => {
        const cell = 40
        const nodes: SceneNode[] = []
        for (let r = 0; r < size; r++)
          for (let c = 0; c < size; c++)
            nodes.push(rect(c * cell, r * cell, cell, cell, { fill: 'bg', stroke: 'grid', strokeWidth: 1 }))
        for (const [j, e] of s.evaders.entries())
          nodes.push(
            circle(e[1] * cell + cell / 2, e[0] * cell + cell / 2, cell / 3, {
              fill: s.free[j] === true ? 'tab3' : 'muted',
              id: `evader-${j}`,
              title: `evader ${j}${s.free[j] === true ? '' : ' (caught)'}`,
            }),
          )
        for (const [i, p] of s.pursuers.entries())
          nodes.push(
            rect(p[1] * cell + cell / 4, p[0] * cell + cell / 4, cell / 2, cell / 2, {
              fill: 'tab0',
              id: `pursuer-${i}`,
              title: `pursuer ${i}`,
            }),
          )
        return scene(size * cell, size * cell, nodes, { title: `Pursuit-evasion, step ${s.t}` })
      },
    }
  },
}

// ------------------------------------------------------------------------------------------ cooperative navigation

/** The state of cooperative navigation: agent positions and velocities, landmark positions and the step count. */
export interface NavigationState {
  readonly positions: readonly (readonly [number, number])[]
  readonly velocities: readonly (readonly [number, number])[]
  readonly landmarks: readonly (readonly [number, number])[]
  readonly t: number
}

/** Options of {@link cooperativeNavigation}. */
export interface NavigationOptions {
  /** Agents and landmarks (default 3). */
  readonly agents?: number
  /** Continuous force actions in [-1, 1]^2 (default false: five discrete moves). */
  readonly continuous?: boolean
  /** Weight of each agent's own collision penalty against the shared reward (default 0.5, as PettingZoo). */
  readonly localRatio?: number
  /** Steps before truncation (default 25). */
  readonly maxCycles?: number
}

/** The physical constants of the multi-agent particle environments (Lowe et al. 2017). */
export const MPE: {
  readonly dt: number
  readonly damping: number
  readonly sensitivity: number
  readonly agentSize: number
  readonly contactForce: number
  readonly contactMargin: number
} = { dt: 0.1, damping: 0.25, sensitivity: 5, agentSize: 0.15, contactForce: 100, contactMargin: 1e-3 }

/**
 * One physics step of the particle world (MPE's `World.step`): actions become forces (times the sensitivity),
 * overlapping agents push each other apart with a softplus contact force, velocities are damped and integrated with
 * the forces (unit mass), then positions with the velocities.
 *
 * @param positions - Agent positions.
 * @param velocities - Agent velocities.
 * @param forces - The control of each agent in [-1, 1]^2.
 * @returns The new positions and velocities.
 * @concept multi-agent
 * @example
 * ```ts
 * import { particleStep } from 'whitebox-engine/methods/rl'
 * particleStep([[0, 0]], [[0, 0]], [[1, 0]]).velocities // => [[0.5, 0]]
 * ```
 */
export function particleStep(
  positions: readonly (readonly [number, number])[],
  velocities: readonly (readonly [number, number])[],
  forces: readonly (readonly number[])[],
): { positions: [number, number][]; velocities: [number, number][] } {
  const n = positions.length
  const f: [number, number][] = forces.map((u) => [(u[0] ?? 0) * MPE.sensitivity, (u[1] ?? 0) * MPE.sensitivity])
  for (let a = 0; a < n; a++)
    for (let b = a + 1; b < n; b++) {
      const pa = positions[a] ?? [0, 0]
      const pb = positions[b] ?? [0, 0]
      const dx = pa[0] - pb[0]
      const dy = pa[1] - pb[1]
      const dist = Math.hypot(dx, dy)
      const k = MPE.contactMargin
      const z = -(dist - 2 * MPE.agentSize) / k
      // log(1 + exp(z)) k, computed stably.
      const penetration = (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z))) * k
      if (dist === 0) continue
      const fx = (MPE.contactForce * dx * penetration) / dist
      const fy = (MPE.contactForce * dy * penetration) / dist
      const fa = f[a] as [number, number]
      const fb = f[b] as [number, number]
      fa[0] += fx
      fa[1] += fy
      fb[0] -= fx
      fb[1] -= fy
    }
  const vel: [number, number][] = velocities.map((v, i) => [
    v[0] * (1 - MPE.damping) + (f[i]?.[0] ?? 0) * MPE.dt,
    v[1] * (1 - MPE.damping) + (f[i]?.[1] ?? 0) * MPE.dt,
  ])
  const pos: [number, number][] = positions.map((p, i) => [
    p[0] + (vel[i]?.[0] ?? 0) * MPE.dt,
    p[1] + (vel[i]?.[1] ?? 0) * MPE.dt,
  ])
  return { positions: pos, velocities: vel }
}

/** The discrete moves of the particle environments: none, left, right, down, up (PettingZoo's MPE order). */
export const PARTICLE_MOVES: readonly (readonly [number, number])[] = [
  [0, 0],
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
]

/**
 * Cooperative navigation (MPE `simple_spread`, Lowe et al. 2017; PettingZoo `simple_spread_v3`): `n` agents in a
 * damped 2-D particle world must cover `n` landmarks without colliding. The shared reward is minus the sum over
 * landmarks of the distance to the nearest agent; each agent also pays 1 per agent it overlaps, mixed with
 * `localRatio`. Actions are five discrete moves or continuous forces in [-1, 1]^2. Observations: own velocity and
 * position, landmark and other-agent positions relative to the agent, and the (silent) communication channels of
 * the others, as in MPE. Agents start uniformly in [-1, 1]^2 and landmarks in [-0.8, 0.8]^2; episodes are truncated
 * after `maxCycles` steps. Each agent's info carries the shared `teamReward`.
 *
 * @concept multi-agent
 * @example
 * ```ts
 * import { cooperativeNavigation } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * cooperativeNavigation.create().reset(key(0)).observations['agent_0']?.length // => 18
 * ```
 */
export const cooperativeNavigation: ParallelDefinition<NavigationState, readonly number[], number | readonly number[]> =
  {
    id: 'CooperativeNavigation-v0',
    description: 'Agents spread out to cover landmarks without colliding (MPE simple_spread).',
    multiAgent: 'parallel',
    options: dict({
      agents: int(1, 8, { default: 3 }),
      continuous: bool({ default: false }),
      localRatio: real(0, 1, { default: 0.5 }),
      maxCycles: int(1, 1000, { default: 25 }),
    }),
    create: (options) => {
      const n = num(options, 'agents', 3)
      const continuous = options?.['continuous'] === true
      const localRatio = num(options, 'localRatio', 0.5)
      const maxCycles = num(options, 'maxCycles', 25)
      const names = Array.from({ length: n }, (_, i) => `agent_${i}`)
      const obsSize = 4 + 2 * n + 4 * (n - 1)
      const obsSpace = box(-Infinity, Infinity, [obsSize])
      const actSpace: Space = continuous
        ? box(-1, 1, [2], { label: 'force (x, y)' })
        : discrete(5, { label: 'none, left, right, down, up' })
      const observe = (s: NavigationState): Record<string, number[]> => {
        const out: Record<string, number[]> = {}
        for (const [i, a] of names.entries()) {
          const p = s.positions[i] ?? [0, 0]
          const v = s.velocities[i] ?? [0, 0]
          const o = [v[0], v[1], p[0], p[1]]
          for (const l of s.landmarks) o.push(l[0] - p[0], l[1] - p[1])
          for (const [j, q] of s.positions.entries()) if (j !== i) o.push(q[0] - p[0], q[1] - p[1])
          for (let j = 0; j < n - 1; j++) o.push(0, 0)
          out[a] = o
        }
        return out
      }
      const live = (s: NavigationState): readonly string[] => (s.t >= maxCycles ? [] : names)
      return {
        spec: {
          id: 'CooperativeNavigation-v0',
          description: 'Agents spread out to cover landmarks without colliding (MPE simple_spread).',
          maxEpisodeSteps: maxCycles,
          deterministic: true,
        },
        possibleAgents: names,
        observationSpaces: Object.fromEntries(names.map((a) => [a, obsSpace])),
        actionSpaces: Object.fromEntries(names.map((a) => [a, actSpace])),
        agents: live,
        reset: (k) => {
          const u = Array.from(uniform(foldIn(k, 0), [n, 2], { low: -1, high: 1 }).data)
          const w = Array.from(uniform(foldIn(k, 1), [n, 2], { low: -1, high: 1 }).data)
          const state: NavigationState = {
            positions: names.map((_, i) => [u[2 * i] ?? 0, u[2 * i + 1] ?? 0]),
            velocities: names.map(() => [0, 0]),
            landmarks: names.map((_, i) => [0.8 * (w[2 * i] ?? 0), 0.8 * (w[2 * i + 1] ?? 0)]),
            t: 0,
          }
          return { state, observations: observe(state), infos: noInfos(names) }
        },
        step: (s, actions) => {
          const forces = names.map((a) => {
            const act = actions[a] ?? 0
            if (typeof act === 'number') return PARTICLE_MOVES[act] ?? [0, 0]
            return [Math.max(-1, Math.min(1, act[0] ?? 0)), Math.max(-1, Math.min(1, act[1] ?? 0))]
          })
          const next = particleStep(s.positions, s.velocities, forces)
          const state: NavigationState = { ...next, landmarks: s.landmarks, t: s.t + 1 }
          const team = navigationCoverage(state)
          const rewards: Record<string, number> = {}
          const infos: Record<string, Readonly<Record<string, number>>> = {}
          for (const [i, a] of names.entries()) {
            const p = state.positions[i] ?? [0, 0]
            let collisions = 0
            for (const [j, q] of state.positions.entries())
              if (j !== i && Math.hypot(p[0] - q[0], p[1] - q[1]) < 2 * MPE.agentSize) collisions += 1
            rewards[a] = team * (1 - localRatio) - collisions * localRatio
            infos[a] = { teamReward: team, collisions }
          }
          const truncated = state.t >= maxCycles
          return {
            state,
            observations: observe(state),
            rewards,
            terminations: flags(names, false),
            truncations: flags(names, truncated),
            infos,
          }
        },
        render: (s) => {
          const px = (x: number): number => 200 + x * 120
          const py = (y: number): number => 200 - y * 120
          const nodes: SceneNode[] = [
            rect(0, 0, 400, 400, { fill: 'bg' }),
            line(0, 200, 400, 200, { stroke: 'grid' }),
            line(200, 0, 200, 400, { stroke: 'grid' }),
          ]
          for (const [i, l] of s.landmarks.entries())
            nodes.push(circle(px(l[0]), py(l[1]), 0.05 * 120, { fill: 'muted', id: `landmark-${i}` }))
          for (const [i, p] of s.positions.entries())
            nodes.push(
              circle(px(p[0]), py(p[1]), MPE.agentSize * 120, {
                fill: `tab${(i % 10) as 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9}`,
                opacity: 0.8,
                id: `agent-${i}`,
                title: `agent ${i}`,
              }),
            )
          nodes.push(text(8, 392, `step ${s.t}, coverage ${navigationCoverage(s).toFixed(2)}`, { size: 12 }))
          return scene(400, 400, nodes, { title: 'Cooperative navigation' })
        },
      }
    },
  }

/**
 * The shared reward of a cooperative-navigation state: minus the sum over landmarks of the distance to the nearest
 * agent (zero when every landmark is covered).
 *
 * @param state - The state.
 * @returns The coverage reward.
 * @concept multi-agent
 * @example
 * ```ts
 * import { navigationCoverage } from 'whitebox-engine/methods/rl'
 * navigationCoverage({ positions: [[0, 0]], velocities: [[0, 0]], landmarks: [[3, 4]], t: 0 }) // => -5
 * ```
 */
export function navigationCoverage(state: NavigationState): number {
  let total = 0
  for (const l of state.landmarks) {
    let best = Infinity
    for (const p of state.positions) best = Math.min(best, Math.hypot(p[0] - l[0], p[1] - l[1]))
    total -= best
  }
  return total
}
