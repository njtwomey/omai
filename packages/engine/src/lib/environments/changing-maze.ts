/**
 * The mazes that change under the agent, Sutton and Barto's tests of Dyna-Q+ (Examples 8.2 and 8.3): the blocking maze,
 * where the only gap in a wall closes and another opens at the far end after a number of steps, and the shortcut maze,
 * where a second, shorter gap opens while the long way stays open. The clock counts steps across episodes (an episode
 * restart keeps it), the change is flagged in the step's info, and the truth is the shortest path before and after.
 */

import { MdmlError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import type { Env, EnvDefinition } from '../rl/index.ts'
import { scene } from '../scene/index.ts'
import { dict, discrete, int } from '../space/index.ts'
import { destination, distanceField, fromAscii, toScene, type World } from '../worlds/index.ts'

/** Which maze changes: the gap moves (`blocking`) or a second gap opens (`shortcut`). */
export type ChangingMazeKind = 'blocking' | 'shortcut'

const LAYOUTS: Readonly<Record<ChangingMazeKind, readonly [readonly string[], readonly string[]]>> = {
  blocking: [
    ['........G', '.........', '########.', '.........', '.........', '...S.....'],
    ['........G', '.........', '.########', '.........', '.........', '...S.....'],
  ],
  shortcut: [
    ['........G', '.........', '.########', '.........', '.........', '...S.....'],
    ['........G', '.........', '.######.#', '.........', '.........', '...S.....'],
  ],
}

/** The step at which each maze changes in the books. */
export const CHANGING_MAZE_DEFAULT_CHANGE: Readonly<Record<ChangingMazeKind, number>> = {
  blocking: 1000,
  shortcut: 3000,
}

/** The state of a changing maze: the agent's cell and the number of steps taken since the first reset. */
export interface ChangingMazeState {
  readonly cell: number
  readonly t: number
}

/** Options of {@link makeChangingMaze}. */
export interface ChangingMazeOptions {
  /** The maze (default `blocking`). */
  readonly kind?: ChangingMazeKind
  /** The step at which the maze changes (default 1000 for blocking, 3000 for the shortcut maze). */
  readonly changeAt?: number
}

/** The truth of a changing maze: both layouts, the shortest path in each and the optimal actions. */
export interface ChangingMazeTruth {
  readonly kind: ChangingMazeKind
  readonly changeAt: number
  /** The maze before and after the change. */
  readonly worlds: readonly [World, World]
  /** The start and goal cells. */
  readonly start: number
  readonly goal: number
  /** The shortest path from the start to the goal, in steps, before and after the change. */
  readonly shortest: readonly [number, number]
  /** The number of steps to the goal from every cell, before and after the change (infinite for walls). */
  readonly distanceToGoal: readonly [Float64Array, Float64Array]
  /** The optimal actions (0 up, 1 right, 2 down, 3 left) from a cell in a phase: those that reach the goal fastest. */
  optimalActions(phase: 0 | 1, cell: number): number[]
}

/**
 * The truth of a changing maze by breadth-first search over each layout.
 *
 * @param kind - Which maze.
 * @param changeAt - The step of the change (default as in the books).
 * @returns The layouts, shortest paths, distance fields and the optimal actions of each phase.
 * @example
 * ```ts
 * import { changingMazeTruth } from 'mdml-engine/environments'
 * changingMazeTruth('blocking').shortest // => [10, 16]
 * changingMazeTruth('shortcut').shortest // => [16, 10]
 * ```
 */
export function changingMazeTruth(kind: ChangingMazeKind = 'blocking', changeAt?: number): ChangingMazeTruth {
  const [a, b] = LAYOUTS[kind]
  const w0 = fromAscii(a)
  const w1 = fromAscii(b)
  const start = w0.starts[0] ?? 0
  const goal = w0.goals[0] ?? 0
  const distances: [Float64Array, Float64Array] = [distanceField(w0, goal), distanceField(w1, goal)]
  return {
    kind,
    changeAt: changeAt ?? CHANGING_MAZE_DEFAULT_CHANGE[kind],
    worlds: [w0, w1],
    start,
    goal,
    shortest: [distances[0][start] ?? Number.POSITIVE_INFINITY, distances[1][start] ?? Number.POSITIVE_INFINITY],
    distanceToGoal: distances,
    optimalActions: (phase, cell) => {
      const w = phase === 0 ? w0 : w1
      const d = distances[phase]
      const here = d[cell] ?? Number.POSITIVE_INFINITY
      const out: number[] = []
      for (let dir = 0; dir < 4; dir++) {
        const to = destination(w, cell, dir)
        if (to >= 0 && (d[to] ?? Number.POSITIVE_INFINITY) === here - 1) out.push(dir)
      }
      return out
    },
  }
}

/**
 * A changing maze as an environment: a 6 by 9 grid with a wall across the third row, the agent starting at the bottom
 * and the goal in the top right corner, which pays 1 (every other step 0) and ends the episode. Observations are the
 * cell index (54 cells), actions are up, right, down and left, and walls block. The clock `t` counts every step since
 * the first reset and survives episode restarts (`resetFrom`), so the maze changes once, at step `changeAt`; the step
 * that makes the change reports `info.changed`, and `info.phase` is 0 before the change and 1 after.
 *
 * @param options - Which maze and when it changes.
 * @returns The environment.
 * @throws MdmlError with code `value` for a negative change step.
 * @example
 * ```ts
 * import { makeChangingMaze } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = makeChangingMaze({ kind: 'blocking' })
 * env.reset(key(0)).observation // => 48
 * ```
 */
export function makeChangingMaze(options: ChangingMazeOptions = {}): Env<ChangingMazeState, number, number> {
  const kind = options.kind ?? 'blocking'
  const truth = changingMazeTruth(kind, options.changeAt)
  if (!(truth.changeAt >= 0)) throw new MdmlError('value', `changing maze: change step ${truth.changeAt} is negative`)
  const phaseOf = (t: number): 0 | 1 => (t >= truth.changeAt ? 1 : 0)
  const start = (t: number): { state: ChangingMazeState; observation: number; info: Record<string, number> } => ({
    state: { cell: truth.start, t },
    observation: truth.start,
    info: { phase: phaseOf(t) },
  })
  const id = kind === 'blocking' ? 'BlockingMaze-v0' : 'ShortcutMaze-v0'
  return {
    spec: {
      id,
      description: 'A maze whose wall changes after a number of steps; the agent must notice and adapt.',
      maxEpisodeSteps: 1000,
    },
    observationSpace: discrete(truth.worlds[0].n, { label: 'cell (row * 9 + column)' }),
    actionSpace: discrete(4, { label: 'up (0), right (1), down (2), left (3)' }),
    reset: (_k: Key) => start(0),
    resetFrom: (previous: ChangingMazeState, _k: Key) => start(previous.t),
    step: (s, action) => {
      const phase = phaseOf(s.t)
      const next = destination(truth.worlds[phase], s.cell, action)
      const cell = next < 0 ? s.cell : next
      const t = s.t + 1
      const reached = cell === truth.goal
      return {
        state: { cell, t },
        observation: cell,
        reward: reached ? 1 : 0,
        terminated: reached,
        truncated: false,
        info: { phase: phaseOf(t), changed: phase === 0 && phaseOf(t) === 1 ? 1 : 0 },
      }
    },
    render: (s) => {
      const pic = toScene(truth.worlds[phaseOf(s.t)], { agent: s.cell, cell: 40 })
      return scene(pic.width, pic.height, pic.children, { title: id })
    },
    observe: (s) => s.cell,
  }
}

/**
 * BlockingMaze-v0 (Sutton and Barto, Example 8.2): the gap in the wall moves from the right end to the left after 1000
 * steps, so the shortest path grows from 10 to 16 steps and the learnt path is blocked.
 *
 * @concept environments
 * @example
 * ```ts
 * import { blockingMaze } from 'mdml-engine/environments'
 * blockingMaze.create().observationSpace.kind // => 'discrete'
 * ```
 */
export const blockingMaze: EnvDefinition<ChangingMazeState, number, number> = {
  id: 'BlockingMaze-v0',
  description: 'A maze whose only gap moves from one end of the wall to the other after a number of steps.',
  options: dict({ changeAt: int(0, 1_000_000, { default: 1000 }) }),
  create: (options) =>
    makeChangingMaze({
      kind: 'blocking',
      ...(typeof options?.['changeAt'] === 'number' ? { changeAt: options['changeAt'] } : {}),
    }),
}

/**
 * ShortcutMaze-v0 (Sutton and Barto, Example 8.3): a second gap opens in the wall after 3000 steps and the shortest
 * path falls from 16 to 10 steps, while the long way learnt before stays open (so only exploration finds the shortcut).
 *
 * @concept environments
 * @example
 * ```ts
 * import { shortcutMaze } from 'mdml-engine/environments'
 * shortcutMaze.create().actionSpace.kind // => 'discrete'
 * ```
 */
export const shortcutMaze: EnvDefinition<ChangingMazeState, number, number> = {
  id: 'ShortcutMaze-v0',
  description: 'A maze in which a shorter gap opens in the wall after a number of steps while the long way stays open.',
  options: dict({ changeAt: int(0, 1_000_000, { default: 3000 }) }),
  create: (options) =>
    makeChangingMaze({
      kind: 'shortcut',
      ...(typeof options?.['changeAt'] === 'number' ? { changeAt: options['changeAt'] } : {}),
    }),
}
