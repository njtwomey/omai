/**
 * Langton's ant and its generalisations: an ant on a grid of coloured cells that turns by a rule written on the cell's
 * colour (`RL` is the classic: right on white, left on black), flips the colour and steps forward. The classic ant
 * wanders chaotically for about ten thousand steps and then builds a diagonal highway forever; other rule strings
 * (`LLRR`, `LRRRRRLLR`; `LR` is the classic's mirror image) make other patterns, and several ants share a board. The ant as a traced algorithm, and the
 * detector that finds the highway from the ant's track.
 */

import { OmaiError } from '../../base/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  event,
  scalar,
  tensor,
} from '../../trace/index.ts'
import { codec, draws, mod, positiveInt } from './_core.ts'

/** Options of {@link langtonsAnt}. */
export interface AntOptions {
  /** The turns by colour, `L`, `R`, `N` (straight) or `U` (turn round): `LR` is Langton's (default `RL`). */
  readonly rule?: string
  /** The board is a torus this wide and high (default 100 by 100). */
  readonly width?: number
  readonly height?: number
  /** How many ants (default 1): the first in the middle facing up, the rest at random places and headings. */
  readonly ants?: number
  /** Moves of every ant per step (default 1). */
  readonly movesPerStep?: number
  /** The highway's period in moves (default 104, the classic ant's). */
  readonly highwayPeriod?: number
}

/** The state of {@link langtonsAnt}. */
export interface AntState {
  /** The colour of every cell, row-major. */
  readonly cells: Uint8Array
  /** Per ant: column, row, heading (0 up, 1 right, 2 down, 3 left). */
  readonly ants: Int32Array
  /** Moves made by each ant. */
  readonly moves: number
  /** The first ant's position on the unwrapped plane (the torus unrolled), where it would be without wrapping. */
  readonly unwrapped: readonly [number, number]
  /** The first ant's unwrapped positions at the last few multiples of the highway period, `[x, y, ...]`, oldest first. */
  readonly track: readonly number[]
  /** The move at which the highway was recognised (-1 until then). */
  readonly highwayAt: number
}

/** The paths of {@link langtonsAnt}. */
export type AntPath = 'ant/board' | 'ant/position' | 'ant/heading' | 'ant/blackCells' | 'ant/highway'

/** The log contract of {@link langtonsAnt}. */
export const antContract: Contract<AntState, AntPath> = defineContract({
  'ant/board': tensor({ role: 'state', shape: ['height', 'width'], description: 'the colour of every cell' }),
  'ant/position': tensor({ role: 'state', shape: ['ants', 3], description: 'column, row and heading of every ant' }),
  'ant/heading': scalar({ description: "the first ant's heading, 0 up, 1 right, 2 down, 3 left" }),
  'ant/blackCells': scalar({ description: 'cells that are not the first colour' }),
  'ant/highway': event({ context: { move: 'the move the highway was recognised at' } }),
})

/**
 * Whether a track is on a highway: the positions one period apart, `repeats` times in a row, all differ by the same
 * nonzero displacement (the ant is translating, not wandering).
 *
 * @param track - Flat unwrapped positions `[x, y, x, y, ...]` sampled once per `period` moves, oldest first.
 * @param repeats - How many equal displacements are wanted (default 3).
 * @returns The displacement `[dx, dy]` per period, or undefined if the track is not on a highway.
 * @concept langtons-ant
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * automata.highwayDetected([0, 0, 2, 2, 4, 4, 6, 6]) // => [2, 2]
 * automata.highwayDetected([0, 0, 2, 2, 3, 3, 6, 6]) // => undefined
 * ```
 */
export function highwayDetected(track: readonly number[], repeats = 3): [number, number] | undefined {
  const n = track.length / 2
  if (n < repeats + 1) return undefined
  const d = (k: number): [number, number] => [
    (track[2 * k + 2] ?? 0) - (track[2 * k] ?? 0),
    (track[2 * k + 3] ?? 0) - (track[2 * k + 1] ?? 0),
  ]
  const last = d(n - 2)
  if (last[0] === 0 && last[1] === 0) return undefined
  for (let k = n - repeats - 1; k < n - 2; k++) {
    const e = d(k)
    if (e[0] !== last[0] || e[1] !== last[1]) return undefined
  }
  return last
}

/**
 * Langton's ant (or several, with any rule string) as a traced algorithm: `movesPerStep` moves of every ant per step.
 * Each move reads the colour under the ant, turns by that colour's letter, advances the colour, and steps forward on
 * the torus. The state serialises. `ant/highway` fires, once, when the first ant's track shows it translating by the
 * same displacement three periods running.
 *
 * @param options - The rule, size, number of ants, moves per step and highway period.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a rule that is not made of `L`, `R`, `N`, `U`, or a bad size.
 * @concept langtons-ant
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // After 5 moves the ant has left three black cells and faces left, one cell left of where it began.
 * const s = runAlgorithm(automata.langtonsAnt({ width: 9, height: 9 }), { seed: 0, maxSteps: 5 }).state
 * [s.cells.reduce((a, b) => a + b, 0), Array.from(s.ants)] // => [3, [3, 4, 3]]
 * ```
 */
export function langtonsAnt(options: AntOptions = {}): Algorithm<AntState, AntPath> {
  const rule = options.rule ?? 'RL'
  const width = options.width ?? 100
  const height = options.height ?? 100
  const count = options.ants ?? 1
  const perStep = options.movesPerStep ?? 1
  const period = options.highwayPeriod ?? 104
  if (!/^[LRNU]{2,}$/.test(rule))
    throw new OmaiError('value', `langtonsAnt: rule '${rule}' must be two or more of L, R, N, U`)
  if (rule.length > 255) throw new OmaiError('value', 'langtonsAnt: too many colours')
  positiveInt('langtonsAnt', 'width', width)
  positiveInt('langtonsAnt', 'height', height)
  positiveInt('langtonsAnt', 'ants', count)
  positiveInt('langtonsAnt', 'movesPerStep', perStep)
  positiveInt('langtonsAnt', 'highwayPeriod', period)
  const turn = Int8Array.from(rule, (ch) => (ch === 'R' ? 1 : ch === 'L' ? 3 : ch === 'U' ? 2 : 0))
  const colours = rule.length
  const keep = 2 * (3 + 1)
  return {
    name: 'dynamics/automata/langtons-ant',
    config: { rule, width, height, ants: count, movesPerStep: perStep, highwayPeriod: period },
    schema: contractSchema(antContract),
    init: (key) => {
      const ants = new Int32Array(3 * count)
      ants[0] = Math.floor(width / 2)
      ants[1] = Math.floor(height / 2)
      ants[2] = 0
      if (count > 1) {
        const u = draws(key, 3 * count)
        for (let a = 1; a < count; a++) {
          ants[3 * a] = Math.floor((u[3 * a] ?? 0) * width)
          ants[3 * a + 1] = Math.floor((u[3 * a + 1] ?? 0) * height)
          ants[3 * a + 2] = Math.floor((u[3 * a + 2] ?? 0) * 4)
        }
      }
      const origin: [number, number] = [ants[0] ?? 0, ants[1] ?? 0]
      return { cells: new Uint8Array(width * height), ants, moves: 0, unwrapped: origin, track: origin, highwayAt: -1 }
    },
    step: (s, ctx) => {
      const cells = Uint8Array.from(s.cells)
      const ants = Int32Array.from(s.ants)
      let track = s.track
      let highwayAt = s.highwayAt
      let moves = s.moves
      let [ux, uy] = s.unwrapped
      let since = moves % period
      for (let m = 0; m < perStep; m++) {
        for (let a = 0; a < count; a++) {
          const x = ants[3 * a] ?? 0
          const y = ants[3 * a + 1] ?? 0
          const idx = y * width + x
          const colour = cells[idx] ?? 0
          const heading = mod((ants[3 * a + 2] ?? 0) + (turn[colour] ?? 0), 4)
          cells[idx] = (colour + 1) % colours
          const dx = heading === 1 ? 1 : heading === 3 ? -1 : 0
          const dy = heading === 2 ? 1 : heading === 0 ? -1 : 0
          ants[3 * a] = mod(x + dx, width)
          ants[3 * a + 1] = mod(y + dy, height)
          ants[3 * a + 2] = heading
          if (a === 0) {
            ux += dx
            uy += dy
          }
        }
        moves += 1
        since += 1
        if (since === period) {
          since = 0
          track = [...track, ux, uy].slice(-keep)
          if (highwayAt < 0 && highwayDetected(track) !== undefined) {
            highwayAt = moves
            ctx.emit('ant/highway', { move: moves })
          }
        }
      }
      return { cells, ants, moves, unwrapped: [ux, uy], track, highwayAt }
    },
    observe: (s) => {
      let black = 0
      for (const c of s.cells) if (c !== 0) black++
      return {
        metrics: { 'ant/heading': s.ants[2] ?? 0, 'ant/blackCells': black },
        tensors: {
          'ant/board': { role: 'state', value: s.cells, shape: [height, width] },
          'ant/position': { role: 'state', value: s.ants, shape: [count, 3] },
        },
      }
    },
    ...codec<AntState>(),
  }
}
