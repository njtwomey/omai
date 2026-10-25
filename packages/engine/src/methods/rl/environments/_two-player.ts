// Shared plumbing of the two-player games (private to methods/rl/environments): seats and agent names, the turn-based
// environment built from a game's rules (with illegal moves forfeiting the game), and board drawing.
import { MdmlError } from '../../../lib/base/index.ts'
import type { Key } from '../../../lib/random/index.ts'
import type { AecEnv, PerAgent } from '../../../lib/rl/index.ts'
import { circle, line, rect, type Scene, type SceneNode, scene, text } from '../../../lib/scene/index.ts'
import { box, discrete, type Space } from '../../../lib/space/index.ts'

/** The two agents of every two-player game; seat 0 moves first. */
export const PLAYERS: readonly ['player_0', 'player_1'] = ['player_0', 'player_1']

/** A seat at the table: 0 for `player_0`, 1 for `player_1`. */
export type Seat = 0 | 1

/** What every two-player game state carries: the seat to move, whether the game is over and any forfeit. */
export interface TurnState {
  readonly toMove: Seat
  readonly over: boolean
  /** The seat that forfeited the game by an illegal move. */
  readonly forfeit?: Seat
}

/** The rules of a two-player game, from which {@link twoPlayerGame} builds the turn-based environment. */
export interface TwoPlayerRules<S extends TurnState> {
  readonly id: string
  readonly description: string
  readonly nActions: number
  readonly actionLabel: string
  readonly observationSize: number
  readonly observationLabel: string
  readonly deterministic: boolean
  reset(key: Key): S
  legal(state: S): readonly number[]
  /** Whether one action is legal (default: membership in `legal`). */
  isLegal?(state: S, action: number): boolean
  /** Play a legal action: the next state and the rewards of the two seats. */
  play(state: S, action: number, key: Key): { readonly state: S; readonly rewards: readonly [number, number] }
  /** The rewards when a seat forfeits (default -1 to it and +1 to the other). */
  forfeitRewards?(state: S, seat: Seat): readonly [number, number]
  observe(state: S, seat: Seat): readonly number[]
  render(state: S): Scene
}

/** The seat of an agent name. */
export function seatOf(agent: string): Seat {
  if (agent === 'player_0') return 0
  if (agent === 'player_1') return 1
  throw new MdmlError('value', `no seat for agent ${agent}`)
}

/** The other seat. */
export const other = (seat: Seat): Seat => (seat === 0 ? 1 : 0)

/** Rewards of the two seats keyed by agent. */
export const perSeat = (r: readonly [number, number]): PerAgent<number> => ({ player_0: r[0], player_1: r[1] })

/** +1 to the winner and -1 to the loser (0 to both for a draw). */
export const winLoss = (winner: Seat | undefined): readonly [number, number] =>
  winner === undefined ? [0, 0] : winner === 0 ? [1, -1] : [-1, 1]

/** The turn-based environment of a two-player game: illegal actions forfeit the game (PettingZoo's convention). */
export function twoPlayerGame<S extends TurnState>(rules: TwoPlayerRules<S>): AecEnv<S, readonly number[], number> {
  const obs: Space = box(0, 1, [rules.observationSize], { label: rules.observationLabel })
  const act: Space = discrete(rules.nActions, { label: rules.actionLabel })
  const isLegal = rules.isLegal ?? ((s: S, a: number) => rules.legal(s).includes(a))
  const flags = (s: S): PerAgent<boolean> => ({ player_0: s.over, player_1: s.over })
  return {
    spec: { id: rules.id, description: rules.description, deterministic: rules.deterministic },
    possibleAgents: PLAYERS,
    observationSpaces: { player_0: obs, player_1: obs },
    actionSpaces: { player_0: act, player_1: act },
    reset: (k) => rules.reset(k),
    currentAgent: (s) => (s.over ? undefined : PLAYERS[s.toMove]),
    observe: (s, agent) => rules.observe(s, seatOf(agent)),
    legalActions: (s, agent) => (s.over || seatOf(agent) !== s.toMove ? [] : rules.legal(s)),
    step: (s, action, k) => {
      if (s.over) throw new MdmlError('value', `${rules.id}: the game is over`)
      if (!Number.isInteger(action) || !isLegal(s, action)) {
        const seat = s.toMove
        const rewards = rules.forfeitRewards?.(s, seat) ?? winLoss(other(seat))
        return { state: { ...s, over: true, forfeit: seat }, rewards: perSeat(rewards) }
      }
      const r = rules.play(s, action, k)
      return { state: r.state, rewards: perSeat(r.rewards) }
    },
    terminations: flags,
    truncations: () => ({ player_0: false, player_1: false }),
    render: (s) => rules.render(s),
  }
}

/** Planes of a board from one seat's perspective: its own pieces, the opponent's, and the empty cells. */
export function boardPlanes(cells: readonly number[], seat: Seat): number[] {
  const mine = seat + 1
  const theirs = 2 - seat
  const n = cells.length
  const out = new Array<number>(3 * n).fill(0)
  for (const [i, c] of cells.entries()) {
    if (c === mine) out[i] = 1
    else if (c === theirs) out[n + i] = 1
    else out[2 * n + i] = 1
  }
  return out
}

/** One-hot encoding of `index` in `n` slots (all zeros when the index is outside). */
export function oneHotList(index: number, n: number): number[] {
  const out = new Array<number>(n).fill(0)
  if (index >= 0 && index < n) out[index] = 1
  return out
}

/** Options of {@link drawBoard}. */
export interface BoardDrawing {
  readonly rows: number
  readonly cols: number
  /** Cell contents row by row: 0 empty, 1 the first player's piece, 2 the second's. */
  readonly cells: readonly number[]
  readonly style: 'marks' | 'discs' | 'stones'
  readonly title: string
  readonly cell?: number
  /** Cells to highlight (a winning line, the last move). */
  readonly highlight?: readonly number[]
  /** A caption under the board. */
  readonly caption?: string
}

/**
 * Draw a board game: noughts and crosses (`marks`), coloured discs in a frame (`discs`), or stones on the
 * intersections of a grid (`stones`).
 */
export function drawBoard(d: BoardDrawing): Scene {
  const cell = d.cell ?? 48
  const pad = cell / 2
  const width = d.cols * cell + 2 * pad
  const height = d.rows * cell + 2 * pad + (d.caption === undefined ? 0 : cell / 2)
  const nodes: SceneNode[] = []
  const lit = new Set(d.highlight ?? [])
  if (d.style === 'stones') {
    nodes.push(rect(0, 0, width, height, { fill: '#dcb35c' }))
    const step = cell
    const x0 = pad + cell / 2
    const y0 = pad + cell / 2
    for (let i = 0; i < d.rows; i++)
      nodes.push(line(x0, y0 + i * step, x0 + (d.cols - 1) * step, y0 + i * step, { stroke: '#000000' }))
    for (let j = 0; j < d.cols; j++)
      nodes.push(line(x0 + j * step, y0, x0 + j * step, y0 + (d.rows - 1) * step, { stroke: '#000000' }))
    for (const [i, c] of d.cells.entries()) {
      if (c === 0) continue
      const r = Math.floor(i / d.cols)
      const q = i % d.cols
      nodes.push(
        circle(x0 + q * step, y0 + r * step, cell * 0.45, {
          fill: c === 1 ? '#000000' : '#ffffff',
          stroke: lit.has(i) ? 'tab3' : '#000000',
          strokeWidth: lit.has(i) ? 3 : 1,
          id: `stone-${i}`,
          title: `${c === 1 ? 'black' : 'white'} stone at row ${r}, column ${q}`,
        }),
      )
    }
  } else {
    if (d.style === 'discs') nodes.push(rect(pad, pad, d.cols * cell, d.rows * cell, { fill: 'tab0' }))
    for (const [i, c] of d.cells.entries()) {
      const r = Math.floor(i / d.cols)
      const q = i % d.cols
      const x = pad + q * cell
      const y = pad + r * cell
      const cx = x + cell / 2
      const cy = y + cell / 2
      if (d.style === 'discs') {
        nodes.push(
          circle(cx, cy, cell * 0.4, {
            fill: c === 0 ? 'bg' : c === 1 ? 'tab3' : 'tab8',
            stroke: lit.has(i) ? 'fg' : 'none',
            strokeWidth: 3,
            id: `cell-${i}`,
          }),
        )
        continue
      }
      nodes.push(
        rect(x, y, cell, cell, {
          fill: lit.has(i) ? 'surface' : 'bg',
          stroke: 'grid',
          strokeWidth: 2,
          id: `cell-${i}`,
        }),
      )
      const m = cell * 0.3
      if (c === 1) {
        nodes.push(line(cx - m, cy - m, cx + m, cy + m, { stroke: 'tab0', strokeWidth: 4 }))
        nodes.push(line(cx - m, cy + m, cx + m, cy - m, { stroke: 'tab0', strokeWidth: 4 }))
      } else if (c === 2) nodes.push(circle(cx, cy, m, { fill: 'none', stroke: 'tab1', strokeWidth: 4 }))
    }
  }
  if (d.caption !== undefined)
    nodes.push(text(width / 2, height - cell / 3, d.caption, { anchor: 'middle', baseline: 'middle', size: cell / 3 }))
  return scene(width, height, nodes, { title: d.title })
}
