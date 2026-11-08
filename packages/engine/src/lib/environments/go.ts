/**
 * Go on small boards under Tromp-Taylor rules (as OpenSpiel's `go`): black and white alternately place stones or pass,
 * groups without liberties are captured, and the game is scored by area. A two-player turn-based environment, its area
 * scores, and a way to set up a position from a text diagram for problems and tests.
 */

import { OmaiError } from '../base/index.ts'
import { key as makeKey, randomBits } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { boardPlanes, drawBoard, other, type Seat, type TurnState, twoPlayerGame, winLoss } from './_two-player.ts'
import type { GameDefinition } from './games.ts'

/**
 * A Go position. `player_0` plays black and moves first; `player_1` plays white. Points are numbered row by row;
 * action `size * size` is a pass.
 */
export interface GoState extends TurnState {
  readonly size: number
  /** The points row by row: 0 empty, 1 black, 2 white. */
  readonly board: readonly number[]
  /** Consecutive passes so far (two end the game). */
  readonly passes: number
  readonly moves: number
  /** The point where an immediate recapture is forbidden (simple ko only). */
  readonly ko?: number
  /** The Zobrist hash of the board. */
  readonly hash: number
  /** Hashes of every board position so far (for positional superko). */
  readonly history: readonly number[]
  readonly winner?: Seat
  /** The final area scores of black and white (white's including komi). */
  readonly score?: readonly [number, number]
}

/** Options of {@link go}. */
export interface GoOptions {
  /** The board size: 5, 7 or 9 (default 9). */
  readonly size?: number
  /** Points added to white's score (default 7.5, as OpenSpiel). */
  readonly komi?: number
  /** `'positional'` superko (default: no move may repeat an earlier board) or `'simple'` ko (no immediate recapture). */
  readonly ko?: 'positional' | 'simple'
  /** The game ends and is scored after this many moves (default `2 * size * size`, as OpenSpiel). */
  readonly maxMoves?: number
}

// Zobrist keys: 26 bits per (point, colour), so a hash fits exactly in a double as hi * 2^26 + lo.
const ZOBRIST_BITS = randomBits(makeKey(0x60), 4 * 81 * 2)
const zobrist = (point: number, colour: number): readonly [number, number] => {
  const i = 2 * (point * 2 + (colour - 1))
  return [(ZOBRIST_BITS[i] ?? 0) & 0x3ffffff, (ZOBRIST_BITS[i + 1] ?? 0) & 0x3ffffff]
}
const split = (h: number): [number, number] => [Math.floor(h / 0x4000000), h % 0x4000000]
const join = (hi: number, lo: number): number => hi * 0x4000000 + lo
const toggle = (h: number, point: number, colour: number): number => {
  const [hi, lo] = split(h)
  const [zh, zl] = zobrist(point, colour)
  return join((hi ^ zh) >>> 0, (lo ^ zl) >>> 0)
}

const neighbourCache = new Map<number, readonly (readonly number[])[]>()
function neighbours(size: number): readonly (readonly number[])[] {
  const cached = neighbourCache.get(size)
  if (cached !== undefined) return cached
  const out = Array.from({ length: size * size }, (_, p) => {
    const r = Math.floor(p / size)
    const c = p % size
    const n: number[] = []
    if (r > 0) n.push(p - size)
    if (r < size - 1) n.push(p + size)
    if (c > 0) n.push(p - 1)
    if (c < size - 1) n.push(p + 1)
    return n
  })
  neighbourCache.set(size, out)
  return out
}

/** The stones of the group at a point and its number of liberties. */
function group(board: readonly number[], size: number, p: number): { stones: number[]; liberties: number } {
  const colour = board[p]
  const nb = neighbours(size)
  const seen = new Set([p])
  const libs = new Set<number>()
  const stack = [p]
  const stones: number[] = []
  while (stack.length > 0) {
    const q = stack.pop() as number
    stones.push(q)
    for (const n of nb[q] ?? []) {
      const v = board[n]
      if (v === 0) libs.add(n)
      else if (v === colour && !seen.has(n)) {
        seen.add(n)
        stack.push(n)
      }
    }
  }
  return { stones, liberties: libs.size }
}

/**
 * Tromp-Taylor area scores: each player's stones plus the empty points that reach only that player's stones.
 *
 * @param board - The points row by row (0 empty, 1 black, 2 white).
 * @param size - The board size.
 * @returns The areas of black and white (no komi).
 * @concept multi-agent
 * @example
 * ```ts
 * import { goArea } from 'omai-engine/environments'
 * goArea([1, 0, 0, 0, 1, 0, 0, 0, 2], 3) // => [2, 1]
 * ```
 */
export function goArea(board: readonly number[], size: number): [number, number] {
  const nb = neighbours(size)
  const area: [number, number] = [0, 0]
  const seen = new Set<number>()
  for (const [p, v] of board.entries()) {
    if (v === 1) area[0] += 1
    else if (v === 2) area[1] += 1
    else if (!seen.has(p)) {
      const stack = [p]
      seen.add(p)
      let count = 0
      let touches = 0
      while (stack.length > 0) {
        const q = stack.pop() as number
        count += 1
        for (const n of nb[q] ?? []) {
          const w = board[n]
          if (w === 0) {
            if (!seen.has(n)) {
              seen.add(n)
              stack.push(n)
            }
          } else if (w !== undefined) touches |= w
        }
      }
      if (touches === 1) area[0] += count
      else if (touches === 2) area[1] += count
    }
  }
  return area
}

/**
 * The result of playing a stone: the new board with captures removed, its hash and the simple-ko point, or
 * `undefined` when the move is illegal (occupied, suicide, or a ko violation under the given rule).
 */
function placeStone(
  s: GoState,
  p: number,
  rule: 'positional' | 'simple',
): { board: number[]; hash: number; ko: number | undefined } | undefined {
  const { size } = s
  if (p < 0 || p >= size * size || s.board[p] !== 0) return undefined
  if (rule === 'simple' && s.ko === p) return undefined
  const colour = s.toMove + 1
  const opponent = 3 - colour
  const board = [...s.board]
  board[p] = colour
  let hash = toggle(s.hash, p, colour)
  const captured: number[] = []
  for (const n of neighbours(size)[p] ?? []) {
    if (board[n] !== opponent) continue
    const g = group(board, size, n)
    if (g.liberties > 0) continue
    for (const q of g.stones) {
      board[q] = 0
      hash = toggle(hash, q, opponent)
      captured.push(q)
    }
  }
  const own = group(board, size, p)
  if (own.liberties === 0) return undefined
  if (rule === 'positional' && s.history.includes(hash)) return undefined
  const ko = captured.length === 1 && own.stones.length === 1 && own.liberties === 1 ? captured[0] : undefined
  return { board, hash, ko }
}

/**
 * A Go position from a diagram: one string per row with `X` (or `B`) for black, `O` (or `W`) for white and `.` for
 * empty points (for setting up problems and tests).
 *
 * @param rows - The rows, top first; the board is square.
 * @param toMove - The seat to move (0 black, default; 1 white).
 * @returns The state, with the diagram as its only previous position.
 * @throws OmaiError with code `value` when the diagram is not square.
 * @concept multi-agent
 * @example
 * ```ts
 * import { goState } from 'omai-engine/environments'
 * goState(['X..', '.O.', '...']).board // => [1, 0, 0, 0, 2, 0, 0, 0, 0]
 * ```
 */
export function goState(rows: readonly string[], toMove: Seat = 0): GoState {
  const size = rows.length
  if (rows.some((r) => r.length !== size)) throw new OmaiError('value', 'goState: the diagram must be square')
  const board = rows.flatMap((r) => [...r].map((ch) => ('XB'.includes(ch) ? 1 : 'OW'.includes(ch) ? 2 : 0)))
  let hash = 0
  for (const [p, v] of board.entries()) if (v !== 0) hash = toggle(hash, p, v)
  return { size, board, toMove, over: false, passes: 0, moves: 0, hash, history: [hash] }
}

/**
 * Go on small boards (OpenSpiel's `go`, Tromp-Taylor rules): black (`player_0`) and white alternately place a stone
 * on an empty point or pass (action `size * size`); groups without liberties are captured; suicide is illegal;
 * positional superko forbids recreating any earlier board (or, with `ko: 'simple'`, only an immediate single-stone
 * recapture is forbidden). Two consecutive passes (or `maxMoves`) end the game, scored by area (stones plus empty
 * regions reaching only one colour) with `komi` added to white; the higher score wins +1 (-1 to the other), equal
 * scores draw. Observations are four planes: own stones, opponent stones, empty points, and ones when the observer
 * plays black.
 *
 * @concept multi-agent
 * @example
 * ```ts
 * import { go } from 'omai-engine/environments'
 * import { playAec } from 'omai-engine/rl'
 * import { key } from 'omai-engine/random'
 * const env = go.create({ size: 5 })
 * const pass = () => 25
 * playAec(env, { player_0: pass, player_1: pass }, key(0)).returns // => { player_0: -1, player_1: 1 }
 * ```
 */
export const go: GameDefinition<GoState> = {
  id: 'Go-v0',
  description: 'Go on a small board with Tromp-Taylor area scoring.',
  multiAgent: 'aec',
  options: dict({
    size: choice([5, 7, 9], { default: 9 }),
    komi: real(0, 10, { default: 7.5, step: 0.5, description: 'points added to white' }),
    ko: choice(['positional', 'simple'], { default: 'positional', description: 'superko or simple ko' }),
    maxMoves: int(1, 1000, { default: 162, description: 'moves before the game is scored' }),
  }),
  create: (options) => {
    const opts = (options ?? {}) as GoOptions
    const size = opts.size ?? 9
    if (![5, 7, 9].includes(size)) throw new OmaiError('value', 'go: size must be 5, 7 or 9')
    const komi = opts.komi ?? 7.5
    const rule = opts.ko ?? 'positional'
    const maxMoves = opts.maxMoves ?? 2 * size * size
    const points = size * size
    const finish = (s: GoState): { state: GoState; rewards: readonly [number, number] } => {
      const [black, white] = goArea(s.board, size)
      const score: [number, number] = [black, white + komi]
      const winner: Seat | undefined = score[0] > score[1] ? 0 : score[1] > score[0] ? 1 : undefined
      return {
        state: { ...s, over: true, score, ...(winner === undefined ? {} : { winner }) },
        rewards: winLoss(winner),
      }
    }
    return twoPlayerGame<GoState>({
      id: 'Go-v0',
      description: 'Go on a small board with Tromp-Taylor area scoring.',
      nActions: points + 1,
      actionLabel: 'point (row * size + column), or size * size to pass',
      observationSize: 4 * points,
      observationLabel: 'own stones, opponent stones, empty points, observer plays black (4 x size x size)',
      deterministic: true,
      reset: () => goState(Array.from({ length: size }, () => '.'.repeat(size))),
      legal: (s) => {
        const out: number[] = []
        for (let p = 0; p < points; p++) if (s.board[p] === 0 && placeStone(s, p, rule) !== undefined) out.push(p)
        out.push(points)
        return out
      },
      isLegal: (s, a) => a === points || placeStone(s, a, rule) !== undefined,
      play: (s, a) => {
        const moves = s.moves + 1
        if (a === points) {
          const { ko: _ko, ...rest } = s
          const next: GoState = { ...rest, toMove: other(s.toMove), passes: s.passes + 1, moves }
          return next.passes >= 2 || moves >= maxMoves ? finish(next) : { state: next, rewards: [0, 0] }
        }
        const placed = placeStone(s, a, rule)
        if (placed === undefined) throw new OmaiError('value', 'go: illegal move')
        const { ko: _ko, ...rest } = s
        const next: GoState = {
          ...rest,
          board: placed.board,
          hash: placed.hash,
          history: [...s.history, placed.hash],
          toMove: other(s.toMove),
          passes: 0,
          moves,
          ...(rule === 'simple' && placed.ko !== undefined ? { ko: placed.ko } : {}),
        }
        return moves >= maxMoves ? finish(next) : { state: next, rewards: [0, 0] }
      },
      observe: (s, seat) => [...boardPlanes(s.board, seat), ...new Array<number>(points).fill(seat === 0 ? 1 : 0)],
      render: (s) =>
        drawBoard({
          rows: size,
          cols: size,
          cells: s.board,
          style: 'stones',
          title: `Go ${size} x ${size}`,
          cell: 40,
          caption:
            s.score === undefined
              ? s.forfeit !== undefined
                ? `${s.forfeit === 0 ? 'black' : 'white'} forfeits`
                : `${s.toMove === 0 ? 'black' : 'white'} to move`
              : `black ${s.score[0]}, white ${s.score[1]}: ${s.winner === undefined ? 'draw' : s.winner === 0 ? 'black wins' : 'white wins'}`,
        }),
    })
  },
}
