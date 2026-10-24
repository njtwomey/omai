import { maxOf, WhiteboxError } from '../../../lib/base/index.ts'
import { permutation } from '../../../lib/random/index.ts'
import type { AecEnv } from '../../../lib/rl/index.ts'
import { rect, type Scene, type SceneNode, scene, text } from '../../../lib/scene/index.ts'
import { bool, type DictSpace, dict, int, tuple } from '../../../lib/space/index.ts'
import {
  boardPlanes,
  drawBoard,
  oneHotList,
  other,
  type Seat,
  type TurnState,
  twoPlayerGame,
  winLoss,
} from './_two-player.ts'

/**
 * A named factory of a turn-based two-player game (the multi-agent counterpart of `EnvDefinition`): `create(options)`
 * returns an {@link AecEnv} with agents `player_0` (who moves first) and `player_1`, legal-action masks, per-agent
 * observations, rewards to both agents (+1 and -1 for a win, 0 for a draw; chips in poker) and a rendered scene. An
 * illegal action forfeits the game, as PettingZoo's illegal-move wrapper does.
 *
 * @concept multi-agent
 */
export interface GameDefinition<S, O = readonly number[], A = number> {
  readonly id: string
  readonly description: string
  /** Marks a multi-agent definition, so registries of single-agent environments skip it. */
  readonly multiAgent: 'aec'
  readonly options?: DictSpace
  create(options?: Readonly<Record<string, unknown>>): AecEnv<S, O, A>
}

const winnerCaption = (s: TurnState & { readonly winner?: Seat }, names: readonly [string, string]): string => {
  if (s.forfeit !== undefined) return `${names[s.forfeit]} forfeits`
  if (!s.over) return `${names[s.toMove]} to move`
  return s.winner === undefined ? 'draw' : `${names[s.winner]} wins`
}

// --------------------------------------------------------------------------------------------------- tic-tac-toe

/** A tic-tac-toe position: nine cells row by row (0 empty, 1 a cross of `player_0`, 2 a nought of `player_1`). */
export interface TicTacToeState extends TurnState {
  readonly board: readonly number[]
  readonly winner?: Seat
}

/** The eight lines of three of a 3 x 3 board. */
export const TIC_TAC_TOE_LINES: readonly (readonly [number, number, number])[] = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
]

const lineOf = (board: readonly number[], piece: number): readonly number[] | undefined =>
  TIC_TAC_TOE_LINES.find((l) => l.every((i) => board[i] === piece))

/**
 * Tic-tac-toe (OpenSpiel's `tic_tac_toe`, PettingZoo's `tictactoe_v3`): `player_0` plays crosses and moves first;
 * actions 0..8 are the cells row by row; three in a row wins +1 (the loser gets -1), a full board is a draw.
 * Observations are 27 numbers: planes of the observer's marks, the opponent's marks and the empty cells.
 *
 * @concept multi-agent
 * @example
 * ```ts
 * import { ticTacToe } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = ticTacToe.create()
 * env.legalActions?.(env.reset(key(0)), 'player_0').length // => 9
 * ```
 */
export const ticTacToe: GameDefinition<TicTacToeState> = {
  id: 'TicTacToe-v0',
  description: 'Three in a row on a 3 x 3 board.',
  multiAgent: 'aec',
  create: () =>
    twoPlayerGame<TicTacToeState>({
      id: 'TicTacToe-v0',
      description: 'Three in a row on a 3 x 3 board.',
      nActions: 9,
      actionLabel: 'cell (row * 3 + column)',
      observationSize: 27,
      observationLabel: 'own marks, opponent marks, empty cells (3 x 9)',
      deterministic: true,
      reset: () => ({ board: new Array<number>(9).fill(0), toMove: 0, over: false }),
      legal: (s) => s.board.flatMap((c, i) => (c === 0 ? [i] : [])),
      isLegal: (s, a) => a >= 0 && a < 9 && s.board[a] === 0,
      play: (s, a) => {
        const board = s.board.map((c, i) => (i === a ? s.toMove + 1 : c))
        const won = lineOf(board, s.toMove + 1) !== undefined
        const full = board.every((c) => c !== 0)
        const over = won || full
        const winner = won ? s.toMove : undefined
        return {
          state: { board, toMove: other(s.toMove), over, ...(winner === undefined ? {} : { winner }) },
          rewards: over ? winLoss(winner) : [0, 0],
        }
      },
      observe: (s, seat) => boardPlanes(s.board, seat),
      render: (s) =>
        drawBoard({
          rows: 3,
          cols: 3,
          cells: s.board,
          style: 'marks',
          title: 'Tic-tac-toe',
          cell: 96,
          highlight: s.winner === undefined ? [] : (lineOf(s.board, s.winner + 1) ?? []),
          caption: winnerCaption(s, ['X', 'O']),
        }),
    }),
}

// -------------------------------------------------------------------------------------------------- connect four

/** A connect-four position: 6 x 7 cells row by row from the top (0 empty, 1 `player_0`, 2 `player_1`). */
export interface ConnectFourState extends TurnState {
  readonly board: readonly number[]
  readonly winner?: Seat
  /** The four (or more) winning cells. */
  readonly line?: readonly number[]
}

const C4_ROWS = 6
const C4_COLS = 7

/**
 * The cells of the longest line through a cell made of one player's pieces, if it has four or more (connect-four win
 * detection).
 *
 * @param board - The 6 x 7 board row by row from the top.
 * @param cell - The cell just played.
 * @returns The winning cells, or `undefined` when no line of four passes through the cell.
 * @concept multi-agent
 * @example
 * ```ts
 * import { connectFourLine } from 'whitebox-engine/methods/rl'
 * const board = new Array(42).fill(0)
 * for (const c of [35, 36, 37, 38]) board[c] = 1
 * connectFourLine(board, 38) // => [35, 36, 37, 38]
 * ```
 */
export function connectFourLine(board: readonly number[], cell: number): number[] | undefined {
  const piece = board[cell] ?? 0
  if (piece === 0) return undefined
  const r0 = Math.floor(cell / C4_COLS)
  const c0 = cell % C4_COLS
  const at = (r: number, c: number): number =>
    r >= 0 && r < C4_ROWS && c >= 0 && c < C4_COLS ? (board[r * C4_COLS + c] ?? 0) : 0
  for (const [dr, dc] of [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ] as const) {
    const cells = [cell]
    for (let k = 1; at(r0 - k * dr, c0 - k * dc) === piece; k++) cells.unshift((r0 - k * dr) * C4_COLS + c0 - k * dc)
    for (let k = 1; at(r0 + k * dr, c0 + k * dc) === piece; k++) cells.push((r0 + k * dr) * C4_COLS + c0 + k * dc)
    if (cells.length >= 4) return cells
  }
  return undefined
}

/**
 * Connect four (OpenSpiel's `connect_four`, PettingZoo's `connect_four_v3`): a 6 x 7 board; actions 0..6 drop a piece
 * into a column that is not full; four in a row horizontally, vertically or diagonally wins (+1 and -1), a full board
 * draws. Observations are 126 numbers: planes of own pieces, opponent pieces and empty cells.
 *
 * @concept multi-agent
 * @example
 * ```ts
 * import { connectFour } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = connectFour.create()
 * env.step(env.reset(key(0)), 3, key(1)).state.board[38] // => 1
 * ```
 */
export const connectFour: GameDefinition<ConnectFourState> = {
  id: 'ConnectFour-v0',
  description: 'Four in a row on a 6 x 7 board, dropping pieces into columns.',
  multiAgent: 'aec',
  create: () => {
    const top = (board: readonly number[], col: number): number => {
      for (let r = C4_ROWS - 1; r >= 0; r--) if (board[r * C4_COLS + col] === 0) return r
      return -1
    }
    return twoPlayerGame<ConnectFourState>({
      id: 'ConnectFour-v0',
      description: 'Four in a row on a 6 x 7 board, dropping pieces into columns.',
      nActions: C4_COLS,
      actionLabel: 'column',
      observationSize: 3 * C4_ROWS * C4_COLS,
      observationLabel: 'own pieces, opponent pieces, empty cells (3 x 6 x 7)',
      deterministic: true,
      reset: () => ({ board: new Array<number>(C4_ROWS * C4_COLS).fill(0), toMove: 0, over: false }),
      legal: (s) => Array.from({ length: C4_COLS }, (_, c) => c).filter((c) => s.board[c] === 0),
      isLegal: (s, a) => a >= 0 && a < C4_COLS && s.board[a] === 0,
      play: (s, col) => {
        const cell = top(s.board, col) * C4_COLS + col
        const board = s.board.map((c, i) => (i === cell ? s.toMove + 1 : c))
        const line = connectFourLine(board, cell)
        const over = line !== undefined || board.every((c) => c !== 0)
        const winner = line === undefined ? undefined : s.toMove
        return {
          state: {
            board,
            toMove: other(s.toMove),
            over,
            ...(winner === undefined ? {} : { winner }),
            ...(line === undefined ? {} : { line }),
          },
          rewards: over ? winLoss(winner) : [0, 0],
        }
      },
      observe: (s, seat) => boardPlanes(s.board, seat),
      render: (s) =>
        drawBoard({
          rows: C4_ROWS,
          cols: C4_COLS,
          cells: s.board,
          style: 'discs',
          title: 'Connect four',
          cell: 64,
          highlight: s.line ?? [],
          caption: winnerCaption(s, ['red', 'yellow']),
        }),
    })
  },
}

// ----------------------------------------------------------------------------------------------------------- Nim

/** A Nim position: the heap sizes. */
export interface NimState extends TurnState {
  readonly heaps: readonly number[]
  readonly winner?: Seat
}

/** Options of {@link nim}. */
export interface NimOptions {
  /** The starting heaps (default `[1, 3, 5, 7]`). */
  readonly heaps?: readonly number[]
  /** Misère play: whoever takes the last object loses (default false: normal play, taking the last object wins). */
  readonly misere?: boolean
}

/**
 * Decode a Nim action: action `heap * maxTake + (take - 1)` takes `take` objects from `heap`, where `maxTake` is the
 * largest starting heap.
 *
 * @param action - The action.
 * @param maxTake - The largest starting heap.
 * @returns The heap and the number of objects taken.
 * @concept multi-agent
 * @example
 * ```ts
 * import { nimMove } from 'whitebox-engine/methods/rl'
 * nimMove(9, 7) // => { heap: 1, take: 3 }
 * ```
 */
export function nimMove(action: number, maxTake: number): { readonly heap: number; readonly take: number } {
  return { heap: Math.floor(action / maxTake), take: (action % maxTake) + 1 }
}

/**
 * Nim (Bouton 1901; OpenSpiel's `nim`): players alternately take one or more objects from a single heap. Under normal
 * play whoever takes the last object wins (+1, the other -1); with `misere` whoever takes it loses. Action
 * `heap * maxTake + (take - 1)` takes `take` objects from `heap` (`maxTake` is the largest starting heap). The mover
 * loses under perfect normal play exactly when the xor of the heaps is zero. Observations are one-hot heap sizes
 * (one block of `start + 1` slots per heap) followed by a 1 when the observer is to move.
 *
 * @concept multi-agent
 * @example
 * ```ts
 * import { nim } from 'whitebox-engine/methods/rl'
 * import { playAec } from 'whitebox-engine/rl'
 * import { key } from 'whitebox-engine/random'
 * playAec(nim.create({ heaps: [3] }), { player_0: () => 0, player_1: () => 0 }, key(0)).returns // => { player_0: 1, player_1: -1 }
 * ```
 */
export const nim: GameDefinition<NimState> = {
  id: 'Nim-v0',
  description: 'Take objects from heaps; normal play: taking the last object wins.',
  multiAgent: 'aec',
  options: dict({
    heaps: tuple(
      [int(1, 31, { default: 1 }), int(1, 31, { default: 3 }), int(1, 31, { default: 5 }), int(1, 31, { default: 7 })],
      {
        description: 'the starting heaps (any number of heaps)',
      },
    ),
    misere: bool({ default: false, description: 'whoever takes the last object loses' }),
  }),
  create: (options) => {
    const opts = (options ?? {}) as NimOptions
    const start = [...(opts.heaps ?? [1, 3, 5, 7])]
    if (start.length === 0 || start.some((h) => !Number.isInteger(h) || h < 1))
      throw new WhiteboxError('value', 'nim: heaps must be positive integers')
    const misere = opts.misere === true
    const maxTake = maxOf(start)
    const size = start.reduce((a, h) => a + h + 1, 0) + 1
    return twoPlayerGame<NimState>({
      id: 'Nim-v0',
      description: 'Take objects from heaps; normal play: taking the last object wins.',
      nActions: start.length * maxTake,
      actionLabel: 'heap * maxTake + (take - 1)',
      observationSize: size,
      observationLabel: 'one-hot heap sizes, then whether the observer is to move',
      deterministic: true,
      reset: () => ({ heaps: start, toMove: 0, over: false }),
      legal: (s) => s.heaps.flatMap((h, i) => Array.from({ length: h }, (_, t) => i * maxTake + t)),
      isLegal: (s, a) => {
        const { heap, take } = nimMove(a, maxTake)
        return a >= 0 && heap < s.heaps.length && take <= (s.heaps[heap] ?? 0)
      },
      play: (s, a) => {
        const { heap, take } = nimMove(a, maxTake)
        const heaps = s.heaps.map((h, i) => (i === heap ? h - take : h))
        const over = heaps.every((h) => h === 0)
        const winner = over ? (misere ? other(s.toMove) : s.toMove) : undefined
        return {
          state: { heaps, toMove: other(s.toMove), over, ...(winner === undefined ? {} : { winner }) },
          rewards: over ? winLoss(winner) : [0, 0],
        }
      },
      observe: (s, seat) => [
        ...s.heaps.flatMap((h, i) => oneHotList(h, (start[i] ?? 0) + 1)),
        s.over || s.toMove !== seat ? 0 : 1,
      ],
      render: (s) => {
        const cell = 32
        const nodes: SceneNode[] = []
        for (const [i, h] of s.heaps.entries()) {
          nodes.push(text(8, (i + 1) * cell * 1.5, `heap ${i}`, { baseline: 'middle', size: 14 }))
          for (let j = 0; j < (start[i] ?? 0); j++) {
            nodes.push(
              rect(80 + j * cell, (i + 1) * cell * 1.5 - cell / 2 + 2, cell - 4, cell - 4, {
                fill: j < h ? 'tab0' : 'none',
                stroke: j < h ? 'fg' : 'grid',
                radius: 4,
              }),
            )
          }
        }
        const height = (s.heaps.length + 1) * cell * 1.5 + cell
        nodes.push(text(8, height - cell / 2, winnerCaption(s, ['player_0', 'player_1']), { size: 14 }))
        return scene(100 + maxTake * cell, height, nodes, { title: misere ? 'Nim (misère)' : 'Nim' })
      },
    })
  },
}

// --------------------------------------------------------------------------------------------------- Kuhn poker

/** A Kuhn poker hand: the private cards (0 jack, 1 queen, 2 king) and the actions so far (0 pass, 1 bet). */
export interface KuhnPokerState extends TurnState {
  readonly cards: readonly [number, number]
  readonly history: readonly number[]
}

const KUHN_TERMINAL = new Set(['0,0', '0,1,0', '0,1,1', '1,0', '1,1'])
const CARD_NAMES = ['J', 'Q', 'K']

/**
 * The start of a Kuhn poker hand with given cards (for exact analysis over the six deals).
 *
 * @param cards - The cards of `player_0` and `player_1` (distinct, in 0..2).
 * @returns The state before the first action.
 * @concept multi-agent
 * @example
 * ```ts
 * import { kuhnPokerState } from 'whitebox-engine/methods/rl'
 * kuhnPokerState([2, 0]).toMove // => 0
 * ```
 */
export function kuhnPokerState(cards: readonly [number, number]): KuhnPokerState {
  return { cards, history: [], toMove: 0, over: false }
}

/**
 * The chips each player wins at the end of a Kuhn poker hand (both ante 1; a bet or call adds 1; a pass facing a bet
 * folds; otherwise the higher card takes the pot).
 *
 * @param cards - The two private cards.
 * @param history - A terminal action sequence (`[0, 0]`, `[0, 1, 0]`, `[0, 1, 1]`, `[1, 0]` or `[1, 1]`).
 * @returns The payoffs of `player_0` and `player_1`.
 * @concept multi-agent
 * @example
 * ```ts
 * import { kuhnPayoff } from 'whitebox-engine/methods/rl'
 * kuhnPayoff([0, 2], [1, 1]) // => [-2, 2]
 * ```
 */
export function kuhnPayoff(cards: readonly [number, number], history: readonly number[]): [number, number] {
  const h = history.join(',')
  if (!KUHN_TERMINAL.has(h)) throw new WhiteboxError('value', `kuhnPayoff: ${h} is not a terminal history`)
  if (h === '0,1,0') return [-1, 1]
  if (h === '1,0') return [1, -1]
  const stake = h === '0,0' ? 1 : 2
  return cards[0] > cards[1] ? [stake, -stake] : [-stake, stake]
}

/**
 * Kuhn poker (Kuhn 1950; OpenSpiel's `kuhn_poker`): a three-card deck (jack, queen, king), one private card each dealt
 * at reset from the key, antes of 1, and one betting round with actions pass (0) and bet (1). The game value for
 * `player_0` under Nash play is -1/18. Observations are OpenSpiel's information-state tensor (11 numbers): the
 * observer's seat (2), its own card (3, the opponent's card stays hidden) and the actions so far (3 x pass/bet).
 *
 * @concept multi-agent
 * @example
 * ```ts
 * import { kuhnPoker } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = kuhnPoker.create()
 * env.observe(env.reset(key(0)), 'player_0').length // => 11
 * ```
 */
export const kuhnPoker: GameDefinition<KuhnPokerState> = {
  id: 'KuhnPoker-v0',
  description: 'Three-card poker with one betting round.',
  multiAgent: 'aec',
  create: () =>
    twoPlayerGame<KuhnPokerState>({
      id: 'KuhnPoker-v0',
      description: 'Three-card poker with one betting round.',
      nActions: 2,
      actionLabel: 'pass (0) or bet (1)',
      observationSize: 11,
      observationLabel: 'seat (2), private card (3), actions (3 x 2)',
      deterministic: true,
      reset: (k) => {
        const deck = permutation(k, 3).data
        return kuhnPokerState([deck[0] ?? 0, deck[1] ?? 1])
      },
      legal: () => [0, 1],
      play: (s, a) => {
        const history = [...s.history, a]
        const over = KUHN_TERMINAL.has(history.join(','))
        return {
          state: { ...s, history, toMove: other(s.toMove), over },
          rewards: over ? kuhnPayoff(s.cards, history) : [0, 0],
        }
      },
      observe: (s, seat) => [
        ...oneHotList(seat, 2),
        ...oneHotList(s.cards[seat], 3),
        ...[0, 1, 2].flatMap((i) => {
          const a = s.history[i]
          return a === undefined ? [0, 0] : oneHotList(a, 2)
        }),
      ],
      render: (s) =>
        renderPoker(
          s.cards,
          undefined,
          s.over,
          s.history.map((a) => (a === 1 ? 'bet' : 'pass')),
          CARD_NAMES,
        ),
    }),
}

// -------------------------------------------------------------------------------------------------- Leduc poker

/** A Leduc poker hand. */
export interface LeducPokerState extends TurnState {
  /** The cards dealt at reset: `player_0`'s, `player_1`'s and the public card (revealed in the second round). */
  readonly cards: readonly [number, number, number]
  readonly round: 0 | 1
  /** The actions of each round (0 fold, 1 check or call, 2 bet or raise). */
  readonly history: readonly [readonly number[], readonly number[]]
  readonly contributions: readonly [number, number]
  readonly raises: number
  readonly folded?: Seat
}

/** The rank of a Leduc card (0 jack, 1 queen, 2 king); the six cards 0..5 come in two suits. */
export const leducRank = (card: number): number => Math.floor(card / 2)

/**
 * The start of a Leduc poker hand with given cards (for exact analysis over the deals).
 *
 * @param cards - The private cards of `player_0` and `player_1` and the public card (distinct, in 0..5).
 * @returns The state before the first action.
 * @concept multi-agent
 * @example
 * ```ts
 * import { leducPokerState } from 'whitebox-engine/methods/rl'
 * leducPokerState([0, 5, 2]).contributions // => [1, 1]
 * ```
 */
export function leducPokerState(cards: readonly [number, number, number]): LeducPokerState {
  return { cards, round: 0, history: [[], []], contributions: [1, 1], raises: 0, toMove: 0, over: false }
}

/**
 * The showdown winner of Leduc poker: a private card pairing the public card wins, otherwise the higher rank; equal
 * ranks split the pot.
 *
 * @param cards - The two private cards and the public card.
 * @returns The winning seat, or `undefined` for a split.
 * @concept multi-agent
 * @example
 * ```ts
 * import { leducShowdown } from 'whitebox-engine/methods/rl'
 * leducShowdown([0, 5, 1]) // => 0
 * ```
 */
export function leducShowdown(cards: readonly [number, number, number]): Seat | undefined {
  const [a, b, pub] = cards.map(leducRank) as [number, number, number]
  if (a === pub && b !== pub) return 0
  if (b === pub && a !== pub) return 1
  if (a === b) return undefined
  return a > b ? 0 : 1
}

const LEDUC_RAISE = [2, 4]
const LEDUC_MAX_RAISES = 2
const LEDUC_NAMES = ['J♠', 'J♥', 'Q♠', 'Q♥', 'K♠', 'K♥']

/**
 * Leduc hold'em (Southey et al. 2005; OpenSpiel's `leduc_poker`): a six-card deck of two suits of jack, queen and
 * king; antes of 1; one private card each and a public card, all dealt at reset from the key (the public card is
 * revealed after the first round). Two betting rounds with bets and raises of 2 in the first and 4 in the second, at
 * most two raises per round. Actions: fold (0, legal only facing a bet), check or call (1), bet or raise (2). A pair
 * with the public card wins the showdown, then the higher card; equal ranks split. Rewards are the chips won or lost.
 * Observations are OpenSpiel's information-state tensor (30 numbers): seat (2), private card (6), public card (6,
 * zero until revealed), and each round's actions as call and raise bits (2 rounds x 4 actions x 2).
 *
 * @concept multi-agent
 * @example
 * ```ts
 * import { leducPoker, leducPokerState } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = leducPoker.create()
 * env.legalActions?.(leducPokerState([0, 5, 2]), 'player_0') // => [1, 2]
 * ```
 */
export const leducPoker: GameDefinition<LeducPokerState> = {
  id: 'LeducPoker-v0',
  description: "Leduc hold'em: six cards, a public card and two betting rounds.",
  multiAgent: 'aec',
  create: () => {
    const legal = (s: LeducPokerState): number[] => {
      const facing = s.contributions[s.toMove] < s.contributions[other(s.toMove)]
      return [...(facing ? [0] : []), 1, ...(s.raises < LEDUC_MAX_RAISES ? [2] : [])]
    }
    return twoPlayerGame<LeducPokerState>({
      id: 'LeducPoker-v0',
      description: "Leduc hold'em: six cards, a public card and two betting rounds.",
      nActions: 3,
      actionLabel: 'fold (0), check or call (1), bet or raise (2)',
      observationSize: 30,
      observationLabel: 'seat (2), private card (6), public card (6), actions (2 rounds x 4 x 2)',
      deterministic: true,
      reset: (k) => {
        const deck = permutation(k, 6).data
        return leducPokerState([deck[0] ?? 0, deck[1] ?? 1, deck[2] ?? 2])
      },
      legal,
      // A fold that faces no bet forfeits like a fold: the folder loses its contribution.
      forfeitRewards: (s, seat) =>
        seat === 0 ? [-s.contributions[0], s.contributions[0]] : [s.contributions[1], -s.contributions[1]],
      play: (s, a) => {
        const me = s.toMove
        const them = other(me)
        const history: [number[], number[]] = [[...s.history[0]], [...s.history[1]]]
        history[s.round].push(a)
        if (a === 0) {
          const lost = s.contributions[me]
          return {
            state: { ...s, history, over: true, folded: me },
            rewards: me === 0 ? [-lost, lost] : [lost, -lost],
          }
        }
        const contributions: [number, number] = [s.contributions[0], s.contributions[1]]
        let raises = s.raises
        if (a === 2) {
          contributions[me] = contributions[them] + (LEDUC_RAISE[s.round] ?? 0)
          raises += 1
        } else contributions[me] = contributions[them]
        // A check or call ends the round unless it opens the round.
        const roundOver = a === 1 && history[s.round].length >= 2
        if (!roundOver) return { state: { ...s, history, contributions, raises, toMove: them }, rewards: [0, 0] }
        if (s.round === 0) {
          return { state: { ...s, history, contributions, raises: 0, round: 1, toMove: 0 }, rewards: [0, 0] }
        }
        const winner = leducShowdown(s.cards)
        const pot = contributions[0]
        return {
          state: { ...s, history, contributions, raises, over: true },
          rewards: winner === undefined ? [0, 0] : winner === 0 ? [pot, -pot] : [-pot, pot],
        }
      },
      observe: (s, seat) => {
        const revealed = s.round === 1 || (s.over && s.folded === undefined)
        return [
          ...oneHotList(seat, 2),
          ...oneHotList(s.cards[seat], 6),
          ...(revealed ? oneHotList(s.cards[2], 6) : new Array<number>(6).fill(0)),
          ...[0, 1].flatMap((r) =>
            [0, 1, 2, 3].flatMap((i) => {
              const a = s.history[r as 0 | 1][i]
              return a === 1 ? [1, 0] : a === 2 ? [0, 1] : [0, 0]
            }),
          ),
        ]
      },
      render: (s) =>
        renderPoker(
          [s.cards[0], s.cards[1]],
          s.round === 1 || s.over ? s.cards[2] : undefined,
          s.over,
          s.history.flatMap((h, r) => h.map((a) => `${r === 0 ? 'r1' : 'r2'} ${['fold', 'call', 'raise'][a] ?? '?'}`)),
          LEDUC_NAMES,
          s.contributions,
        ),
    })
  },
}

function renderPoker(
  cards: readonly [number, number],
  publicCard: number | undefined,
  showdown: boolean,
  actions: readonly string[],
  names: readonly string[],
  contributions?: readonly [number, number],
): Scene {
  const card = (x: number, y: number, label: string, hidden: boolean, id: string): SceneNode[] => [
    rect(x, y, 60, 84, { fill: hidden ? 'tab0' : 'bg', stroke: 'fg', strokeWidth: 1, radius: 6, id }),
    ...(hidden ? [] : [text(x + 30, y + 42, label, { anchor: 'middle', baseline: 'middle', size: 22 })]),
  ]
  const nodes: SceneNode[] = [
    rect(0, 0, 420, 300, { fill: '#076324' }),
    text(20, 24, 'player_0', { fill: '#ffffff', size: 16 }),
    ...card(20, 36, names[cards[0]] ?? '?', false, 'card-0'),
    text(320, 24, 'player_1', { fill: '#ffffff', size: 16 }),
    ...card(320, 36, names[cards[1]] ?? '?', !showdown, 'card-1'),
    ...(publicCard === undefined ? [] : card(180, 36, names[publicCard] ?? '?', false, 'public')),
    text(20, 160, actions.length === 0 ? 'no actions yet' : actions.join(', '), { fill: '#ffffff', size: 14 }),
  ]
  if (contributions !== undefined)
    nodes.push(text(20, 190, `pot ${contributions[0] + contributions[1]}`, { fill: '#ffffff', size: 14 }))
  return scene(420, 300, nodes, { title: 'Poker' })
}
