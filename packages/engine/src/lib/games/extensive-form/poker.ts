/**
 * Small poker games used as benchmarks for imperfect-information solvers, with OpenSpiel's rules and information
 * states: Kuhn poker (three cards, one betting round) with its known family of Nash equilibria, and Leduc poker (six
 * cards, two betting rounds and a public card).
 */

import { OmaiError } from '../../base/index.ts'
import { CHANCE, TERMINAL } from './_tree.ts'
import type { ExtensiveGame, TabularPolicy } from './tree.ts'

/** A Kuhn poker state: the cards dealt so far and the betting actions (0 pass, 1 bet). */
export interface KuhnState {
  readonly cards: readonly number[]
  readonly history: readonly number[]
}

const KUHN_TERMINAL = ['00', '010', '011', '10', '11']

/**
 * Kuhn poker (Kuhn 1950) with OpenSpiel's rules and information states (`kuhn_poker`): a deck of three cards (0 jack,
 * 1 queen, 2 king), antes of 1, one private card each (chance deals player 0's card, then player 1's), and one
 * betting round with actions pass (0) and bet (1); a pass facing a bet folds, otherwise the higher card takes the
 * pot. Information states are the player's card followed by the actions as `p` and `b` (`'1pb'`). The value for
 * player 0 is -1/18.
 *
 * @returns The game.
 * @concept kuhn-poker
 * @example
 * ```ts
 * import { gameTree, kuhnPoker } from 'omai-engine/games'
 * gameTree(kuhnPoker()).infosets.map((i) => i.key).slice(0, 4) // => ['0', '1p', '0pb', '1b']
 * ```
 */
export function kuhnPoker(): ExtensiveGame<KuhnState> {
  const done = (h: readonly number[]): boolean => KUHN_TERMINAL.includes(h.join(''))
  return {
    name: 'kuhn_poker',
    numPlayers: 2,
    initial: () => ({ cards: [], history: [] }),
    currentPlayer: (s) => (s.cards.length < 2 ? CHANCE : done(s.history) ? TERMINAL : s.history.length % 2),
    legalActions: (s) => (s.cards.length < 2 || done(s.history) ? [] : [0, 1]),
    chanceOutcomes: (s) => {
      if (s.cards.length >= 2) return []
      const left = [0, 1, 2].filter((c) => !s.cards.includes(c))
      return left.map((c) => [c, 1 / left.length] as const)
    },
    apply: (s, a) =>
      s.cards.length < 2
        ? { cards: [...s.cards, a], history: s.history }
        : { cards: s.cards, history: [...s.history, a] },
    returns: (s) => {
      const h = s.history.join('')
      if (!done(s.history)) return [0, 0]
      const winner = (s.cards[0] ?? 0) > (s.cards[1] ?? 0) ? 0 : 1
      // A fold: the bettor takes the other's ante.
      if (h === '10') return [1, -1]
      if (h === '010') return [-1, 1]
      const stake = h === '00' ? 1 : 2
      return winner === 0 ? [stake, -stake] : [-stake, stake]
    },
    informationState: (s, p) => `${s.cards[p] ?? ''}${s.history.map((a) => (a === 0 ? 'p' : 'b')).join('')}`,
    actionName: (s, a) => (s.cards.length < 2 ? `deal ${a}` : a === 0 ? 'pass' : 'bet'),
  }
}

/**
 * Kuhn's (1950) family of Nash equilibria of {@link kuhnPoker}, indexed by player 0's bluffing frequency `alpha` in
 * `[0, 1/3]`: player 0 bets a jack with probability alpha and a king with 3 alpha, checks a queen and calls a bet on
 * it with alpha + 1/3; player 1 bets a jack after a check with 1/3, calls a bet with a queen with 1/3, and always
 * bets and calls with a king.
 *
 * @param alpha - Player 0's jack-betting probability (default 0).
 * @returns The equilibrium policy of both players.
 * @throws OmaiError with code `value` for alpha outside `[0, 1/3]`.
 * @concept kuhn-poker
 * @example
 * ```ts
 * import { expectedReturns, kuhnEquilibrium, kuhnPoker } from 'omai-engine/games'
 * Math.round(expectedReturns(kuhnPoker(), kuhnEquilibrium(0.1))[0] * 18 * 1e9) / 1e9 // => -1
 * ```
 */
export function kuhnEquilibrium(alpha = 0): TabularPolicy {
  if (!(alpha >= 0 && alpha <= 1 / 3)) throw new OmaiError('value', 'kuhnEquilibrium: alpha must be in [0, 1/3]')
  const call = alpha + 1 / 3
  return {
    '0': [1 - alpha, alpha],
    '1': [1, 0],
    '2': [1 - 3 * alpha, 3 * alpha],
    '0pb': [1, 0],
    '1pb': [1 - call, call],
    '2pb': [0, 1],
    '0p': [2 / 3, 1 / 3],
    '0b': [1, 0],
    '1p': [1, 0],
    '1b': [2 / 3, 1 / 3],
    '2p': [0, 1],
    '2b': [0, 1],
  }
}

/** A Leduc poker state: the private cards, the public card, and the betting actions of each round (0 fold, 1 call, 2 raise). */
export interface LeducState {
  readonly cards: readonly number[]
  readonly publicCard: number | null
  readonly rounds: readonly (readonly number[])[]
  /** Chips each player has put in the pot (antes included). */
  readonly contributions: readonly [number, number]
  readonly folded: number | null
}

const RAISE = [2, 4]

/** Whether a betting round is over: a fold, or a call that is not the round's first action. */
function roundOver(actions: readonly number[]): boolean {
  const last = actions[actions.length - 1]
  return last === 0 || (last === 1 && actions.length > 1)
}

function raises(actions: readonly number[]): number {
  let n = 0
  for (const a of actions) if (a === 2) n += 1
  return n
}

/**
 * Leduc poker (Southey et al. 2005) with OpenSpiel's rules (`leduc_poker`, two players): a deck of six cards (two
 * suits of three ranks; card c has rank `floor(c / 2)`), antes of 1, one private card each, then a betting round with
 * raises of 2, a public card, and a second round with raises of 4; at most two raises per round, fold (0) only when
 * facing a raise, call (1, a check when nothing is owed) and raise (2). A pair with the public card wins, else the
 * higher rank; equal ranks split. Information states are `card:public:round1/round2` with the actions as `f`, `c`
 * and `r` (for example `'4:1:rc/r'`). The game value for player 0 is about -0.0856.
 *
 * @returns The game.
 * @concept leduc-poker
 * @example
 * ```ts
 * import { gameTree, leducPoker } from 'omai-engine/games'
 * const t = gameTree(leducPoker())
 * [t.nodes.length, t.infosets.length] // => [9457, 936]
 * ```
 */
export function leducPoker(): ExtensiveGame<LeducState> {
  const round = (s: LeducState): number => (s.publicCard === null ? 0 : 1)
  const current = (s: LeducState): readonly number[] => s.rounds[round(s)] ?? []
  const terminal = (s: LeducState): boolean =>
    s.folded !== null || (s.publicCard !== null && roundOver(s.rounds[1] ?? []))
  const chance = (s: LeducState): boolean =>
    s.cards.length < 2 || (s.publicCard === null && roundOver(s.rounds[0] ?? []) && s.folded === null)
  const letters = (xs: readonly number[]): string => xs.map((a) => (a === 0 ? 'f' : a === 1 ? 'c' : 'r')).join('')
  return {
    name: 'leduc_poker',
    numPlayers: 2,
    initial: () => ({ cards: [], publicCard: null, rounds: [[], []], contributions: [1, 1], folded: null }),
    currentPlayer: (s) => (chance(s) ? CHANCE : terminal(s) ? TERMINAL : current(s).length % 2),
    legalActions: (s) => {
      if (chance(s) || terminal(s)) return []
      const acts = current(s)
      const facing = s.contributions[0] !== s.contributions[1]
      return [...(facing ? [0] : []), 1, ...(raises(acts) < 2 ? [2] : [])]
    },
    chanceOutcomes: (s) => {
      if (!chance(s)) return []
      const left = [0, 1, 2, 3, 4, 5].filter((c) => !s.cards.includes(c))
      return left.map((c) => [c, 1 / left.length] as const)
    },
    apply: (s, a) => {
      if (s.cards.length < 2) return { ...s, cards: [...s.cards, a] }
      if (chance(s)) return { ...s, publicCard: a }
      const r = round(s)
      const p = current(s).length % 2
      const c: [number, number] = [s.contributions[0], s.contributions[1]]
      const top = Math.max(c[0], c[1])
      if (a === 1) c[p] = top
      else if (a === 2) c[p] = top + (RAISE[r] ?? 0)
      const rounds = s.rounds.map((x, i) => (i === r ? [...x, a] : x))
      return { ...s, rounds, contributions: c, folded: a === 0 ? p : null }
    },
    returns: (s) => {
      if (!terminal(s)) return [0, 0]
      if (s.folded !== null) {
        const lost = s.contributions[s.folded] ?? 0
        return s.folded === 0 ? [-lost, lost] : [lost, -lost]
      }
      const pub = Math.floor((s.publicCard ?? 0) / 2)
      const rank = (p: number): number => {
        const own = Math.floor((s.cards[p] ?? 0) / 2)
        return own === pub ? 10 + own : own
      }
      const d = rank(0) - rank(1)
      const pot = s.contributions[1]
      return d > 0 ? [pot, -pot] : d < 0 ? [-pot, pot] : [0, 0]
    },
    informationState: (s, p) =>
      `${s.cards[p] ?? ''}:${s.publicCard ?? ''}:${letters(s.rounds[0] ?? [])}/${letters(s.rounds[1] ?? [])}`,
    actionName: (s, a) => (chance(s) ? `deal ${a}` : a === 0 ? 'fold' : a === 1 ? 'call' : 'raise'),
  }
}
