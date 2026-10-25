/**
 * The textbook normal-form games: the prisoner's dilemma, matching pennies, rock-paper-scissors, battle of the sexes,
 * stag hunt, chicken, hawk-dove, Shapley's game and pure coordination, with named actions, plus random games with
 * uniform payoffs.
 */

import { WhiteboxError } from '../../base/index.ts'
import { type Key, uniform } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { bimatrixGame, type NormalFormGame, normalFormGame } from './game.ts'

/**
 * The prisoner's dilemma with temptation 5, reward 3, punishment 1 and sucker's payoff 0 (actions cooperate, defect).
 * Defection strictly dominates, so the unique equilibrium (1, 1) is worse for both than mutual cooperation.
 *
 * @returns The game.
 * @concept prisoners-dilemma
 * @example
 * ```ts
 * import { prisonersDilemma, pureNashEquilibria } from 'whitebox-engine/games'
 * pureNashEquilibria(prisonersDilemma())[0].payoffs // => [1, 1]
 * ```
 */
export function prisonersDilemma(): NormalFormGame {
  return bimatrixGame(
    [
      [3, 0],
      [5, 1],
    ],
    [
      [3, 5],
      [0, 1],
    ],
    {
      name: "prisoner's dilemma",
      actionNames: [
        ['cooperate', 'defect'],
        ['cooperate', 'defect'],
      ],
    },
  )
}

/**
 * Matching pennies: the row player wins 1 when the coins match and loses 1 otherwise (zero-sum, actions heads,
 * tails). The only equilibrium is both mixing evenly.
 *
 * @returns The game.
 * @example
 * ```ts
 * import { matchingPennies, pureNashEquilibria } from 'whitebox-engine/games'
 * pureNashEquilibria(matchingPennies()).length // => 0
 * ```
 */
export function matchingPennies(): NormalFormGame {
  return bimatrixGame(
    [
      [1, -1],
      [-1, 1],
    ],
    undefined,
    {
      name: 'matching pennies',
      actionNames: [
        ['heads', 'tails'],
        ['heads', 'tails'],
      ],
    },
  )
}

/**
 * Rock-paper-scissors: win 1, lose 1, draw 0 (zero-sum, actions rock, paper, scissors). The unique equilibrium is
 * uniform play.
 *
 * @returns The game.
 * @example
 * ```ts
 * import { rockPaperScissors } from 'whitebox-engine/games'
 * rockPaperScissors().payoffs[0][1] // => -1
 * ```
 */
export function rockPaperScissors(): NormalFormGame {
  const names = ['rock', 'paper', 'scissors']
  return bimatrixGame(
    [
      [0, -1, 1],
      [1, 0, -1],
      [-1, 1, 0],
    ],
    undefined,
    { name: 'rock-paper-scissors', actionNames: [names, names] },
  )
}

/**
 * The battle of the sexes: both prefer to coincide, the row player at the opera (3 against 2) and the column player
 * at the football (actions opera, football). Two pure equilibria and one mixed, `[[3/5, 2/5], [2/5, 3/5]]`.
 *
 * @returns The game.
 * @example
 * ```ts
 * import { battleOfTheSexes } from 'whitebox-engine/games'
 * battleOfTheSexes().actionNames[0] // => ['opera', 'football']
 * ```
 */
export function battleOfTheSexes(): NormalFormGame {
  return bimatrixGame(
    [
      [3, 0],
      [0, 2],
    ],
    [
      [2, 0],
      [0, 3],
    ],
    {
      name: 'battle of the sexes',
      actionNames: [
        ['opera', 'football'],
        ['opera', 'football'],
      ],
    },
  )
}

/**
 * The stag hunt: hunting stag together pays 4 each, hare pays 3 alone or 2 when both hunt hare, stag alone pays 0
 * (actions stag, hare). Two pure equilibria (payoff- and risk-dominant) and one mixed.
 *
 * @returns The game.
 * @example
 * ```ts
 * import { pureNashEquilibria, stagHunt } from 'whitebox-engine/games'
 * pureNashEquilibria(stagHunt()).map((e) => e.payoffs) // => [[4, 4], [2, 2]]
 * ```
 */
export function stagHunt(): NormalFormGame {
  return bimatrixGame(
    [
      [4, 0],
      [3, 2],
    ],
    [
      [4, 3],
      [0, 2],
    ],
    {
      name: 'stag hunt',
      actionNames: [
        ['stag', 'hare'],
        ['stag', 'hare'],
      ],
    },
  )
}

/**
 * Chicken with Aumann's payoffs (actions swerve, straight): both swerving pay 6, the one going straight against a
 * swerver gets 7 against 2, and a crash pays 0. Aumann's (1974) example where a correlated equilibrium (a third on
 * each outcome but the crash, or better a half on both swerving) beats every Nash equilibrium in welfare.
 *
 * @returns The game.
 * @concept correlated-equilibrium
 * @example
 * ```ts
 * import { chicken, pureNashEquilibria } from 'whitebox-engine/games'
 * pureNashEquilibria(chicken()).map((e) => e.payoffs) // => [[2, 7], [7, 2]]
 * ```
 */
export function chicken(): NormalFormGame {
  return bimatrixGame(
    [
      [6, 2],
      [7, 0],
    ],
    [
      [6, 7],
      [2, 0],
    ],
    {
      name: 'chicken',
      actionNames: [
        ['swerve', 'straight'],
        ['swerve', 'straight'],
      ],
    },
  )
}

/**
 * The hawk-dove game with resource value `v` and fight cost `c` (actions hawk, dove): hawks against hawks get
 * `(v - c) / 2`, a hawk takes `v` from a dove, doves share `v / 2`. For `c > v` the evolutionarily stable mix plays
 * hawk with probability `v / c`.
 *
 * @param v - The value of the resource (default 2).
 * @param c - The cost of a fight (default 4).
 * @returns The symmetric game.
 * @example
 * ```ts
 * import { hawkDove } from 'whitebox-engine/games'
 * hawkDove(2, 4).payoffs[0] // => Float64Array.from([-1, 2, 0, 1])
 * ```
 */
export function hawkDove(v = 2, c = 4): NormalFormGame {
  const a = [
    [(v - c) / 2, v],
    [0, v / 2],
  ]
  return bimatrixGame(
    a,
    [
      [a[0]?.[0] ?? 0, a[1]?.[0] ?? 0],
      [a[0]?.[1] ?? 0, a[1]?.[1] ?? 0],
    ],
    {
      name: 'hawk-dove',
      actionNames: [
        ['hawk', 'dove'],
        ['hawk', 'dove'],
      ],
    },
  )
}

/**
 * Shapley's game (1964): a variant of rock-paper-scissors where the winner gets 1 and the loser and draws get 0.
 * Fictitious play cycles on it with ever longer runs instead of converging to the unique equilibrium (uniform play).
 *
 * @returns The game.
 * @concept fictitious-play
 * @example
 * ```ts
 * import { shapleyGame } from 'whitebox-engine/games'
 * shapleyGame().payoffs[1] // => Float64Array.from([0, 1, 0, 0, 0, 1, 1, 0, 0])
 * ```
 */
export function shapleyGame(): NormalFormGame {
  return bimatrixGame(
    [
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
    ],
    [
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
    ],
    { name: "Shapley's game" },
  )
}

/**
 * A pure coordination game: every player gets 1 when all pick the same action and 0 otherwise.
 *
 * @param players - The number of players (default 2).
 * @param actions - The number of actions of each (default 2).
 * @returns The game.
 * @example
 * ```ts
 * import { coordinationGame, pureNashEquilibria } from 'whitebox-engine/games'
 * pureNashEquilibria(coordinationGame(3, 2)).length // => 2
 * ```
 */
export function coordinationGame(players = 2, actions = 2): NormalFormGame {
  const shape = new Array<number>(players).fill(actions)
  const total = actions ** players
  const u = new Float64Array(total)
  // A profile is unanimous when every digit (base `actions`) equals the first.
  for (let s = 0; s < total; s++) {
    let r = s
    const first = r % actions
    let same = true
    for (let p = 0; p < players; p++) {
      if (r % actions !== first) same = false
      r = Math.floor(r / actions)
    }
    u[s] = same ? 1 : 0
  }
  const tensor = (): number[] | number[][] => nest(Array.from(u), shape)
  return normalFormGame(
    shape.map(() => tensor()),
    { name: 'coordination' },
  )
}

function nest(flat: number[], shape: readonly number[]): number[] | number[][] {
  if (shape.length <= 1) return flat
  const [first, ...rest] = shape
  const size = flat.length / (first ?? 1)
  return Array.from({ length: first ?? 1 }, (_, i) => nest(flat.slice(i * size, (i + 1) * size), rest)) as number[][]
}

/**
 * A game with independent uniform payoffs in `[low, high)` (generic: non-degenerate with probability one).
 *
 * @param key - The random key.
 * @param shape - The number of actions of each player.
 * @param options - The payoff range (default `[0, 1)`).
 * @returns The game.
 * @throws WhiteboxError with code `value` for an empty shape.
 * @example
 * ```ts
 * import { randomGame } from 'whitebox-engine/games'
 * import { key } from 'whitebox-engine/random'
 * randomGame(key(0), [2, 3]).payoffs.length // => 2
 * ```
 */
export function randomGame(
  key: Key,
  shape: readonly number[],
  options: { readonly low?: number; readonly high?: number } = {},
): NormalFormGame {
  if (shape.length === 0) throw new WhiteboxError('value', 'randomGame: at least one player is needed')
  const draws = toFlat(uniform(key, [shape.length, ...shape], { low: options.low ?? 0, high: options.high ?? 1 }))
  const size = draws.length / shape.length
  return normalFormGame(
    shape.map((_, p) => nest(draws.slice(p * size, (p + 1) * size), shape)),
    { name: 'random game' },
  )
}
