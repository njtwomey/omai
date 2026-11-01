/**
 * Cooperative (transferable-utility) games, where coalitions of players earn a joint worth to be shared. Building games
 * from a characteristic function, with classic examples (weighted voting, glove, bankruptcy, unanimity), and the ways
 * of dividing the worth: the exact Shapley value (each player's average marginal contribution over all orders of
 * arrival), its permutation-sampling estimate, the pairwise Shapley interaction index and the Banzhaf index, plus tests
 * for superadditivity and convexity.
 */

import { MdmlError, meanOf, popcount, varianceOf } from '../base/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../random/index.ts'
import { dict, type OptionTable, optionTable } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'

/**
 * A coalitional (transferable-utility) game: the worth `v(S)` of every coalition S of `n` players, indexed by the
 * bitmask of its members (bit i for player i), with `v(empty) = 0`.
 *
 * @concept coalitional-game
 */
export interface CoalitionalGame {
  readonly n: number
  /** The worth of every coalition, `values[mask]`, length `2^n`. */
  readonly values: Float64Array
  readonly players: readonly string[]
}

/** The largest number of players with exact (exponential) computations. */
const MAX_PLAYERS = 24

/**
 * The members of a coalition given by its bitmask (bit i for player i), ascending.
 *
 * @param mask - The coalition's bitmask.
 * @param n - The number of players.
 * @returns The members.
 * @example
 * ```ts
 * import { membersOf } from 'mdml-engine/games'
 * membersOf(0b101, 3) // => [0, 2]
 * ```
 */
export function membersOf(mask: number, n: number): number[] {
  const out: number[] = []
  for (let i = 0; i < n; i++) if ((mask >> i) & 1) out.push(i)
  return out
}

/**
 * A coalitional game from its characteristic function, given as a function of the members (ascending) or as the
 * table of worths by bitmask.
 *
 * @param n - The number of players (at most 24).
 * @param value - `v(members)`, or the `2^n` worths by bitmask.
 * @param options - Player names (default `'0', '1', ...`).
 * @returns The game (the empty coalition is worth 0).
 * @throws MdmlError with code `value` for too many players or a table of the wrong length.
 * @concept coalitional-game
 * @example
 * ```ts
 * import { coalitionalGame } from 'mdml-engine/games'
 * const g = coalitionalGame(2, (s) => (s.length === 2 ? 10 : 0))
 * Array.from(g.values) // => [0, 0, 0, 10]
 * ```
 */
export function coalitionalGame(
  n: number,
  value: ((members: readonly number[]) => number) | ArrayLike<number>,
  options: { readonly players?: readonly string[] } = {},
): CoalitionalGame {
  if (!Number.isInteger(n) || n < 1 || n > MAX_PLAYERS)
    throw new MdmlError('value', `coalitionalGame: between 1 and ${MAX_PLAYERS} players`)
  const size = 2 ** n
  const values = new Float64Array(size)
  if (typeof value === 'function') for (let m = 1; m < size; m++) values[m] = value(membersOf(m, n))
  else {
    if (value.length !== size) throw new MdmlError('value', `coalitionalGame: ${size} worths expected`)
    for (let m = 1; m < size; m++) values[m] = value[m] ?? 0
  }
  const players = options.players ?? Array.from({ length: n }, (_, i) => String(i))
  if (players.length !== n) throw new MdmlError('value', 'coalitionalGame: one name per player')
  return { n, values, players }
}

/**
 * The worth of a coalition given by its members.
 *
 * @param game - The game.
 * @param members - The players in the coalition.
 * @returns `v(members)`.
 * @example
 * ```ts
 * import { coalitionValue, gloveGame } from 'mdml-engine/games'
 * coalitionValue(gloveGame(1, 2), [0, 2]) // => 1
 * ```
 */
export function coalitionValue(game: CoalitionalGame, members: readonly number[]): number {
  let mask = 0
  for (const i of members) mask |= 1 << i
  return game.values[mask] ?? 0
}

/**
 * A weighted voting game `[quota; w_1, ..., w_n]`: a coalition wins (worth 1) when its weights reach the quota.
 *
 * @param quota - The weight needed to win.
 * @param weights - Every player's weight.
 * @param options - Player names.
 * @returns The simple game.
 * @concept voting-power
 * @example
 * ```ts
 * import { shapleyValue, weightedVotingGame } from 'mdml-engine/games'
 * shapleyValue(weightedVotingGame(51, [50, 49, 1])) // => [0.6666666666666666, 0.16666666666666666, 0.16666666666666666]
 * ```
 */
export function weightedVotingGame(
  quota: number,
  weights: readonly number[],
  options: { readonly players?: readonly string[] } = {},
): CoalitionalGame {
  return coalitionalGame(
    weights.length,
    (s) => {
      let w = 0
      for (const i of s) w += weights[i] ?? 0
      return w >= quota ? 1 : 0
    },
    options,
  )
}

/**
 * The glove game: players `0..left-1` own a left glove, the next `right` a right glove, and a coalition is worth the
 * number of pairs it can make.
 *
 * @param left - Owners of left gloves.
 * @param right - Owners of right gloves.
 * @returns The game.
 * @example
 * ```ts
 * import { gloveGame, shapleyValue } from 'mdml-engine/games'
 * shapleyValue(gloveGame(1, 2)).map((v) => Math.round(v * 1e9) / 1e9) // => [0.666666667, 0.166666667, 0.166666667]
 * ```
 */
export function gloveGame(left: number, right: number): CoalitionalGame {
  return coalitionalGame(left + right, (s) => {
    let l = 0
    for (const i of s) if (i < left) l += 1
    return Math.min(l, s.length - l)
  })
}

/**
 * The bankruptcy game of an estate E and claims c (O'Neill 1982): a coalition is worth what remains after the others'
 * claims are paid in full, `max(0, E - sum of the others' claims)`. Its nucleolus is the Talmud rule (Aumann and
 * Maschler 1985).
 *
 * @param estate - The estate.
 * @param claims - Every creditor's claim.
 * @returns The game.
 * @concept nucleolus
 * @example
 * ```ts
 * import { bankruptcyGame, coalitionValue } from 'mdml-engine/games'
 * coalitionValue(bankruptcyGame(200, [100, 200, 300]), [2]) // => 0
 * ```
 */
export function bankruptcyGame(estate: number, claims: readonly number[]): CoalitionalGame {
  let total = 0
  for (const c of claims) total += c
  return coalitionalGame(claims.length, (s) => {
    let own = 0
    for (const i of s) own += claims[i] ?? 0
    return Math.max(0, estate - (total - own))
  })
}

/**
 * The unanimity game of a coalition T: worth 1 for every coalition containing T, 0 otherwise. Every game is a unique
 * linear combination of unanimity games (with the Harsanyi dividends as coefficients).
 *
 * @param n - The number of players.
 * @param coalition - The members of T.
 * @returns The game.
 * @example
 * ```ts
 * import { shapleyValue, unanimityGame } from 'mdml-engine/games'
 * shapleyValue(unanimityGame(3, [0, 2])) // => [0.5, 0, 0.5]
 * ```
 */
export function unanimityGame(n: number, coalition: readonly number[]): CoalitionalGame {
  let t = 0
  for (const i of coalition) t |= 1 << i
  return coalitionalGame(n, (s) => {
    let mask = 0
    for (const i of s) mask |= 1 << i
    return (mask & t) === t ? 1 : 0
  })
}

/** The weight |S|! (n - |S| - 1)! / n! of a coalition of size s not containing the player. */
function shapleyWeights(n: number): Float64Array {
  const w = new Float64Array(n)
  // w[s] = 1 / (n * C(n - 1, s)).
  let c = 1
  for (let s = 0; s < n; s++) {
    w[s] = 1 / (n * c)
    c = (c * (n - 1 - s)) / (s + 1)
  }
  return w
}

/**
 * The Shapley value (Shapley 1953): every player's average marginal contribution over all orders of arrival,
 * `phi_i = sum over S not containing i of |S|! (n - |S| - 1)! / n! (v(S + i) - v(S))`, computed exactly over the
 * `2^n` coalitions. It is the unique efficient, symmetric, additive value giving null players nothing; on weighted
 * voting games it is the Shapley-Shubik power index.
 *
 * @param game - The game.
 * @returns One value per player (summing to `v(N)`).
 * @concept shapley-value
 * @example
 * ```ts
 * import { coalitionalGame, shapleyValue } from 'mdml-engine/games'
 * shapleyValue(coalitionalGame(3, (s) => (s.length >= 2 ? 1 : 0))).map((v) => Math.round(v * 1e12) / 1e12) // => [0.333333333333, 0.333333333333, 0.333333333333]
 * ```
 */
export function shapleyValue(game: CoalitionalGame): number[] {
  const { n, values } = game
  const w = shapleyWeights(n)
  const phi = new Array<number>(n).fill(0)
  for (let m = 0; m < values.length; m++) {
    const s = popcount(m)
    for (let i = 0; i < n; i++) {
      if ((m >> i) & 1) continue
      phi[i] = (phi[i] ?? 0) + (w[s] ?? 0) * ((values[m | (1 << i)] ?? 0) - (values[m] ?? 0))
    }
  }
  return phi
}

/**
 * The Shapley interaction index (Grabisch and Roubens 1999) of every pair of players:
 * `I_ij = sum over S not containing i or j of |S|! (n - |S| - 2)! / (n - 1)! (v(S + ij) - v(S + i) - v(S + j) + v(S))`,
 * positive for complements and negative for substitutes, with the Shapley values on the diagonal (`I_ii = phi_i`, the
 * index of a singleton). SHAP's interaction values (`explain.shapleyInteractionValues`) are built from it.
 *
 * @param game - The game.
 * @returns The symmetric `n x n` matrix, as nested arrays.
 * @concept shapley-value
 * @example
 * ```ts
 * import { coalitionalGame, shapleyInteractionIndex } from 'mdml-engine/games'
 * // Two players worth something only together: complements.
 * shapleyInteractionIndex(coalitionalGame(2, (s) => (s.length === 2 ? 1 : 0))) // => [[0.5, 1], [1, 0.5]]
 * ```
 */
export function shapleyInteractionIndex(game: CoalitionalGame): number[][] {
  const { n, values } = game
  const out = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  if (n >= 2) {
    // |S|! (n - |S| - 2)! / (n - 1)! is the Shapley weight of a game with n - 1 players.
    const w = shapleyWeights(n - 1)
    for (let m = 0; m < values.length; m++) {
      const s = popcount(m)
      if (s > n - 2) continue
      for (let i = 0; i < n; i++) {
        if ((m >> i) & 1) continue
        for (let j = i + 1; j < n; j++) {
          if ((m >> j) & 1) continue
          const d =
            (values[m | (1 << i) | (1 << j)] ?? 0) -
            (values[m | (1 << i)] ?? 0) -
            (values[m | (1 << j)] ?? 0) +
            (values[m] ?? 0)
          const rowI = out[i] ?? []
          const rowJ = out[j] ?? []
          rowI[j] = (rowI[j] ?? 0) + (w[s] ?? 0) * d
          rowJ[i] = (rowJ[i] ?? 0) + (w[s] ?? 0) * d
        }
      }
    }
  }
  const phi = shapleyValue(game)
  for (const [i, row] of out.entries()) row[i] = phi[i] ?? 0
  return out
}

/** How the random orders of arrival are drawn: the order at sample `t` of `n` players. */
export type OrderSampler = (key: Key, n: number) => (t: number) => number[]

/** What an order sampler can do, for validity rules and selectors. */
export interface OrderSamplerProperties {
  /** Whether consecutive samples are negatively correlated by construction (a variance reduction). */
  readonly antithetic: boolean
}

/**
 * The samplers of arrival orders as one slot: `independent` (a fresh uniform permutation per sample) and `antithetic`
 * (each uniform permutation is followed by its reverse, so a player early in one order is late in the next, which
 * lowers the variance of the average marginal contribution).
 *
 * @concept shapley-value
 * @example
 * ```ts
 * import { orderSamplers } from 'mdml-engine/games'
 * orderSamplers.where((p) => p.antithetic) // => ['antithetic']
 * ```
 */
export const orderSamplers: OptionTable<OrderSampler, OrderSamplerProperties> = optionTable(
  {
    independent: {
      description: 'a fresh uniform permutation for every sample',
      properties: { antithetic: false },
      space: dict({}),
      make: () => (base, n) => (t) => Array.from(toFlat(permutation(foldIn(base, t), n))),
    },
    antithetic: {
      description: 'each uniform permutation followed by its reverse',
      properties: { antithetic: true },
      space: dict({}),
      make: () => (base, n) => (t) => {
        const order = Array.from(toFlat(permutation(foldIn(base, Math.floor(t / 2)), n)))
        return t % 2 === 0 ? order : order.reverse()
      },
    },
  },
  {
    default: 'independent',
    label: 'order sampler',
    description: 'How the random orders of arrival are drawn.',
  },
)

/** Options for {@link sampledShapley}. */
export interface SampledShapleyOptions {
  /** Random orders to average over (default 1000). */
  readonly samples?: number
  /** The random key (default `key(0)`). */
  readonly key?: Key
  /** Pair every order with its reverse (default false; see {@link orderSamplers}). */
  readonly antithetic?: boolean
}

/** The outcome of {@link sampledShapley}: estimates with their standard errors. */
export interface SampledShapleyResult {
  readonly values: readonly number[]
  readonly standardErrors: readonly number[]
  readonly samples: number
}

/** The paths sampled Shapley estimation logs, as a type. */
export type ShapleyPath = 'shapley/phi' | 'shapley/stderr' | 'shapley/permutation'

/**
 * What sampled Shapley estimation logs, by path: the order of arrival sampled at this step, and the estimate and its
 * standard error for every player so far.
 */
export const shapleyContract: Contract<unknown, ShapleyPath> = defineContract({
  'shapley/permutation': tensor({
    role: 'state',
    shape: ['players'],
    description: 'the order of arrival sampled at this step',
  }),
  'shapley/phi': tensor({
    role: 'state',
    shape: ['players'],
    description: 'the Shapley estimate of every player so far',
  }),
  'shapley/stderr': tensor({
    role: 'state',
    shape: ['players'],
    description: 'the standard error of every estimate so far (0 until two samples)',
  }),
})

/** The state of {@link sampledShapleyAlgorithm}. The draws grow in place; copy a state to keep it. */
export interface SampledShapleyState {
  /** The marginal contribution of each player in each sample so far: `draws[player][sample]`. */
  readonly draws: readonly (readonly number[])[]
  /** Samples drawn so far. */
  readonly t: number
  /** The order of arrival sampled at the last step (empty before the first). */
  readonly order: readonly number[]
}

/** Estimates and standard errors from the draws so far. */
function shapleyEstimate(draws: readonly (readonly number[])[]): { values: number[]; standardErrors: number[] } {
  return {
    values: draws.map((d) => meanOf(d)),
    standardErrors: draws.map((d) => (d.length > 1 ? Math.sqrt(varianceOf(d, 1) / d.length) : 0)),
  }
}

/**
 * Permutation-sampling Shapley estimation as a steppable algorithm: each step draws one random order of arrival, lets
 * the players join in it and records every player's marginal contribution. The estimates and their standard errors can
 * be watched converging.
 *
 * @param n - The number of players.
 * @param value - The worth of a coalition given its members (ascending), or a tabulated game.
 * @param options - The number of orders, the key and `antithetic` pairing.
 * @returns The algorithm; it is done after `samples` steps.
 * @throws MdmlError with code `value` for fewer than one sample.
 * @concept shapley-value
 * @example
 * ```ts
 * import { gloveGame, sampledShapleyAlgorithm } from 'mdml-engine/games'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(sampledShapleyAlgorithm(3, gloveGame(1, 2), { samples: 50 }), { maxSteps: 50 })
 * [run.state.t, run.state.draws.length] // => [50, 3]
 * ```
 */
export function sampledShapleyAlgorithm(
  n: number,
  value: ((members: readonly number[]) => number) | CoalitionalGame,
  options: SampledShapleyOptions = {},
): Algorithm<SampledShapleyState, ShapleyPath> {
  const samples = options.samples ?? 1000
  if (!(samples >= 1)) throw new MdmlError('value', 'sampledShapley: at least one sample is needed')
  const worth =
    typeof value === 'function'
      ? (members: readonly number[]): number => value([...members].sort((a, b) => a - b))
      : (members: readonly number[]): number => coalitionValue(value, members)
  const orderAt = orderSamplers.make({ kind: options.antithetic === true ? 'antithetic' : 'independent' })(
    options.key ?? makeKey(0),
    n,
  )
  let empty: number | undefined
  return {
    name: 'shapley-sampling',
    config: { n, samples, antithetic: options.antithetic ?? false },
    init: () => ({ draws: Array.from({ length: n }, () => []), t: 0, order: [] }),
    step: (s, ctx) => {
      empty ??= worth([])
      const order = orderAt(s.t)
      const draws = s.draws as number[][]
      const members: number[] = []
      let previous = empty
      for (const i of order) {
        members.push(i)
        const v = worth(members)
        draws[i]?.push(v - previous)
        previous = v
      }
      if (ctx.wants('shapley/permutation')) ctx.log('shapley/permutation', Int32Array.from(order))
      if (ctx.wants('shapley/phi') || ctx.wants('shapley/stderr')) {
        const { values, standardErrors } = shapleyEstimate(draws)
        if (ctx.wants('shapley/phi')) ctx.log('shapley/phi', Float64Array.from(values))
        if (ctx.wants('shapley/stderr')) ctx.log('shapley/stderr', Float64Array.from(standardErrors))
      }
      return { draws, t: s.t + 1, order }
    },
    done: (s) => (s.t >= samples ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SampledShapleyState,
  }
}

/**
 * The Shapley value estimated by permutation sampling (Castro, Gomez and Tejada 2009): average every player's
 * marginal contribution over random orders of arrival. Unbiased, with standard errors from the sample variance; it
 * needs only `samples * n` evaluations of the worth, so it serves games too large to tabulate (model explanation).
 *
 * @param n - The number of players.
 * @param value - The worth of a coalition given its members (ascending), or a tabulated game.
 * @param options - The number of orders and the key.
 * @returns Estimates and standard errors.
 * @throws MdmlError with code `value` for fewer than one sample.
 * @concept shapley-value
 * @example
 * ```ts
 * import { gloveGame, sampledShapley } from 'mdml-engine/games'
 * const r = sampledShapley(3, gloveGame(1, 2), { samples: 2000 })
 * Math.abs(r.values[0] - 2 / 3) < 4 * r.standardErrors[0] // => true
 * ```
 */
export function sampledShapley(
  n: number,
  value: ((members: readonly number[]) => number) | CoalitionalGame,
  options: SampledShapleyOptions = {},
): SampledShapleyResult {
  const samples = options.samples ?? 1000
  const { state } = runAlgorithm(sampledShapleyAlgorithm(n, value, options), { maxSteps: samples })
  return { ...shapleyEstimate(state.draws), samples }
}

/**
 * The Banzhaf index (Banzhaf 1965; Penrose 1946): every player's average marginal contribution over the coalitions
 * of the others, `beta_i = 2^-(n-1) sum over S not containing i of (v(S + i) - v(S))`; normalised to sum to 1 when
 * asked (the normalised Banzhaf index of voting power).
 *
 * @param game - The game.
 * @param options - `normalised` (default false: the raw Banzhaf value).
 * @returns One index per player.
 * @concept voting-power
 * @example
 * ```ts
 * import { banzhafIndex, weightedVotingGame } from 'mdml-engine/games'
 * banzhafIndex(weightedVotingGame(51, [50, 49, 1]), { normalised: true }) // => [0.6, 0.2, 0.2]
 * ```
 */
export function banzhafIndex(game: CoalitionalGame, options: { readonly normalised?: boolean } = {}): number[] {
  const { n, values } = game
  const beta = new Array<number>(n).fill(0)
  for (let m = 0; m < values.length; m++)
    for (let i = 0; i < n; i++)
      if (!((m >> i) & 1)) beta[i] = (beta[i] ?? 0) + (values[m | (1 << i)] ?? 0) - (values[m] ?? 0)
  const scale = 2 ** (n - 1)
  const raw = beta.map((b) => b / scale)
  if (options.normalised !== true) return raw
  let total = 0
  for (const b of raw) total += b
  return total === 0 ? raw : raw.map((b) => b / total)
}

/**
 * Whether the game is superadditive: `v(S + T) >= v(S) + v(T)` for disjoint coalitions (merging never hurts).
 *
 * @param game - The game.
 * @param tol - Tolerance (default 1e-12).
 * @returns Whether every disjoint pair satisfies the inequality.
 * @example
 * ```ts
 * import { gloveGame, isSuperadditive } from 'mdml-engine/games'
 * isSuperadditive(gloveGame(2, 2)) // => true
 * ```
 */
export function isSuperadditive(game: CoalitionalGame, tol = 1e-12): boolean {
  const size = game.values.length
  const full = size - 1
  for (let s = 1; s < size; s++) {
    const rest = full & ~s
    // Every non-empty submask t of the complement of s.
    for (let t = rest; t > 0; t = (t - 1) & rest)
      if ((game.values[s | t] ?? 0) < (game.values[s] ?? 0) + (game.values[t] ?? 0) - tol) return false
  }
  return true
}

/**
 * Whether the game is convex (supermodular, Shapley 1971): marginal contributions grow with the coalition,
 * `v(S + i + j) - v(S + j) >= v(S + i) - v(S)`. Convex games have a non-empty core containing the Shapley value.
 *
 * @param game - The game.
 * @param tol - Tolerance (default 1e-12).
 * @returns Whether the game is convex.
 * @concept core
 * @example
 * ```ts
 * import { coalitionalGame, isConvex } from 'mdml-engine/games'
 * [isConvex(coalitionalGame(3, (s) => s.length ** 2)), isConvex(coalitionalGame(3, (s) => (s.length >= 2 ? 1 : 0)))] // => [true, false]
 * ```
 */
export function isConvex(game: CoalitionalGame, tol = 1e-12): boolean {
  const { n, values } = game
  for (let m = 0; m < values.length; m++)
    for (let i = 0; i < n; i++) {
      if ((m >> i) & 1) continue
      for (let j = i + 1; j < n; j++) {
        if ((m >> j) & 1) continue
        const a = (values[m | (1 << i) | (1 << j)] ?? 0) - (values[m | (1 << j)] ?? 0)
        const b = (values[m | (1 << i)] ?? 0) - (values[m] ?? 0)
        if (a < b - tol) return false
      }
    }
  return true
}
