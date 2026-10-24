import { meanOf, varianceOf, WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'

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
 * import { membersOf } from 'whitebox-engine/games'
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
 * @throws WhiteboxError with code `value` for too many players or a table of the wrong length.
 * @concept coalitional-game
 * @example
 * ```ts
 * import { coalitionalGame } from 'whitebox-engine/games'
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
    throw new WhiteboxError('value', `coalitionalGame: between 1 and ${MAX_PLAYERS} players`)
  const size = 2 ** n
  const values = new Float64Array(size)
  if (typeof value === 'function') for (let m = 1; m < size; m++) values[m] = value(membersOf(m, n))
  else {
    if (value.length !== size) throw new WhiteboxError('value', `coalitionalGame: ${size} worths expected`)
    for (let m = 1; m < size; m++) values[m] = value[m] ?? 0
  }
  const players = options.players ?? Array.from({ length: n }, (_, i) => String(i))
  if (players.length !== n) throw new WhiteboxError('value', 'coalitionalGame: one name per player')
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
 * import { coalitionValue, gloveGame } from 'whitebox-engine/games'
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
 * import { shapleyValue, weightedVotingGame } from 'whitebox-engine/games'
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
 * import { gloveGame, shapleyValue } from 'whitebox-engine/games'
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
 * import { bankruptcyGame, coalitionValue } from 'whitebox-engine/games'
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
 * import { shapleyValue, unanimityGame } from 'whitebox-engine/games'
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

function popcount(m: number): number {
  let c = 0
  let x = m
  while (x !== 0) {
    x &= x - 1
    c += 1
  }
  return c
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
 * import { coalitionalGame, shapleyValue } from 'whitebox-engine/games'
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

/** Options for {@link sampledShapley}. */
export interface SampledShapleyOptions {
  /** Random orders to average over (default 1000). */
  readonly samples?: number
  /** The random key (default `key(0)`). */
  readonly key?: Key
}

/** The outcome of {@link sampledShapley}: estimates with their standard errors. */
export interface SampledShapleyResult {
  readonly values: readonly number[]
  readonly standardErrors: readonly number[]
  readonly samples: number
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
 * @throws WhiteboxError with code `value` for fewer than one sample.
 * @concept shapley-value
 * @example
 * ```ts
 * import { gloveGame, sampledShapley } from 'whitebox-engine/games'
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
  if (!(samples >= 1)) throw new WhiteboxError('value', 'sampledShapley: at least one sample is needed')
  const worth =
    typeof value === 'function'
      ? (members: readonly number[]): number => value([...members].sort((a, b) => a - b))
      : (members: readonly number[]): number => coalitionValue(value, members)
  const base = options.key ?? makeKey(0)
  const draws: number[][] = Array.from({ length: n }, () => [])
  const empty = worth([])
  for (let t = 0; t < samples; t++) {
    const order = toFlat(permutation(foldIn(base, t), n))
    const members: number[] = []
    let previous = empty
    for (const i of order) {
      members.push(i)
      const v = worth(members)
      draws[i]?.push(v - previous)
      previous = v
    }
  }
  return {
    values: draws.map((d) => meanOf(d)),
    standardErrors: draws.map((d) => (d.length > 1 ? Math.sqrt(varianceOf(d, 1) / d.length) : 0)),
    samples,
  }
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
 * import { banzhafIndex, weightedVotingGame } from 'whitebox-engine/games'
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
 * import { gloveGame, isSuperadditive } from 'whitebox-engine/games'
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
 * import { coalitionalGame, isConvex } from 'whitebox-engine/games'
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
