/**
 * The normal-form game: players choose actions simultaneously and receive payoffs from a tensor indexed by the joint
 * profile. Building games (any number of players, or a bimatrix pair as in nashpy), expected payoffs and action values
 * under mixed strategies, best responses, deviation gains and NashConv, Nash equilibrium checks and the pure
 * equilibria.
 */

import { MdmlError, maxOf } from '../../base/index.ts'
import { asTensor, stridesOf, type TensorLike, toFlat } from '../../tensor/index.ts'
import type { RunOptions } from '../../trace/index.ts'
import { decodeProfile, encodeProfile, profileCount, profileWeights } from './_profiles.ts'

/** A mixed strategy: one probability per action. */
export type Strategy = readonly number[]

/**
 * A finite game in normal (strategic) form: every player picks an action at once and receives a payoff that depends
 * on the joint profile. Payoffs are stored per player, row-major over profiles with player 0's action the slowest
 * index (so a two-player game's payoffs are the familiar bimatrix `A[i][j]`, `B[i][j]`).
 *
 * @concept normal-form-game
 */
export interface NormalFormGame {
  readonly name: string
  /** The number of actions of each player. */
  readonly shape: readonly number[]
  /** Each player's payoff for every profile, row-major. */
  readonly payoffs: readonly Float64Array[]
  /** Names of every player's actions. */
  readonly actionNames: readonly (readonly string[])[]
}

/** A Nash (or other) equilibrium: one mixed strategy per player and the expected payoffs they give. */
export interface Equilibrium {
  readonly strategies: readonly Strategy[]
  readonly payoffs: readonly number[]
}

/** Tracing options shared by the iterative game solvers: a tracer and policy (or a parent run), a run id, a seed or key. */
export type GameRunOptions = Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId' | 'seed' | 'key'>>

/** Options for {@link normalFormGame} and {@link bimatrixGame}. */
export interface GameOptions {
  readonly name?: string
  readonly actionNames?: readonly (readonly string[])[]
}

/**
 * A normal-form game from one payoff tensor per player (all of shape `[actions of player 0, ..., of player n-1]`).
 *
 * @param payoffs - Player i's payoff tensor for each player i.
 * @param options - A name and action names (default `'0', '1', ...`).
 * @returns The game.
 * @throws MdmlError with code `shape` when the tensors disagree with each other or with the number of players.
 * @concept normal-form-game
 * @index structure A normal-form game from payoff tensors.
 * @example
 * ```ts
 * import { normalFormGame } from 'mdml-engine/games'
 * const g = normalFormGame([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], { name: 'coordination' })
 * g.shape // => [2, 2]
 * ```
 */
export function normalFormGame(payoffs: readonly TensorLike[], options: GameOptions = {}): NormalFormGame {
  if (payoffs.length === 0) throw new MdmlError('shape', 'normalFormGame: at least one player is needed')
  const tensors = payoffs.map((p) => asTensor(p))
  const shape = [...(tensors[0]?.shape ?? [])]
  if (shape.length !== payoffs.length || shape.some((k) => k < 1))
    throw new MdmlError(
      'shape',
      `normalFormGame: ${payoffs.length} players need payoff tensors with ${payoffs.length} axes`,
    )
  for (const t of tensors)
    if (t.shape.length !== shape.length || t.shape.some((k, i) => k !== shape[i]))
      throw new MdmlError('shape', 'normalFormGame: every payoff tensor must have the same shape')
  const actionNames = shape.map((k, p) => {
    const names = options.actionNames?.[p]
    if (names !== undefined && names.length !== k)
      throw new MdmlError('shape', `normalFormGame: player ${p} has ${k} actions but ${names.length} names`)
    return names ?? Array.from({ length: k }, (_, a) => String(a))
  })
  return {
    name: options.name ?? 'game',
    shape,
    payoffs: tensors.map((t) => Float64Array.from(toFlat(t))),
    actionNames,
  }
}

/**
 * A two-player game from the row player's payoff matrix `A` and the column player's `B` (default `-A`, a zero-sum
 * game), as in nashpy's `Game(A, B)`.
 *
 * @param a - The row player's payoffs, m x n.
 * @param b - The column player's payoffs, m x n (default `-a`).
 * @param options - A name and action names.
 * @returns The game.
 * @concept normal-form-game
 * @example
 * ```ts
 * import { bimatrixGame, payoffMatrices } from 'mdml-engine/games'
 * payoffMatrices(bimatrixGame([[1, -1], [-1, 1]]))[1] // => [[-1, 1], [1, -1]]
 * ```
 */
export function bimatrixGame(a: TensorLike, b?: TensorLike, options: GameOptions = {}): NormalFormGame {
  const at = asTensor(a)
  if (at.shape.length !== 2) throw new MdmlError('shape', 'bimatrixGame: payoffs must be matrices')
  const bt = b === undefined ? Float64Array.from(toFlat(at), (v) => -v) : toFlat(asTensor(b))
  const [m, n] = at.shape as [number, number]
  const rows = (flat: ArrayLike<number>): number[][] =>
    Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => flat[i * n + j] ?? 0))
  if (bt.length !== m * n) throw new MdmlError('shape', 'bimatrixGame: A and B must have the same shape')
  return normalFormGame([rows(toFlat(at)), rows(bt)], options)
}

/**
 * The payoff matrices `[A, B]` of a two-player game.
 *
 * @param game - A two-player game.
 * @returns The row and column players' payoff matrices.
 * @throws MdmlError with code `value` for games with other than two players.
 * @example
 * ```ts
 * import { bimatrixGame, payoffMatrices } from 'mdml-engine/games'
 * payoffMatrices(bimatrixGame([[3, 0], [5, 1]], [[3, 5], [0, 1]]))[0] // => [[3, 0], [5, 1]]
 * ```
 */
export function payoffMatrices(game: NormalFormGame): [number[][], number[][]] {
  if (game.shape.length !== 2) throw new MdmlError('value', `${game.name}: payoff matrices need a two-player game`)
  const [m, n] = game.shape as [number, number]
  const mat = (p: number): number[][] =>
    Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => game.payoffs[p]?.[i * n + j] ?? 0))
  return [mat(0), mat(1)]
}

function checkStrategies(game: NormalFormGame, strategies: readonly ArrayLike<number>[]): void {
  if (strategies.length !== game.shape.length || strategies.some((s, p) => s.length !== game.shape[p]))
    throw new MdmlError('shape', `${game.name}: one strategy per player, one probability per action`)
}

/**
 * Every player's payoff at a pure profile.
 *
 * @param game - The game.
 * @param profile - One action per player.
 * @returns The payoffs.
 * @example
 * ```ts
 * import { prisonersDilemma, profilePayoffs } from 'mdml-engine/games'
 * profilePayoffs(prisonersDilemma(), [1, 0]) // => [5, 0]
 * ```
 */
export function profilePayoffs(game: NormalFormGame, profile: readonly number[]): number[] {
  const idx = encodeProfile(profile, stridesOf(game.shape))
  return game.payoffs.map((u) => u[idx] ?? 0)
}

/**
 * Every player's expected payoff when all play independent mixed strategies.
 *
 * @param game - The game.
 * @param strategies - One mixed strategy per player.
 * @returns The expected payoffs.
 * @example
 * ```ts
 * import { expectedPayoffs, matchingPennies } from 'mdml-engine/games'
 * expectedPayoffs(matchingPennies(), [[1, 0], [0.5, 0.5]]) // => [0, 0]
 * ```
 */
export function expectedPayoffs(game: NormalFormGame, strategies: readonly ArrayLike<number>[]): number[] {
  checkStrategies(game, strategies)
  const w = profileWeights(game.shape, strategies)
  return game.payoffs.map((u) => {
    let s = 0
    for (let k = 0; k < w.length; k++) s += (w[k] ?? 0) * (u[k] ?? 0)
    return s
  })
}

/**
 * The expected payoff to `player` of each of its pure actions, the others playing their mixed strategies (the
 * player's own entry of `strategies` is ignored).
 *
 * @param game - The game.
 * @param player - The player.
 * @param strategies - One mixed strategy per player.
 * @returns One value per action of `player`.
 * @example
 * ```ts
 * import { actionValues, rockPaperScissors } from 'mdml-engine/games'
 * actionValues(rockPaperScissors(), 0, [[1, 0, 0], [1, 0, 0]]) // => [0, 1, -1]
 * ```
 */
export function actionValues(game: NormalFormGame, player: number, strategies: readonly ArrayLike<number>[]): number[] {
  checkStrategies(game, strategies)
  const k = game.shape[player]
  if (k === undefined) throw new MdmlError('value', `${game.name}: no player ${player}`)
  const stride = stridesOf(game.shape)[player] ?? 1
  const w = profileWeights(game.shape, strategies, player)
  const u = game.payoffs[player] ?? new Float64Array(0)
  const out = new Array<number>(k).fill(0)
  for (let s = 0; s < w.length; s++) {
    const a = Math.floor(s / stride) % k
    out[a] = (out[a] ?? 0) + (w[s] ?? 0) * (u[s] ?? 0)
  }
  return out
}

/**
 * The pure best responses of `player` to the others' mixed strategies: every action within `tol` of the best value.
 *
 * @param game - The game.
 * @param player - The player.
 * @param strategies - One mixed strategy per player (the player's own is ignored).
 * @param tol - Absolute tolerance for ties (default 1e-9).
 * @returns The best actions, ascending.
 * @concept best-response
 * @example
 * ```ts
 * import { bestResponses, matchingPennies } from 'mdml-engine/games'
 * bestResponses(matchingPennies(), 0, [[1, 0], [0.5, 0.5]]) // => [0, 1]
 * ```
 */
export function bestResponses(
  game: NormalFormGame,
  player: number,
  strategies: readonly ArrayLike<number>[],
  tol = 1e-9,
): number[] {
  const v = actionValues(game, player, strategies)
  const best = maxOf(v)
  const out: number[] = []
  for (let a = 0; a < v.length; a++) if ((v[a] ?? -Infinity) >= best - tol) out.push(a)
  return out
}

/**
 * How much each player gains by switching to a best response, the others' strategies fixed (zero for every player
 * exactly at a Nash equilibrium).
 *
 * @param game - The game.
 * @param strategies - One mixed strategy per player.
 * @returns One non-negative gain per player.
 * @example
 * ```ts
 * import { deviationGains, prisonersDilemma } from 'mdml-engine/games'
 * deviationGains(prisonersDilemma(), [[1, 0], [1, 0]]) // => [2, 2]
 * ```
 */
export function deviationGains(game: NormalFormGame, strategies: readonly ArrayLike<number>[]): number[] {
  const values = expectedPayoffs(game, strategies)
  return game.shape.map((_, p) => Math.max(0, maxOf(actionValues(game, p, strategies)) - (values[p] ?? 0)))
}

/**
 * The NashConv of a profile (OpenSpiel's `nash_conv`): the sum over players of the gain from best responding. It is
 * zero exactly at Nash equilibria; for a two-player zero-sum game it is the duality gap.
 *
 * @param game - The game.
 * @param strategies - One mixed strategy per player.
 * @returns The NashConv.
 * @concept nash-equilibrium
 * @example
 * ```ts
 * import { nashConv, rockPaperScissors } from 'mdml-engine/games'
 * nashConv(rockPaperScissors(), [[1, 0, 0], [1 / 3, 1 / 3, 1 / 3]]) // => 1
 * ```
 */
export function nashConv(game: NormalFormGame, strategies: readonly ArrayLike<number>[]): number {
  let s = 0
  for (const g of deviationGains(game, strategies)) s += g
  return s
}

/**
 * Whether a profile is a Nash equilibrium: valid mixed strategies from which no player gains more than `tol` by a
 * unilateral deviation.
 *
 * @param game - The game.
 * @param strategies - One mixed strategy per player.
 * @param tol - Tolerance for the gains and for the probabilities (default 1e-8).
 * @returns Whether every strategy is a best response to the others.
 * @concept nash-equilibrium
 * @example
 * ```ts
 * import { isNash, matchingPennies } from 'mdml-engine/games'
 * [isNash(matchingPennies(), [[0.5, 0.5], [0.5, 0.5]]), isNash(matchingPennies(), [[1, 0], [1, 0]])] // => [true, false]
 * ```
 */
export function isNash(game: NormalFormGame, strategies: readonly ArrayLike<number>[], tol = 1e-8): boolean {
  checkStrategies(game, strategies)
  for (const s of strategies) {
    let total = 0
    for (let a = 0; a < s.length; a++) {
      const p = s[a] ?? 0
      if (p < -tol) return false
      total += p
    }
    if (Math.abs(total - 1) > tol) return false
  }
  return deviationGains(game, strategies).every((g) => g <= tol)
}

/**
 * Every pure-strategy Nash equilibrium, in profile order: the profiles where each action is a best response to the
 * others.
 *
 * @param game - The game.
 * @returns The equilibria (strategies as indicator vectors) with their payoffs.
 * @concept nash-equilibrium
 * @example
 * ```ts
 * import { battleOfTheSexes, pureNashEquilibria } from 'mdml-engine/games'
 * pureNashEquilibria(battleOfTheSexes()).map((e) => e.payoffs) // => [[3, 2], [2, 3]]
 * ```
 */
export function pureNashEquilibria(game: NormalFormGame): Equilibrium[] {
  const strides = stridesOf(game.shape)
  const total = profileCount(game.shape)
  const out: Equilibrium[] = []
  for (let s = 0; s < total; s++) {
    const profile = decodeProfile(s, game.shape)
    let stable = true
    for (let p = 0; p < game.shape.length && stable; p++) {
      const u = game.payoffs[p] ?? new Float64Array(0)
      const own = u[s] ?? 0
      const base = s - (profile[p] ?? 0) * (strides[p] ?? 0)
      for (let a = 0; a < (game.shape[p] ?? 0); a++) if ((u[base + a * (strides[p] ?? 0)] ?? 0) > own) stable = false
    }
    if (stable)
      out.push({
        strategies: profile.map((a, p) => Array.from({ length: game.shape[p] ?? 0 }, (_, b) => (b === a ? 1 : 0))),
        payoffs: game.payoffs.map((u) => u[s] ?? 0),
      })
  }
  return out
}

/**
 * Whether the payoffs sum to the same constant at every profile (a constant-sum game, zero-sum when the constant is 0).
 *
 * @param game - The game.
 * @param tol - Absolute tolerance (default 1e-12).
 * @returns Whether the game is constant-sum.
 * @example
 * ```ts
 * import { isConstantSum, matchingPennies, prisonersDilemma } from 'mdml-engine/games'
 * [isConstantSum(matchingPennies()), isConstantSum(prisonersDilemma())] // => [true, false]
 * ```
 */
export function isConstantSum(game: NormalFormGame, tol = 1e-12): boolean {
  const total = profileCount(game.shape)
  let first = Number.NaN
  for (let s = 0; s < total; s++) {
    let sum = 0
    for (const u of game.payoffs) sum += u[s] ?? 0
    if (s === 0) first = sum
    else if (Math.abs(sum - first) > tol) return false
  }
  return true
}

/**
 * The uniform mixed strategy of every player.
 *
 * @param game - The game.
 * @returns One uniform strategy per player.
 * @example
 * ```ts
 * import { rockPaperScissors, uniformProfile } from 'mdml-engine/games'
 * uniformProfile(rockPaperScissors())[0].length // => 3
 * ```
 */
export function uniformProfile(game: NormalFormGame): number[][] {
  return game.shape.map((k) => new Array<number>(k).fill(1 / k))
}
