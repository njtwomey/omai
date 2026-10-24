import { WhiteboxError } from '../../base/index.ts'
import { type LinprogOptions, linprog } from '../../optim/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { decodeProfile, profileCount, stridesOf } from './_profiles.ts'
import type { NormalFormGame } from './game.ts'

/** Options for {@link correlatedEquilibrium}. */
export interface CorrelatedOptions extends LinprogOptions {
  /**
   * What to maximise over the polytope of equilibria: the sum of payoffs (`'welfare'`, the default), the smallest
   * payoff (`'egalitarian'`), a weighted sum (one weight per player) or nothing (`'feasible'`, any equilibrium).
   */
  readonly objective?: 'welfare' | 'egalitarian' | 'feasible' | readonly number[]
  /** A coarse correlated equilibrium (deviations decided before seeing the recommendation) instead (default false). */
  readonly coarse?: boolean
}

/** A correlated equilibrium: a distribution over joint profiles (in profile order) and its expected payoffs. */
export interface CorrelatedEquilibrium {
  /** The probability of each joint profile, row-major (player 0's action slowest). */
  readonly distribution: readonly number[]
  readonly payoffs: readonly number[]
  /** The optimised objective. */
  readonly value: number
}

/**
 * The incentive rows: for each player, each recommended action (or none, for coarse) and each deviation, the gain
 * `u_i(a', s_-i) - u_i(s)` of every profile s where the recommendation applies (zero elsewhere).
 */
function incentiveRows(game: NormalFormGame, coarse: boolean): { player: number; row: Float64Array }[] {
  const total = profileCount(game.shape)
  const strides = stridesOf(game.shape)
  const out: { player: number; row: Float64Array }[] = []
  for (let p = 0; p < game.shape.length; p++) {
    const k = game.shape[p] ?? 0
    const stride = strides[p] ?? 1
    const u = game.payoffs[p] ?? new Float64Array(0)
    const recommended = coarse ? [-1] : Array.from({ length: k }, (_, a) => a)
    for (const a of recommended) {
      for (let d = 0; d < k; d++) {
        if (d === a) continue
        const row = new Float64Array(total)
        for (let s = 0; s < total; s++) {
          const own = Math.floor(s / stride) % k
          if (a >= 0 && own !== a) continue
          row[s] = (u[s + (d - own) * stride] ?? 0) - (u[s] ?? 0)
        }
        out.push({ player: p, row })
      }
    }
  }
  return out
}

/**
 * A correlated equilibrium (Aumann 1974) of any finite game by linear programming: a distribution over joint
 * profiles such that no player gains by deviating from a recommended action, which maximises the chosen objective.
 * With `coarse`, a coarse correlated equilibrium (Hannan set): no player gains by ignoring the recommendation
 * altogether. Nash equilibria are correlated equilibria, and the set is a polytope, so the optimum is found exactly.
 *
 * @param game - The game.
 * @param options - The objective, coarse or not, and options for the linear program.
 * @returns The distribution, the payoffs and the objective's value.
 * @throws WhiteboxError with code `convergence` when the linear program fails.
 * @concept correlated-equilibrium
 * @reference Aumann (1974), Subjectivity and correlation in randomized strategies.
 * @example
 * ```ts
 * import { chicken, correlatedEquilibrium } from 'whitebox-engine/games'
 * const ce = correlatedEquilibrium(chicken())
 * [ce.distribution.map((p) => Math.round(p * 1e9) / 1e9), Math.round(ce.value * 1e9) / 1e9] // => [[0.5, 0.25, 0.25, 0], 10.5]
 * ```
 */
export function correlatedEquilibrium(game: NormalFormGame, options: CorrelatedOptions = {}): CorrelatedEquilibrium {
  const { objective = 'welfare', coarse = false, ...lp } = options
  const total = profileCount(game.shape)
  const players = game.shape.length
  const rows = incentiveRows(game, coarse)
  const egalitarian = objective === 'egalitarian'
  const n = total + (egalitarian ? 1 : 0)
  const c = new Array<number>(n).fill(0)
  if (Array.isArray(objective) || objective === 'welfare') {
    const weights = Array.isArray(objective) ? objective : new Array<number>(players).fill(1)
    if (weights.length !== players) throw new WhiteboxError('shape', `${game.name}: one weight per player`)
    for (let s = 0; s < total; s++)
      for (let p = 0; p < players; p++) c[s] = (c[s] ?? 0) - (weights[p] ?? 0) * (game.payoffs[p]?.[s] ?? 0)
  } else if (egalitarian) c[total] = -1
  // Each incentive row is homogeneous (right-hand side 0), so scaling it to unit size changes nothing but the
  // conditioning of the program; payoff differences of very different sizes otherwise strain the simplex tolerances.
  const aUb = rows.map(({ row }) => {
    let big = 0
    for (const v of row) big = Math.max(big, Math.abs(v))
    const scaled = Array.from(row, (v) => (big > 0 ? v / big : 0))
    return egalitarian ? [...scaled, 0] : scaled
  })
  // Egalitarian: t <= E[u_p] for every player p.
  if (egalitarian) for (let p = 0; p < players; p++) aUb.push([...Array.from(game.payoffs[p] ?? [], (v) => -v), 1])
  const r = linprog(
    {
      c,
      aUb,
      bUb: new Array<number>(aUb.length).fill(0),
      aEq: [[...new Array<number>(total).fill(1), ...(egalitarian ? [0] : [])]],
      bEq: [1],
      bounds: [
        ...new Array<readonly [number, null]>(total).fill([0, null]),
        ...(egalitarian ? [[null, null] as const] : []),
      ],
    },
    lp,
  )
  if (!r.success)
    throw new WhiteboxError('convergence', `correlatedEquilibrium: the linear program ended with '${r.status}'`)
  const z = toFlat(r.x)
  const distribution = z.slice(0, total).map((v) => Math.max(0, v))
  const payoffs = game.payoffs.map((u) => distribution.reduce((s, v, k) => s + v * (u[k] ?? 0), 0))
  return { distribution, payoffs, value: objective === 'feasible' ? 0 : -r.fun }
}

/**
 * How far a joint distribution is from a (coarse) correlated equilibrium, per player: for a correlated equilibrium
 * the swap regret (the sum over recommended actions of the best gain from a deviation), for a coarse one the external
 * regret (the best gain from always playing one action). Zero for every player exactly at an equilibrium.
 *
 * @param game - The game.
 * @param distribution - The probability of each joint profile, row-major.
 * @param options - `coarse` for the coarse notion.
 * @returns One non-negative gain per player.
 * @throws WhiteboxError with code `shape` for a distribution of the wrong length.
 * @concept correlated-equilibrium
 * @example
 * ```ts
 * import { correlatedGains, prisonersDilemma } from 'whitebox-engine/games'
 * correlatedGains(prisonersDilemma(), [1, 0, 0, 0]) // => [2, 2]
 * ```
 */
export function correlatedGains(
  game: NormalFormGame,
  distribution: ArrayLike<number>,
  options: { readonly coarse?: boolean } = {},
): number[] {
  const total = profileCount(game.shape)
  if (distribution.length !== total)
    throw new WhiteboxError('shape', `${game.name}: ${total} profile probabilities expected`)
  const strides = stridesOf(game.shape)
  return game.shape.map((k, p) => {
    const stride = strides[p] ?? 1
    const u = game.payoffs[p] ?? new Float64Array(0)
    // gain[a][d]: the gain of playing d whenever a is recommended (a = 0 only, for coarse).
    const groups = options.coarse === true ? 1 : k
    const gain = Array.from({ length: groups }, () => new Array<number>(k).fill(0))
    for (let s = 0; s < total; s++) {
      const w = distribution[s] ?? 0
      if (w === 0) continue
      const own = Math.floor(s / stride) % k
      const g = gain[options.coarse === true ? 0 : own] ?? []
      for (let d = 0; d < k; d++) g[d] = (g[d] ?? 0) + w * ((u[s + (d - own) * stride] ?? 0) - (u[s] ?? 0))
    }
    let total2 = 0
    for (const g of gain) {
      let best = 0
      for (const v of g) best = Math.max(best, v)
      total2 += best
    }
    return total2
  })
}

/**
 * Whether a joint distribution is a (coarse) correlated equilibrium, within `tol`.
 *
 * @param game - The game.
 * @param distribution - The probability of each joint profile, row-major.
 * @param options - `coarse` for the coarse notion, and the tolerance (default 1e-8).
 * @returns Whether no player gains more than `tol`.
 * @concept correlated-equilibrium
 * @example
 * ```ts
 * import { chicken, isCorrelatedEquilibrium } from 'whitebox-engine/games'
 * isCorrelatedEquilibrium(chicken(), [1 / 3, 1 / 3, 1 / 3, 0]) // => true
 * ```
 */
export function isCorrelatedEquilibrium(
  game: NormalFormGame,
  distribution: ArrayLike<number>,
  options: { readonly coarse?: boolean; readonly tol?: number } = {},
): boolean {
  const tol = options.tol ?? 1e-8
  let mass = 0
  for (let s = 0; s < distribution.length; s++) {
    if ((distribution[s] ?? 0) < -tol) return false
    mass += distribution[s] ?? 0
  }
  if (Math.abs(mass - 1) > tol) return false
  return correlatedGains(game, distribution, options).every((g) => g <= tol)
}

/**
 * The joint distribution of independent mixed strategies (the product distribution), row-major: the correlated
 * equilibrium a Nash equilibrium induces.
 *
 * @param game - The game.
 * @param strategies - One mixed strategy per player.
 * @returns The probability of each joint profile.
 * @example
 * ```ts
 * import { jointDistribution, matchingPennies } from 'whitebox-engine/games'
 * jointDistribution(matchingPennies(), [[0.5, 0.5], [1, 0]]) // => [0.5, 0, 0.5, 0]
 * ```
 */
export function jointDistribution(game: NormalFormGame, strategies: readonly ArrayLike<number>[]): number[] {
  const total = profileCount(game.shape)
  return Array.from({ length: total }, (_, s) =>
    decodeProfile(s, game.shape).reduce((w, a, p) => w * (strategies[p]?.[a] ?? 0), 1),
  )
}
