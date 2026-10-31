/**
 * Skill-rating systems as filters over a sequence of games: Elo (Elo 1978), Glicko (Glickman 1999) and Glicko-2
 * (Glickman 2012) with their rating deviations and volatilities, and TrueSkill (Herbrich, Minka and Graepel 2007) for
 * two players or two teams, where the factor graph of skills, performances and the game outcome is solved by Gaussian
 * moment matching (expectation propagation, exact for two sides). Each update is a pure function of the ratings and the
 * result, and `ratingAlgorithm` runs any of them as an `Algorithm` with one game (or, for the Glicko systems, one rating
 * period) per step and a serialisable state. The Bradley-Terry and Plackett-Luce models, which fit all games at once, are
 * in `choice-models.ts`.
 */

import { MdmlError } from '../base/index.ts'
import { ndtr, ndtri } from '../numerics/special/index.ts'
import { item } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'

const Phi = (x: number): number => item(ndtr(x))
const phi = (x: number): number => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)

/**
 * The expected score of A against B under Elo's logistic curve: 1 / (1 + 10^((rb - ra) / scale)).
 *
 * @param ra - A's rating.
 * @param rb - B's rating.
 * @param scale - The rating difference at which the odds are ten to one (default 400).
 * @returns A's expected score, in (0, 1).
 * @example
 * ```ts
 * import { eloExpected } from 'mdml-engine/pgm'
 * eloExpected(1500, 1500) // => 0.5
 * ```
 */
export function eloExpected(ra: number, rb: number, scale = 400): number {
  return 1 / (1 + 10 ** ((rb - ra) / scale))
}

/**
 * One Elo update: both ratings move by K times the surprise, in opposite directions.
 *
 * @param ra - A's rating.
 * @param rb - B's rating.
 * @param score - A's score: 1 a win, 0.5 a draw, 0 a loss.
 * @param k - The step size (default 32).
 * @param scale - The logistic scale (default 400).
 * @returns The new ratings.
 * @example
 * ```ts
 * import { eloUpdate } from 'mdml-engine/pgm'
 * eloUpdate(1500, 1500, 1) // => [1516, 1484]
 * ```
 */
export function eloUpdate(ra: number, rb: number, score: number, k = 32, scale = 400): [number, number] {
  const d = k * (score - eloExpected(ra, rb, scale))
  return [ra + d, rb - d]
}

/** A Glicko rating: the rating and its deviation. */
export interface GlickoRating {
  readonly r: number
  readonly rd: number
}

/** A result against one opponent in a rating period. */
export interface GlickoResult {
  readonly opponent: GlickoRating
  /** 1 a win, 0.5 a draw, 0 a loss. */
  readonly score: number
}

const Q = Math.log(10) / 400

/**
 * The Glicko-1 update over one rating period (Glickman 1999, steps 2 and 3): the new rating and deviation from the results
 * against the period's opponents. With no games the deviation only grows (use {@link glickoInflate} for the period gap).
 *
 * @param player - The rating before the period.
 * @param results - The games of the period.
 * @returns The new rating.
 * @example
 * ```ts
 * import { glickoUpdate } from 'mdml-engine/pgm'
 * const p = glickoUpdate({ r: 1500, rd: 200 }, [
 *   { opponent: { r: 1400, rd: 30 }, score: 1 },
 *   { opponent: { r: 1550, rd: 100 }, score: 0 },
 *   { opponent: { r: 1700, rd: 300 }, score: 0 },
 * ])
 * [Math.round(p.r), Math.round(p.rd * 10) / 10] // => [1464, 151.4]
 * ```
 */
export function glickoUpdate(player: GlickoRating, results: readonly GlickoResult[]): GlickoRating {
  if (results.length === 0) return player
  const g = (rd: number): number => 1 / Math.sqrt(1 + (3 * Q * Q * rd * rd) / (Math.PI * Math.PI))
  let invD2 = 0
  let sum = 0
  for (const { opponent, score } of results) {
    const gj = g(opponent.rd)
    const e = 1 / (1 + 10 ** ((-gj * (player.r - opponent.r)) / 400))
    invD2 += Q * Q * gj * gj * e * (1 - e)
    sum += gj * (score - e)
  }
  const denom = 1 / (player.rd * player.rd) + invD2
  return { r: player.r + (Q / denom) * sum, rd: Math.sqrt(1 / denom) }
}

/**
 * The Glicko deviation after idle rating periods: sqrt(rd^2 + c^2 t), capped at the unrated deviation.
 *
 * @param rd - The deviation.
 * @param c - The per-period volatility constant.
 * @param periods - Periods without play.
 * @param cap - The unrated deviation (default 350).
 * @returns The new deviation.
 * @example
 * ```ts
 * import { glickoInflate } from 'mdml-engine/pgm'
 * glickoInflate(50, 10, 3) // => 52.91502622129181
 * ```
 */
export function glickoInflate(rd: number, c: number, periods = 1, cap = 350): number {
  return Math.min(Math.sqrt(rd * rd + c * c * periods), cap)
}

/** A Glicko-2 rating: rating, deviation and volatility. */
export interface Glicko2Rating extends GlickoRating {
  readonly sigma: number
}

const SCALE2 = 173.7178

/**
 * The Glicko-2 update over one rating period (Glickman 2012): the volatility by the Illinois iteration on the log
 * variance, then the deviation and rating.
 *
 * @param player - The rating before the period.
 * @param results - The games of the period.
 * @param tau - The system constant limiting the volatility's change (default 0.5).
 * @returns The new rating, deviation and volatility.
 * @example
 * ```ts
 * import { glicko2Update } from 'mdml-engine/pgm'
 * const p = glicko2Update({ r: 1500, rd: 200, sigma: 0.06 }, [
 *   { opponent: { r: 1400, rd: 30 }, score: 1 },
 *   { opponent: { r: 1550, rd: 100 }, score: 0 },
 *   { opponent: { r: 1700, rd: 300 }, score: 0 },
 * ])
 * [Math.round(p.r * 100) / 100, Math.round(p.rd * 100) / 100, Math.round(p.sigma * 1e5) / 1e5] // => [1464.05, 151.52, 0.06]
 * ```
 */
export function glicko2Update(player: Glicko2Rating, results: readonly GlickoResult[], tau = 0.5): Glicko2Rating {
  const mu = (player.r - 1500) / SCALE2
  const phi0 = player.rd / SCALE2
  if (results.length === 0)
    return { r: player.r, rd: Math.sqrt(phi0 * phi0 + player.sigma ** 2) * SCALE2, sigma: player.sigma }
  const g = (p: number): number => 1 / Math.sqrt(1 + (3 * p * p) / (Math.PI * Math.PI))
  let vInv = 0
  let delta = 0
  for (const { opponent, score } of results) {
    const gj = g(opponent.rd / SCALE2)
    const e = 1 / (1 + Math.exp(-gj * (mu - (opponent.r - 1500) / SCALE2)))
    vInv += gj * gj * e * (1 - e)
    delta += gj * (score - e)
  }
  const v = 1 / vInv
  delta *= v
  const a = Math.log(player.sigma ** 2)
  const f = (x: number): number => {
    const ex = Math.exp(x)
    return (ex * (delta * delta - phi0 * phi0 - v - ex)) / (2 * (phi0 * phi0 + v + ex) ** 2) - (x - a) / (tau * tau)
  }
  let A = a
  let B: number
  if (delta * delta > phi0 * phi0 + v) B = Math.log(delta * delta - phi0 * phi0 - v)
  else {
    let k = 1
    while (f(a - k * tau) < 0) k++
    B = a - k * tau
  }
  let fA = f(A)
  let fB = f(B)
  while (Math.abs(B - A) > 1e-9) {
    const C = A + ((A - B) * fA) / (fB - fA)
    const fC = f(C)
    if (fC * fB <= 0) {
      A = B
      fA = fB
    } else fA /= 2
    B = C
    fB = fC
  }
  const sigma = Math.exp(A / 2)
  const phiStar = Math.sqrt(phi0 * phi0 + sigma * sigma)
  const phiNew = 1 / Math.sqrt(1 / (phiStar * phiStar) + 1 / v)
  let sum = 0
  for (const { opponent, score } of results) {
    const gj = g(opponent.rd / SCALE2)
    const e = 1 / (1 + Math.exp(-gj * (mu - (opponent.r - 1500) / SCALE2)))
    sum += gj * (score - e)
  }
  const muNew = mu + phiNew * phiNew * sum
  return { r: SCALE2 * muNew + 1500, rd: SCALE2 * phiNew, sigma }
}

/** A TrueSkill rating: a Gaussian belief about the skill. */
export interface TrueSkillRating {
  readonly mu: number
  readonly sigma: number
}

/** The parameters of the TrueSkill model. */
export interface TrueSkillEnvironment {
  /** The prior mean (default 25). */
  readonly mu?: number
  /** The prior deviation (default mu / 3). */
  readonly sigma?: number
  /** The performance noise (default sigma / 2). */
  readonly beta?: number
  /** The dynamics: skill drift added to the variance before each game (default sigma / 100). */
  readonly tau?: number
  /** The chance of a draw between equals (default 0.1). */
  readonly drawProbability?: number
}

function environment(env: TrueSkillEnvironment): Required<TrueSkillEnvironment> {
  const mu = env.mu ?? 25
  const sigma = env.sigma ?? mu / 3
  return {
    mu,
    sigma,
    beta: env.beta ?? sigma / 2,
    tau: env.tau ?? sigma / 100,
    drawProbability: env.drawProbability ?? 0.1,
  }
}

/** The additive correction v and multiplicative correction w of the truncated Gaussian, for a win or a draw. */
function vw(t: number, eps: number, draw: boolean): { v: number; w: number } {
  if (!draw) {
    const d = Phi(t - eps)
    const v = d < 2.222758749e-162 ? eps - t : phi(t - eps) / d
    return { v, w: v * (v + t - eps) }
  }
  const absT = Math.abs(t)
  const denom = Phi(eps - absT) - Phi(-eps - absT)
  const num = phi(-eps - absT) - phi(eps - absT)
  const v = denom < 2.222758749e-162 ? (t < 0 ? -t - eps : -t + eps) : t < 0 ? -num / denom : num / denom
  const w =
    denom < 2.222758749e-162
      ? 1
      : (denom > 0 ? ((eps - absT) * phi(eps - absT) + (eps + absT) * phi(eps + absT)) / denom : 1) + (num / denom) ** 2
  return { v, w }
}

/**
 * The TrueSkill update of a game between two teams (Herbrich et al. 2007): each player's skill is a Gaussian, a team's
 * performance is the sum of its players' performances (each skill plus noise beta), and the outcome truncates the
 * difference of the team performances. Moment matching gives the new beliefs in closed form.
 *
 * @param winners - The winning team's players (the first team on a draw).
 * @param losers - The losing team's players.
 * @param draw - Whether the game was drawn (default false).
 * @param env - The model parameters.
 * @returns The new beliefs, in the order given.
 * @example
 * ```ts
 * import { trueskillUpdate } from 'mdml-engine/pgm'
 * const [[w], [l]] = trueskillUpdate([{ mu: 25, sigma: 25 / 3 }], [{ mu: 25, sigma: 25 / 3 }])
 * [Math.round((w?.mu ?? 0) * 1000) / 1000, Math.round((l?.mu ?? 0) * 1000) / 1000] // => [29.396, 20.604]
 * ```
 */
export function trueskillUpdate(
  winners: readonly TrueSkillRating[],
  losers: readonly TrueSkillRating[],
  draw = false,
  env: TrueSkillEnvironment = {},
): [TrueSkillRating[], TrueSkillRating[]] {
  if (winners.length === 0 || losers.length === 0) throw new MdmlError('value', 'trueskillUpdate: empty team')
  const e = environment(env)
  const dynamic = (r: TrueSkillRating): number => r.sigma * r.sigma + e.tau * e.tau
  const sumVar = (team: readonly TrueSkillRating[]): number => team.reduce((a, r) => a + dynamic(r), 0)
  const players = winners.length + losers.length
  const c = Math.sqrt(sumVar(winners) + sumVar(losers) + players * e.beta * e.beta)
  const muW = winners.reduce((a, r) => a + r.mu, 0)
  const muL = losers.reduce((a, r) => a + r.mu, 0)
  const epsilon = item(ndtri((e.drawProbability + 1) / 2)) * Math.sqrt(players) * e.beta
  const { v, w } = vw((muW - muL) / c, epsilon / c, draw)
  const move = (team: readonly TrueSkillRating[], sign: 1 | -1): TrueSkillRating[] =>
    team.map((r) => {
      const s2 = dynamic(r)
      return { mu: r.mu + sign * ((s2 / c) * v), sigma: Math.sqrt(s2 * Math.max(1 - (s2 / (c * c)) * w, 1e-12)) }
    })
  return [move(winners, 1), move(losers, -1)]
}

/** The rating system an algorithm runs. */
export type RatingSystem = 'elo' | 'glicko' | 'glicko2' | 'trueskill'

/** A game between players a and b. */
export interface RatedGame {
  readonly a: number
  readonly b: number
  /** The score of a: 1 a win, 0.5 a draw, 0 a loss. */
  readonly score: number
  /** The rating period (the Glicko systems update all games of a period together; default each game its own). */
  readonly period?: number
}

/** The ratings of every player after some games. */
export interface RatingState {
  /** Games or periods processed. */
  readonly t: number
  readonly rating: readonly number[]
  /** The deviation (Glicko, Glicko-2) or sigma (TrueSkill); zero for Elo. */
  readonly deviation: readonly number[]
  /** The volatility (Glicko-2). */
  readonly volatility: readonly number[]
  /** The log-loss of the predictions made before each game so far. */
  readonly logLoss: number
}

/** Options of {@link ratingAlgorithm}. */
export interface RatingOptions {
  /** Elo's step size (default 32). */
  readonly k?: number
  /** The Glicko-2 system constant (default 0.5). */
  readonly tau?: number
  /** TrueSkill's parameters. */
  readonly trueskill?: TrueSkillEnvironment
}

/**
 * A rating system as a filter over games: one game per step (one rating period for the Glicko systems), tracking every
 * player's rating and uncertainty, and the log-loss of the win probability each system predicts before seeing the game
 * (the prequential score of the ratings).
 *
 * @param system - Elo, Glicko, Glicko-2 or TrueSkill.
 * @param games - The games in order.
 * @param nPlayers - The number of players.
 * @param options - The system's parameters.
 * @returns The algorithm and the number of steps.
 * @example
 * ```ts
 * import { ratingAlgorithm } from 'mdml-engine/pgm'
 * ratingAlgorithm('elo', [{ a: 0, b: 1, score: 1 }], 2).steps // => 1
 * ```
 */
export function ratingAlgorithm(
  system: RatingSystem,
  games: readonly RatedGame[],
  nPlayers: number,
  options: RatingOptions = {},
): { readonly algorithm: Algorithm<RatingState>; readonly steps: number } {
  const periodic = system === 'glicko' || system === 'glicko2'
  const groups: RatedGame[][] = []
  if (periodic) {
    const seen = new Map<number, RatedGame[]>()
    games.forEach((g, i) => {
      const p = g.period ?? i
      if (!seen.has(p)) seen.set(p, [])
      seen.get(p)?.push(g)
    })
    for (const p of [...seen.keys()].sort((a, b) => a - b)) groups.push(seen.get(p) as RatedGame[])
  } else for (const g of games) groups.push([g])
  const ts = environment(options.trueskill ?? {})
  const start = (): RatingState => ({
    t: 0,
    rating: new Array<number>(nPlayers).fill(system === 'trueskill' ? ts.mu : 1500),
    deviation: new Array<number>(nPlayers).fill(system === 'elo' ? 0 : system === 'trueskill' ? ts.sigma : 350),
    volatility: new Array<number>(nPlayers).fill(0.06),
    logLoss: 0,
  })
  const winProbability = (s: RatingState, a: number, b: number): number => {
    const ra = s.rating[a] ?? 0
    const rb = s.rating[b] ?? 0
    if (system === 'elo') return eloExpected(ra, rb)
    if (system === 'trueskill') {
      const c = Math.sqrt(2 * ts.beta * ts.beta + (s.deviation[a] ?? 0) ** 2 + (s.deviation[b] ?? 0) ** 2)
      return Phi((ra - rb) / c)
    }
    const gf = (rd: number): number =>
      1 / Math.sqrt(1 + (3 * Q * Q * (rd * rd + (s.deviation[b] ?? 0) ** 2)) / (Math.PI * Math.PI))
    return 1 / (1 + 10 ** ((-gf(s.deviation[a] ?? 0) * (ra - rb)) / 400))
  }
  const algorithm: Algorithm<RatingState> = {
    name: `rating-${system}`,
    config: { system, nPlayers, nGames: games.length, k: options.k ?? 32, tau: options.tau ?? 0.5 },
    schema: channels({
      metrics: { logLoss: { description: 'cumulative log-loss of the pre-game win predictions', direction: 'lower' } },
      tensors: { rating: { role: 'state', description: 'the rating of every player', shape: [nPlayers] } },
    }),
    init: start,
    step: (s) => {
      const group = groups[s.t]
      if (group === undefined) return s
      const rating = [...s.rating]
      const deviation = [...s.deviation]
      const volatility = [...s.volatility]
      let logLoss = s.logLoss
      for (const g of group) {
        const p = Math.min(Math.max(winProbability(s, g.a, g.b), 1e-12), 1 - 1e-12)
        logLoss -= g.score * Math.log(p) + (1 - g.score) * Math.log(1 - p)
      }
      if (system === 'elo') {
        for (const g of group) {
          const [na, nb] = eloUpdate(rating[g.a] ?? 1500, rating[g.b] ?? 1500, g.score, options.k ?? 32)
          rating[g.a] = na
          rating[g.b] = nb
        }
      } else if (system === 'trueskill') {
        for (const g of group) {
          const a = { mu: rating[g.a] ?? ts.mu, sigma: deviation[g.a] ?? ts.sigma }
          const b = { mu: rating[g.b] ?? ts.mu, sigma: deviation[g.b] ?? ts.sigma }
          const draw = g.score === 0.5
          const [x, y] = g.score >= 0.5 ? trueskillUpdate([a], [b], draw, ts) : trueskillUpdate([b], [a], false, ts)
          const [na, nb] = g.score >= 0.5 ? [x?.[0], y?.[0]] : [y?.[0], x?.[0]]
          rating[g.a] = na?.mu ?? a.mu
          deviation[g.a] = na?.sigma ?? a.sigma
          rating[g.b] = nb?.mu ?? b.mu
          deviation[g.b] = nb?.sigma ?? b.sigma
        }
      } else {
        // Both players of every game are updated from the ratings before the period.
        const players = new Set(group.flatMap((g) => [g.a, g.b]))
        for (const i of players) {
          const results: GlickoResult[] = group.flatMap((g) =>
            g.a === i
              ? [{ opponent: { r: s.rating[g.b] ?? 1500, rd: s.deviation[g.b] ?? 350 }, score: g.score }]
              : g.b === i
                ? [{ opponent: { r: s.rating[g.a] ?? 1500, rd: s.deviation[g.a] ?? 350 }, score: 1 - g.score }]
                : [],
          )
          if (system === 'glicko') {
            const n = glickoUpdate({ r: s.rating[i] ?? 1500, rd: s.deviation[i] ?? 350 }, results)
            rating[i] = n.r
            deviation[i] = n.rd
          } else {
            const n = glicko2Update(
              { r: s.rating[i] ?? 1500, rd: s.deviation[i] ?? 350, sigma: s.volatility[i] ?? 0.06 },
              results,
              options.tau ?? 0.5,
            )
            rating[i] = n.r
            deviation[i] = n.rd
            volatility[i] = n.sigma
          }
        }
      }
      return { t: s.t + 1, rating, deviation, volatility, logLoss }
    },
    done: (s) => (s.t >= groups.length ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { logLoss: s.logLoss },
      tensors: { rating: { role: 'state', value: [...s.rating], shape: [nPlayers] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as RatingState,
  }
  return { algorithm, steps: groups.length }
}
