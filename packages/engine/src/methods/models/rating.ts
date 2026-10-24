// TODO: Glicko-2 (volatility), TrueSkill on its factor graph by EP for teams and free-for-all races (here races are
// split into pairs), TrueSkill through time, Plackett-Luce, Massey and Colley ratings.
import { WhiteboxError } from '../../lib/base/index.ts'
import type { ComparisonData } from '../../lib/data/index.ts'
import {
  coefficientTable,
  type Explained,
  type Exposition,
  learningCurve,
  posteriorGrid,
} from '../../lib/exposition/index.ts'
import { ndtr, ndtri } from '../../lib/numerics/special/index.ts'
import { item, toFlat } from '../../lib/tensor/index.ts'

/** One rating system: a player's state, the win probability of one player over another, and the update of a game. */
export interface RatingSystem<S> {
  readonly name: string
  readonly initial: () => S
  /** P(a beats b) before the game (draws counted as half). */
  readonly expected: (a: S, b: S) => number
  /** The states after a game; `score` is 1 when a won, 0.5 for a draw, 0 when b won. */
  readonly update: (a: S, b: S, score: number, game: { readonly gamesA: number; readonly gamesB: number }) => [S, S]
  readonly rating: (s: S) => number
  /** The rating's standard deviation (0 for systems without one). */
  readonly deviation: (s: S) => number
}

/** The result of rating a sequence of games. */
export interface Ratings {
  readonly system: string
  /** Final rating and deviation per player. */
  readonly ratings: readonly number[]
  readonly deviations: readonly number[]
  /** Players from best to worst. */
  readonly order: readonly number[]
  /** P(first player wins) before each game, as predicted then. */
  readonly predictions: readonly number[]
  /** Games played per player. */
  readonly games: readonly number[]
  /** history[g][p]: player p's rating after game g. */
  readonly history: readonly (readonly number[])[]
}

/** Options of {@link rateGames}. */
export interface RateOptions {
  /** Players whose trajectories become `posteriorGrid` records (default the top 3). */
  readonly trajectories?: number
  /** Most points of the learning curve (default 100). */
  readonly curvePoints?: number
}

/** The pairwise games of comparison data, in order: a race of k players becomes its k (k - 1) / 2 pairs. */
function gamesOf(data: ComparisonData): { a: number; b: number; score: number }[] {
  const part = toFlat(data.participants)
  const ranks = toFlat(data.ranks)
  const k = data.participants.shape[1] ?? 2
  const n = data.participants.shape[0] ?? 0
  const out: { a: number; b: number; score: number }[] = []
  for (let g = 0; g < n; g++)
    for (let i = 0; i < k; i++)
      for (let j = i + 1; j < k; j++) {
        const ri = ranks[g * k + i] ?? 0
        const rj = ranks[g * k + j] ?? 0
        out.push({ a: part[g * k + i] ?? 0, b: part[g * k + j] ?? 0, score: ri < rj ? 1 : ri === rj ? 0.5 : 0 })
      }
  return out
}

/**
 * Rate players from comparison data with a rating system, game by game (races split into pairs): every game's
 * prediction is made before its result updates both players. Exposition: a coefficient table of the final ratings
 * (best first, with deviations and two-deviation intervals when the system has them), a `learningCurve` of the
 * running log loss and accuracy of the predictions against games played, and the rating trajectories of the top
 * players as `posteriorGrid` records (rating after every game, variance the squared deviation).
 *
 * @param data - Comparison data (participants and finishing places).
 * @param system - The rating system.
 * @param options - Trajectories and curve points.
 * @returns The ratings, predictions and history, with the exposition records.
 */
export function rateGames<S>(
  data: ComparisonData,
  system: RatingSystem<S>,
  options: RateOptions = {},
): Explained<Ratings> {
  const games = gamesOf(data)
  if (games.length === 0) throw new WhiteboxError('value', 'rateGames: no games')
  const players = Math.max(data.playerNames.length, ...games.flatMap((g) => [g.a + 1, g.b + 1]))
  const states = Array.from({ length: players }, () => system.initial())
  const played = new Array<number>(players).fill(0)
  const predictions: number[] = []
  const history: number[][] = []
  const deviations: number[][] = []
  for (const g of games) {
    const sa = states[g.a] as S
    const sb = states[g.b] as S
    predictions.push(system.expected(sa, sb))
    const [na, nb] = system.update(sa, sb, g.score, { gamesA: played[g.a] ?? 0, gamesB: played[g.b] ?? 0 })
    states[g.a] = na
    states[g.b] = nb
    played[g.a] = (played[g.a] ?? 0) + 1
    played[g.b] = (played[g.b] ?? 0) + 1
    history.push(states.map(system.rating))
    deviations.push(states.map(system.deviation))
  }
  const ratings = states.map(system.rating)
  const dev = states.map(system.deviation)
  const order = ratings.map((_, p) => p).sort((p, q) => (ratings[q] ?? 0) - (ratings[p] ?? 0) || p - q)
  // Prequential scores: running log loss and accuracy over decisive games.
  const sizes: number[] = []
  const logLoss: number[] = []
  const accuracy: number[] = []
  const every = Math.max(1, Math.ceil(games.length / (options.curvePoints ?? 100)))
  let ll = 0
  let right = 0
  let decisive = 0
  for (const [i, g] of games.entries()) {
    const p = Math.min(1 - 1e-12, Math.max(1e-12, predictions[i] ?? 0.5))
    if (g.score !== 0.5) {
      decisive++
      ll -= Math.log(g.score === 1 ? p : 1 - p)
      if ((p > 0.5 && g.score === 1) || (p < 0.5 && g.score === 0)) right++
      else if (p === 0.5) right += 0.5
    }
    if ((i + 1) % every === 0 || i === games.length - 1) {
      sizes.push(i + 1)
      logLoss.push(decisive === 0 ? Math.LN2 : ll / decisive)
      accuracy.push(decisive === 0 ? 0.5 : right / decisive)
    }
  }
  const names = (p: number) => data.playerNames[p] ?? `player ${p}`
  const hasDeviation = dev.some((d) => d > 0)
  const records: Exposition[] = [
    coefficientTable({
      names: order.map(names),
      estimates: order.map((p) => ratings[p] ?? 0),
      ...(hasDeviation
        ? {
            standardErrors: order.map((p) => dev[p] ?? 0),
            intervals: order.map(
              (p) => [(ratings[p] ?? 0) - 2 * (dev[p] ?? 0), (ratings[p] ?? 0) + 2 * (dev[p] ?? 0)] as [number, number],
            ),
            level: 0.95,
          }
        : {}),
    }),
    learningCurve({
      sizes,
      train: logLoss,
      validation: accuracy,
      resource: 'games',
      labels: { train: 'running log loss', validation: 'running accuracy' },
    }),
    ...order.slice(0, options.trajectories ?? 3).map((p) =>
      posteriorGrid({
        inputs: history.map((_, g) => [g + 1]),
        mean: history.map((h) => h[p] ?? 0),
        variance: deviations.map((d) => (d[p] ?? 0) ** 2),
      }),
    ),
  ]
  return {
    system: system.name,
    ratings,
    deviations: dev,
    order,
    predictions,
    games: played,
    history,
    exposition: records,
  }
}

/** Options of {@link eloSystem}. */
export interface EloOptions {
  /** The starting rating (default 1500). */
  readonly initial?: number
  /** The K-factor (default 32), or FIDE's: 40 for a player's first 30 games, then 20, and 10 once rated 2400. */
  readonly k?: number | 'fide'
  /** The logistic scale: a difference of `scale` points means odds of ten to one (default 400). */
  readonly scale?: number
}

/** An Elo player: the rating and whether the player has reached 2400 (FIDE's K-factor). */
export interface EloState {
  readonly rating: number
  readonly reached2400: boolean
}

/**
 * The Elo rating system (Elo 1978, with the chess conventions): E = 1 / (1 + 10^((R_b - R_a) / 400)) and
 * R_a += K (S_a - E_a).
 *
 * @param options - Starting rating, K-factor and scale.
 * @returns The rating system.
 */
export function eloSystem(options: EloOptions = {}): RatingSystem<EloState> {
  const scale = options.scale ?? 400
  const kOf = (s: EloState, games: number): number => {
    const k = options.k ?? 32
    if (k !== 'fide') return k
    return games < 30 ? 40 : s.reached2400 ? 10 : 20
  }
  const expected = (a: EloState, b: EloState): number => 1 / (1 + 10 ** ((b.rating - a.rating) / scale))
  return {
    name: 'elo',
    initial: () => ({ rating: options.initial ?? 1500, reached2400: false }),
    expected,
    update: (a, b, score, g) => {
      const e = expected(a, b)
      const ra = a.rating + kOf(a, g.gamesA) * (score - e)
      const rb = b.rating + kOf(b, g.gamesB) * (1 - score - (1 - e))
      return [
        { rating: ra, reached2400: a.reached2400 || ra >= 2400 },
        { rating: rb, reached2400: b.reached2400 || rb >= 2400 },
      ]
    },
    rating: (s) => s.rating,
    deviation: () => 0,
  }
}

/** Options of {@link glickoSystem}. */
export interface GlickoOptions {
  /** The starting rating and deviation (default 1500 and 350). */
  readonly initial?: number
  readonly deviation?: number
  /** Deviation growth per game a player sits out, sqrt(RD^2 + c^2 t) (default 0). */
  readonly c?: number
}

/** A Glicko player: rating, deviation and the game count when last seen. */
export interface GlickoState {
  readonly rating: number
  readonly rd: number
}

const Q = Math.LN10 / 400

/**
 * The Glicko rating system (Glickman 1999), each game its own rating period: ratings carry a deviation RD that
 * shrinks as a player plays; g(RD) = 1 / sqrt(1 + 3 q^2 RD^2 / pi^2) discounts results against uncertain opponents.
 *
 * @param options - Starting rating and deviation, and the per-game growth c.
 * @returns The rating system.
 */
export function glickoSystem(options: GlickoOptions = {}): RatingSystem<GlickoState> {
  const rd0 = options.deviation ?? 350
  const c = options.c ?? 0
  const g = (rd: number): number => 1 / Math.sqrt(1 + (3 * Q * Q * rd * rd) / (Math.PI * Math.PI))
  const e = (a: GlickoState, b: GlickoState): number => 1 / (1 + 10 ** ((-g(b.rd) * (a.rating - b.rating)) / 400))
  const one = (a: GlickoState, b: GlickoState, score: number): GlickoState => {
    const rd = Math.min(Math.sqrt(a.rd * a.rd + c * c), rd0)
    const ea = e({ ...a, rd }, b)
    const gb = g(b.rd)
    const d2 = 1 / (Q * Q * gb * gb * ea * (1 - ea))
    const denom = 1 / (rd * rd) + 1 / d2
    return { rating: a.rating + (Q / denom) * gb * (score - ea), rd: Math.sqrt(1 / denom) }
  }
  return {
    name: 'glicko',
    initial: () => ({ rating: options.initial ?? 1500, rd: rd0 }),
    expected: (a, b) => 1 / (1 + 10 ** ((-g(Math.sqrt(a.rd * a.rd + b.rd * b.rd)) * (a.rating - b.rating)) / 400)),
    update: (a, b, score) => [one(a, b, score), one(b, a, 1 - score)],
    rating: (s) => s.rating,
    deviation: (s) => s.rd,
  }
}

/** Options of {@link trueSkillSystem}. */
export interface TrueSkillOptions {
  /** The prior skill mean and standard deviation (default 25 and 25 / 3). */
  readonly mu?: number
  readonly sigma?: number
  /** The performance noise (default sigma / 2). */
  readonly beta?: number
  /** The dynamics noise added before each game (default sigma / 100). */
  readonly tau?: number
  /** The probability of a draw between equals (default 0.1). */
  readonly drawProbability?: number
}

/** A TrueSkill player: a Gaussian belief over skill. */
export interface TrueSkillState {
  readonly mu: number
  readonly sigma: number
}

const pdf = (x: number): number => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
const cdf = (x: number): number => item(ndtr(x))

/**
 * TrueSkill for two players (Herbrich, Minka and Graepel 2007; the `trueskill` package's 1v1 rating): skills are
 * Gaussian beliefs, performances add noise beta, and a game's result (a win beyond the draw margin, or a draw within
 * it) updates both beliefs by moment matching the truncated Gaussian (the exact EP update for two players).
 *
 * @param options - Prior, performance and dynamics noise, and the draw probability.
 * @returns The rating system (the rating is the mean; conservative ratings are mu - 3 sigma).
 */
export function trueSkillSystem(options: TrueSkillOptions = {}): RatingSystem<TrueSkillState> {
  const sigma0 = options.sigma ?? 25 / 3
  const beta = options.beta ?? sigma0 / 2
  const tau = options.tau ?? sigma0 / 100
  const draw = options.drawProbability ?? 0.1
  const margin = item(ndtri((draw + 1) / 2)) * Math.SQRT2 * beta
  return {
    name: 'trueskill',
    initial: () => ({ mu: options.mu ?? 25, sigma: sigma0 }),
    expected: (a, b) => cdf((a.mu - b.mu) / Math.sqrt(2 * beta * beta + a.sigma ** 2 + b.sigma ** 2)),
    update: (a0, b0, score) => {
      // The first is the winner (swapped when b won).
      const [w0, l0] = score === 0 ? [b0, a0] : [a0, b0]
      const w = { mu: w0.mu, s2: w0.sigma ** 2 + tau * tau }
      const l = { mu: l0.mu, s2: l0.sigma ** 2 + tau * tau }
      const c = Math.sqrt(2 * beta * beta + w.s2 + l.s2)
      const t = (w.mu - l.mu) / c
      const e = margin / c
      let v: number
      let wv: number
      if (score === 0.5) {
        const z = cdf(e - t) - cdf(-e - t)
        v = (pdf(-e - t) - pdf(e - t)) / z
        wv = v * v + ((e - t) * pdf(e - t) + (e + t) * pdf(e + t)) / z
      } else {
        const x = t - e
        const z = cdf(x)
        v = z > 1e-300 ? pdf(x) / z : -x
        wv = v * (v + x)
      }
      const nw = { mu: w.mu + (w.s2 / c) * v, sigma: Math.sqrt(w.s2 * (1 - (w.s2 / (c * c)) * wv)) }
      const nl = { mu: l.mu - (l.s2 / c) * v, sigma: Math.sqrt(l.s2 * (1 - (l.s2 / (c * c)) * wv)) }
      return score === 0 ? [nl, nw] : [nw, nl]
    },
    rating: (s) => s.mu,
    deviation: (s) => s.sigma,
  }
}

/**
 * Elo ratings of comparison data (see {@link eloSystem} and {@link rateGames}).
 *
 * @param data - Comparison data.
 * @param options - Elo's options and the exposition options.
 * @returns The ratings with their exposition.
 */
export function eloRatings(data: ComparisonData, options: EloOptions & RateOptions = {}): Explained<Ratings> {
  return rateGames(data, eloSystem(options), options)
}

/**
 * Glicko ratings of comparison data (see {@link glickoSystem} and {@link rateGames}).
 *
 * @param data - Comparison data.
 * @param options - Glicko's options and the exposition options.
 * @returns The ratings with their exposition.
 */
export function glickoRatings(data: ComparisonData, options: GlickoOptions & RateOptions = {}): Explained<Ratings> {
  return rateGames(data, glickoSystem(options), options)
}

/**
 * TrueSkill ratings of comparison data (see {@link trueSkillSystem} and {@link rateGames}).
 *
 * @param data - Comparison data.
 * @param options - TrueSkill's options and the exposition options.
 * @returns The ratings with their exposition.
 */
export function trueSkillRatings(
  data: ComparisonData,
  options: TrueSkillOptions & RateOptions = {},
): Explained<Ratings> {
  return rateGames(data, trueSkillSystem(options), options)
}

/**
 * Bradley-Terry strengths by Hunter's MM algorithm (2004): the maximum-likelihood static ratings of all games at
 * once (draws counted as half a win each), on the Elo scale (400 log10 strength, mean 1500). Exposition: the
 * coefficient table of the ratings and a `learningCurve` of the log likelihood per game over MM iterations.
 *
 * @param data - Comparison data.
 * @param options - Iterations (default 200) and tolerance (default 1e-10).
 * @returns The ratings (no predictions; history holds the ratings per iteration), with the exposition.
 */
export function bradleyTerry(
  data: ComparisonData,
  options: { readonly maxIter?: number; readonly tol?: number } = {},
): Explained<Ratings> {
  const games = gamesOf(data)
  const players = Math.max(data.playerNames.length, ...games.flatMap((g) => [g.a + 1, g.b + 1]))
  const wins = new Array<number>(players).fill(0)
  const pairs = new Map<string, number>()
  const played = new Array<number>(players).fill(0)
  for (const g of games) {
    wins[g.a] = (wins[g.a] ?? 0) + g.score
    wins[g.b] = (wins[g.b] ?? 0) + 1 - g.score
    played[g.a] = (played[g.a] ?? 0) + 1
    played[g.b] = (played[g.b] ?? 0) + 1
    const key = g.a < g.b ? `${g.a},${g.b}` : `${g.b},${g.a}`
    pairs.set(key, (pairs.get(key) ?? 0) + 1)
  }
  // A tiny pseudo-win against the field keeps unbeaten or winless players finite.
  const eps = 1e-3
  let p = new Array<number>(players).fill(1)
  const history: number[][] = []
  const ll: number[] = []
  const logLik = (q: readonly number[]): number =>
    games.reduce((s, g) => {
      const pa = (q[g.a] ?? 1) / ((q[g.a] ?? 1) + (q[g.b] ?? 1))
      return s + g.score * Math.log(pa) + (1 - g.score) * Math.log(1 - pa)
    }, 0) / games.length
  for (let it = 0; it < (options.maxIter ?? 200); it++) {
    const denom = new Array<number>(players).fill(0)
    for (const [key, n] of pairs) {
      const [i, j] = key.split(',').map(Number) as [number, number]
      const s = n / ((p[i] ?? 1) + (p[j] ?? 1))
      denom[i] = (denom[i] ?? 0) + s
      denom[j] = (denom[j] ?? 0) + s
    }
    const next = p.map((_, i) => ((wins[i] ?? 0) + eps) / ((denom[i] ?? 0) + (2 * eps) / ((p[i] ?? 1) + 1)))
    const gm = Math.exp(next.reduce((a, v) => a + Math.log(v), 0) / players)
    const scaled = next.map((v) => v / gm)
    const change = scaled.reduce((a, v, i) => Math.max(a, Math.abs(Math.log(v) - Math.log(p[i] ?? 1))), 0)
    p = scaled
    history.push(p.map((v) => 1500 + 400 * Math.log10(v)))
    ll.push(logLik(p))
    if (change < (options.tol ?? 1e-10)) break
  }
  const ratings = p.map((v) => 1500 + 400 * Math.log10(v))
  const order = ratings.map((_, i) => i).sort((a, b) => (ratings[b] ?? 0) - (ratings[a] ?? 0) || a - b)
  const names = (i: number) => data.playerNames[i] ?? `player ${i}`
  return {
    system: 'bradley-terry',
    ratings,
    deviations: ratings.map(() => 0),
    order,
    predictions: [],
    games: played,
    history,
    exposition: [
      coefficientTable({ names: order.map(names), estimates: order.map((i) => ratings[i] ?? 0) }),
      learningCurve({
        sizes: ll.map((_, i) => i + 1),
        train: ll,
        validation: ll,
        resource: 'MM iterations',
        labels: { train: 'log likelihood per game', validation: 'log likelihood per game' },
      }),
    ],
  }
}
