import { WhiteboxError } from '../base/index.ts'
import {
  type DatasetGenerator,
  type DatasetRef,
  type InteractionData,
  interactions,
  parseCsv,
  unzip,
} from '../data/index.ts'
import { categorical, foldIn, gumbel, type Key, normal, poisson, uniform } from '../random/index.ts'
import { bool, choice, type DictSpace, dict, int, real } from '../space/index.ts'
import { stamp } from './_util.ts'

/** Row-major factor matrix `[rows, rank]` of standard normals scaled by `1 / sqrt(rank)`. */
function factors(k: Key, rows: number, rank: number, scale: number): Float64Array {
  return Float64Array.from(normal(k, [rows, rank]).data, (v) => (scale * v) / Math.sqrt(rank))
}

function dotRow(a: Float64Array, i: number, b: Float64Array, j: number, rank: number): number {
  let s = 0
  for (let r = 0; r < rank; r++) s += (a[i * rank + r] ?? 0) * (b[j * rank + r] ?? 0)
  return s
}

/**
 * Explicit ratings with low-rank truth (the matrix-completion setting): `rating = mu + b_user + b_item + u·v + noise`
 * with rank-`rank` factors, rounded and clipped to the 1 to 5 scale when `discrete`, and observed either completely at
 * random or more often for popular items and high ratings (missing not at random). The truth has the factors, the
 * biases and the full expected rating function.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { lowRankRatings } from 'whitebox-engine/datasets'
 * generate(lowRankRatings, { knobs: { users: 20, items: 30 } }).data.nItems // => 30
 * ```
 */
export const lowRankRatings: DatasetGenerator<InteractionData> = {
  name: 'lowRankRatings',
  kind: 'interactions',
  description: 'User-item ratings from a low-rank model with biases, observed at random or not at random.',
  knobs: dict({
    users: int(1, 100_000, { label: 'users', description: 'The number of users.', default: 100 }),
    items: int(1, 100_000, { label: 'items', description: 'The number of items.', default: 200 }),
    rank: int(1, 100, { label: 'rank', description: 'The number of latent factors.', default: 3 }),
    density: real(0.0001, 1, {
      description: 'The fraction of user-item ratings observed.',
      default: 0.1,
      label: 'fraction of ratings observed',
    }),
    noise: real(0, 5, {
      label: 'noise standard deviation',
      description: 'The standard deviation of the rating noise.',
      default: 0.3,
    }),
    discrete: bool({
      description: 'Round ratings to whole stars from 1 to 5.',
      default: true,
      label: 'round to 1 to 5 stars',
    }),
    missing: choice(['MCAR', 'MNAR'], {
      description: 'Ratings missing completely at random, or more often when low (not at random).',
      default: 'MCAR',
      label: 'observation mechanism',
    }),
  }),
  generate: (k, knobs) => {
    const nu = knobs['users'] as number
    const ni = knobs['items'] as number
    const rank = knobs['rank'] as number
    const noise = knobs['noise'] as number
    const discrete = knobs['discrete'] !== false
    const mu = 3.5
    const u = factors(foldIn(k, 0), nu, rank, 1)
    const v = factors(foldIn(k, 1), ni, rank, 1)
    const bu = Float64Array.from(normal(foldIn(k, 2), [nu]).data, (x) => 0.3 * x)
    const bi = Float64Array.from(normal(foldIn(k, 3), [ni]).data, (x) => 0.5 * x)
    const expected = (user: number, item: number) =>
      mu + (bu[user] ?? 0) + (bi[item] ?? 0) + dotRow(u, user, v, item, rank)
    const density = knobs['density'] as number
    const sel = uniform(foldIn(k, 4), [nu * ni]).data
    const z = normal(foldIn(k, 5), [nu * ni]).data
    const users: number[] = []
    const items: number[] = []
    const values: number[] = []
    const props: number[] = []
    for (let a = 0; a < nu; a++) {
      for (let b = 0; b < ni; b++) {
        const e = expected(a, b)
        // MNAR: popular items and high ratings are observed more often (a logistic in the item bias and the rating).
        const p =
          knobs['missing'] === 'MNAR'
            ? Math.min(1, density * 2 * (1 / (1 + Math.exp(-(2 * (bi[b] ?? 0) + (e - mu))))))
            : density
        if ((sel[a * ni + b] ?? 1) >= p) continue
        const raw = e + noise * (z[a * ni + b] ?? 0)
        users.push(a)
        items.push(b)
        values.push(discrete ? Math.min(5, Math.max(1, Math.round(raw))) : raw)
        props.push(p)
      }
    }
    return {
      data: interactions(
        { users, items, values, propensities: props, nUsers: nu, nItems: ni },
        stamp('lowRankRatings', lowRankRatings.description, knobs),
      ),
      truth: {
        noiseScale: noise,
        latents: { userFactors: u, itemFactors: v, userBias: bu, itemBias: bi },
        extra: { expected, globalMean: mu, rank, observationProbability: props },
      },
    }
  },
}

/** The knobs of the click-log generator. */
function clickKnobs(): DictSpace {
  return dict({
    sessions: int(1, 1_000_000, {
      default: 500,
      label: 'sessions',
      description: 'The number of search sessions logged.',
    }),
    items: int(2, 100_000, { default: 50, label: 'items', description: 'The number of items that can be shown.' }),
    listLength: int(1, 100, {
      default: 10,
      label: 'items shown per session',
      description: 'How many items each ranked list shows.',
    }),
    model: choice(['position', 'cascade'], {
      default: 'position',
      label: 'click model',
      description: 'Position-based (examination by rank) or cascade (scan down until a click).',
    }),
    eta: real(0, 5, {
      default: 1,
      label: 'position-bias exponent',
      description: 'How fast examination decays with rank: P(examine rank r) = r^-η.',
    }),
    temperature: real(0.01, 100, {
      default: 1,
      label: 'logging policy temperature',
      description: 'The temperature of the Plackett-Luce logging policy: higher is more random.',
    }),
  })
}

/**
 * Click logs from a stochastic logging policy and a known click model, for unbiased learning to rank and off-policy
 * evaluation. Each session shows `listLength` items ranked by a Plackett-Luce policy (Gumbel-perturbed scores
 * `relevance / temperature`); clicks follow the position-based model (`P(click) = (1 / rank)^eta × relevance`) or the
 * cascade model (scan down, click the first attractive item, stop). Every logged row carries its position and the
 * policy's probability of having placed that item at that position (estimated exactly for the top position, and by
 * Monte Carlo below it). The truth has every item's relevance and the examination probabilities.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { clickLogs } from 'whitebox-engine/datasets'
 * generate(clickLogs, { knobs: { sessions: 10, listLength: 5 } }).data.positions?.shape // => [50]
 * ```
 */
export const clickLogs: DatasetGenerator<InteractionData> = {
  name: 'clickLogs',
  kind: 'interactions',
  description: 'Ranked lists shown by a stochastic logging policy with clicks from a position-based or cascade model.',
  knobs: clickKnobs(),
  generate: (k, knobs) => {
    const sessions = knobs['sessions'] as number
    const ni = knobs['items'] as number
    const len = Math.min(knobs['listLength'] as number, ni)
    const eta = knobs['eta'] as number
    const temp = knobs['temperature'] as number
    const cascade = knobs['model'] === 'cascade'
    const relevance = Float64Array.from(uniform(foldIn(k, 0), [ni]).data, (x) => x ** 2)
    const examine = Array.from({ length: len }, (_, r) => (1 / (r + 1)) ** eta)
    // Placement probabilities of the Plackett-Luce policy, by Monte Carlo over 4000 rankings.
    const sims = 4000
    const placed = new Float64Array(ni * len)
    for (let s = 0; s < sims; s++) {
      const g = gumbel(foldIn(foldIn(k, 1), s), [ni]).data
      const order = Array.from({ length: ni }, (_, i) => i).sort(
        (a, b) => (relevance[b] ?? 0) / temp + (g[b] ?? 0) - ((relevance[a] ?? 0) / temp + (g[a] ?? 0)),
      )
      for (let r = 0; r < len; r++)
        placed[(order[r] ?? 0) * len + r] = (placed[(order[r] ?? 0) * len + r] ?? 0) + 1 / sims
    }
    const users: number[] = []
    const items: number[] = []
    const values: number[] = []
    const positions: number[] = []
    const props: number[] = []
    const sess: number[] = []
    for (let s = 0; s < sessions; s++) {
      const g = gumbel(foldIn(foldIn(k, 2), s), [ni]).data
      const order = Array.from({ length: ni }, (_, i) => i).sort(
        (a, b) => (relevance[b] ?? 0) / temp + (g[b] ?? 0) - ((relevance[a] ?? 0) / temp + (g[a] ?? 0)),
      )
      const u = uniform(foldIn(foldIn(k, 3), s), [len]).data
      let stopped = false
      for (let r = 0; r < len; r++) {
        const item = order[r] ?? 0
        let click = 0
        if (cascade) {
          if (!stopped && (u[r] ?? 1) < (relevance[item] ?? 0)) {
            click = 1
            stopped = true
          }
        } else if ((u[r] ?? 1) < (examine[r] ?? 0) * (relevance[item] ?? 0)) click = 1
        users.push(s)
        items.push(item)
        values.push(click)
        positions.push(r)
        props.push(Math.max(placed[item * len + r] ?? 0, 1 / sims))
        sess.push(s)
      }
    }
    return {
      data: interactions(
        { users, items, values, positions, propensities: props, sessions: sess, nUsers: sessions, nItems: ni },
        stamp('clickLogs', clickLogs.description, knobs),
      ),
      truth: {
        latents: { relevance },
        extra: {
          examination: cascade ? undefined : examine,
          model: cascade ? 'cascade' : 'position',
          placement: placed,
        },
      },
    }
  },
}

/**
 * Implicit feedback (consumption counts) from Poisson matrix factorisation: `count ~ Poisson(sum_r theta_ur beta_ir)`
 * with gamma-distributed non-negative factors; zero counts are not logged, as in play counts or purchases. The truth
 * has the factors and the rate function.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { implicitFeedback } from 'whitebox-engine/datasets'
 * generate(implicitFeedback, { knobs: { users: 10, items: 10 } }).data.nUsers // => 10
 * ```
 */
export const implicitFeedback: DatasetGenerator<InteractionData> = {
  name: 'implicitFeedback',
  kind: 'interactions',
  description: 'Consumption counts from Poisson matrix factorisation with gamma factors; zeros are unobserved.',
  knobs: dict({
    users: int(1, 100_000, { label: 'users', description: 'The number of users.', default: 100 }),
    items: int(1, 100_000, { label: 'items', description: 'The number of items.', default: 100 }),
    rank: int(1, 100, { label: 'rank', description: 'The number of latent gamma factors.', default: 3 }),
    activity: real(0.001, 100, {
      description: 'The mean consumption count per user-item pair.',
      default: 0.3,
      label: 'mean rate per user-item pair',
    }),
  }),
  generate: (k, knobs) => {
    const nu = knobs['users'] as number
    const ni = knobs['items'] as number
    const rank = knobs['rank'] as number
    const scale = Math.sqrt((knobs['activity'] as number) / rank)
    // Gamma(0.5) factors with mean `scale` give sparse, heavy-tailed preferences.
    const theta = Float64Array.from(normal(foldIn(k, 0), [nu, rank]).data, (z) => scale * z * z)
    const beta = Float64Array.from(normal(foldIn(k, 1), [ni, rank]).data, (z) => scale * z * z)
    const rate = (a: number, b: number) => dotRow(theta, a, beta, b, rank)
    const users: number[] = []
    const items: number[] = []
    const values: number[] = []
    for (let a = 0; a < nu; a++) {
      const lam = Array.from({ length: ni }, (_, b) => rate(a, b))
      for (const [b, l] of lam.entries()) {
        if (l <= 0) continue
        const c = poisson(foldIn(foldIn(k, 2), a * ni + b), l, [1]).data[0] ?? 0
        if (c > 0) {
          users.push(a)
          items.push(b)
          values.push(c)
        }
      }
    }
    return {
      data: interactions(
        { users, items, values, nUsers: nu, nItems: ni },
        stamp('implicitFeedback', implicitFeedback.description, knobs),
      ),
      truth: { latents: { userFactors: theta, itemFactors: beta }, extra: { rate } },
    }
  },
}

/**
 * Interactions with a Zipfian (long-tail) catalogue: item `i` (0-based popularity rank) is chosen with probability
 * proportional to `(i + 1)^-s`, and users by their own Zipfian activity, the popularity skew behind cold-start and
 * popularity-bias studies. The truth gives both distributions.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { zipfianCatalogue } from 'whitebox-engine/datasets'
 * generate(zipfianCatalogue, { knobs: { interactions: 100 } }).data.values.shape // => [100]
 * ```
 */
export const zipfianCatalogue: DatasetGenerator<InteractionData> = {
  name: 'zipfianCatalogue',
  kind: 'interactions',
  description: 'Interactions whose items and users follow Zipf popularity laws.',
  knobs: dict({
    interactions: int(1, 10_000_000, {
      label: 'interactions',
      description: 'The number of user-item interactions.',
      default: 5000,
    }),
    users: int(1, 1_000_000, { label: 'users', description: 'The number of users.', default: 500 }),
    items: int(1, 1_000_000, { label: 'items', description: 'The number of items.', default: 1000 }),
    itemExponent: real(0, 5, {
      description: 'The Zipf exponent of item popularity: larger concentrates on the head.',
      default: 1.1,
      label: 'Zipf exponent of items',
    }),
    userExponent: real(0, 5, {
      description: 'The Zipf exponent of user activity.',
      default: 0.8,
      label: 'Zipf exponent of users',
    }),
  }),
  generate: (k, knobs) => {
    const m = knobs['interactions'] as number
    const nu = knobs['users'] as number
    const ni = knobs['items'] as number
    const zipf = (n: number, s: number) => {
      const w = Array.from({ length: n }, (_, i) => (i + 1) ** -s)
      const z = w.reduce((a, b) => a + b, 0)
      return w.map((v) => v / z)
    }
    const pi = zipf(ni, knobs['itemExponent'] as number)
    const pu = zipf(nu, knobs['userExponent'] as number)
    const items = Array.from(categorical(foldIn(k, 0), pi, [m]).data)
    const users = Array.from(categorical(foldIn(k, 1), pu, [m]).data)
    const times = Array.from(uniform(foldIn(k, 2), [m]).data).sort((a, b) => a - b)
    return {
      data: interactions(
        { users, items, values: new Array<number>(m).fill(1), times, nUsers: nu, nItems: ni },
        stamp('zipfianCatalogue', zipfianCatalogue.description, knobs),
      ),
      truth: { extra: { itemPopularity: pi, userActivity: pu } },
    }
  },
}

/**
 * Parse GroupLens' `ml-latest-small.zip` (100836 ratings of 9742 movies by 610 users, half-star scale) into
 * interactions with rating times and movie titles as item names. Users and items are re-indexed from 0 in order of
 * first appearance in the files' id order.
 *
 * @param bytes - The archive.
 * @returns The interactions.
 * @throws WhiteboxError with code `value` when the archive lacks `ratings.csv`.
 * @example
 * ```ts
 * import { parseMovieLens } from 'whitebox-engine/datasets'
 * typeof parseMovieLens // => 'function'
 * ```
 */
export async function parseMovieLens(bytes: Uint8Array): Promise<InteractionData> {
  const files = await unzip(bytes)
  const find = (suffix: string) => [...files.entries()].find(([name]) => name.endsWith(suffix))?.[1]
  const ratings = find('ratings.csv')
  if (ratings === undefined) throw new WhiteboxError('value', 'parseMovieLens: no ratings.csv in the archive')
  const movies = find('movies.csv')
  const rows = parseCsv(new TextDecoder().decode(ratings)).rows
  const titles = new Map<string, string>()
  if (movies !== undefined)
    for (const r of parseCsv(new TextDecoder().decode(movies)).rows) titles.set(r[0] ?? '', r[1] ?? '')
  const userIndex = new Map<string, number>()
  const itemIndex = new Map<string, number>()
  const index = (map: Map<string, number>, key: string) => {
    let v = map.get(key)
    if (v === undefined) {
      v = map.size
      map.set(key, v)
    }
    return v
  }
  const users = rows.map((r) => index(userIndex, r[0] ?? ''))
  const items = rows.map((r) => index(itemIndex, r[1] ?? ''))
  return interactions(
    {
      users,
      items,
      values: rows.map((r) => Number(r[2])),
      times: rows.map((r) => Number(r[3])),
      itemNames: [...itemIndex.keys()].map((id) => titles.get(id) ?? id),
      userNames: [...userIndex.keys()],
    },
    {
      name: 'movielensSmall',
      description: 'MovieLens latest-small: 100k half-star ratings of 9.7k movies by 610 users (GroupLens).',
      source: 'GroupLens ml-latest-small.zip',
      url: 'https://grouplens.org/datasets/movielens/latest/',
      citation: 'Harper and Konstan (2015), The MovieLens datasets: history and context, ACM TiiS 5(4).',
      licence: 'GroupLens research licence (no redistribution; fetched from the source)',
    },
  )
}

/**
 * A small MovieLens (`ml-latest-small`, 978 kB) as a remote asset pinned by its SHA-256.
 *
 * @example
 * ```ts
 * import { movielensSmall } from 'whitebox-engine/datasets'
 * movielensSmall.kind // => 'remote'
 * ```
 */
export const movielensSmall: DatasetRef<InteractionData> = {
  kind: 'remote',
  name: 'movielensSmall',
  url: 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
  sha256: '696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436',
  size: 978_202,
  licence: 'GroupLens research licence (no redistribution; fetched from the source)',
  parse: parseMovieLens,
}
