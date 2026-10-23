import { describe, expect, it } from 'vitest'
import { argmaxOf, maxOf } from '../../lib/base/index.ts'
import { generate, load, memoryCache, rowCount, sha256Hex } from '../../lib/data/index.ts'
import { key } from '../../lib/random/index.ts'
import { type Tensor, toFlat } from '../../lib/tensor/index.ts'
import {
  additionTask,
  associativeRecall,
  clickLogs,
  copyTask,
  driftingTopics,
  dyckTask,
  eloGames,
  implicitFeedback,
  lowRankRatings,
  modularArithmetic,
  multiplayerRaces,
  naiveBayesCorpus,
  nearDuplicateCorpus,
  nurseryRhymes,
  parityTask,
  reverseTask,
  roundRobin,
  sortTask,
  tinyShakespeare,
  topicCorpus,
  workedExampleCorpus,
  zipfianCatalogue,
} from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

describe('comparisons', () => {
  it('Elo games follow the Davidson model', () => {
    const { data, truth } = generate(eloGames, {
      key: key(0),
      knobs: { players: 2, games: 4000, spread: 300, drawWeight: 0.5 },
    })
    const skills = toFlat(truth.latents?.['skills'] as Tensor)
    const probability = truth.extra?.['probabilities'] as (a: number, b: number) => { win: number; draw: number }
    const probs = probability(skills[0] ?? 0, skills[1] ?? 0)
    const p = toFlat(data.participants)
    const r = toFlat(data.ranks)
    let wins0 = 0
    let draws = 0
    for (let g = 0; g < 4000; g++) {
      if (r[2 * g] === r[2 * g + 1]) draws++
      else if ((p[2 * g + (r[2 * g] === 0 ? 0 : 1)] ?? 0) === 0) wins0++
    }
    expect(Math.abs(draws / 4000 - probs.draw)).toBeLessThan(4 * Math.sqrt(0.25 / 4000))
    expect(Math.abs(wins0 / 4000 - probs.win)).toBeLessThan(4 * Math.sqrt(0.25 / 4000))
    expect(p.every((v, i) => i % 2 === 0 || v !== p[i - 1])).toBe(true)
  })
  it('round robins schedule every pair and races rank by performance', () => {
    const { data, truth } = generate(roundRobin, { key: key(1), knobs: { players: 5, rounds: 3 } })
    expect(rowCount(data)).toBe(30)
    expect(truth.extra?.['order']).toHaveLength(5)
    const race = multiplayerRaces(key(2), { players: 12, races: 50, size: 5 })
    const perf = toFlat(race.truth.latents?.['performances'] as Tensor)
    const ranks = toFlat(race.data.ranks)
    for (let r = 0; r < 50; r++) {
      const winner = ranks.slice(r * 5, r * 5 + 5).indexOf(0)
      const best = perf.slice(r * 5, r * 5 + 5)
      expect(best[winner]).toBe(maxOf(best))
    }
  })
})

describe('ranking', () => {
  it('low-rank ratings recover their expected values', () => {
    const { data, truth } = generate(lowRankRatings, {
      key: key(3),
      knobs: { users: 60, items: 80, density: 0.5, discrete: false, noise: 0.2 },
    })
    const f = truth.extra?.['expected'] as (u: number, i: number) => number
    const u = toFlat(data.users)
    const i = toFlat(data.items)
    const r = toFlat(data.values).map((v, j) => v - f(u[j] ?? 0, i[j] ?? 0))
    expect(Math.sqrt(mean(r.map((v) => v * v)))).toBeCloseTo(0.2, 1)
    expect(Math.abs(rowCount(data) / (60 * 80) - 0.5)).toBeLessThan(0.03)
    const stars = generate(lowRankRatings, { key: key(4), knobs: { users: 20, items: 20, missing: 'MNAR' } })
    expect(toFlat(stars.data.values).every((v) => Number.isInteger(v) && v >= 1 && v <= 5)).toBe(true)
  })
  it('click logs: position bias and propensities', () => {
    const { data, truth } = generate(clickLogs, {
      key: key(5),
      knobs: { sessions: 3000, items: 20, listLength: 5, temperature: 0.2 },
    })
    const pos = toFlat(data.positions as Tensor)
    const clicks = toFlat(data.values)
    const ctr = [0, 1, 2, 3, 4].map((r) => mean(clicks.filter((_, j) => pos[j] === r)))
    expect(ctr[0]).toBeGreaterThan(ctr[4] ?? 1)
    const props = toFlat(data.propensities as Tensor)
    expect(props.every((p) => p > 0 && p <= 1)).toBe(true)
    // Each session's positions are a ranking: placement probabilities sum to one at every position.
    const placed = truth.extra?.['placement'] as Float64Array
    for (let r = 0; r < 5; r++) {
      let s = 0
      for (let it = 0; it < 20; it++) s += placed[it * 5 + r] ?? 0
      expect(s).toBeCloseTo(1, 10)
    }
    const cascade = generate(clickLogs, { key: key(6), knobs: { sessions: 200, model: 'cascade' } })
    const c = toFlat(cascade.data.values)
    for (let s = 0; s < 200; s++) expect(c.slice(s * 10, s * 10 + 10).reduce((a, b) => a + b, 0)).toBeLessThanOrEqual(1)
  })
  it('implicit feedback and Zipfian catalogues', () => {
    const imp = generate(implicitFeedback, { key: key(7), knobs: { users: 30, items: 30 } })
    expect(toFlat(imp.data.values).every((v) => v >= 1)).toBe(true)
    const z = generate(zipfianCatalogue, { key: key(8), knobs: { interactions: 20000, items: 100, itemExponent: 1 } })
    const items = toFlat(z.data.items)
    const top = items.filter((i) => i === 0).length / 20000
    const pi = z.truth.extra?.['itemPopularity'] as number[]
    expect(Math.abs(top - (pi[0] ?? 0))).toBeLessThan(4 * Math.sqrt(((pi[0] ?? 0) * (1 - (pi[0] ?? 0))) / 20000))
  })
})

describe('corpora', () => {
  it('topic corpora: bars topics and an exact log-likelihood', () => {
    const { data, truth } = generate(topicCorpus, { key: key(9), knobs: { topics: 'bars', nDocs: 50, docLength: 30 } })
    const topics = truth.latents?.['topics'] as number[][]
    expect(topics.length).toBe(10)
    for (const t of topics) expect(t.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const z = truth.latents?.['assignments'] as number[][]
    // Each token lies in its topic's bar.
    for (const [d, doc] of (data.tokens ?? []).entries())
      for (const [i, w] of doc.entries()) expect(topics[z[d]?.[i] ?? 0]?.[w]).toBeGreaterThan(0)
    expect(Number.isFinite(truth.extra?.['logLikelihood'] as number)).toBe(true)
  })
  it('naive Bayes corpora have an exact posterior that classifies well', () => {
    const { data, truth } = generate(naiveBayesCorpus, { key: key(10), knobs: { nDocs: 300, signal: 0.5 } })
    const post = truth.extra?.['posterior'] as (d: readonly number[]) => number[]
    const y = toFlat(data.labels as Tensor)
    const acc = mean((data.tokens ?? []).map((d, i) => (argmaxOf(post(d)) === y[i] ? 1 : 0)))
    expect(acc).toBeGreaterThan(0.95)
  })
  it('drifting topics move between slices', () => {
    const { truth } = generate(driftingTopics, { key: key(11), knobs: { slices: 3, drift: 1 } })
    const t = truth.latents?.['topics'] as number[][][]
    const tv = (a: number[], b: number[]) => 0.5 * a.reduce((s, v, i) => s + Math.abs(v - (b[i] ?? 0)), 0)
    expect(tv(t[0]?.[0] ?? [], t[2]?.[0] ?? [])).toBeGreaterThan(0.05)
  })
  it('real and worked-example corpora', () => {
    expect(workedExampleCorpus('deerwester').classes).toEqual(['hci', 'graphs'])
    expect(workedExampleCorpus('caesar').documents?.[1]).toContain('noble Brutus')
    expect(nurseryRhymes().documents?.length).toBe(16)
  })
  it('near-duplicates are planted with their similarity', () => {
    const { data, truth } = generate(nearDuplicateCorpus, {
      key: key(12),
      knobs: { nOriginals: 40, duplicated: 0.25, copies: 2, editRate: 0.05 },
    })
    expect(data.documents?.length).toBe(60)
    const pairs = truth.extra?.['pairs'] as { original: number; copy: number; jaccard: number }[]
    expect(pairs.length).toBe(20)
    const clusters = toFlat(truth.clusters as Tensor)
    for (const p of pairs) {
      expect(clusters[p.original]).toBe(clusters[p.copy])
      expect(p.jaccard).toBeGreaterThan(0.4)
    }
    const zero = generate(nearDuplicateCorpus, {
      key: key(13),
      knobs: { nOriginals: 5, duplicated: 1, copies: 1, editRate: 0 },
    })
    const zeroPairs = (zero.truth.extra?.['pairs'] ?? []) as { jaccard: number }[]
    expect(zeroPairs.every((p) => p.jaccard === 1)).toBe(true)
  })
  it('Tiny Shakespeare parses into passages of character ids', async () => {
    const bytes = new TextEncoder().encode('First Citizen:\nBefore we proceed.\n\nAll:\nSpeak, speak.\n')
    const ref = { ...tinyShakespeare, sha256: await sha256Hex(bytes) }
    const c = await load(ref, {
      cache: memoryCache(),
      fetch: async () => ({ ok: true, status: 200, arrayBuffer: async () => bytes.slice().buffer }),
    })
    expect(c.documents?.length).toBe(2)
    expect(c.tokens?.[1]?.map((t) => c.vocabulary?.[t]).join('')).toBe('All:\nSpeak, speak.\n')
  })
})

describe('sequence tasks', () => {
  it('targets solve the tasks', () => {
    for (const g of [
      copyTask,
      reverseTask,
      sortTask,
      dyckTask,
      additionTask,
      associativeRecall,
      parityTask,
      modularArithmetic,
    ]) {
      const { data, truth } = generate(g, {
        key: key(14),
        knobs: g === modularArithmetic ? { modulus: 11 } : { n: 50 },
      })
      const solve = truth.extra?.['solve'] as (s: readonly number[]) => number[]
      const [n = 0, t = 0] = data.inputs.shape
      const [, u = 0] = data.targets.shape
      for (let i = 0; i < n; i++) {
        const input = Array.from(data.inputs.data.slice(i * t, (i + 1) * t)).filter(
          (_, j) => data.inputMask.data[i * t + j] === 1,
        )
        const target = Array.from(data.targets.data.slice(i * u, (i + 1) * u)).filter(
          (_, j) => data.targetMask.data[i * u + j] === 1,
        )
        expect(solve(input), g.name).toEqual(target)
      }
      const acc = truth.extra?.['accuracy'] as (p: readonly (readonly number[])[]) => number
      expect(acc([])).toBe(0)
    }
  })
  it('Dyck strings are balanced exactly half the time, sums are right, the table is complete', () => {
    const { data } = generate(dyckTask, { key: key(15), knobs: { n: 200 } })
    const yes = toFlat(data.targets).filter((v) => v === 3).length
    expect(yes).toBeGreaterThanOrEqual(99)
    expect(yes).toBeLessThanOrEqual(101)
    const add = generate(additionTask, { key: key(16), knobs: { n: 1, digits: 5 } })
    const solve = add.truth.extra?.['solve'] as (s: readonly number[]) => number[]
    // 999 + 1 = 1000: tokens are digit + 2, '+' is 12.
    expect(solve([11, 11, 11, 12, 3])).toEqual([3, 2, 2, 2])
    const table = generate(modularArithmetic, { key: key(17), knobs: { modulus: 7, operation: '/' } })
    expect(table.data.inputs.shape[0]).toBe(42)
    const div = table.truth.extra?.['solve'] as (s: readonly number[]) => number[]
    // 3 / 5 mod 7 = 3 * 3 = 2 (5 * 3 = 15 = 1 mod 7); numbers are offset by the two special tokens.
    expect(div([5, 9, 7, 10])).toEqual([4])
  })
})
