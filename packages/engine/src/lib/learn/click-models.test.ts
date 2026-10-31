import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { clickLogs, dbnClickLogs } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  type ClickSession,
  cascadeClickProbabilities,
  cascadeModel,
  clickSessions,
  dbnAlgorithm,
  dbnClickProbabilities,
  dbnModel,
  positionBasedAlgorithm,
  positionBasedClickProbabilities,
  positionBasedModel,
} from './click-models.ts'

const correlation = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  const n = a.length
  const ma = Array.from(a).reduce((x, y) => x + y, 0) / n
  const mb = Array.from(b).reduce((x, y) => x + y, 0) / n
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    sab += ((a[i] ?? 0) - ma) * ((b[i] ?? 0) - mb)
    saa += ((a[i] ?? 0) - ma) ** 2
    sbb += ((b[i] ?? 0) - mb) ** 2
  }
  return sab / Math.sqrt(saa * sbb)
}

describe('sessions from a log', () => {
  it('groups rows by session and orders them by display position', () => {
    const { data } = generate(clickLogs, { key: key(1), knobs: { sessions: 5, listLength: 4, items: 10 } })
    const s = clickSessions(data)
    expect(s).toHaveLength(5)
    for (const x of s) {
      expect(x.items).toHaveLength(4)
      expect(new Set(x.items).size).toBe(4)
    }
    expect(s.flatMap((x) => x.clicks)).toEqual(Array.from(toFlat(data.values)))
  })
})

describe('the position-based model', () => {
  const { data, truth } = generate(clickLogs, {
    key: key(2),
    knobs: { sessions: 6000, items: 20, listLength: 8, model: 'position', eta: 1, temperature: 2 },
  })
  const sessions = clickSessions(data)
  const relevance = truth.latents?.['relevance'] as Float64Array
  const fit = positionBasedModel(sessions, data.nItems)

  it('recovers the examination curve and the relevance of the log', () => {
    const truthCurve = Array.from({ length: 8 }, (_, r) => 1 / (r + 1))
    for (let r = 0; r < 8; r++)
      expect(Math.abs((fit.model.examination[r] ?? 0) - (truthCurve[r] ?? 0))).toBeLessThan(0.1)
    expect(correlation(fit.model.attractiveness, relevance)).toBeGreaterThan(0.95)
  })

  it('raises the likelihood at every iteration', () => {
    const alg = positionBasedAlgorithm(sessions, data.nItems)
    let s = alg.init(key(0))
    let last = s.logLikelihood
    for (let i = 0; i < 15; i++) {
      s = alg.step(s, { t: i, key: key(0) } as never)
      expect(s.logLikelihood).toBeGreaterThanOrEqual(last - 1e-8)
      last = s.logLikelihood
    }
    const back = alg.deserialise?.(alg.serialise?.(s) as never)
    expect(back?.alpha).toEqual(s.alpha)
    expect(runAlgorithm(alg, { seed: 0, maxSteps: 3 }).steps).toBe(3)
  })

  it('predicts click probabilities that match the observed click rate by rank', () => {
    const observed = new Array<number>(8).fill(0)
    const predicted = new Array<number>(8).fill(0)
    for (const s of sessions) {
      const p = positionBasedClickProbabilities(fit.model, s.items)
      for (let r = 0; r < 8; r++) {
        observed[r] = (observed[r] ?? 0) + (s.clicks[r] ?? 0) / sessions.length
        predicted[r] = (predicted[r] ?? 0) + (p[r] ?? 0) / sessions.length
      }
    }
    for (let r = 0; r < 8; r++) expect(Math.abs((observed[r] ?? 0) - (predicted[r] ?? 0))).toBeLessThan(0.02)
  })
})

describe('the cascade model', () => {
  it('is the clicks over the examinations, hand-computed', () => {
    const s: ClickSession[] = [
      { items: [0, 1], clicks: [0, 1] },
      { items: [0, 1], clicks: [1, 0] },
      { items: [1, 0], clicks: [0, 0] },
    ]
    // Examinations: session 1: items 0, 1 (click on 1); session 2: item 0 only (clicked); session 3: both, no click.
    // item 0: 1 click in 3 looks; item 1: 1 click in 2 looks.
    expect(cascadeModel(s, 2, [0, 0]).attractiveness).toEqual([1 / 3, 1 / 2])
    expect(cascadeClickProbabilities({ attractiveness: [0.5, 0.5], logLikelihood: 0 }, [0, 1])).toEqual([0.5, 0.25])
  })

  it('recovers the relevance of a cascade log', () => {
    const { data, truth } = generate(clickLogs, {
      key: key(3),
      knobs: { sessions: 6000, items: 20, listLength: 8, model: 'cascade', temperature: 2 },
    })
    const relevance = truth.latents?.['relevance'] as Float64Array
    const m = cascadeModel(clickSessions(data), data.nItems)
    expect(correlation(m.attractiveness, relevance)).toBeGreaterThan(0.95)
  })
})

describe('the dynamic Bayesian network model', () => {
  it('computes the likelihood of a session by the forward pass, as enumeration of the examination chain does', () => {
    const sessions: ClickSession[] = [
      { items: [0, 1, 2], clicks: [0, 1, 0] },
      { items: [2, 1, 0], clicks: [1, 0, 1] },
      { items: [1, 0, 2], clicks: [0, 0, 0] },
    ]
    const a = [0.3, 0.6, 0.45]
    const sat = [0.2, 0.7, 0.5]
    const gamma = 0.8
    const enumerate = (s: ClickSession, r: number, examining: boolean): number => {
      if (r === s.items.length) return 1
      const item = s.items[r] as number
      if (!examining) return s.clicks[r] === 0 ? enumerate(s, r + 1, false) : 0
      if (s.clicks[r] === 0)
        return (
          (1 - (a[item] as number)) * gamma * enumerate(s, r + 1, true) +
          (1 - (a[item] as number)) * (1 - gamma) * enumerate(s, r + 1, false)
        )
      const ai = a[item] as number
      const si = sat[item] as number
      return (
        ai *
        (si * enumerate(s, r + 1, false) +
          (1 - si) * (gamma * enumerate(s, r + 1, true) + (1 - gamma) * enumerate(s, r + 1, false)))
      )
    }
    const expected = sessions.reduce((acc, s) => acc + Math.log(enumerate(s, 0, true)), 0)
    // The algorithm's initial state has a = s = 0.5 and gamma = 0.9; compare under those parameters.
    const flat = (s: ClickSession): number => {
      const e = (r: number, examining: boolean): number => {
        if (r === s.items.length) return 1
        if (!examining) return s.clicks[r] === 0 ? e(r + 1, false) : 0
        if (s.clicks[r] === 0) return 0.5 * 0.9 * e(r + 1, true) + 0.5 * 0.1 * e(r + 1, false)
        return 0.5 * (0.5 * e(r + 1, false) + 0.5 * (0.9 * e(r + 1, true) + 0.1 * e(r + 1, false)))
      }
      return e(0, true)
    }
    expect(dbnAlgorithm(sessions, 3).init(key(0)).logLikelihood).toBeCloseTo(
      sessions.reduce((acc, s) => acc + Math.log(flat(s)), 0),
      12,
    )
    expect(expected).toBeLessThan(0)
  })

  const { data, truth } = generate(dbnClickLogs, {
    key: key(4),
    knobs: { sessions: 8000, items: 15, listLength: 8, continuation: 0.85 },
  })
  const sessions = clickSessions(data)
  const latents = truth.latents ?? {}
  const fit = dbnModel(sessions, data.nItems, { maxIter: 300 })

  it('recovers the attractiveness, the satisfaction and the continuation of the planted network', () => {
    expect(Math.abs(fit.model.continuation - 0.85)).toBeLessThan(0.05)
    expect(correlation(fit.model.attractiveness, latents['attractiveness'] as Float64Array)).toBeGreaterThan(0.95)
    expect(correlation(fit.model.satisfaction, latents['satisfaction'] as Float64Array)).toBeGreaterThan(0.8)
  })

  it('raises the likelihood at every iteration and serialises', () => {
    const alg = dbnAlgorithm(sessions, data.nItems)
    let s = alg.init(key(0))
    let last = s.logLikelihood
    for (let i = 0; i < 10; i++) {
      s = alg.step(s, { t: i, key: key(0) } as never)
      expect(s.logLikelihood).toBeGreaterThanOrEqual(last - 1e-7)
      last = s.logLikelihood
    }
    expect(alg.deserialise?.(alg.serialise?.(s) as never)?.continuation).toBe(s.continuation)
  })

  it('predicts the click rate by rank', () => {
    const observed = new Array<number>(8).fill(0)
    const predicted = new Array<number>(8).fill(0)
    for (const s of sessions) {
      const p = dbnClickProbabilities(fit.model, s.items)
      for (let r = 0; r < 8; r++) {
        observed[r] = (observed[r] ?? 0) + (s.clicks[r] ?? 0) / sessions.length
        predicted[r] = (predicted[r] ?? 0) + (p[r] ?? 0) / sessions.length
      }
    }
    for (let r = 0; r < 8; r++) expect(Math.abs((observed[r] ?? 0) - (predicted[r] ?? 0))).toBeLessThan(0.02)
  })
})
