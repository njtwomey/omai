import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { quantileSketches } from './quantile-options.ts'
import {
  gkSummary,
  gkSummaryAlgorithm,
  kllSketch,
  kllSketchAlgorithm,
  p2Algorithm,
  quantileSketchContract,
} from './quantiles.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const items = Array.from({ length: 1000 }, (_, i) => ((i * 7919) % 1000) + 1)

describe('gkSummaryAlgorithm', () => {
  it('adds one value per step, keeps the rank bound and equals gkSummary', () => {
    const eps = 0.02
    const run = runAlgorithm(gkSummaryAlgorithm({ epsilon: eps, items }), { maxSteps: 2000 })
    expect([run.status, run.steps, run.state.count]).toEqual(['completed', 1000, 1000])
    const direct = gkSummary(eps)
    for (const v of items) direct.add(v)
    expect(run.state.tuples).toEqual(direct.tuples())
    expect(run.state.tuples.length).toBeLessThan(400)
    // Every stored tuple keeps g + delta within 2 epsilon n, and the median is within epsilon n ranks of the truth.
    for (const t of run.state.tuples.slice(1, -1))
      expect(t.g + t.delta).toBeLessThanOrEqual(Math.max(1, 2 * eps * 1000) + 1)
    expect(Math.abs(direct.quantile(0.5) - 500)).toBeLessThanOrEqual(eps * 1000 + 1)
  })

  it('serialises, with NaN extremes before the first item, and resumes', () => {
    const alg = gkSummaryAlgorithm({ epsilon: 0.05, items })
    const start = viaJson(alg.serialise?.(alg.init(key(0))) ?? null)
    expect(alg.deserialise?.(start)?.min).toBeNaN()
    const whole = runAlgorithm(alg, { maxSteps: 2000 })
    const part = runAlgorithm(alg, { maxSteps: 333 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 2000,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 333, serialised: true },
    })
    expect(rest.state).toEqual(whole.state)
  })

  it('logs the size, merges, bound and quantiles', () => {
    const rec = recorder()
    runAlgorithm(gkSummaryAlgorithm({ epsilon: 0.05, items: items.slice(0, 200), quantiles: [0.5] }), {
      maxSteps: 300,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('sketch/size')).toHaveLength(200)
    expect(metric('sketch/error').at(-1)).toBeCloseTo(0.05 * 200, 12)
    expect(metric('sketch/compress').some((c) => c > 0)).toBe(true)
    expect(Object.keys(quantileSketchContract)).toHaveLength(4)
    expect(() => gkSummaryAlgorithm({ epsilon: 0, items })).toThrow(/epsilon/)
  })
})

describe('kllSketchAlgorithm', () => {
  it('compacts as the stream grows and equals kllSketch with the same key', () => {
    const run = runAlgorithm(kllSketchAlgorithm({ k: 32, key: key(4), items }), { maxSteps: 2000 })
    expect(run.state.count).toBe(1000)
    expect(run.state.compactions).toBeGreaterThan(0)
    const direct = kllSketch({ k: 32, key: key(4) })
    for (const v of items) direct.add(v)
    expect(run.state.levels).toEqual(direct.levels())
    const stored = run.state.levels.reduce((s, l) => s + l.length, 0)
    expect(stored).toBeLessThan(200)
    // Total weight is preserved: sum of 2^h over stored values equals the count.
    expect(run.state.levels.reduce((s, l, h) => s + l.length * 2 ** h, 0)).toBe(1000)
  })

  it('resumes the coin stream exactly from a JSON state', () => {
    const alg = kllSketchAlgorithm({ k: 16, key: key(9), items })
    const whole = runAlgorithm(alg, { maxSteps: 2000 })
    for (const cut of [100, 517]) {
      const part = runAlgorithm(alg, { maxSteps: cut }).state
      const rest = runAlgorithm(alg, {
        maxSteps: 2000,
        start: { state: viaJson(alg.serialise?.(part) ?? null), step: cut, serialised: true },
      })
      expect(rest.state.levels).toEqual(whole.state.levels)
      expect(rest.state.coins).toBe(whole.state.coins)
    }
    expect(() => kllSketchAlgorithm({ k: 4, key: key(0), items })).toThrow(/at least 8/)
  })
})

describe('p2Algorithm', () => {
  it('tracks the median of a stream with five markers', () => {
    const run = runAlgorithm(p2Algorithm({ p: 0.5, items }), { maxSteps: 2000 })
    expect(run.state.n).toBe(1000)
    expect(run.state.heights).toHaveLength(5)
    expect(Math.abs((run.state.heights[2] ?? 0) - 500)).toBeLessThan(25)
    const alg = p2Algorithm({ p: 0.5, items })
    const part = runAlgorithm(alg, { maxSteps: 40 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 2000,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 40, serialised: true },
    })
    expect(rest.state).toEqual(run.state)
    expect(() => p2Algorithm({ p: 1, items })).toThrow(/p must/)
  })
})

describe('quantileSketches', () => {
  it('names the sketches, builds a stepped algorithm for each and reports their properties', () => {
    expect(quantileSketches.names).toEqual(['greenwald-khanna', 'kll', 'p2'])
    for (const kind of quantileSketches.names) {
      const run = runAlgorithm(quantileSketches.make({ kind })({ items: items.slice(0, 300), key: key(1) }), {
        maxSteps: 400,
      })
      expect(run.status).toBe('completed')
    }
    expect(quantileSketches.properties({ kind: 'kll' }).randomised).toBe(true)
    expect(quantileSketches.where((p) => p.guaranteed)).toEqual(['greenwald-khanna'])
  })
})
