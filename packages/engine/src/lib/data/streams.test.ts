import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import { type DataStream, type StreamOptions, stream, streamAlgorithm } from './streams.ts'

/** The points of steps [from, to) as rows: features, target, the concept each came from, and the step. */
function rows(s: DataStream, from: number, to: number): { x: number[]; y: number; concept: number; t: number }[] {
  const out: { x: number[]; y: number; concept: number; t: number }[] = []
  for (let t = from; t < to; t++) {
    const b = s.next(t)
    const x = toFlat(b.x)
    const y = toFlat(b.y)
    for (let i = 0; i < y.length; i++)
      out.push({
        x: x.slice(i * s.truth.dim, (i + 1) * s.truth.dim),
        y: y[i] as number,
        concept: b.concept[i] as number,
        t,
      })
  }
  return out
}

const share = (xs: readonly boolean[]): number => xs.filter(Boolean).length / xs.length

describe('streams: abrupt drift in a classification stream', () => {
  const s = stream(key(3), { drift: 'abrupt', points: [100, 200], length: 300, batch: 5, dim: 4, classes: 3 })

  it('plants the drift where it says, in the truth', () => {
    expect(s.truth.changepoints).toEqual([100, 200])
    expect(s.truth.drifts.map((d) => [d.point, d.start, d.end, d.from, d.to, d.recurring])).toEqual([
      [100, 100, 100, 0, 1, false],
      [200, 200, 200, 1, 2, false],
    ])
    expect([0, 99, 100, 199, 200, 299].map((t) => s.truth.conceptAt(t))).toEqual([0, 0, 1, 1, 2, 2])
    expect(s.truth.transitionAt(150)).toEqual({ from: 1, to: 1, mix: 0 })
  })

  it('draws the data from the concept in force: its rule holds exactly before and after, and the old rule fails after', () => {
    for (const [from, to, concept] of [
      [0, 100, 0],
      [100, 200, 1],
      [200, 300, 2],
    ] as const) {
      const r = rows(s, from, to)
      expect(share(r.map((p) => p.y === s.truth.predict(concept, p.x))), `concept ${concept}`).toBe(1)
      expect(r.every((p) => p.concept === concept)).toBe(true)
    }
    const after = rows(s, 100, 200)
    expect(share(after.map((p) => p.y === s.truth.predict(0, p.x)))).toBeLessThan(0.8)
    // Where the stream is read for a change: the error of the first concept's rule is nil before, large after.
    const err = (from: number, to: number): number =>
      1 - share(rows(s, from, to).map((p) => p.y === s.truth.predict(0, p.x)))
    expect(err(0, 100)).toBe(0)
    expect(err(100, 120)).toBeGreaterThan(0.15)
    const first = Array.from({ length: 300 }, (_, t) => t).find((t) => err(t, t + 1) > 0 && t >= 90)
    expect(first).toBeGreaterThanOrEqual(100)
    expect(first).toBeLessThan(105)
  })

  it('has the shape of its batches', () => {
    const b = s.next(7)
    expect(b.x.shape).toEqual([5, 4])
    expect(b.y.shape).toEqual([5])
    expect(b.y.dtype).toBe('int32')
    expect(toFlat(b.y).every((c) => c >= 0 && c < 3)).toBe(true)
  })
})

describe('streams: what shifts', () => {
  it('covariate shift moves the inputs and leaves the rule alone', () => {
    const s = stream(key(4), { shift: 'covariate', drift: 'abrupt', points: [500], length: 1000, dim: 3, batch: 2 })
    const mean = (r: ReturnType<typeof rows>, j: number): number =>
      r.reduce((a, p) => a + (p.x[j] as number), 0) / r.length
    const before = rows(s, 0, 500)
    const after = rows(s, 500, 1000)
    const target = s.truth.parameters(1).mean
    for (let j = 0; j < 3; j++) {
      expect(mean(before, j)).toBeCloseTo(0, 0)
      expect(Math.abs(mean(after, j) - (target[j] as number))).toBeLessThan(0.15)
    }
    expect(Math.hypot(...target)).toBeCloseTo(3, 9)
    // The same rule in both concepts, and the data follow it throughout.
    expect(after.every((p) => p.y === s.truth.predict(0, p.x) && p.y === s.truth.predict(1, p.x))).toBe(true)
  })

  it('label shift moves the class frequencies and leaves what a class looks like alone', () => {
    const s = stream(key(5), { shift: 'label', drift: 'abrupt', points: [1000], length: 2000, classes: 3, batch: 2 })
    const freq = (r: ReturnType<typeof rows>): number[] => [0, 1, 2].map((c) => share(r.map((p) => p.y === c)))
    const before = freq(rows(s, 0, 1000))
    const after = freq(rows(s, 1000, 2000))
    const priors = s.truth.parameters(1).priors
    for (let c = 0; c < 3; c++) {
      expect(before[c]).toBeGreaterThan(0.28)
      expect(before[c]).toBeLessThan(0.39)
      expect(Math.abs((after[c] as number) - (priors[c] as number))).toBeLessThan(0.04)
    }
    expect(Math.max(...after)).toBeGreaterThan(0.8)
  })

  it('shifts a regression stream by concept, by covariate and by target', () => {
    const concept = stream(key(6), {
      task: 'regression',
      drift: 'abrupt',
      points: [50],
      length: 100,
      dim: 3,
      batch: 10,
    })
    const clean = (s: DataStream, from: number, to: number, c: number): boolean =>
      rows(s, from, to).every((p) => Math.abs(p.y - s.truth.predict(c, p.x)) < 1e-12)
    expect(clean(concept, 0, 50, 0)).toBe(true)
    expect(clean(concept, 50, 100, 1)).toBe(true)
    expect(clean(concept, 50, 100, 0)).toBe(false)
    expect(concept.next(0).y.dtype).toBe('float64')

    const covariate = stream(key(7), {
      task: 'regression',
      shift: 'covariate',
      drift: 'abrupt',
      points: [500],
      length: 1000,
      batch: 4,
    })
    const xMean = (r: ReturnType<typeof rows>): number => r.reduce((a, p) => a + (p.x[0] as number), 0) / r.length
    expect(
      Math.abs(xMean(rows(covariate, 500, 1000)) - (covariate.truth.parameters(1).mean[0] as number)),
    ).toBeLessThan(0.15)

    const label = stream(key(8), {
      task: 'regression',
      shift: 'label',
      drift: 'abrupt',
      points: [500],
      length: 1000,
      batch: 4,
    })
    const yMean = (r: ReturnType<typeof rows>): number => r.reduce((a, p) => a + p.y, 0) / r.length
    expect(yMean(rows(label, 0, 500))).toBeCloseTo(0, 0)
    expect(Math.abs(yMean(rows(label, 500, 1000)) - (label.truth.parameters(1).priors[0] as number))).toBeLessThan(0.15)
    expect(Math.abs(label.truth.parameters(1).priors[0] as number)).toBeCloseTo(3, 9)
  })

  it('adds label noise in a classification stream and noise on a regression target', () => {
    const noisy = stream(key(9), { drift: 'none', length: 400, batch: 5, noise: 0.3 })
    const wrong = 1 - share(rows(noisy, 0, 400).map((p) => p.y === noisy.truth.predict(0, p.x)))
    expect(wrong).toBeGreaterThan(0.08)
    expect(wrong).toBeLessThan(0.22)
    const regression = stream(key(9), { task: 'regression', drift: 'none', length: 400, batch: 5, noise: 0.5 })
    const r = rows(regression, 0, 400).map((p) => p.y - regression.truth.predict(0, p.x))
    expect(Math.sqrt(r.reduce((a, v) => a + v * v, 0) / r.length)).toBeCloseTo(0.5, 1)
  })
})

describe('streams: how the drift happens', () => {
  it('abrupt drift is a step: no point in the window belongs to the other concept', () => {
    const s = stream(key(10), { drift: 'abrupt', points: [500], length: 1000, batch: 4 })
    expect(rows(s, 0, 500).every((p) => p.concept === 0)).toBe(true)
    expect(rows(s, 500, 1000).every((p) => p.concept === 1)).toBe(true)
  })

  it('gradual drift mixes the two concepts, point by point, in a window around the drift point', () => {
    const s = stream(key(11), { drift: 'gradual', points: [500], width: 100, length: 1000, batch: 10 })
    const d = s.truth.drifts[0]
    expect([d?.start, d?.point, d?.end]).toEqual([450, 500, 550])
    const from1 = (a: number, b: number): number => share(rows(s, a, b).map((p) => p.concept === 1))
    expect(from1(0, 450)).toBe(0)
    expect(from1(551, 1000)).toBe(1)
    expect(Math.abs(from1(490, 510) - 0.5)).toBeLessThan(0.1)
    // The share of the new concept rises through the window.
    const ramp = [451, 475, 500, 525, 549].map((t) => from1(t, t + 1))
    expect(from1(451, 470)).toBeLessThan(from1(480, 500))
    expect(from1(480, 500)).toBeLessThan(from1(500, 520))
    expect(from1(500, 520)).toBeLessThan(from1(530, 549))
    expect(ramp.length).toBe(5)
    // Within the window the rule of each point is its own concept's.
    expect(rows(s, 450, 550).every((p) => p.y === s.truth.predict(p.concept, p.x))).toBe(true)
    expect(s.truth.transitionAt(500).mix).toBeCloseTo(0.5, 12)
    expect(s.truth.conceptAt(499)).toBe(0)
    expect(s.truth.conceptAt(500)).toBe(1)
  })

  it('incremental drift moves the concept itself: the target is the blend of the two concepts’ at the step’s mix', () => {
    const s = stream(key(12), {
      task: 'regression',
      drift: 'incremental',
      points: [500],
      width: 200,
      length: 1000,
      batch: 3,
    })
    for (const t of [0, 399, 400, 450, 500, 550, 599, 600, 999]) {
      const { from, to, mix } = s.truth.transitionAt(t)
      for (const p of rows(s, t, t + 1)) {
        const blended = (1 - mix) * s.truth.predict(from, p.x) + mix * s.truth.predict(to, p.x)
        expect(p.y, `step ${t}`).toBeCloseTo(blended, 10)
      }
    }
    expect(s.truth.transitionAt(450).mix).toBeCloseTo(0.25, 12)
    expect(s.truth.transitionAt(600).mix).toBe(0)
  })

  it('recurring drift returns to the concepts it has been in, on its period', () => {
    const s = stream(key(13), { drift: 'recurring', period: 100, length: 450 })
    expect(s.truth.changepoints).toEqual([100, 200, 300, 400])
    expect([0, 100, 200, 300, 400].map((t) => s.truth.conceptAt(t))).toEqual([0, 1, 0, 1, 0])
    expect(s.truth.drifts.map((d) => d.recurring)).toEqual([false, true, true, true])
    const three = stream(key(13), { drift: 'recurring', period: 100, length: 450, concepts: 3 })
    expect([0, 100, 200, 300, 400].map((t) => three.truth.conceptAt(t))).toEqual([0, 1, 2, 0, 1])
    // The concept that returns is the same one: the same rule on the same inputs.
    const a = rows(s, 0, 100)
    expect(a.every((p) => p.y === s.truth.predict(0, p.x))).toBe(true)
    expect(rows(s, 200, 300).every((p) => p.y === s.truth.predict(0, p.x))).toBe(true)
  })

  it('spaces the drifts evenly by default and plants none for no drift', () => {
    expect(stream(key(0), { drift: 'abrupt', count: 3, length: 1000 }).truth.changepoints).toEqual([250, 500, 750])
    const none = stream(key(0), { drift: 'none', length: 100 })
    expect(none.truth.changepoints).toEqual([])
    expect(none.truth.concepts).toBe(1)
    expect(rows(none, 0, 100).every((p) => p.concept === 0)).toBe(true)
  })

  it('at severity 0 a new concept is the old one', () => {
    const s = stream(key(14), { drift: 'abrupt', points: [10], length: 20, severity: 0, dim: 3 })
    expect(s.truth.parameters(1)).toEqual(s.truth.parameters(0))
  })
})

describe('streams: determinism and random access', () => {
  const options: StreamOptions = { drift: 'gradual', points: [60], width: 40, length: 120, batch: 3, dim: 3 }

  it('gives the same stream for the same key, whatever order the steps are asked in', () => {
    const a = stream(key(1), options)
    const b = stream(key(1), options)
    const forward = [0, 10, 59, 60, 119].map((t) => toFlat(a.next(t).x))
    const backward = [119, 60, 59, 10, 0].map((t) => toFlat(b.next(t).x)).reverse()
    expect(backward).toEqual(forward)
    expect(a.next(60).concept).toEqual(b.next(60).concept)
    expect(toFlat(a.next(61).y)).toEqual(toFlat(a.next(61).y))
    expect(a.truth.parameters(1)).toEqual(b.truth.parameters(1))
  })

  it('gives another stream for another key', () => {
    const a = stream(key(1), options)
    const c = stream(key(2), options)
    expect(toFlat(a.next(0).x)).not.toEqual(toFlat(c.next(0).x))
    expect(a.truth.parameters(1)).not.toEqual(c.truth.parameters(1))
    expect(a.truth.changepoints).toEqual(c.truth.changepoints)
  })

  it('refuses nonsense', () => {
    expect(() => stream(key(0), { length: 0 })).toThrow(/length/)
    expect(() => stream(key(0), { points: [0], length: 10 })).toThrow(/outside/)
    expect(() => stream(key(0), { points: [10], length: 10 })).toThrow(/outside/)
    expect(() => stream(key(0), { drift: 'gradual', points: [50, 60], width: 30, length: 100 })).toThrow(
      /closer than a transition/,
    )
    expect(() => stream(key(0), { classes: 1 })).toThrow(/two classes/)
    expect(() => stream(key(0), { severity: 2 })).toThrow(/severity/)
    expect(() => stream(key(0), {}).next(1000)).toThrow(/outside/)
  })
})

describe('streams as an algorithm', () => {
  const options: StreamOptions = { drift: 'abrupt', points: [12], length: 30, batch: 2 }
  const alg = streamAlgorithm(options)

  it('delivers one step per step, with the truth logged beside the data', () => {
    const rec = recorder()
    const { state, steps, status } = runAlgorithm(alg, { seed: 4, maxSteps: 100, tracer: rec })
    expect([steps, status, state.t]).toEqual([30, 'completed', 30])
    const run = rec.run()
    const concept = metricSeries(run, 'stream/concept').values
    const truth = stream(state.key, options).truth
    expect(concept).toEqual(Array.from({ length: 30 }, (_, t) => truth.conceptAt(t)))
    const drift = metricSeries(run, 'stream/drift')
    expect(drift.steps.filter((_, i) => drift.values[i] === 1)).toEqual([12])
    expect(toFlat(state.batch?.x as never)).toEqual(toFlat(stream(state.key, options).next(29).x))
  })

  it('draws the stream of its run’s key: the same seed the same points, another seed others', () => {
    const last = (seed: number): number[] => toFlat(runAlgorithm(alg, { seed, maxSteps: 30 }).state.batch?.y as never)
    expect(last(1)).toEqual(last(1))
    const a = runAlgorithm(alg, { seed: 1, maxSteps: 5 }).state.batch
    const b = runAlgorithm(alg, { seed: 2, maxSteps: 5 }).state.batch
    expect(toFlat(a?.x as never)).not.toEqual(toFlat(b?.x as never))
  })

  it('serialises, so a run resumes from a checkpoint exactly', () => {
    const straight = runAlgorithm(alg, { seed: 6, maxSteps: 30 })
    const half = runAlgorithm(alg, { seed: 6, maxSteps: 14 })
    const json = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
    const resumed = runAlgorithm(alg, { seed: 6, maxSteps: 16, start: { state: json, step: 14, serialised: true } })
    expect(resumed.state.t).toBe(30)
    expect(toFlat(resumed.state.batch?.x as never)).toEqual(toFlat(straight.state.batch?.x as never))
    expect(resumed.state.batch?.concept).toEqual(straight.state.batch?.concept)
  })
})
