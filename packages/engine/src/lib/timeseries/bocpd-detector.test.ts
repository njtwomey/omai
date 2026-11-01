import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { changepointSeries, conjugateModelOf, type SegmentModel } from '../datasets/index.ts'
import {
  bocpd,
  hazards,
  hazardValue,
  mapChangepoints,
  segmentModels,
  segmentModelValue,
  segmentModelValueOf,
} from '../dynamics/filtering/index.ts'
import { recordsOf } from '../exposition/index.ts'
import { key } from '../random/index.ts'
import { contains, defaults, type SpaceValue } from '../space/index.ts'
import { metricSeries } from '../trace/index.ts'
import { bocpdDetector, changepointData } from './bocpd-detector.ts'

const series = [0, 0.1, -0.1, 0, 0.1, 5, 5.2, 4.9, 5.1, 5, 5.1, 0.1, 0, -0.1, 0.2]
const mean = { kind: 'mean', sigma: 0.5, priorMean: 0, priorStd: 5 } as const
const detector = bocpdDetector.with({ model: mean, hazard: { kind: 'constant', rate: 0.1 } })

describe('bocpdDetector', () => {
  it('fits a plain series: the same numbers as bocpd, and the changepoints', () => {
    const fitted = detector.fit(changepointData(series))
    const direct = bocpd(series, { kind: 'normal-known-variance', sigma: 0.5, priorMean: 0, priorStd: 5 }, 0.1)
    expect(fitted.params.logEvidence).toBe(direct.logEvidence)
    expect(fitted.params.changeProbability).toEqual(direct.changeProbability)
    expect(fitted.params.mapRunLength).toEqual(direct.mapRunLength)
    expect(fitted.params.steps).toBe(series.length)
    expect(fitted.changepoints()).toEqual([5, 11])
    expect(fitted.detections(0.5).every((d) => d.detectedAt >= d.index)).toBe(true)
    // A stricter threshold detects no more than a looser one.
    expect(fitted.changepoints(0.99).length).toBeLessThanOrEqual(fitted.changepoints(0.2).length)
  })

  it('exposes the run-length posterior as a strip, with the true changepoints when the data has them', () => {
    const plain = detector.fit(changepointData(series))
    const [strip] = recordsOf(plain.exposition, 'strip')
    expect(strip?.tokens).toHaveLength(series.length)
    expect(strip?.marks).toBeUndefined()
    const known = detector.fit(changepointData(series, [5, 11]))
    expect(recordsOf(known.exposition, 'strip')[0]?.marks).toEqual([5, 11])
    // The path is the MAP run length (as a track index); a changepoint drops it to the first track.
    expect(known.params.mapRunLength[5]).toBe(1)
    expect(recordsOf(known.exposition, 'strip')[0]?.path?.[5]).toBe(0)
  })

  it('takes a generated time series and reads the truth from its segments', () => {
    const { data, truth } = generate(changepointSeries, {
      key: key(11),
      knobs: { n: 240, kind: 'mean', hazard: 0.02, noise: 0.5, shift: 4 },
    })
    const prepared = changepointData(data)
    expect(prepared.changepoints).toEqual(truth.changepoints)
    const fitted = bocpdDetector
      .with({
        model: { kind: 'mean', sigma: 0.5, priorMean: 0, priorStd: 4 },
        hazard: { kind: 'constant', rate: 0.02 },
      })
      .fit(prepared)
    const found = fitted.changepoints(0.5)
    const truthIdx = truth.changepoints ?? []
    const hit = truthIdx.filter((c) => found.some((f) => Math.abs(f - c) <= 3)).length
    expect(hit / Math.max(1, truthIdx.length)).toBeGreaterThanOrEqual(0.7)
    expect(recordsOf(fitted.exposition, 'strip')[0]?.marks).toEqual(truthIdx)
  })

  it('runs every kind of shift on data of its kind', () => {
    for (const kind of ['mean', 'variance', 'rate', 'ar', 'trend'] as const) {
      const { data } = generate(changepointSeries, { key: key(5), knobs: { n: 150, kind, hazard: 0.03 } })
      const fitted = bocpdDetector
        .with({ model: segmentModelValue(kind), hazard: hazardValue('constant', { rate: 0.03 }) })
        .fit(changepointData(data))
      expect(Number.isFinite(fitted.params.logEvidence), kind).toBe(true)
      expect(fitted.params.steps).toBe(150)
    }
    // Bits and the normal-gamma model.
    const bits = series.map((v) => (v > 2.5 ? 1 : 0))
    expect(
      bocpdDetector
        .with({ model: segmentModelValue('bernoulli'), hazard: { kind: 'constant', rate: 0.1 } })
        .fit(changepointData(bits))
        .changepoints()[0],
    ).toBe(5)
    expect(
      bocpdDetector.with({ model: segmentModelValue('mean-variance') }).fit(changepointData(series)).params.logEvidence,
    ).toBeLessThan(0)
  })

  it('the model that generated the data explains it better than another kind of shift', () => {
    const { data, truth } = generate(changepointSeries, { key: key(2), knobs: { n: 200, kind: 'mean', hazard: 0.03 } })
    const matched = segmentModelValueOf(conjugateModelOf(truth.extra?.['model'] as SegmentModel))
    const hazard = hazardValue('constant', { rate: 0.03 })
    const evidence = (model: SpaceValue): number =>
      bocpdDetector.with({ model, hazard }).fit(changepointData(data)).params.logEvidence
    expect(segmentModels.make(matched).kind).toBe('normal-known-variance')
    expect(evidence(matched)).toBeGreaterThan(evidence(segmentModelValue('variance')))
    expect(evidence(matched)).toBeGreaterThan(evidence(segmentModelValue('trend')))
  })

  it('refuses a series its model cannot score', () => {
    const counts = bocpdDetector.with({ model: segmentModelValue('rate') })
    expect(() => counts.fit(changepointData([1, 2, -1]))).toThrow(/counts/)
    expect(() => counts.fit(changepointData([1, 2.5]))).toThrow(/counts/)
    expect(() => bocpdDetector.with({ model: segmentModelValue('bernoulli') }).fit(changepointData([0, 2]))).toThrow(
      /bits/,
    )
    expect(() => detector.fit(changepointData([0, Number.NaN]))).toThrow(/finite/)
  })

  it('has a hyperparameter space with the model and the hazard as unions', () => {
    const space = detector.hyperSpace
    expect(space?.kind).toBe('dict')
    const value = defaults(space as NonNullable<typeof space>)
    expect(contains(space as NonNullable<typeof space>, value)).toBe(true)
    expect((value as Record<string, unknown>)['model']).toEqual(defaults(segmentModels.space))
    expect((value as Record<string, unknown>)['hazard']).toEqual(defaults(hazards.space))
  })

  it('runs live as a session: stepping, the fitted value at any step, and the paths of the contract', () => {
    const data = changepointData(series, [5, 11])
    const s = detector.session(data, { mode: 'live' })
    expect(s.t).toBe(0)
    s.advance({ steps: 8 })
    expect(s.t).toBe(8)
    const at8 = s.fittedAt(8)
    expect(at8.params.steps).toBe(8)
    expect(at8.changepoints()).toEqual([5])
    s.advance()
    expect(s.t).toBe(series.length)
    expect(s.fittedAt(s.t).changepoints()).toEqual([5, 11])
    expect(s.fittedAt(8).params.steps).toBe(8)
    // The default subscription logs the change probability each step, and the run-length rows in full.
    const run = s.run()
    expect(metricSeries(run, 'bocpd/change-probability').values).toHaveLength(series.length)
    expect(metricSeries(run, 'bocpd/map-run-length').values.at(5)).toBe(1)
    const rows = run.events.flatMap((e) => (e.type === 'tensor' && e.name === 'bocpd/run-length' ? [e] : []))
    expect(rows).toHaveLength(series.length)
    expect(rows[3]?.values).toHaveLength(5)
    // Derived paths: the strip so far and the detections so far.
    const strip = s.probe(8, 'model/run-length') as { kind: string; tokens: string[]; marks: number[] }
    expect(strip.kind).toBe('strip')
    expect(strip.tokens).toHaveLength(8)
    expect(strip.marks).toEqual([5])
    expect(s.probe(8, 'model/changepoints')).toEqual(mapChangepoints(at8.params, 0.5))
  })

  it('a session changed to another model and hazard in its configuration steps with those', () => {
    const data = changepointData(series)
    const s = bocpdDetector.session(data, {
      mode: 'eager',
      logs: 'off',
    })
    const fixed = bocpdDetector.session(data, { mode: 'eager', logs: 'off' }).fittedAt(series.length)
    expect(s.fittedAt(series.length).params.logEvidence).toBe(fixed.params.logEvidence)
    const other = bocpdDetector
      .with({ hazard: hazardValue('minimum-segment', { rate: 0.1, minSegment: 3 }), model: mean })
      .session(data, { mode: 'eager', logs: 'off' })
      .fittedAt(series.length)
    expect(other.params.logEvidence).not.toBe(fixed.params.logEvidence)
    expect(other.params.mapRunLength).toHaveLength(series.length)
  })
})
