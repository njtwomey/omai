import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { normal } from '../random/index.ts'
import type { Algorithm } from './algorithm.ts'
import { runAlgorithm } from './algorithm.ts'
import { type ChannelRef, onSchedule, type RecordSchedule, shouldRecord, tracePolicy } from './policy.ts'
import { metricSeries } from './query.ts'
import { recorder } from './tracer.ts'

const schedule: fc.Arbitrary<RecordSchedule> = fc.oneof(
  fc.integer({ min: 1, max: 12 }).map((every) => ({ every })),
  fc.constant({ logSpaced: true as const }),
  fc.constant({ firstAndLast: true as const }),
)

const walk: Algorithm<number> = {
  name: 'walk',
  config: {},
  init: () => 0,
  step: (x, ctx) => x + (normal(ctx.key, [1]).data[0] ?? 0),
  observe: (x) => ({ metrics: { x, ignored: -x } }),
}

describe('shouldRecord', () => {
  it('matches the schedule for selected channels and is false for unselected ones', () => {
    fc.assert(
      fc.property(schedule, fc.integer({ min: 0, max: 5000 }), (s, step) => {
        const policy = tracePolicy({ schedule: s, metrics: ['x'] })
        const x: ChannelRef = { kind: 'metric', name: 'x' }
        expect(shouldRecord(policy, x, step)).toBe(onSchedule(s, step))
        expect(shouldRecord(policy, x, step, { last: true })).toBe(true)
        expect(shouldRecord(policy, { kind: 'metric', name: 'other' }, step, { last: true })).toBe(false)
      }),
    )
  })

  it('every-n and log-spaced schedules pick the expected steps', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 20 }), fc.integer({ min: 0, max: 2000 }), (n, step) => {
        expect(shouldRecord(tracePolicy({ schedule: { every: n } }), { kind: 'metric', name: 'm' }, step)).toBe(
          step % n === 0,
        )
        const logSpaced = step === 0 || Number.isInteger(Math.log2(step))
        expect(shouldRecord(tracePolicy({ schedule: { logSpaced: true } }), { kind: 'metric', name: 'm' }, step)).toBe(
          logSpaced,
        )
      }),
    )
  })

  it('the runner records metrics exactly at the steps shouldRecord picks, plus the last', () => {
    fc.assert(
      fc.property(schedule, fc.integer({ min: 1, max: 40 }), fc.integer({ min: 0, max: 99 }), (s, maxSteps, seed) => {
        const policy = tracePolicy({ schedule: s, metrics: ['x'] })
        const rec = recorder()
        runAlgorithm(walk, { seed, maxSteps, tracer: rec, policy })
        const expected: number[] = []
        for (let t = 0; t <= maxSteps; t++) {
          if (shouldRecord(policy, { kind: 'metric', name: 'x' }, t, { last: t === maxSteps })) expected.push(t)
        }
        expect(metricSeries(rec.run(), 'x').steps).toEqual(expected)
        expect(metricSeries(rec.run(), 'ignored').steps).toEqual([])
      }),
    )
  })

  it('on-change schedules record a value only when it moved by more than the tolerance', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 3 }), { minLength: 1, maxLength: 30 }),
        fc.constantFrom(0, 1.5),
        (values, tolerance) => {
          const policy = tracePolicy({ schedule: { onChange: true, tolerance } })
          const kept: number[] = []
          let previous: number | undefined
          values.forEach((v, t) => {
            if (shouldRecord(policy, { kind: 'metric', name: 'v' }, t, { previous, current: v })) {
              kept.push(t)
              previous = v
            }
          })
          const rec = recorder()
          const replayed: Algorithm<number> = {
            name: 'values',
            config: {},
            init: () => 0,
            step: (t) => t + 1,
            observe: (t) => ({ metrics: { v: values[Math.min(t, values.length - 1)] ?? 0 } }),
          }
          runAlgorithm(replayed, { maxSteps: values.length - 1, tracer: rec, policy })
          expect(metricSeries(rec.run(), 'v').steps).toEqual(kept)
        },
      ),
    )
  })
})
