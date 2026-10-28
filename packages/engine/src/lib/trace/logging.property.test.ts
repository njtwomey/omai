import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { Log, resolve, scalar, tensor } from './contract.ts'
import { gated, logContext, scheduler } from './logging.ts'
import { metricSeries, tensorSeries } from './query.ts'
import { recorder, startRun } from './tracer.ts'

const contract = { 'a/x': scalar(), 'b/t': tensor({ role: 'state' }) }

describe('keep', () => {
  it('holds at most n events of a path, the latest due ones, and leaves the other paths and the order alone', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 8 }),
        fc.integer({ min: 0, max: 40 }),
        fc.integer({ min: 1, max: 4 }),
        (n, steps, every) => {
          const r = resolve(contract, { 'a/x': Log({ keep: n, every }), 'b/t': Log() })
          const s = scheduler(r)
          const rec = recorder()
          const run = startRun(gated(rec, r, s), { name: 'x', seed: 0, config: {}, policy: r.policy })
          const due: number[] = []
          for (let t = 0; t < steps; t++) {
            const at = logContext(run, r, s, t, { stop: () => {} })
            at.log('a/x', t)
            at.log('b/t', [t])
            if (every <= 1 || t % every === 0) due.push(t)
          }
          const kept = metricSeries(rec.run(), 'a/x').steps
          expect(kept.length).toBeLessThanOrEqual(n)
          expect(kept).toEqual(due.slice(-n))
          expect(tensorSeries(rec.run(), 'b/t').length).toBe(steps)
          const order = rec.events.flatMap((e) => ('step' in e ? [e.step] : []))
          expect(order).toEqual([...order].sort((a, b) => a - b))
        },
      ),
    )
  })
})
