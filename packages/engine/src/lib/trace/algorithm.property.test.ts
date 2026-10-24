import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { normal } from '../random/index.ts'
import { type Algorithm, runAlgorithm, runFor, type Slice } from './algorithm.ts'
import { tracePolicy } from './policy.ts'
import { recorder, startRun } from './tracer.ts'

const noisy: Algorithm<number> = {
  name: 'noisy',
  config: {},
  init: (k) => normal(k, [1]).data[0] ?? 0,
  step: (x, ctx) => 0.9 * x + (normal(ctx.key, [1]).data[0] ?? 0),
  done: (x) => (Math.abs(x) > 6 ? 'diverged' : undefined),
  observe: (x) => ({ metrics: { x } }),
}

describe('runFor', () => {
  it('any sequence of slices computes and records what one uninterrupted run does', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 50 }),
        fc.integer({ min: 1, max: 40 }),
        fc.array(fc.integer({ min: 1, max: 7 }), { minLength: 1, maxLength: 60 }),
        fc.integer({ min: 1, max: 4 }),
        (seed, maxSteps, sizes, every) => {
          const policy = tracePolicy({ schedule: { every } })
          const whole = recorder()
          const full = runAlgorithm(noisy, { seed, maxSteps, tracer: whole, policy })
          const sliced = recorder()
          const run = startRun(sliced, { name: noisy.name, seed, config: noisy.config, policy })
          let slice: Slice<number> | undefined
          let i = 0
          while (slice?.status === undefined) {
            const steps = sizes[i % sizes.length] ?? 1
            slice = runFor(noisy, { steps }, { seed, maxSteps, run, ...(slice === undefined ? {} : { from: slice }) })
            i += 1
          }
          expect(slice.state).toBe(full.state)
          expect(slice.t).toBe(full.steps)
          expect(slice.status).toBe(full.status)
          const metrics = (r: ReturnType<typeof recorder>): unknown[] => r.events.filter((e) => e.type === 'metric')
          expect(metrics(sliced)).toEqual(metrics(whole))
        },
      ),
    )
  })
})
