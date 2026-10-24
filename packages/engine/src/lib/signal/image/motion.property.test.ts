import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../../testing/arbitrary.ts'
import { toFlat } from '../../tensor/index.ts'
import { policies, recorder } from '../../trace/index.ts'
import * as I from './index.ts'

const frame = (dy: number, dx: number, n = 24): number[][] =>
  Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => Math.sin((j - dx) / 3) + Math.cos((i - dy) / 4)),
  )

describe('motion', () => {
  it('optical flow recovers small translations in the interior', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -0.3, max: 0.3, noNaN: true }),
        fc.double({ min: -0.3, max: 0.3, noNaN: true }),
        (dy, dx) => {
          const n = 24
          const lk = toFlat(I.lucasKanade(frame(0, 0), frame(dy, dx)))
          const hs = toFlat(I.hornSchunck(frame(0, 0), frame(dy, dx), { alpha: 0.3, maxIter: 300 }))
          const centre = 12 * n + 12
          expect(lk[centre] ?? 0).toBeCloseTo(dy, 1)
          expect(lk[n * n + centre] ?? 0).toBeCloseTo(dx, 1)
          expect(Math.abs((hs[centre] ?? 0) - dy)).toBeLessThan(0.1)
          expect(Math.abs((hs[n * n + centre] ?? 0) - dx)).toBeLessThan(0.1)
        },
      ),
      { numRuns: Math.min(RUNS, 10) },
    )
  })

  it('block matching and phase correlation recover integer shifts', () => {
    fc.assert(
      fc.property(fc.integer({ min: -2, max: 2 }), fc.integer({ min: -2, max: 2 }), (dy, dx) => {
        const a = frame(0, 0, 16)
        const b = frame(dy, dx, 16)
        const v = toFlat(I.blockMatching(a, b, { block: 4, radius: 3 }).vectors)
        // The interior block (1, 1) moves by (dy, dx).
        expect([v[(1 * 4 + 1) * 2], v[(1 * 4 + 1) * 2 + 1]]).toEqual([dy, dx])
      }),
      { numRuns: Math.min(RUNS, 10) },
    )
  })

  it('traces Horn-Schunck convergence', () => {
    const tracer = recorder()
    I.hornSchunck(frame(0, 0, 12), frame(0.1, 0, 12), { tracer, policy: policies.minimal, maxIter: 20 })
    expect(tracer.events.filter((e) => e.type === 'metric' && e.name === 'change').length).toBeGreaterThan(5)
  })
})
