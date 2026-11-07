import { describe, expect, it } from 'vitest'
import * as R from './index.ts'

const finders = { bisect: R.bisect, brentq: R.brentq, ridder: R.ridder, toms748: R.toms748 } as const
type Finder = keyof typeof finders

describe('the bracketing root finders at tiny scale and on NaN', () => {
  const scales = [1e-100, 1e-160, 1e-200, 1e-300, 1e300]
  it.each(scales.flatMap((s) => (Object.keys(finders) as Finder[]).map((name) => [name, s] as const)))(
    '%s finds the root of s (x - 0.3) for s = %s',
    (name, s) => {
      const r = finders[name]((x) => s * (x - 0.3), 0, 1)
      expect(r.converged).toBe(true)
      expect(Math.abs(r.root - 0.3)).toBeLessThan(1e-10)
    },
  )

  const nan = (): number => Number.NaN
  const hole = (x: number): number => (x > 0.2 && x < 0.4 ? Number.NaN : x - 0.3)
  it.each([
    ['f is NaN everywhere', nan, -1, 1],
    ['f is NaN at one end', (x: number) => (x < 0 ? Number.NaN : x), -1, 1],
    ['f is NaN inside the bracket', hole, 0, 1],
  ])('every finder throws a value error when %s', (_what, f, a, b) => {
    for (const name of Object.keys(finders) as Finder[]) {
      expect(() => finders[name](f, a, b), name).toThrow(expect.objectContaining({ code: 'value' }))
    }
  })
})
