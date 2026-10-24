import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { tensor } from '../tensor/index.ts'
import { metricSeries, recorder } from '../trace/index.ts'
import * as O from './index.ts'

const FUNCTIONS: Readonly<Record<string, O.ScalarFunction>> = {
  shifted: (x) => (x - 2) ** 2 + 1,
  cosh1: (x) => Math.cosh(x - 1),
  quartic: (x) => x ** 4 - 3 * x ** 3 + 2,
  expline: (x) => Math.exp(x) - 2 * x,
  kink: (x) => Math.abs(x - 0.3) + 0.1 * x * x,
  wave: (x) => Math.sin(x) + 0.1 * x,
}

const METHODS: Readonly<Record<string, O.ScalarMethod>> = { brent: O.brent, golden: O.golden, bounded: O.fminbound }

describe('scalar minimisation against scipy.optimize', () => {
  for (const c of loadFixture('optim/scalar').cases) {
    it(c.name, () => {
      const f = FUNCTIONS[c.params['function'] as string] as O.ScalarFunction
      if (c.fn === 'bracket') {
        const b = O.bracket(f, { xa: c.params['xa'] as number, xb: c.params['xb'] as number })
        expectMatches(tensor([b.xa, b.xb, b.xc, b.fa, b.fb, b.fc, b.nfev]), c.output, c.rtol, c.atol, c.name)
        return
      }
      const br = c.params['bracket'] as [number, number] | [number, number, number] | undefined
      const bounds = c.params['bounds'] as [number, number] | undefined
      const tol = c.params['tol'] as number | undefined
      const r = O.minimizeScalar(f, {
        method: METHODS[c.fn] as O.ScalarMethod,
        ...(br === undefined ? {} : { bracket: br }),
        ...(bounds === undefined ? {} : { bounds }),
        ...(tol === undefined ? {} : { tol }),
      })
      expect(r.converged).toBe(true)
      // scipy's bounded method reports its evaluation count as nit.
      const nit = c.fn === 'bounded' ? r.nfev : r.iterations
      expectMatches(tensor([r.x, r.fun, nit, r.nfev]), c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('scalar minimisers', () => {
  it('trace the interval of uncertainty, shrinking to the tolerance', () => {
    const rec = recorder()
    O.brent((x) => (x - 1) ** 2, { tracer: rec })
    const width = metricSeries(rec.run(), 'width').values
    expect(width.length).toBeGreaterThan(3)
    expect((width.at(-1) ?? 1) < (width[0] ?? 0)).toBe(true)
  })
  it('reject invalid brackets and bounds', () => {
    expect(() => O.brent((x) => x * x, { bracket: [0, 1, 2] })).toThrow(/f\(xb\)/)
    expect(() => O.fminbound((x) => x)).toThrow(/bounds/)
    expect(() => O.bracket((x) => -x, { maxIterations: 5 })).toThrow()
  })
  it('report the iteration cap', () => {
    const r = O.golden((x) => (x - 2) ** 2, { maxIterations: 3 })
    expect(r.converged).toBe(false)
    expect(r.iterations).toBe(3)
  })
})
