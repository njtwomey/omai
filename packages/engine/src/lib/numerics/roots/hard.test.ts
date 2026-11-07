import { describe, expect, it } from 'vitest'
import * as R from './index.ts'

// Hard functions for the scalar root finders: a triple root, a step, a steep tanh, Wilkinson's polynomial, tiny values.
// The expected [root, iterations, converged] of bisect, brentq and ridder are scipy.optimize's (1.18, same algorithms,
// the same last digits); toms748 is a different implementation of the same method, so it is held to the true root.
type Scalar = (x: number) => number
const triple: Scalar = (x) => {
  const d = x - 1
  return d * d * d
}
const wilkinson: Scalar = (x) => {
  let p = 1
  for (let k = 1; k <= 20; k++) p *= x - k
  return p
}
const functions: Readonly<Record<string, readonly [Scalar, number, number, number]>> = {
  triple: [triple, 0, 3, 1],
  cubic: [(x) => x * x * x - x - 1, 1, 2, 1.324717957244746],
  wilkinson: [wilkinson, 19.5, 21, 20],
  step: [(x) => (x < 0.3 ? -1 : 1), 0, 1, 0.3],
  steep: [(x) => Math.tanh(1e3 * (x - 0.3)), -1, 1, 0.3],
  tiny: [(x) => (x - 0.5) * (x * x + 1) * 1e-10, 0, 2, 0.5],
}

// [function, finder, root, iterations, converged] from scipy.optimize.
const scipyRows: readonly (readonly [string, 'bisect' | 'brentq' | 'ridder' | 'toms748', number, number, boolean])[] = [
  ['triple', 'bisect', 1.0000000000004547, 41, true],
  ['triple', 'brentq', 1.0000000002456189, 100, false],
  ['triple', 'ridder', 0.9999999999995511, 34, true],
  ['triple', 'toms748', 1.0000000000001854, 31, true],
  ['cubic', 'bisect', 1.324717957244502, 39, true],
  ['cubic', 'brentq', 1.324717957244746, 9, true],
  ['cubic', 'ridder', 1.3247179572457464, 6, true],
  ['cubic', 'toms748', 1.3247179572437455, 5, true],
  ['wilkinson', 'bisect', 19.999999999999545, 40, true],
  ['wilkinson', 'brentq', 20.0, 12, true],
  ['wilkinson', 'ridder', 20.000000000000746, 6, true],
  ['wilkinson', 'toms748', 20.0, 6, true],
  ['step', 'bisect', 0.3000000000010914, 39, true],
  ['step', 'brentq', 0.3000000000010914, 40, true],
  ['step', 'ridder', 0.30000000000002985, 22, true],
  ['step', 'toms748', 0.2999999999996076, 20, true],
  ['steep', 'bisect', 0.3000000000010914, 40, true],
  ['steep', 'brentq', 0.30000000000000937, 15, true],
  ['steep', 'ridder', 0.3000000000006571, 12, true],
  ['steep', 'toms748', 0.3, 9, true],
  ['tiny', 'bisect', 0.5, 2, true],
  ['tiny', 'brentq', 0.49999999999999994, 8, true],
  ['tiny', 'ridder', 0.5, 5, true],
  ['tiny', 'toms748', 0.5, 4, true],
]

describe('scalar root finders on hard functions', () => {
  it('bisect, brentq, ridder and toms748 find the root scipy finds (brentq runs out of iterations on the triple root, as scipy)', () => {
    const failures: string[] = []
    for (const [name, finder, root, iterations, converged] of scipyRows) {
      const [f, a, b, truth] = functions[name] as readonly [Scalar, number, number, number]
      const r = R[finder](f, a, b)
      const label = `${finder} on ${name}`
      if (r.converged !== converged) failures.push(`${label}: converged ${r.converged}, scipy ${converged}`)
      if (r.status !== (converged ? 'converged' : 'completed')) failures.push(`${label}: status ${r.status}`)
      if (finder === 'toms748') {
        // Same method, other implementation: within the default xtol (2e-12) of the true root.
        if (Math.abs(r.root - truth) > 1e-11) failures.push(`${label}: root ${r.root}, true ${truth}`)
      } else {
        if (Math.abs(r.root - root) > 1e-13 * Math.max(1, Math.abs(root)))
          failures.push(`${label}: root ${r.root}, scipy ${root}`)
        if (r.iterations !== iterations) failures.push(`${label}: ${r.iterations} iterations, scipy ${iterations}`)
      }
    }
    expect(failures).toEqual([])
  })

  it('end points that are roots, reversed brackets, a cube root and exp(x) = 1e100 are handled by all four', () => {
    for (const finder of ['bisect', 'brentq', 'ridder', 'toms748'] as const) {
      const at = R[finder]((x) => x * (x - 1), 0, 1)
      expect([finder, at.root, at.iterations, at.converged]).toEqual([finder, 0, 0, true])
      expect(R[finder]((x) => x - 0.5, 1, 0).root, `${finder} reversed`).toBeCloseTo(0.5, 11)
      // The root 0 of a function with an infinite slope there, found to the absolute tolerance.
      expect(Math.abs(R[finder](Math.cbrt, -1, 8).root), `${finder} cube root`).toBeLessThan(1e-11)
      // f(1000) is Infinity; ridder's interpolation cannot use it, the others do.
      if (finder !== 'ridder') {
        const r = R[finder]((x) => Math.exp(x) - 1e100, 0, 1000)
        expect(r.converged, `${finder} exp`).toBe(true)
        expect(r.root, `${finder} exp`).toBeCloseTo(100 * Math.LN10, 9)
      }
    }
  })

  it('refuse a bracket without a sign change and an infinite bracket, with a value error that names the finder', () => {
    for (const finder of ['bisect', 'brentq', 'ridder', 'toms748'] as const) {
      expect(() => R[finder]((x) => x * x + 1, -1, 1), `${finder} same signs`).toThrow(
        expect.objectContaining({
          code: 'value',
          message: expect.stringContaining(`${finder}: f(a) and f(b) must have different signs`),
        }),
      )
      for (const [a, b] of [
        [0, Number.POSITIVE_INFINITY],
        [Number.NEGATIVE_INFINITY, 1],
      ] as const) {
        expect(() => R[finder]((x) => x - 0.3, a, b), `${finder} on [${a}, ${b}]`).toThrow(
          expect.objectContaining({ code: 'value', message: `${finder}: the bracket must be finite` }),
        )
      }
    }
  })
})

describe('newton, secant and halley', () => {
  it('converge as scipy.optimize.newton does, and report a zero derivative, a divergence and the iteration limit', () => {
    // [rows of scipy.optimize.newton: root, iterations]: x^2 = 2 by newton from 1 (5), cos x = x by secant (5), x^3 = 2 by
    // halley (four evaluations), and (x - 1)^3 from 3, where convergence is only linear (35 iterations to the tolerance 1e-6).
    const newton = R.newton((x) => x * x - 2, 1, { fprime: (x) => 2 * x })
    expect([newton.root, newton.iterations, newton.status]).toEqual([1.4142135623730951, 5, 'converged'])
    const secant = R.newton((x) => Math.cos(x) - x, 1)
    expect([secant.root, secant.iterations]).toEqual([expect.closeTo(0.7390851332151608, 12), 5])
    const halley = R.halley(
      (x) => x * x * x - 2,
      1,
      (x) => 3 * x * x,
      (x) => 6 * x,
    )
    // The same four evaluations as scipy (scipy's `iterations` is 3: it does not count the step that finds f = 0).
    expect([halley.root, halley.functionCalls]).toEqual([expect.closeTo(1.2599210498948732, 12), 4])
    const multiple = R.newton((x) => (x - 1) ** 3, 3, { fprime: (x) => 3 * (x - 1) ** 2, tol: 1e-6 })
    expect([multiple.root, multiple.iterations]).toEqual([expect.closeTo(1.0000013735229765, 10), 35])
    expect(R.secant((x) => x * x * x - 2, 1).root).toBeCloseTo(2 ** (1 / 3), 8)
    // A zero derivative at the start (x^3 - 1 at 0), a step that leaves the double range (atan from 2), the limit.
    const flat = R.newton((x) => x * x * x - 1, 0, { fprime: (x) => 3 * x * x })
    expect([flat.flag, flat.converged, flat.status]).toEqual(['derivative was zero', false, 'failed'])
    const diverge = R.newton(Math.atan, 2, { fprime: (x) => 1 / (1 + x * x), maxIterations: 20 })
    expect([diverge.converged, Math.abs(diverge.root) > 1e100]).toEqual([false, true])
    const limited = R.newton((x) => x * x - 2, 1, { fprime: (x) => 2 * x, maxIterations: 2 })
    expect([limited.root, limited.flag, limited.status]).toEqual([
      1.4166666666666667,
      'maximum iterations',
      'completed',
    ])
  })
})
