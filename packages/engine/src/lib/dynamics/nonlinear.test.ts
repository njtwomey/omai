import { describe, expect, it } from 'vitest'
import { add, mul, neg, sliceAxis, stack, sub, sum, toFlat, type Value } from '../tensor/index.ts'
import {
  continueEquilibria,
  correlationDimension,
  delayEmbedding,
  floquetMultipliers,
  limitCycle,
  lyapunovSpectrum,
  poincareSection,
  recurrenceQuantification,
} from './nonlinear.ts'

const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))
const lorenz = (_t: number, y: Value): Value => {
  const [x, v, z] = [at(y, 0), at(y, 1), at(y, 2)]
  return stack([mul(sub(v, x), 10), sub(mul(x, sub(28, z)), v), sub(mul(x, v), mul(z, 8 / 3))])
}
const hopf = (x: Value, p: Value): Value => {
  // The supercritical Hopf normal form in Cartesian coordinates.
  const [a, b] = [at(x, 0), at(x, 1)]
  const r2 = add(mul(a, a), mul(b, b))
  return stack([sub(sub(mul(p, a), b), mul(a, r2)), sub(add(a, mul(p, b)), mul(b, r2))])
}

describe('nonlinear dynamics of flows', () => {
  it('Lorenz: the spectrum sums to the divergence and the largest exponent is positive', () => {
    const r = lyapunovSpectrum(lorenz, [1, 1, 20], { tEnd: 20, interval: 0.5, transient: 5 })
    const [l1, l2, l3] = r.exponents
    expect((l1 ?? 0) + (l2 ?? 0) + (l3 ?? 0)).toBeCloseTo(-(10 + 1 + 8 / 3), 1)
    expect(l1).toBeGreaterThan(0.5)
    expect(Math.abs(l2 ?? 1)).toBeLessThan(0.15)
  })

  it('the Hopf normal form: a stable limit cycle of radius sqrt(p) with multipliers 1 and exp(-4 pi p)', () => {
    const p = 0.25
    const f = (_t: number, y: Value): Value => hopfAt(y, p)
    const c = limitCycle(f, [0.6, 0], 6.0, { fixedIndex: 1 })
    expect(c.converged).toBe(true)
    expect(Math.abs(c.period - 2 * Math.PI)).toBeLessThan(1e-7)
    expect(Math.abs(Math.hypot(...toFlat(c.y0)) - Math.sqrt(p))).toBeLessThan(1e-7)
    expect(c.stable).toBe(true)
    const mags = c.multipliers.re.map((re, i) => Math.hypot(re, c.multipliers.im[i] ?? 0)).sort((a, b) => a - b)
    expect(Math.abs((mags[0] ?? 0) - Math.exp(-4 * Math.PI * p))).toBeLessThan(1e-6)
    expect(Math.abs((mags[1] ?? 0) - 1)).toBeLessThan(1e-6)
    expect(floquetMultipliers(f, c.y0, c.period).re.length).toBe(2)
  })

  it('Poincare sections of the forced oscillator are periodic in the forcing period', () => {
    const f = (t: number, y: Value): Value =>
      stack([at(y, 1), add(sub(neg(at(y, 0)), mul(at(y, 1), 0.2)), Math.cos(t))])
    const s = poincareSection(f, [0, 0], (y) => y.data[1] ?? 0, { crossings: 6, direction: -1 })
    expect(s.t.length).toBe(6)
  })

  it('continuation finds the fold of x^2 + p and the Hopf point of the normal form', () => {
    const fold = continueEquilibria((x, p) => add(mul(x, x), p), [-1], -1, { ds: 0.05, steps: 60 })
    expect(fold.bifurcations[0]?.type).toBe('fold')
    expect(Math.abs(fold.bifurcations[0]?.p ?? 1)).toBeLessThan(5e-3)
    const h = continueEquilibria(hopf, [0, 0], -0.5, { ds: 0.1, steps: 12 })
    const hp = h.bifurcations.find((b) => b.type === 'hopf')
    expect(Math.abs(hp?.p ?? 1)).toBeLessThan(1e-6)
    expect(h.branch[0]?.stable).toBe(true)
    expect(h.branch.at(-1)?.stable).toBe(false)
  })

  it('recurrence quantification and embeddings', () => {
    expect(delayEmbedding([1, 2, 3], 2)).toEqual([
      [1, 2],
      [2, 3],
    ])
    expect(() => delayEmbedding([1], 3)).toThrow()
    let x = 0.3
    const chaos = Array.from({ length: 200 }, () => (x = 4 * x * (1 - x)))
    const periodic = Array.from({ length: 200 }, (_, i) => Math.sin((2 * Math.PI * i) / 20))
    const rc = recurrenceQuantification(chaos, 0.05, { dimension: 2 })
    const rp = recurrenceQuantification(periodic, 0.05, { dimension: 2 })
    expect(rp.determinism).toBeGreaterThan(rc.determinism)
    expect(rp.maxDiagonal).toBeGreaterThan(rc.maxDiagonal)
    expect(rc.recurrenceRate).toBeGreaterThan(0)
    // The Henon attractor has correlation dimension about 1.2.
    let a = 0.1
    let b = 0.1
    const hx: number[] = []
    for (let i = 0; i < 600; i++) {
      ;[a, b] = [1 - 1.4 * a * a + b, 0.3 * a]
      if (i >= 100) hx.push(a)
    }
    const d = correlationDimension(hx, 2).dimension
    expect(d).toBeGreaterThan(0.9)
    expect(d).toBeLessThan(1.5)
  })
})

function hopfAt(y: Value, p: number): Value {
  const [a, b] = [at(y, 0), at(y, 1)]
  const r2 = add(mul(a, a), mul(b, b))
  return stack([sub(sub(mul(a, p), b), mul(a, r2)), sub(add(a, mul(b, p)), mul(b, r2))])
}
