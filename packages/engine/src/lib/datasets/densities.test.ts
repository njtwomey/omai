import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { maxOf } from '../base/index.ts'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import {
  ackley,
  bananaDensity,
  beale,
  branin,
  braninSurface,
  checkerboardDensity,
  type Density,
  densitySamples,
  funnelDensity,
  gaussianMixtureDensity,
  himmelblau,
  learningRateSurface,
  pinwheelDensity,
  quadraticBowl,
  rastrigin,
  ringDensity,
  rosenbrock,
} from './index.ts'

/** Integrate exp(logProb) on a grid over a box (midpoint rule). */
function integrate(d: Density, box: readonly (readonly [number, number])[], steps = 300): number {
  const [[a0, a1], [b0, b1]] = box as [[number, number], [number, number]]
  const ha = (a1 - a0) / steps
  const hb = (b1 - b0) / steps
  const pts: number[][] = []
  for (let i = 0; i < steps; i++) for (let j = 0; j < steps; j++) pts.push([a0 + (i + 0.5) * ha, b0 + (j + 0.5) * hb])
  return toFlat(d.logProb(tensor(pts))).reduce((s, v) => s + Math.exp(v), 0) * ha * hb
}

/** The fraction of samples in a box against the density's integral over it. */
function massCheck(d: Density, box: readonly (readonly [number, number])[], n = 6000): void {
  const { x } = d.sample(key(3), n)
  const p = toFlat(x)
  const [[a0, a1], [b0, b1]] = box as [[number, number], [number, number]]
  let inside = 0
  for (let i = 0; i < n; i++) {
    const a = p[2 * i] ?? 0
    const b = p[2 * i + 1] ?? 0
    if (a >= a0 && a < a1 && b >= b0 && b < b1) inside++
  }
  const mass = integrate(d, box, 200)
  expect(Math.abs(inside / n - mass), d.name).toBeLessThan(5 * Math.sqrt(Math.max(mass * (1 - mass), 1e-4) / n) + 0.005)
}

describe('densities', () => {
  const all: Density[] = [
    ringDensity({ radii: [1, 2.5], sigma: 0.2 }),
    gaussianMixtureDensity(),
    gaussianMixtureDensity({ layout: 'grid', count: 3, radius: 1.5, sigma: 0.3 }),
    pinwheelDensity(),
    bananaDensity({ a: 1, b: 0.5 }),
  ]
  for (const d of all) {
    it(`${d.name} integrates to one and its sampler matches it`, () => {
      expect(integrate(d, d.bounds)).toBeCloseTo(1, 2)
      const [[a0, a1], [b0]] = d.bounds as [[number, number], [number, number]]
      massCheck(d, [
        [a0, (a0 + a1) / 2 + 0.3],
        [b0, 0.4],
      ])
    })
  }

  it('the funnel is normalised and has its neck', () => {
    const f = funnelDensity(1)
    expect(
      integrate(
        f,
        [
          [-6, 6],
          [-25, 25],
        ],
        600,
      ),
    ).toBeCloseTo(1, 2)
    const { x } = f.sample(key(1), 4000)
    const p = toFlat(x)
    const narrow = p.filter((_, i) => i % 2 === 1 && (p[i - 1] ?? 0) < -2).map(Math.abs)
    expect(maxOf(narrow)).toBeLessThan(1.5)
  })

  it('the checkerboard is uniform on its dark squares', () => {
    const c = checkerboardDensity()
    expect(integrate(c, c.bounds, 400)).toBeCloseTo(1, 6)
    const { x } = c.sample(key(2), 500)
    expect(toFlat(c.logProb(x)).every((v) => Math.abs(v + Math.log(32)) < 1e-12)).toBe(true)
  })

  it('generates labelled samples with the exact log density', () => {
    const { data, truth } = generate(densitySamples, { key: key(4), knobs: { density: 'grid', n: 100 } })
    expect(data.classes.length).toBe(25)
    expect(toFlat(truth.logProb?.(data.x) as Tensor).every(Number.isFinite)).toBe(true)
  })
})

describe('test functions', () => {
  for (const c of loadFixture('datasets/objectives').cases) {
    it(c.name, () => {
      const x = decode(c.inputs[0]) as Tensor
      if (c.fn === 'rosenbrock') {
        const f = rosenbrock(x.shape[0] ?? 2)
        const got = c.params['part'] === 'f' ? f.f(x) : c.params['part'] === 'g' ? f.gradient(x) : f.hessian(x)
        expectMatches(got, c.output, c.rtol, 1e-10, c.name)
        return
      }
      const f = c.fn === 'himmelblau' ? himmelblau() : branin()
      expectMatches(f.f(x), c.output, c.rtol, 1e-12, c.name)
    })
  }

  it('reach their minimum at every minimiser with zero gradient', () => {
    for (const f of [
      rosenbrock(3),
      himmelblau(),
      beale(),
      rastrigin(3),
      ackley(2),
      branin(),
      quadraticBowl({ dim: 3, key: key(0) }),
    ]) {
      for (const m of f.minimizers) {
        expect(toFlat(f.f(tensor(m as number[])))[0], f.name).toBeCloseTo(f.minimum, f.name === 'branin' ? 4 : 10)
        if (f.name !== 'ackley' && f.name !== 'branin')
          for (const g of toFlat(f.gradient(m as number[]))) expect(Math.abs(g), f.name).toBeLessThan(1e-6)
      }
    }
  })

  it('evaluate batches, and the quadratic bowl has the asked conditioning', () => {
    const r = rastrigin(2)
    expect(
      toFlat(
        r.f(
          tensor([
            [0, 0],
            [1, 1],
          ]),
        ),
      ),
    ).toEqual([0, 2])
    const q = quadraticBowl({ dim: 4, condition: 100, key: key(1) })
    const h = toFlat(q.hessian([0.3, -0.2, 1, 2]))
    const a = toFlat(q.matrix)
    h.forEach((v, i) => {
      expect(v).toBeCloseTo(a[i] ?? 0, 10)
    })
  })

  it('tuning surfaces have known optima and learning curves', () => {
    const s = learningRateSurface()
    expect(s.mean({ learningRate: 1e-1, weightDecay: 1e-4 })).toBeGreaterThan(s.optimalValue)
    expect(s.mean(s.optimum, 1)).toBeGreaterThan(s.mean(s.optimum, 10))
    const draws = Array.from({ length: 400 }, (_, i) => s.evaluate(s.optimum, key(i)))
    const m = draws.reduce((a, b) => a + b, 0) / draws.length
    expect(Math.abs(m - s.optimalValue)).toBeLessThan(4 * (0.02 / Math.sqrt(400)))
    const b = braninSurface()
    expect(b.mean(b.optimum)).toBeCloseTo(b.optimalValue, 5)
  })
})
