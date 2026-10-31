import { describe, expect, it } from 'vitest'
import { tensor, toFlat } from '../../tensor/index.ts'
import { metricSeries, policies, recorder, runAlgorithm, tensorSeries } from '../../trace/index.ts'
import {
  classifyLagrangePoints,
  effectivePotential,
  jacobiConstant,
  lagrangePoints,
  pseudoPotential,
  ROUTH_CRITICAL_MASS_RATIO,
  restrictedThreeBodyAlgorithm,
  restrictedThreeBodyField,
  zeroVelocityCurves,
} from './three-body.ts'

const EARTH_MOON = 0.012150585609624

describe('the restricted three-body field', () => {
  it('is the gradient of the pseudo-potential plus the Coriolis terms', () => {
    const mu = 0.2
    const f = restrictedThreeBodyField(mu, 3)
    const s = [0.3, -0.4, 0.2, 0.1, -0.5, 0.25]
    const out = toFlat(f(0, tensor(s)))
    const h = 1e-6
    const grad = [0, 1, 2].map((k) => {
      const up = [s[0] as number, s[1] as number, s[2] as number]
      const down = [...up]
      up[k] = (up[k] as number) + h
      down[k] = (down[k] as number) - h
      return (
        (pseudoPotential(mu, ...(up as [number, number, number])) -
          pseudoPotential(mu, ...(down as [number, number, number]))) /
        (2 * h)
      )
    })
    expect(out.slice(0, 3)).toEqual([0.1, -0.5, 0.25])
    expect(out[3]).toBeCloseTo(2 * (s[4] as number) + (grad[0] as number), 6)
    expect(out[4]).toBeCloseTo(-2 * (s[3] as number) + (grad[1] as number), 6)
    expect(out[5]).toBeCloseTo(grad[2] as number, 6)
  })

  it('agrees between the planar and spatial forms at z = 0', () => {
    const planar = toFlat(restrictedThreeBodyField(0.3, 2)(0, tensor([0.4, 0.9, -0.2, 0.3])))
    const spatial = toFlat(restrictedThreeBodyField(0.3, 3)(0, tensor([0.4, 0.9, 0, -0.2, 0.3, 0])))
    expect([spatial[0], spatial[1], spatial[3], spatial[4]]).toEqual(planar)
    expect(spatial[5]).toBeCloseTo(0, 15)
  })

  it('has the effective potential as minus the pseudo-potential and the Jacobi constant as 2 Omega - v^2', () => {
    expect(effectivePotential(0.1, 0.3, 0.4)).toBe(-pseudoPotential(0.1, 0.3, 0.4))
    expect(jacobiConstant(0.1, [0.3, 0.4, 1, 2])).toBeCloseTo(2 * pseudoPotential(0.1, 0.3, 0.4) - 5, 14)
    expect(jacobiConstant(0.1, [0.3, 0.4, 0.5, 1, 2, 2])).toBeCloseTo(2 * pseudoPotential(0.1, 0.3, 0.4, 0.5) - 9, 14)
    expect(() => jacobiConstant(0.1, [1, 2, 3])).toThrow()
    expect(() => restrictedThreeBodyField(0)).toThrow()
    expect(() => restrictedThreeBodyField(0.7)).toThrow()
  })
})

describe('Lagrange points', () => {
  it('matches the published positions for the Earth-Moon system', () => {
    const p = lagrangePoints(EARTH_MOON)
    expect(p.L1[0]).toBeCloseTo(0.836915, 5)
    expect(p.L2[0]).toBeCloseTo(1.155682, 5)
    expect(p.L3[0]).toBeCloseTo(-1.005063, 5)
    expect(p.L4[0]).toBeCloseTo(0.5 - EARTH_MOON, 14)
    expect(p.L4[1]).toBeCloseTo(Math.sqrt(3) / 2, 14)
    expect(p.L5[1]).toBeCloseTo(-Math.sqrt(3) / 2, 14)
  })

  it('makes each an equilibrium: the field vanishes there, for every mass ratio', () => {
    for (const mu of [1e-6, 0.01, 0.0385, 0.2, 0.5]) {
      const f = restrictedThreeBodyField(mu)
      for (const [x, y] of Object.values(lagrangePoints(mu))) {
        const out = toFlat(f(0, tensor([x, y, 0, 0])))
        expect(Math.hypot(...out)).toBeLessThan(1e-10)
      }
    }
  })

  it('puts L4 and L5 at unit distance from both primaries, and the collinear ones on the axis in order', () => {
    const mu = 0.3
    const p = lagrangePoints(mu)
    expect(Math.hypot(p.L4[0] + mu, p.L4[1])).toBeCloseTo(1, 14)
    expect(Math.hypot(p.L4[0] - 1 + mu, p.L4[1])).toBeCloseTo(1, 14)
    expect(p.L3[0]).toBeLessThan(-mu)
    expect(p.L1[0]).toBeGreaterThan(-mu)
    expect(p.L1[0]).toBeLessThan(1 - mu)
    expect(p.L2[0]).toBeGreaterThan(1 - mu)
  })

  it('has L1 and L2 a Hill-radius (mu / 3)^(1/3) from the small primary when it is light', () => {
    const mu = 1e-6
    const p = lagrangePoints(mu)
    const hill = (mu / 3) ** (1 / 3)
    expect(1 - mu - p.L1[0]).toBeCloseTo(hill, 3)
    expect(p.L2[0] - (1 - mu)).toBeCloseTo(hill, 3)
  })

  it('classifies L1-L3 as saddles and L4, L5 as centres below Routh’s mass ratio and unstable above', () => {
    expect(ROUTH_CRITICAL_MASS_RATIO).toBeCloseTo(0.0385208965, 9)
    const below = classifyLagrangePoints(0.03)
    expect(below.map((p) => p.type)).toEqual(['saddle', 'saddle', 'saddle', 'center', 'center'])
    expect(below.map((p) => p.stable)).toEqual([false, false, false, true, true])
    const above = classifyLagrangePoints(0.04)
    expect(above.slice(3).map((p) => p.stable)).toEqual([false, false])
    expect(above.slice(3).map((p) => Math.max(...p.eigenvalues.re))).toEqual([expect.any(Number), expect.any(Number)])
    expect(Math.max(...(above[3]?.eigenvalues.re ?? []))).toBeGreaterThan(1e-3)
  })

  it('has the eigenvalues at L4 that the linear theory gives: lambda^2 = (-1 +- sqrt(1 - 27 mu (1 - mu))) / 2', () => {
    const mu = 0.02
    const root = Math.sqrt(1 - 27 * mu * (1 - mu))
    const l4 = classifyLagrangePoints(mu)[3]
    const freqs = (l4?.eigenvalues.im ?? []).map(Math.abs).sort((a, b) => a - b)
    const expected = [Math.sqrt((1 - root) / 2), Math.sqrt((1 + root) / 2)]
    expect(freqs[0]).toBeCloseTo(expected[0] as number, 7)
    expect(freqs[3]).toBeCloseTo(expected[1] as number, 7)
  })
})

describe('trajectories', () => {
  const mu = 0.01215
  const start = [0.8, 0, 0, 0.35]
  const drift = (dt: number, steps: number): number => {
    const rec = recorder()
    runAlgorithm(restrictedThreeBodyAlgorithm(mu, start, { dt, steps }), { maxSteps: steps, tracer: rec })
    return Math.max(...metricSeries(rec.run(), 'mass/jacobiDrift').values.map(Math.abs))
  }

  it('conserves the Jacobi constant to the order of the integrator', () => {
    const coarse = drift(0.02, 500)
    const fine = drift(0.01, 1000)
    expect(coarse).toBeLessThan(1e-4)
    // A fourth-order method: halving the step divides the error by about 16.
    expect(coarse / fine).toBeGreaterThan(8)
    expect(coarse / fine).toBeLessThan(40)
  })

  it('keeps a mass released at rest near L4 librating there (the tadpole) when mu is below Routh’s, and drives it away above', () => {
    const farthest = (mu4: number): number => {
      const [x4, y4] = lagrangePoints(mu4).L4
      const rec = recorder()
      runAlgorithm(restrictedThreeBodyAlgorithm(mu4, [x4 + 0.01, y4, 0, 0], { dt: 0.01, steps: 6000 }), {
        maxSteps: 6000,
        tracer: rec,
        policy: policies.teaching,
      })
      return Math.max(
        ...tensorSeries(rec.run(), 'mass/position').map((e) =>
          Math.hypot((e.values?.[0] ?? 0) - x4, (e.values?.[1] ?? 0) - y4),
        ),
      )
    }
    expect(farthest(0.01)).toBeLessThan(0.2)
    expect(farthest(0.2)).toBeGreaterThan(0.5)
  })

  it('drives a mass released at rest next to L1 away along the axis (the saddle)', () => {
    const mu1 = 0.01
    const [x1] = lagrangePoints(mu1).L1
    const near = runAlgorithm(restrictedThreeBodyAlgorithm(mu1, [x1 + 1e-6, 0, 0, 0], { dt: 0.01, steps: 1500 }), {
      maxSteps: 1500,
    }).state
    expect(Math.abs((near.y[0] as number) - x1)).toBeGreaterThan(0.05)
  })

  it('cannot cross its zero-velocity curve: the forbidden region stays forbidden', () => {
    const C = jacobiConstant(mu, start)
    const rec = recorder()
    runAlgorithm(restrictedThreeBodyAlgorithm(mu, start, { dt: 0.01, steps: 3000 }), {
      maxSteps: 3000,
      tracer: rec,
      policy: policies.teaching,
    })
    const positions = tensorSeries(rec.run(), 'mass/position')
    expect(positions.length).toBe(3000)
    for (const e of positions) {
      const [x, y] = e.values ?? []
      expect(2 * pseudoPotential(mu, x as number, y as number) - C).toBeGreaterThan(-1e-6)
    }
  })

  it('logs time, position, velocity, the Jacobi constant and its drift, and resumes from a serialised state', () => {
    const alg = restrictedThreeBodyAlgorithm(mu, start, { dt: 0.05, steps: 40 })
    const rec = recorder()
    const straight = runAlgorithm(alg, { maxSteps: 100, tracer: rec }).state
    const r = rec.run()
    expect(metricSeries(r, 'mass/jacobi').values.length).toBe(40)
    expect(metricSeries(r, 'mass/t').values.at(-1)).toBeCloseTo(2, 12)
    expect(metricSeries(r, 'mass/jacobiDrift').values[0]).toBeCloseTo(
      (metricSeries(r, 'mass/jacobi').values[0] as number) - jacobiConstant(mu, start),
      14,
    )
    const half = runAlgorithm(alg, { maxSteps: 15 }).state
    let s = alg.deserialise?.(alg.serialise?.(half) as never) as typeof half
    const ctx = { log: () => undefined, emit: () => undefined, wants: () => false } as never
    while (alg.done?.(s, s.step) === undefined) s = alg.step(s, ctx)
    expect(Array.from(s.y)).toEqual(Array.from(straight.y))
  })
})

describe('zero-velocity curves', () => {
  it('traces the level set 2 Omega = C', () => {
    const mu = 0.1
    const C = 3.6
    const curves = zeroVelocityCurves(mu, C, { n: 300 })
    expect(curves.length).toBeGreaterThan(0)
    for (const curve of curves) {
      for (const [x, y] of curve) expect(Math.abs(2 * pseudoPotential(mu, x, y) - C)).toBeLessThan(0.05)
    }
  })

  it('opens at the Lagrange points as the constant falls: closed around the primaries above C(L1), joined below', () => {
    const mu = 0.1
    const points = lagrangePoints(mu)
    const cOf = ([x, y]: readonly [number, number]): number => 2 * pseudoPotential(mu, x, y)
    expect(cOf(points.L1)).toBeGreaterThan(cOf(points.L2))
    expect(cOf(points.L2)).toBeGreaterThan(cOf(points.L3))
    expect(cOf(points.L3)).toBeGreaterThan(cOf(points.L4))
    const closedAt = (c: number): number =>
      zeroVelocityCurves(mu, c, { n: 400, window: [-0.6, 1.4, -1, 1] }).filter(
        (p) => p.length > 3 && p[0]?.[0] === p.at(-1)?.[0],
      ).length
    // Above C(L1) the two lobes around the primaries are separate closed curves; below, they have merged into one.
    expect(closedAt(cOf(points.L1) + 0.2)).toBeGreaterThanOrEqual(2)
    expect(closedAt(cOf(points.L1) - 0.2)).toBeLessThan(closedAt(cOf(points.L1) + 0.2))
  })
})
