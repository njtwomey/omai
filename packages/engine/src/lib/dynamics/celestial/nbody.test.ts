import { describe, expect, it } from 'vitest'
import { fromStorage, toFlat } from '../../tensor/index.ts'
import { messages, metricSeries, policies, recorder, runAlgorithm, tensorSeries } from '../../trace/index.ts'
import { energyDrift } from '../ode/index.ts'
import {
  configurationState,
  configurationSystem,
  FIGURE_EIGHT,
  GAUSS_K,
  lagrangeTriangle,
  type NBodyConfiguration,
  PLANETS,
  PYTHAGOREAN,
  solarSystem,
  twoBody,
} from './configurations.ts'
import { keplerState, orbitalPeriod } from './elements.ts'
import { type NBodyIntegrator, nBodyAlgorithm, nBodyHamiltonian, nBodyStep } from './nbody.ts'

const tensorOf = (x: ArrayLike<number>, shape: number[]) => fromStorage(Float64Array.from(x), shape, 'float64')

/** Run a configuration; returns the final state and the recorded run. */
function run(c: NBodyConfiguration, dt: number, steps: number, integrator: NBodyIntegrator, closeBelow?: number) {
  const system = configurationSystem(c)
  const { q, p } = configurationState(c)
  const rec = recorder()
  const alg = nBodyAlgorithm(system, tensorOf(q, [c.masses.length, c.dim]), tensorOf(p, [c.masses.length, c.dim]), {
    dt,
    steps,
    integrator,
    ...(closeBelow === undefined ? {} : { closeBelow }),
  })
  const { state } = runAlgorithm(alg, { maxSteps: steps + 1, tracer: rec, policy: policies.teaching })
  return { state, run: rec.run(), system }
}

describe('the n-body Hamiltonian', () => {
  it('has a force that is minus the gradient of the potential energy', () => {
    const masses = [1, 2.5, 0.7, 1.3]
    for (const dim of [2, 3] as const) {
      const sys = nBodyHamiltonian(masses, dim, { G: 1.7, softening: 0.05 })
      const q = Float64Array.from({ length: masses.length * dim }, (_, i) => Math.sin(1.3 * i + 0.4) * 2)
      const f = toFlat(sys.force(tensorOf(q, [masses.length, dim])))
      const h = 1e-6
      for (let i = 0; i < q.length; i++) {
        const up = q.slice()
        const down = q.slice()
        up[i] = (up[i] as number) + h
        down[i] = (down[i] as number) - h
        const numeric = -(sys.potential(up) - sys.potential(down)) / (2 * h)
        expect(f[i]).toBeCloseTo(numeric, 6)
      }
    }
  })

  it('gives equal and opposite forces (Newton’s third law) and the Newtonian magnitude G m1 m2 / r^2', () => {
    const sys = nBodyHamiltonian([2, 3], 3, { G: 4 })
    const f = toFlat(sys.force(tensorOf([0, 0, 0, 3, 0, 4], [2, 3])))
    for (const [k, x] of f.slice(0, 3).entries()) expect(-x).toBeCloseTo(f[3 + k] as number, 14)
    expect(Math.hypot(...f.slice(0, 3))).toBeCloseTo((4 * 2 * 3) / 25, 12)
    // Body 0 is pulled towards body 1 at (3, 0, 4).
    expect(f[0]).toBeCloseTo(((4 * 2 * 3) / 25) * (3 / 5), 12)
  })

  it('computes the momentum, angular momentum, centre of mass and closest pair', () => {
    const sys = nBodyHamiltonian([1, 3], 2)
    expect(sys.momentum([1, 2, 3, 4])).toEqual([4, 6])
    expect(sys.angularMomentum([1, 0, 0, 2], [0, 1, 3, 0])).toEqual([1 * 1 - 0 * 0 + (0 * 0 - 2 * 3)])
    expect(sys.centreOfMass([0, 0, 4, 0])).toEqual([3, 0])
    expect(sys.closest([0, 0, 3, 4])).toMatchObject({ distance: 5, i: 0, j: 1 })
    expect(() => nBodyHamiltonian([1], 2)).toThrow()
    expect(() => nBodyHamiltonian([1, -1], 2)).toThrow()
  })

  it('softens close encounters: a finite potential at zero separation', () => {
    const sys = nBodyHamiltonian([1, 1], 2, { softening: 0.1 })
    expect(sys.potential([0, 0, 0, 0])).toBeCloseTo(-10, 12)
  })
})

describe('integrators on a bound two-body orbit', () => {
  const c = twoBody(1.1)
  const T = c.period as number

  it('carries its truth: the period, the energy, and the orbit the analytic solution gives', () => {
    expect(c.kind).toBe('elliptic')
    const sys = configurationSystem(c)
    const { q, p } = configurationState(c)
    expect(sys.kinetic(p) + sys.potential(q)).toBeCloseTo(c.energy as number, 12)
    expect(T).toBeCloseTo(orbitalPeriod(2, 1 / (2 - 1.21)), 12)
    // The relative orbit after a quarter period against the Kepler solution of the reduced problem.
    const steps = 4000
    const { state } = run(c, T / 4 / steps, steps, 'yoshida4')
    const rel = [(state.q[2] as number) - (state.q[0] as number), (state.q[3] as number) - (state.q[1] as number)]
    const a = 1 / (2 - 1.21)
    const exact = keplerState(2, { a, e: 0.21, i: 0, raan: 0, argp: 0, nu: 0 }, T / 4)
    expect(rel[0]).toBeCloseTo(exact.r[0], 7)
    expect(rel[1]).toBeCloseTo(exact.r[1], 7)
  })

  it('conserves the momentum, the angular momentum and the centre of mass', () => {
    const { state, system } = run(c, T / 400, 800, 'stormer-verlet')
    const { q, p } = configurationState(c)
    expect(Math.hypot(...system.momentum(state.p))).toBeLessThan(1e-12)
    expect(system.angularMomentum(state.q, state.p)[0]).toBeCloseTo(system.angularMomentum(q, p)[0] as number, 12)
    expect(Math.hypot(...system.centreOfMass(state.q))).toBeLessThan(1e-12)
  })

  it('shows the energy of Stormer-Verlet bounded and that of RK4 drifting over many orbits', () => {
    const orbits = 200
    const stepsPerOrbit = 25
    const errorOf = (integrator: NBodyIntegrator): { max: number; slope: number } => {
      const { run: r } = run(c, T / stepsPerOrbit, orbits * stepsPerOrbit, integrator)
      const energy = metricSeries(r, 'bodies/energy').values
      const t = metricSeries(r, 'bodies/t').values
      const d = energyDrift(t, energy)
      return { max: d.maxRelative, slope: d.slope }
    }
    const verlet = errorOf('stormer-verlet')
    const rk4 = errorOf('rk4')
    expect(verlet.max).toBeLessThan(0.02)
    expect(rk4.max).toBeGreaterThan(10 * verlet.max)
    // RK4 loses energy steadily (its orbit spirals in); Verlet's trend is nowhere near that.
    expect(Math.abs(rk4.slope)).toBeGreaterThan(20 * Math.abs(verlet.slope))
  })

  it('orders the splitting methods by their order: a smaller energy error from the higher-order ones', () => {
    const err = (m: NBodyIntegrator): number => {
      const { run: r } = run(c, T / 100, 300, m)
      return Math.max(...metricSeries(r, 'bodies/energyDrift').values)
    }
    const euler = err('symplectic-euler')
    const verlet = err('stormer-verlet')
    const yoshida = err('yoshida4')
    const forestRuth = err('forest-ruth')
    expect(euler).toBeGreaterThan(5 * verlet)
    expect(verlet).toBeGreaterThan(5 * yoshida)
    expect(verlet).toBeGreaterThan(5 * forestRuth)
  })
})

describe('configurations with a known truth', () => {
  it('has the figure-eight close after its period, up to the digits the initial conditions carry', () => {
    const steps = 3000
    const T = FIGURE_EIGHT.period as number
    const { state, system } = run(FIGURE_EIGHT, T / steps, steps, 'yoshida4')
    const { q, p } = configurationState(FIGURE_EIGHT)
    for (let i = 0; i < q.length; i++) expect(Math.abs((state.q[i] as number) - (q[i] as number))).toBeLessThan(2e-6)
    expect(Math.hypot(...system.momentum(p))).toBeLessThan(1e-7)
    expect(Math.hypot(...system.centreOfMass(q))).toBeLessThan(1e-7)
    // A choreography: a third of a period later each body is where the next one started.
    const third = run(FIGURE_EIGHT, T / steps, steps / 3, 'yoshida4').state
    for (let i = 0; i < 3; i++) {
      const j = (i + 1) % 3
      expect(
        Math.hypot(
          (third.q[2 * i] as number) - (q[2 * j] as number),
          (third.q[2 * i + 1] as number) - (q[2 * j + 1] as number),
        ),
      ).toBeLessThan(1e-6)
    }
  })

  it('has the Pythagorean problem start at the energy its masses and triangle give', () => {
    const sys = configurationSystem(PYTHAGOREAN)
    const { q, p } = configurationState(PYTHAGOREAN)
    expect(sys.kinetic(p) + sys.potential(q)).toBeCloseTo(-(12 / 5 + 15 / 4 + 20 / 3), 12)
    expect(PYTHAGOREAN.energy).toBeCloseTo(-12.816666666666666, 12)
    // Opposite the sides of length 3, 4, 5 sit the masses 3, 4, 5.
    const d = (i: number, j: number): number =>
      Math.hypot((q[2 * i] as number) - (q[2 * j] as number), (q[2 * i + 1] as number) - (q[2 * j + 1] as number))
    expect([d(1, 2), d(0, 2), d(0, 1)]).toEqual([3, 4, 5])
  })

  it('logs the close encounters of the Pythagorean problem as events, each pair once as it enters the threshold', () => {
    // To t = 1.5, before the singular encounter near t = 1.88 that a fixed step cannot follow.
    const threshold = 2
    const { run: r, state } = run(PYTHAGOREAN, 2e-3, 750, 'yoshida4', threshold)
    const nearest = metricSeries(r, 'bodies/minSeparation').values
    const events = messages(r, 'bodies/close')
    // One event per entry into the threshold, on the step the separation first falls below it.
    const entries = nearest
      .map((d, i) => (d < threshold && (i === 0 || (nearest[i - 1] as number) >= threshold) ? i : -1))
      .filter((s) => s >= 0)
    expect(entries.length).toBeGreaterThan(0)
    expect(events.map((e) => e.step)).toEqual(entries)
    expect(events[0]?.payload).toMatchObject({ distance: expect.any(Number) })
    // The energy is kept to a few digits through the first, non-singular part of the dance.
    expect(Math.max(...metricSeries(r, 'bodies/energyDrift').values)).toBeLessThan(1e-4)
    expect(state.step).toBe(750)
  })

  it('rotates the Lagrange triangle rigidly', () => {
    const c = lagrangeTriangle(2, 3, 1.5)
    const T = c.period as number
    const steps = 800
    const quarter = run(c, T / 4 / steps, steps, 'yoshida4').state
    const { q } = configurationState(c)
    // A quarter turn takes (x, y) to (-y, x) for every body, and the side stays 2.
    for (let i = 0; i < 3; i++) {
      expect(quarter.q[2 * i]).toBeCloseTo(-(q[2 * i + 1] as number), 6)
      expect(quarter.q[2 * i + 1]).toBeCloseTo(q[2 * i] as number, 6)
    }
    const sys = configurationSystem(c)
    expect(sys.kinetic(configurationState(c).p) + sys.potential(q)).toBeCloseTo(c.energy as number, 12)
  })

  it('launches two bodies as circular, elliptic, parabolic and hyperbolic by the speed', () => {
    const kinds = [0.8, 1, 1.2, Math.SQRT2, 1.6].map((f) => twoBody(f).kind)
    expect(kinds).toEqual(['elliptic', 'circular', 'elliptic', 'parabolic', 'hyperbolic'])
    expect(twoBody(Math.SQRT2).energy).toBe(0)
    expect(twoBody(1.6).period).toBeUndefined()
    // The hyperbola leaves for ever: the separation grows without bound while the energy stays positive.
    const c = twoBody(1.6)
    const { state, system } = run(c, 0.01, 2000, 'yoshida4')
    const { q, p } = configurationState(c)
    expect(system.kinetic(p) + system.potential(q)).toBeGreaterThan(0)
    expect(system.closest(state.q).distance).toBeGreaterThan(20)
  })

  it('puts the planets on their orbits: Earth returns after a year, and the energy is conserved', () => {
    const c = solarSystem(['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter'])
    expect(c.masses.length).toBe(6)
    const dt = 1
    const days = 365
    const { state, run: r } = run(c, dt, days, 'yoshida4')
    const { q } = configurationState(c)
    // Earth is the third planet, body 3 (the Sun is body 0): heliocentric positions.
    const helio = (s: { q: Float64Array }, b: number): number[] =>
      [0, 1, 2].map((k) => (s.q[3 * b + k] as number) - (s.q[k] as number))
    const start = helio({ q }, 3)
    const end = helio(state, 3)
    const radius = Math.hypot(...start)
    expect(Math.hypot(...end.map((v, k) => v - (start[k] as number))) / radius).toBeLessThan(0.02)
    expect(Math.max(...metricSeries(r, 'bodies/energyDrift').values)).toBeLessThan(2e-7)
    expect(PLANETS.length).toBe(8)
    expect(GAUSS_K).toBeCloseTo(0.01720209895, 12)
    expect(() => solarSystem(['Pluto'])).toThrow()
  })
})

describe('the n-body algorithm', () => {
  const c = twoBody(1.1)
  const system = configurationSystem(c)
  const { q, p } = configurationState(c)
  const make = (steps: number, integrator: NBodyIntegrator = 'stormer-verlet') =>
    nBodyAlgorithm(system, tensorOf(q, [2, 2]), tensorOf(p, [2, 2]), { dt: 0.05, steps, integrator })

  it('logs every path of its contract each step', () => {
    const rec = recorder()
    runAlgorithm(make(10), { maxSteps: 20, tracer: rec, policy: policies.teaching })
    const r = rec.run()
    for (const name of [
      'bodies/t',
      'bodies/energy',
      'bodies/energyDrift',
      'bodies/angularMomentum',
      'bodies/momentum',
      'bodies/minSeparation',
    ]) {
      expect(metricSeries(r, name).values.length).toBe(10)
    }
    expect(tensorSeries(r, 'bodies/positions').length).toBe(10)
    expect(metricSeries(r, 'bodies/t').values.at(-1)).toBeCloseTo(0.5, 12)
  })

  it('serialises a state and resumes to the same result as an uninterrupted run', () => {
    const alg = make(40)
    const straight = runAlgorithm(alg, { maxSteps: 100 }).state
    const half = runAlgorithm(alg, { maxSteps: 17 }).state
    const restored = alg.deserialise?.(alg.serialise?.(half) as never) as typeof half
    let s = restored
    const ctx = { log: () => undefined, emit: () => undefined, wants: () => false } as never
    while (alg.done?.(s, s.step) === undefined) s = alg.step(s, ctx)
    expect(Array.from(s.q)).toEqual(Array.from(straight.q))
    expect(s.step).toBe(40)
  })

  it('refuses a bad step, an unknown integrator and a state of the wrong shape', () => {
    expect(() => nBodyAlgorithm(system, tensorOf(q, [2, 2]), tensorOf(p, [2, 2]), { dt: 0, steps: 1 })).toThrow()
    expect(() => nBodyStep(system, q, p, 0.1, 'leapfrog' as never)).toThrow()
    expect(() => nBodyAlgorithm(system, tensorOf([0, 0, 1], [3]), tensorOf(p, [2, 2]), { dt: 0.1, steps: 1 })).toThrow()
  })
})
