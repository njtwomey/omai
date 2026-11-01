import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import {
  advection1d,
  advection1dAlgorithm,
  heat1d,
  heat1dAlgorithm,
  pdeContract,
  type TimeStepState,
  wave1d,
  wave1dAlgorithm,
} from './finite-difference.ts'
import { advectionSchemes, heatSchemes, initialProfiles } from './finite-difference-options.ts'
import { dirichlet, grid1d, neumann } from './grids.ts'

const g = grid1d(0, 1, 21)
const u0 = (x: number): number => Math.sin(Math.PI * x) + 0.3 * Math.sin(3 * Math.PI * x)
const w = grid1d(0, 1, 41)
const a = grid1d(0, 1, 51)
const bump = (x: number): number => Math.exp(-100 * (x - 0.3) ** 2)

const heatOpts = { dt: 0.0005, tEnd: 0.02, left: dirichlet(), right: dirichlet(), every: 8 } as const
const waveOpts = { dt: 0.02, tEnd: 0.5, c: 1.2, left: dirichlet(), right: dirichlet(), every: 5 } as const

const cases = {
  'heat-explicit': { alg: () => heat1dAlgorithm(u0, g, { ...heatOpts, scheme: 'explicit' }), steps: 40 },
  'heat-implicit': { alg: () => heat1dAlgorithm(u0, g, { ...heatOpts, scheme: 'implicit' }), steps: 40 },
  'heat-crank-nicolson': { alg: () => heat1dAlgorithm(u0, g, heatOpts), steps: 40 },
  wave: {
    alg: () =>
      wave1dAlgorithm(
        (x) => Math.sin(Math.PI * x),
        (x) => 0.2 * x,
        w,
        waveOpts,
      ),
    steps: 25,
  },
  'adv-upwind': {
    alg: () => advection1dAlgorithm(bump, a, { velocity: 0.8, dt: 0.01, tEnd: 0.3, every: 10 }),
    steps: 30,
  },
  'adv-lax-friedrichs': {
    alg: () =>
      advection1dAlgorithm(bump, a, { velocity: 0.8, dt: 0.01, tEnd: 0.3, scheme: 'lax-friedrichs', every: 10 }),
    steps: 30,
  },
  'adv-lax-wendroff': {
    alg: () => advection1dAlgorithm(bump, a, { velocity: 0.8, dt: 0.01, tEnd: 0.3, scheme: 'lax-wendroff', every: 10 }),
    steps: 30,
  },
} as const

// The final field of each case before the steppers became algorithms: five nodes (1, n/4, n/2, 3n/4, n-2) and the sum.
const OLD: Record<keyof typeof cases, { at: number[]; sum: number; snapshots: number }> = {
  'heat-explicit': {
    at: [0.1512997738918, 0.6160592782662, 0.7703648971447, 0.6160592782662, 0.1512997738918],
    sum: 10.63936630597,
    snapshots: 6,
  },
  'heat-implicit': {
    at: [0.153234996183, 0.6194426686062, 0.7671733607065, 0.6194426686062, 0.153234996183],
    sum: 10.66610069659,
    snapshots: 6,
  },
  'heat-crank-nicolson': {
    at: [0.1522700489022, 0.6177555322942, 0.7687644501481, 0.6177555322942, 0.1522700489022],
    sum: 10.65276790775,
    snapshots: 6,
  },
  wave: {
    at: [-0.02174235809915, -0.1934824543903, -0.2753819561282, -0.2018355577301, -0.02257973612676],
    sum: -7.064129630544,
    snapshots: 6,
  },
  'adv-upwind': {
    at: [4.074300059386e-7, 0.002369434128195, 0.7217173907942, 0.03767917772181, 0.000004153915091325],
    sum: 8.862222308833,
    snapshots: 4,
  },
  'adv-lax-friedrichs': {
    at: [0.0002260336064601, 0.0320128118488, 0.5369497881248, 0.1156823720959, 0.0004747719203808],
    sum: 8.86249852479,
    snapshots: 4,
  },
  'adv-lax-wendroff': {
    at: [6.627595275123e-7, -0.00002269513194651, 0.8899284513684, 0.01312094401776, -3.483274193342e-8],
    sum: 8.862221827482,
    snapshots: 4,
  },
}

const rel = (x: number, y: number): void => {
  expect(Math.abs(x - y)).toBeLessThan(1e-11 * Math.max(1, Math.abs(y)))
}

describe('finite-difference time steppers as algorithms', () => {
  it('are the solvers: run to the end they give the numbers the functions gave before', () => {
    for (const [name, c] of Object.entries(cases) as [keyof typeof cases, (typeof cases)[keyof typeof cases]][]) {
      const s = runAlgorithm(c.alg(), { maxSteps: c.steps }).state
      const n = s.u.length
      const idx = [1, Math.floor(n / 4), Math.floor(n / 2), Math.floor((3 * n) / 4), n - 2]
      const old = OLD[name]
      idx.forEach((i, k) => {
        rel(s.u[i] ?? 0, old.at[k] ?? 0)
      })
      rel(
        s.u.reduce((x, y) => x + y, 0),
        old.sum,
      )
      expect(s.ts).toHaveLength(old.snapshots)
      expect(s.step).toBe(c.steps)
    }
    // And the functions are that run: same snapshots.
    const sol = heat1d(u0, g, { ...heatOpts, scheme: 'implicit' })
    const alg = runAlgorithm(cases['heat-implicit'].alg(), { maxSteps: 40 }).state
    expect(Array.from(toFlat(sol.u)).slice(-21)).toEqual(Array.from(alg.u))
    expect(
      wave1d(
        (x) => Math.sin(Math.PI * x),
        (x) => 0.2 * x,
        w,
        waveOpts,
      ).t,
    ).toEqual([...runAlgorithm(cases.wave.alg(), { maxSteps: 25 }).state.ts])
    expect(advection1d(bump, a, { velocity: 0.8, dt: 0.01, tEnd: 0.3, every: 10 }).status).toBe('success')
  })

  it('stops on an unstable explicit step with status diverged, as the solver reports failure', () => {
    const unstable = { dt: 0.2, tEnd: 80, scheme: 'explicit', left: dirichlet(), right: dirichlet() } as const
    const run = runAlgorithm(heat1dAlgorithm(u0, g, unstable), { maxSteps: 400 })
    expect(run.status).toBe('diverged')
    expect(run.state.failed).toBe(true)
    expect(run.state.ts).toHaveLength(131)
    expect(heat1d(u0, g, unstable).status).toBe('failed')
  })

  it('have a state that is JSON, and a run resumes from a checkpoint to the same end (the leapfrog keeps its second level)', () => {
    for (const c of Object.values(cases)) {
      const alg = c.alg()
      const whole = runAlgorithm(alg, { maxSteps: c.steps }).state
      const half = runAlgorithm(alg, { maxSteps: 7 }).state
      expect(half.step).toBe(7)
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
      const rest = runAlgorithm(alg, { maxSteps: c.steps - 7, start: { state: stored, step: 7, serialised: true } })
      expect(rest.steps).toBe(c.steps)
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole))
    }
  })

  it('log their contract by path: time, the field, its norm, its largest value and the failure flag', () => {
    const rec = recorder()
    runAlgorithm(cases['heat-crank-nicolson'].alg(), {
      maxSteps: 5,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('pde/t')).toHaveLength(5)
    expect(metric('pde/t')[4]).toBeCloseTo(0.0025, 12)
    expect(metric('pde/norm').every((v) => v > 0)).toBe(true)
    expect(metric('pde/failed')).toEqual([0, 0, 0, 0, 0])
    expect(rec.events.flatMap((e) => (e.type === 'tensor' && e.name === 'pde/u' ? [e.shape[0]] : []))).toEqual(
      Array(5).fill(21),
    )
    expect(Object.keys(pdeContract)).toEqual(['pde/t', 'pde/u', 'pde/norm', 'pde/max', 'pde/failed'])
  })

  it('conserve what the scheme conserves: the l2 norm falls under diffusion, the mass is steady under periodic upwind', () => {
    for (const scheme of ['implicit', 'crank-nicolson'] as const) {
      const alg = heat1dAlgorithm(u0, g, { dt: 0.002, tEnd: 0.1, scheme, left: dirichlet(), right: dirichlet() })
      const run = runAlgorithm(alg, { maxSteps: 50 }).state
      const norms = run.us.map((u) => Math.hypot(...u))
      expect(norms.every((v, i) => i === 0 || v <= (norms[i - 1] ?? 0) + 1e-12)).toBe(true)
    }
    const adv = runAlgorithm(advection1dAlgorithm(bump, a, { velocity: 0.8, dt: 0.01, tEnd: 0.5 }), {
      maxSteps: 50,
    }).state
    const mass = (u: Float64Array): number => u.slice(0, a.n - 1).reduce((x, y) => x + y, 0)
    expect(Math.abs(mass(adv.u) - mass(adv.us[0] as Float64Array))).toBeLessThan(1e-9)
  })

  it('run every option of the scheme and profile tables', () => {
    for (const name of heatSchemes.names) {
      const s = runAlgorithm(
        heat1dAlgorithm(u0, g, { dt: 0.0005, tEnd: 0.005, scheme: name as never, left: neumann(), right: dirichlet() }),
        { maxSteps: 10 },
      ).state
      expect([name, s.failed]).toEqual([name, false])
    }
    expect(heatSchemes.make({ kind: 'theta', theta: 0.9 })).toBe(0.9)
    expect(heatSchemes.where((p) => p.order === 2)).toEqual(['crank-nicolson'])
    for (const name of advectionSchemes.names) {
      const s: TimeStepState = runAlgorithm(
        advection1dAlgorithm(initialProfiles.make({ kind: 'gaussian' }), a, {
          velocity: 1,
          dt: 0.01,
          tEnd: 0.1,
          scheme: name as never,
        }),
        { maxSteps: 10 },
      ).state
      expect(s.failed).toBe(false)
    }
    expect(advectionSchemes.where((p) => p.oscillates)).toEqual(['lax-wendroff'])
    for (const name of initialProfiles.names) {
      const f = initialProfiles.make({ kind: name })
      const v = Array.from(g.x, f)
      expect(v.every(Number.isFinite)).toBe(true)
    }
  })
})
