import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import { assemble1d, heatFem, heatFemAlgorithm, heatFemContract } from './finite-element.ts'
import {
  BURGERS,
  conservationLaw,
  conservationLawAlgorithm,
  fvContract,
  LIMITERS,
  linearFlux,
  rusanovFlux,
} from './finite-volume.ts'
import { fokkerPlanck, fokkerPlanckAlgorithm, fokkerPlanckContract } from './fokker-planck.ts'
import { fourierWavenumbers, spectralContract, spectralEtdrk4, spectralEtdrk4Algorithm } from './spectral.ts'

// A sample of the final field of each solver as it was before the solvers became algorithms: nodes 1, n/4, n/2,
// 3n/4 and n-2 (13 significant digits).
const OLD = {
  fem: [0.09539638043526, 0.4312056648711, 0.6098168994328, 0.4312056648711, 0.09539638043526],
  fv: [0.09932475710725, 0.8497503111427, 0.9999999974954, 0, 0],
  fp: [9.080674894342e-15, 0.000002209015371693, 0.2819897583484, 0.01902658261772, 5.136370623304e-9],
  sp: [0.140781367244, 0.3678794411714, 7.263824954556e-17, -0.3678794411714, -0.2601300475114],
}
const sample = (u: ArrayLike<number>): number[] => {
  const n = u.length
  return [1, Math.floor(n / 4), Math.floor(n / 2), Math.floor((3 * n) / 4), n - 2].map((i) => u[i] ?? 0)
}
const close = (a: number[], b: number[]): void => {
  b.forEach((v, i) => {
    expect(Math.abs((a[i] ?? 0) - v)).toBeLessThan(1e-11 * Math.max(1e-3, Math.abs(v)) + 1e-14)
  })
}

const mesh = Array.from({ length: 21 }, (_, i) => i / 20)
const femAlg = () =>
  heatFemAlgorithm(assemble1d(mesh), (x) => Math.sin(Math.PI * x), { dt: 0.005, tEnd: 0.05, every: 3 })
const step = (x: number): number => (x < 0.5 ? 1 : 0)
const fvOptions = { flux: BURGERS, tEnd: 0.3, scheme: 'muscl', limiter: 'minmod', every: 4 } as const
const fvAlg = () => conservationLawAlgorithm(step, [0, 1], 60, fvOptions)
const process = { drift: (x: number) => -x, sigma: () => 1 }
const bump = (x: number): number => Math.exp(-((x - 1) ** 2) / 0.02)
const fpAlg = () => fokkerPlanckAlgorithm(process, bump, [-4, 4], 80, { dt: 0.02, tEnd: 0.3, every: 3 })
const n = 16
const wavenumbers = fourierWavenumbers(n, 2 * Math.PI)
const xs = Float64Array.from({ length: n }, (_, j) => (2 * Math.PI * j) / n)
const problem = {
  linear: { re: Array.from(wavenumbers, (q) => -q * q), im: new Array(n).fill(0) },
  nonlinear: (v: { re: Float64Array; im: Float64Array }) => ({ re: v.re.map((q) => 0.01 * q), im: v.im.map(() => 0) }),
}
const spAlg = () => spectralEtdrk4Algorithm(problem, Array.from(xs, Math.sin), { dt: 0.05, tEnd: 1, every: 5 })

describe('finite-element, finite-volume, Fokker-Planck and spectral time steppers as algorithms', () => {
  it('are the solvers: run to the end they give the numbers the functions gave before', () => {
    const fem = runAlgorithm(femAlg(), { maxSteps: 10 }).state
    close(sample(fem.u), OLD.fem)
    expect(fem.ts).toHaveLength(5)
    const fv = runAlgorithm(fvAlg(), { maxSteps: 1000 }).state
    close(sample(fv.u), OLD.fv)
    expect(fv.ts).toHaveLength(10)
    const fp = runAlgorithm(fpAlg(), { maxSteps: 15 }).state
    close(sample(fp.p), OLD.fp)
    expect(fp.ts).toHaveLength(6)
    const sp = runAlgorithm(spAlg(), { maxSteps: 20 }).state
    close(sample(sp.us.at(-1) as Float64Array), OLD.sp)
    expect(sp.ts).toHaveLength(5)
    // The functions are those runs.
    const femFn = heatFem(assemble1d(mesh), (x) => Math.sin(Math.PI * x), { dt: 0.005, tEnd: 0.05, every: 3 })
    expect(Array.from(toFlat(femFn.u)).slice(-21)).toEqual(Array.from(fem.u))
    const fvFn = conservationLaw(step, [0, 1], 60, fvOptions)
    expect(fvFn.mass.at(-1)).toBeCloseTo(0.5, 12)
    expect(fokkerPlanck(process, bump, [-4, 4], 80, { dt: 0.02, tEnd: 0.3, every: 3 }).mean.at(-1)).toBeCloseTo(
      0.7437424398452,
      10,
    )
    const spFn = spectralEtdrk4(problem, Array.from(xs, Math.sin), xs, { dt: 0.05, tEnd: 1, every: 5 })
    expect(spFn.status).toBe('success')
  })

  it('have a state that is JSON, and a run resumes from a checkpoint to the same end', () => {
    for (const [alg, steps, split] of [
      [femAlg(), 10, 4],
      [fpAlg(), 15, 6],
      [spAlg(), 20, 7],
    ] as const) {
      const whole = runAlgorithm(alg as never, { maxSteps: steps }).state
      const half = runAlgorithm(alg as never, { maxSteps: split }).state
      const stored = JSON.parse(JSON.stringify((alg as { serialise: (s: unknown) => unknown }).serialise(half)))
      const rest = runAlgorithm(alg as never, {
        maxSteps: steps - split,
        start: { state: stored, step: split, serialised: true },
      })
      expect(rest.steps).toBe(steps)
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole))
    }
    // The finite-volume step is set by the CFL number, so its step count is not known: resume by steps taken.
    const alg = fvAlg()
    const whole = runAlgorithm(alg, { maxSteps: 1000 })
    const half = runAlgorithm(alg, { maxSteps: 12 })
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
    const rest = runAlgorithm(alg, { maxSteps: 1000, start: { state: stored, step: 12, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
  })

  it('log their contracts by path', () => {
    const cases = [
      [femAlg(), 2, heatFemContract, 'pde/u', [21]],
      [fvAlg(), 2, fvContract, 'pde/u', [60]],
      [fpAlg(), 2, fokkerPlanckContract, 'pde/density', [80]],
      [spAlg(), 2, spectralContract, 'pde/u', [16]],
    ] as const
    for (const [alg, steps, contract, field, shape] of cases) {
      const rec = recorder()
      runAlgorithm(alg as never, {
        maxSteps: steps,
        tracer: rec,
        policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
      })
      const shapes = rec.events.flatMap((e) => (e.type === 'tensor' && e.name === field ? [[...e.shape]] : []))
      expect(shapes).toEqual(Array(steps).fill(shape))
      const times = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'pde/t' ? [e.value] : []))
      expect(times).toHaveLength(steps)
      expect(Object.keys(contract)).toContain('pde/t')
    }
  })

  it('keep what they conserve: mass in the finite volumes and the density, decay of the spectral heat mode', () => {
    const fv = runAlgorithm(fvAlg(), { maxSteps: 1000 }).state
    const mass = (u: Float64Array): number => u.reduce((a, b) => a + b, 0) / 60
    expect(Math.abs(mass(fv.us[0] as Float64Array) - mass(fv.u))).toBeLessThan(1e-12)
    const fp = runAlgorithm(fpAlg(), { maxSteps: 15 }).state
    const dx = 8 / 80
    const total = (p: Float64Array): number => p.reduce((a, b) => a + b * dx, 0)
    expect(Math.abs(total(fp.ps[0] as Float64Array) - 1)).toBeLessThan(1e-12)
    expect(Math.abs(total(fp.p) - 1)).toBeLessThan(1e-9)
    // The pure heat mode sin(x) decays as exp(-t): the spectral step is exact for the linear part.
    const heat = {
      linear: problem.linear,
      nonlinear: (v: { re: Float64Array; im: Float64Array }) => ({ re: v.re.map(() => 0), im: v.im.map(() => 0) }),
    }
    const sp = runAlgorithm(spectralEtdrk4Algorithm(heat, Array.from(xs, Math.sin), { dt: 0.1, tEnd: 1 }), {
      maxSteps: 10,
    }).state
    expect(Math.abs((sp.us.at(-1)?.[4] ?? 0) - Math.exp(-1))).toBeLessThan(1e-12)
  })

  it('run every limiter and Riemann solver of the finite volumes', () => {
    for (const limiter of Object.keys(LIMITERS) as (keyof typeof LIMITERS)[]) {
      for (const riemann of ['godunov', 'rusanov'] as const) {
        const alg = conservationLawAlgorithm(step, [0, 1], 40, { ...fvOptions, tEnd: 0.1, limiter, riemann })
        const s = runAlgorithm(alg, { maxSteps: 1000 }).state
        expect(s.failed).toBe(false)
        expect(Math.abs(s.u.reduce((a, b) => a + b, 0) / 40 - 0.5)).toBeLessThan(1e-12)
      }
    }
    expect(rusanovFlux(linearFlux(1), 1, 1)).toBe(1)
  })
})
