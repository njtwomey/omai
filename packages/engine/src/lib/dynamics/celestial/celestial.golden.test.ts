import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../../testing/systems.ts'
import type { Vec3 } from '../../base/index.ts'
import { fromStorage, type Tensor, tensor, toFlat } from '../../tensor/index.ts'
import { runAlgorithm, type StepContext } from '../../trace/index.ts'
import { elementsFromState, keplerState } from './elements.ts'
import { deflectionAngle, lightRayAlgorithm, measuredDeflection } from './geodesics.ts'
import { solveKepler } from './kepler.ts'
import { nBodyHamiltonian, nBodyStep } from './nbody.ts'
import { lagrangePoints, restrictedThreeBodyAlgorithm } from './three-body.ts'

const flat = (t: Tensor): number[] => toFlat(t)
const quiet = { log: () => undefined, emit: () => undefined, wants: () => false } as unknown as StepContext

const FNS: Readonly<Record<string, GoldenFn>> = {
  'kepler-elliptic': ([m], p) =>
    tensor(flat(m as Tensor).map((M) => solveKepler(M, p['e'] as number, { tol: 1e-15 }).anomaly)),
  'kepler-hyperbolic': ([m], p) =>
    tensor(flat(m as Tensor).map((M) => solveKepler(M, p['e'] as number, { tol: 1e-15 }).anomaly)),
  'kepler-parabolic': ([m]) => tensor(flat(m as Tensor).map((M) => solveKepler(M, 1).trueAnomaly)),
  'two-body': ([y0, times], p) => {
    const y = flat(y0 as Tensor)
    const mu = p['mu'] as number
    const el = elementsFromState(mu, y.slice(0, 3) as unknown as Vec3, y.slice(3) as unknown as Vec3)
    const rows = flat(times as Tensor).flatMap((t) => {
      const s = keplerState(mu, el, t)
      return [...s.r, ...s.v]
    })
    return fromStorage(Float64Array.from(rows), [rows.length / 6, 6], 'float64')
  },
  nbody: ([masses, q0, p0], p) => {
    const m = flat(masses as Tensor)
    const system = nBodyHamiltonian(m, 3, { G: p['G'] as number })
    const dt = p['dt'] as number
    const steps = Math.round((p['t'] as number) / dt)
    let q: ArrayLike<number> = flat(q0 as Tensor)
    let mom: ArrayLike<number> = flat(p0 as Tensor)
    for (let k = 0; k < steps; k++) ({ q, p: mom } = nBodyStep(system, q, mom, dt, 'yoshida4'))
    return tensor([...Array.from(q), ...Array.from(mom)])
  },
  'lagrange-collinear': ([mus]) => {
    const rows = flat(mus as Tensor).map((mu) => {
      const l = lagrangePoints(mu)
      return [l.L1[0], l.L2[0], l.L3[0]]
    })
    return fromStorage(Float64Array.from(rows.flat()), [rows.length, 3], 'float64')
  },
  deflection: ([bs]) => tensor(flat(bs as Tensor).map((b) => deflectionAngle(b, 1))),
  'deflection-ray': ([bs]) =>
    tensor(
      flat(bs as Tensor).map((b) => {
        const { state } = runAlgorithm(lightRayAlgorithm(1, b, { dphi: 0.002, startRadius: 400 * b }), {
          maxSteps: 60000,
        })
        return measuredDeflection(state, b, 1, 400 * b)
      }),
    ),
  cr3bp: ([y0, times], p) => {
    const t = flat(times as Tensor)
    const dt = p['dt'] as number
    const per = Math.round(((t[1] as number) - (t[0] as number)) / dt)
    const alg = restrictedThreeBodyAlgorithm(p['mu'] as number, flat(y0 as Tensor), { dt, steps: per * (t.length - 1) })
    let s = alg.init(null as never)
    const rows: number[] = [...s.y]
    for (let k = 1; k <= per * (t.length - 1); k++) {
      s = alg.step(s, quiet)
      if (k % per === 0) rows.push(...s.y)
    }
    return fromStorage(Float64Array.from(rows), [t.length, 4], 'float64')
  },
}

describe('dynamics/celestial against scipy', () => {
  runFixture('dynamics/celestial', FNS, it)
})
