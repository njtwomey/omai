import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { fromStorage, type Tensor, toFlat, type Value } from '../tensor/index.ts'
import { doublePendulumField, doublePendulumFlipTime, doublePendulumLagrangian, lagrangianField } from './mechanics.ts'
import { rosslerField, thomasField } from './named-systems.ts'
import { type FlowState, flowAlgorithm } from './ode/flow.ts'
import type { OdeVectorField } from './ode/problem.ts'
import { kuramotoField } from './synchronisation.ts'

const quiet = { log: () => undefined, emit: () => undefined, wants: () => false } as never

/** RK4 states at the given times (multiples of dt), one row each. */
function states(field: OdeVectorField, y0: Tensor, times: Tensor, dt: number): Value {
  const t = toFlat(times)
  const per = Math.round(((t[1] as number) - (t[0] as number)) / dt)
  const total = per * (t.length - 1)
  const alg = flowAlgorithm(field, y0, { dt, steps: total })
  let s: FlowState = alg.init(null as never)
  const rows: number[] = [...s.y]
  for (let k = 1; k <= total; k++) {
    s = alg.step(s, quiet)
    if (k % per === 0) rows.push(...s.y)
  }
  return fromStorage(Float64Array.from(rows), [t.length, s.y.length], 'float64')
}

const pendulum = (p: Readonly<Record<string, unknown>>): Record<'m1' | 'm2' | 'l1' | 'l2' | 'g', number> => ({
  m1: p['m1'] as number,
  m2: p['m2'] as number,
  l1: p['l1'] as number,
  l2: p['l2'] as number,
  g: p['g'] as number,
})

const FNS: Readonly<Record<string, GoldenFn>> = {
  rossler: ([y0, t], p) => states(rosslerField(), y0 as Tensor, t as Tensor, p['dt'] as number),
  thomas: ([y0, t], p) => states(thomasField(), y0 as Tensor, t as Tensor, p['dt'] as number),
  'double-pendulum': ([y0, t], p) =>
    states(doublePendulumField(pendulum(p)), y0 as Tensor, t as Tensor, p['dt'] as number),
  'double-pendulum-lagrangian': ([y0, t], p) =>
    states(lagrangianField(doublePendulumLagrangian(pendulum(p)), 2), y0 as Tensor, t as Tensor, p['dt'] as number),
  kuramoto: ([omega, theta0, t], p) =>
    states(
      kuramotoField({ omega: toFlat(omega as Tensor), K: p['K'] as number }),
      theta0 as Tensor,
      t as Tensor,
      p['dt'] as number,
    ),
  'flip-time': ([starts]) => {
    const s = starts as Tensor
    const flat = toFlat(s)
    const rows = (s.shape[0] ?? 0) as number
    const out = Array.from({ length: rows }, (_, i) =>
      doublePendulumFlipTime(
        flat[2 * i] as number,
        flat[2 * i + 1] as number,
        { m1: 1, m2: 1, l1: 1, l2: 1, g: 9.81 },
        { tMax: 10, dt: 1 / 1000 },
      ),
    )
    return fromStorage(Float64Array.from(out), [rows], 'float64')
  },
}

describe('dynamics/mechanics against scipy.integrate', { timeout: 60_000 }, () => {
  runFixture('dynamics/mechanics', FNS, it)
})
