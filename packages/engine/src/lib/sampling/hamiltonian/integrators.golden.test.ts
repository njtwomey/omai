import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, tensor, type Value } from '../../tensor/index.ts'
import { numericTarget } from '../target.ts'
import {
  euclideanMetric,
  hamiltonianTrajectory,
  type IntegratorScheme,
  integrator,
  integratorState,
} from './integrators.ts'

type Params = Readonly<Record<string, unknown>>

// The density of the oracle: x0 ~ N(0, 4), x1 | x0 ~ N(x0^2 / 2, 1).
const banana = numericTarget((x) => -0.5 * ((x[0] ?? 0) ** 2 / 4 + ((x[1] ?? 0) - 0.5 * (x[0] ?? 0) ** 2) ** 2), 2, {
  grad: (x) => {
    const a = x[0] ?? 0
    const r = (x[1] ?? 0) - 0.5 * a * a
    return [-a / 4 + r * a, -r]
  },
})

const rows = (t: Tensor): number[][] => {
  const n = t.shape[0] ?? 0
  const m = t.shape[1] ?? 0
  return Array.from({ length: n }, (_, i) => Array.from(t.data.slice(i * m, (i + 1) * m) as ArrayLike<number>))
}

const CALLS: Readonly<Record<string, (xs: readonly Tensor[], p: Params) => Value>> = {
  trajectory: ([x0, p0, imm], p) => {
    const metric = euclideanMetric(imm as Tensor, 2)
    const step = integrator(banana, metric, p['scheme'] as IntegratorScheme)
    const states = hamiltonianTrajectory(
      step,
      integratorState(banana, (x0 as Tensor).data as ArrayLike<number>, (p0 as Tensor).data as ArrayLike<number>),
      p['stepSize'] as number,
      p['numSteps'] as number,
    )
    return tensor(states.map((s) => [...s.position, ...s.momentum]))
  },
  kineticEnergy: ([ps, imm]) => {
    const metric = euclideanMetric(imm as Tensor, 2)
    return tensor(rows(ps as Tensor).map((q) => metric.kineticEnergy(q)))
  },
  isTurning: ([triples, imm]) => {
    const metric = euclideanMetric(imm as Tensor, 2)
    const t = triples as Tensor
    const n = t.shape[0] ?? 0
    const d = t.data as ArrayLike<number>
    const at = (i: number, j: number): number[] => [d[i * 6 + j * 2] ?? 0, d[i * 6 + j * 2 + 1] ?? 0]
    return tensor(
      Array.from({ length: n }, (_, i) => {
        const [l, r, mid] = [at(i, 0), at(i, 1), at(i, 2)]
        const sum = mid.map((v, k) => v + (l[k] ?? 0) + (r[k] ?? 0))
        return metric.isTurning(l, r, sum) ? 1 : 0
      }),
    )
  },
}

describe('sampling/hamiltonian/integrators against BlackJAX', () => {
  for (const c of loadFixture('sampling/hamiltonian/integrators').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(
        call(
          c.inputs.map((x) => decode(x) as Tensor),
          c.params,
        ),
        c.output,
        c.rtol,
        c.atol,
        c.name,
      )
    })
  }
})
