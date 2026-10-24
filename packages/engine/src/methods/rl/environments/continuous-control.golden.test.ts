import { describe, it } from 'vitest'
import { type Tensor, tensor, toArray, toFlat } from '../../../lib/tensor/index.ts'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import {
  type CartDoublePendulumOptions,
  cartDoublePendulumDerivatives,
  cartDoublePendulumDynamics,
  cartDoublePendulumParameters,
  doubleIntegratorMatrices,
  planarQuadrotorDerivatives,
  planarQuadrotorDynamics,
  planarQuadrotorParameters,
  pointMassDynamics,
  pointMassParameters,
  type Reacher2DOptions,
  reacher2dDerivatives,
  reacher2dDynamics,
  reacher2dParameters,
} from './continuous-control.ts'

const vec = (x: unknown): number[] => toFlat(decode(x) as Tensor)
const rows = (x: unknown): number[][] => toArray(decode(x) as Tensor) as number[][]

interface System {
  derivatives(s: readonly number[], u: readonly number[]): number[]
  step(s: readonly number[], u: readonly number[]): readonly number[]
}

function system(fn: string, options: Readonly<Record<string, unknown>>): System | undefined {
  if (fn.startsWith('cartDoublePendulum')) {
    // The fixtures' states have no time limit or track: widen the force limit so no clipping applies.
    const p = cartDoublePendulumParameters({ ...(options as CartDoublePendulumOptions), maxForce: 1e9 })
    return {
      derivatives: (s, u) => cartDoublePendulumDerivatives(s, u[0] ?? 0, p),
      step: (s, u) => cartDoublePendulumDynamics(s, u, p),
    }
  }
  if (fn.startsWith('reacher2d')) {
    const p = reacher2dParameters(options as Reacher2DOptions)
    return {
      derivatives: (s, u) => reacher2dDerivatives(s, u, p),
      step: (s, u) => reacher2dDynamics([...s, 0, 0], u, p).slice(0, 4),
    }
  }
  if (fn.startsWith('planarQuadrotor')) {
    const p = planarQuadrotorParameters()
    return {
      derivatives: (s, u) => planarQuadrotorDerivatives(s, u, p),
      step: (s, u) => planarQuadrotorDynamics([...s, 0, 0], u, p).slice(0, 6),
    }
  }
  return undefined
}

describe('continuous-control dynamics against scipy, sympy and numpy references', () => {
  for (const c of loadFixture('methods/rl/environments/continuous_control').cases) {
    it(c.name, () => {
      const prm = c.params as Readonly<Record<string, unknown>>
      let got: Tensor
      if (c.fn === 'doubleIntegratorMatrices') {
        const { A, B } = doubleIntegratorMatrices({ mass: prm['mass'] as number, dt: prm['dt'] as number })
        got = tensor(A.map((r, i) => [...r, ...(B[i] ?? [])]))
      } else if (c.fn === 'pointMassStep') {
        const p = pointMassParameters({
          mass: prm['mass'] as number,
          damping: prm['damping'] as number,
          gain: prm['gain'] as number,
          dt: prm['dt'] as number,
        })
        got = tensor(pointMassDynamics([...vec(c.inputs[0]), 0, 0], vec(c.inputs[1]), p).slice(0, 4))
      } else {
        const sys = system(c.fn, (prm['options'] ?? {}) as Readonly<Record<string, unknown>>)
        if (sys === undefined) throw new Error(`no system for ${c.fn}`)
        if (c.fn.endsWith('Derivatives')) got = tensor(sys.derivatives(vec(c.inputs[0]), vec(c.inputs[1])))
        else if (c.fn.endsWith('Step')) got = tensor([...sys.step(vec(c.inputs[0]), vec(c.inputs[1]))])
        else {
          let s: readonly number[] = vec(c.inputs[0])
          const out: number[][] = []
          for (const u of rows(c.inputs[1])) {
            s = sys.step(s, u)
            out.push([...s])
          }
          got = tensor(out)
        }
      }
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})
