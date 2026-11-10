import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { transformed } from '../dists/compose/index.ts'
import { bijectors, halfCauchy, normal, vonMises } from '../dists/index.ts'
import { item, type Tensor } from '../tensor/index.ts'
import { logJoint } from './density.ts'
import { type Model, observe, plate, sample } from './primitives.ts'
import { circular, locScale, reparam, transformReparam } from './reparam.ts'

const SIGMA = [15, 10, 16, 11]
const SCHOOLS = [28, 8, -3, 7]

const theta: Model = () => {
  const t = sample('theta', normal(1.5, 2.5))
  observe('y', normal(t, 1), 0.7)
}
const schools: Model = () => {
  const mu = sample('mu', normal(0, 5))
  const tau = sample('tau', halfCauchy(5))
  plate('J', 4, () => {
    const t = sample('theta', normal(mu, tau))
    observe('y', normal(t, SIGMA), SCHOOLS)
  })
}
const lognormal: Model = () => {
  const s = sample('s', transformed(normal(0.2, 0.7), bijectors.exp()))
  observe('y', normal(s, 1), 1.3)
}
const angle: Model = () => {
  const phi = sample('phi', vonMises(0.3, 2))
  observe('y', normal(phi, 1), 0.1)
}

const lj = (m: Model, v: Record<string, number | Tensor>): number => item(logJoint(m, undefined, v).logJoint)

describe('probprog/reparam against NumPyro and Pyro', () => {
  for (const c of loadFixture('probprog/reparam').cases) {
    it(c.name, () => {
      let got: number
      switch (c.fn) {
        case 'locScaleTheta':
          got = lj(reparam(theta, { theta: locScale(c.params['centered'] as number) }), {
            theta_decentered: c.params['z'] as number,
          })
          break
        case 'locScaleSchools': {
          const input = c.inputs[0] as Record<string, unknown>
          got = lj(reparam(schools, { theta: locScale(c.params['centered'] as number) }), {
            mu: decode(input['mu']) as Tensor,
            tau: decode(input['tau']) as Tensor,
            theta_decentered: decode(input['theta_decentered']) as Tensor,
          })
          break
        }
        case 'transformLognormal':
          got = lj(reparam(lognormal, { s: transformReparam() }), { s_base: c.params['z'] as number })
          break
        case 'circular':
          got = lj(reparam(angle, { phi: circular() }), { phi_unwrapped: c.params['value'] as number })
          break
        default:
          throw new Error(`no function ${c.fn}`)
      }
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})
