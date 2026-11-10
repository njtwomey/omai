import { describe, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { exponential, normal } from '../dists/index.ts'
import { item } from '../tensor/index.ts'
import { logJoint } from './density.ts'
import { lift } from './lift.ts'
import { type Model, observe, param } from './primitives.ts'

const program: Model = () => {
  const s = param('s', 1, { support: { kind: 'positive' } })
  observe('y', normal(0, s), 0.5)
}

describe('probprog/lift against NumPyro and Pyro', () => {
  for (const c of loadFixture('probprog/lift').cases) {
    it(c.name, () => {
      const lifted = lift(program, { s: exponential(c.params['rate'] as number) })
      const got = item(logJoint(lifted, undefined, { s: c.params['s'] as number }).logJoint)
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})
