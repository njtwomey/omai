import { type Custom, describeDistributions } from '../../../testing/distributions.ts'
import type { Tensor } from '../../tensor/index.ts'
import * as B from './index.ts'

const BIJECTORS: Readonly<Record<string, () => B.Bijector>> = {
  exp: B.exp,
  softplus: B.softplus,
  sigmoid: B.sigmoid,
  tanh: B.tanh,
  affine: () => B.affine(0.5, -2),
  power: () => B.power(1.7),
  stickBreaking: B.stickBreaking,
  correlationCholesky: B.correlationCholesky,
}
const METHODS = ['forward', 'inverse', 'forwardLogDetJacobian'] as const

const custom: Record<string, Custom> = {}
for (const [name, make] of Object.entries(BIJECTORS))
  for (const method of METHODS) custom[`${name}.${method}`] = (xs) => make()[method](xs[0] as Tensor)

for (const file of ['elementwise', 'structured']) describeDistributions(`dists/bijectors/${file}`, {}, custom)
