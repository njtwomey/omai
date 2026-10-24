import { goldenSuite } from '../../testing/suite.ts'
import { key } from '../random/index.ts'
import {
  add,
  concat,
  cos,
  mul,
  neg,
  reshape,
  sin,
  sliceAxis,
  stack,
  type Tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Sde, type SdeMethod, sdeint } from './sde.ts'

// The system of oracle/cases/stochastic/sde.py, on states [paths, 2].
const col = (y: Value, i: number): Value => sliceAxis(y, 1, i, i + 1)
const drift = (t: number, y: Value): Value =>
  concat(
    [add(neg(col(y, 0)), sin(col(y, 1))), add(add(mul(col(y, 1), -0.5), mul(mul(col(y, 0), col(y, 1)), 0.1)), 0.2 * t)],
    1,
  )
const general = (_t: number, y: Value): Value => {
  const zero = mul(col(y, 0), 0)
  const row0 = concat([mul(col(y, 0), 0.3), add(zero, 0.1)], 1)
  const row1 = concat([add(zero, 0.05), mul(cos(col(y, 1)), 0.2)], 1)
  return stack([row0, row1], 1)
}
const diagonal = (_t: number, y: Value): Value => concat([mul(col(y, 0), 0.3), mul(cos(col(y, 1)), 0.2)], 1)

goldenSuite('stochastic/sde against sdeint with the same Brownian increments', 'stochastic/sde', {
  sdeint: ([y0, dw, tspan], p) => {
    const noise = p['noise'] as 'general' | 'diagonal'
    const method = p['method'] as SdeMethod
    const sde: Sde = {
      drift,
      diffusion: noise === 'general' ? general : diagonal,
      noise,
      ...(noise === 'general' ? { noiseDim: 2 } : {}),
      calculus: method === 'heun' ? 'stratonovich' : 'ito',
    }
    const steps = dw.shape[0] ?? 0
    // the key is unused: the increments are given
    const paths = sdeint(key(0), sde, reshape(y0, [1, 2]), toFlat(tspan as Tensor), {
      method,
      increments: reshape(dw, [1, steps, 2]),
    })
    return reshape(paths.values, [steps + 1, 2])
  },
})
