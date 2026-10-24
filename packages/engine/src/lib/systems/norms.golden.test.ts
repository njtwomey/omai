import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import type { Tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const sys = (xs: Tensor[], dt: unknown = 0): S.StateSpace =>
  S.ss(xs[0] as Tensor, xs[1] as Tensor, xs[2] as Tensor, xs[3], { dt: dt as number })

const FNS: Readonly<Record<string, GoldenFn>> = {
  h2: (xs, p) => S.h2Norm(sys(xs, p['dt'])),
  hinf: (xs, p) => S.hinfNorm(sys(xs, p['dt'])).norm,
}

describe('systems/norms against python-control', () => {
  runFixture('systems/norms', FNS, it)
})
