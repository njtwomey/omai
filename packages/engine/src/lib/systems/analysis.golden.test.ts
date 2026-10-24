import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import type { Tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const sys = (xs: Tensor[], dt: unknown = 0): S.StateSpace =>
  S.ss(xs[0] as Tensor, xs[1] as Tensor, xs[2] as Tensor, xs[3], { dt: dt as number })

const FNS: Readonly<Record<string, GoldenFn>> = {
  ctrb: ([a, b]) => S.ctrb(a as Tensor, b as Tensor),
  obsv: ([a, c]) => S.obsv(a as Tensor, c as Tensor),
  gram: (xs, p) => S.gram(sys(xs, p['dt']), p['type'] as S.GramianType),
}

describe('systems/analysis against python-control', () => {
  runFixture('systems/analysis', FNS, it)
})
