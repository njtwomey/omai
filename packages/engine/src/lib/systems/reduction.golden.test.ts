import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { type ComplexValue, reshape, stack, type Tensor, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const W = [0.1, 0.31622776601683794, 1, 3.1622776601683795, 10]
const parts = (h: ComplexValue): Value => stack([reshape(h.re, [-1]), reshape(h.im, [-1])], 0)
const sys = (xs: Tensor[], dt: unknown = 0): S.StateSpace =>
  S.ss(xs[0] as Tensor, xs[1] as Tensor, xs[2] as Tensor, xs[3], { dt: dt as number })

const FNS: Readonly<Record<string, GoldenFn>> = {
  hsvd: (xs) => S.hsvd(sys(xs)),
  balred: (xs, p) =>
    parts(S.freqresp(S.balred(sys(xs), p['order'] as number, { method: p['method'] as 'truncate' }), W).h),
}

describe('systems/reduction against python-control', () => {
  runFixture('systems/reduction', FNS, it)
})
