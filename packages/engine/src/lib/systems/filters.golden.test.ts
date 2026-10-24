import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { type ComplexValue, reshape, stack, type Tensor, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const parts = (h: ComplexValue): Value => stack([reshape(h.re, [-1]), reshape(h.im, [-1])], 0)
const freqzOptions = (p: Readonly<Record<string, unknown>>): S.FreqzOptions => ({
  ...(p['whole'] === undefined ? {} : { whole: p['whole'] as boolean }),
  ...(p['fs'] === undefined ? {} : { fs: p['fs'] as number }),
  ...(p['include_nyquist'] === undefined ? {} : { includeNyquist: p['include_nyquist'] as boolean }),
})

const FNS: Readonly<Record<string, GoldenFn>> = {
  'freqs-w': ([n, d], p) => S.freqs(n as Tensor, d as Tensor, p['n'] as number).w,
  'freqs-h': ([n, d], p) => parts(S.freqs(n as Tensor, d as Tensor, p['n'] as number).h),
  'freqz-w': ([b, a], p) => S.freqz(b as Tensor, a as Tensor, p['worN'] as number, freqzOptions(p)).w,
  'freqz-h': ([b, a], p) => parts(S.freqz(b as Tensor, a as Tensor, p['worN'] as number, freqzOptions(p)).h),
  'freqz-given': ([b, a, w]) => parts(S.freqz(b as Tensor, a as Tensor, w as Tensor).h),
}

describe('systems/filters against scipy.signal', () => {
  runFixture('systems/filters', FNS, it)
})
