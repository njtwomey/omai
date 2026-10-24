import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { type ComplexValue, reshape, stack, type Tensor, tensor, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const parts = (h: ComplexValue): Value => stack([reshape(h.re, [-1]), reshape(h.im, [-1])], 0)
const sys = (num: Tensor, den: Tensor, form: unknown): S.Lti => {
  const g = S.tf(num, den)
  return form === 'zpk' ? S.toZpk(g) : form === 'ss' ? S.toSs(g) : g
}
const bodePart = (r: S.BodeResponse, part: unknown): Value => (part === 'w' ? r.w : part === 'mag' ? r.mag : r.phase)

const FNS: Readonly<Record<string, GoldenFn>> = {
  'freqresp-tf': ([n, d, w], p) => parts(S.freqresp(sys(n as Tensor, d as Tensor, p['form']), w).h),
  'freqresp-w': ([n, d], p) => S.freqresp(S.tf(n as Tensor, d as Tensor), undefined, { n: p['n'] as number }).w,
  'freqresp-h': ([n, d], p) => parts(S.freqresp(S.tf(n as Tensor, d as Tensor), undefined, { n: p['n'] as number }).h),
  bode: ([n, d], p) => bodePart(S.bode(S.tf(n as Tensor, d as Tensor), undefined, { n: p['n'] as number }), p['part']),
  dfreqresp: ([n, d, w], p) => parts(S.freqresp(S.tf(n as Tensor, d as Tensor, { dt: p['dt'] as number }), w).h),
  dbode: ([n, d, w], p) => bodePart(S.bode(S.tf(n as Tensor, d as Tensor, { dt: p['dt'] as number }), w), p['part']),
  'freqresp-ss': ([a, b, c, d, w]) => parts(S.freqresp(S.ss(a as Tensor, b as Tensor, c as Tensor, d), w).h),
  'freqresp-ss-d': ([a, b, c, d, w], p) =>
    parts(S.freqresp(S.ss(a as Tensor, b as Tensor, c as Tensor, d, { dt: p['dt'] as number }), w).h),
  'dcgain-ss': ([a, b, c, d]) => S.dcgain(S.ss(a as Tensor, b as Tensor, c as Tensor, d)),
  'dcgain-ss-d': ([a, b, c, d], p) =>
    S.dcgain(S.ss(a as Tensor, b as Tensor, c as Tensor, d, { dt: p['dt'] as number })),
  'dcgain-tf': ([n, d]) => S.dcgain(S.tf(n as Tensor, d as Tensor)),
  evalfr: ([n, d], p) =>
    parts(S.evalfr(S.tf(n as Tensor, d as Tensor), { re: p['re'] as number, im: p['im'] as number })),
  findfreqs: ([n, d], p) => S.findfreqs(n as Tensor, d as Tensor, p['n'] as number),
  nyquist: ([n, d], p) => tensor(S.nyquist(S.tf(n as Tensor, d as Tensor, { dt: p['dt'] as number })).count),
}

describe('systems/frequency against scipy.signal and python-control', () => {
  runFixture('systems/frequency', FNS, it)
})
