import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { reshape, type Tensor, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const pick = (r: S.TimeResponse, part: unknown): Value => (part === 'x' ? r.x : r.y)
// scipy's dlsim, dstep and dimpulse keep the output axis for one output.
const col = (y: Value): Value => (y.shape.length === 1 ? reshape(y, [-1, 1]) : y)

const FNS: Readonly<Record<string, GoldenFn>> = {
  lsim: ([a, b, c, d, u, t, x0], p) =>
    pick(
      S.lsim(S.ss(a as Tensor, b as Tensor, c as Tensor, d), u as Tensor, t, {
        x0: x0 as Tensor,
        interp: p['interp'] as boolean,
      }),
      p['part'],
    ),
  'lsim-tf': ([num, den, u, t, x0], p) =>
    pick(S.lsim(S.tf(num as Tensor, den as Tensor), u as Tensor, t, { x0: x0 as Tensor }), p['part']),
  dlsim: ([a, b, c, d, u, x0], p) => {
    const r = S.lsim(
      S.ss(a as Tensor, b as Tensor, c as Tensor, d, { dt: p['dt'] as number }),
      u as Tensor,
      undefined,
      { x0: x0 as Tensor },
    )
    return p['part'] === 'x' ? r.x : col(r.y)
  },
  'dlsim-t': ([a, b, c, d, u, t], p) => {
    const r = S.lsim(S.ss(a as Tensor, b as Tensor, c as Tensor, d, { dt: p['dt'] as number }), u as Tensor, t)
    return col(r.y)
  },
  'step-t': ([num, den]) => S.step(S.tf(num as Tensor, den as Tensor)).t,
  step: ([num, den]) => S.step(S.tf(num as Tensor, den as Tensor)).y,
  impulse: ([num, den], p) => S.impulse(S.tf(num as Tensor, den as Tensor), { n: p['n'] as number }).y,
  'step-ss': ([a, b, c, d, x0, t]) =>
    S.step(S.ss(a as Tensor, b as Tensor, c as Tensor, d), { x0: x0 as Tensor, t: t as Tensor }).y,
  'impulse-ss': ([a, b, c, d, x0, t]) =>
    S.impulse(S.ss(a as Tensor, b as Tensor, c as Tensor, d), { x0: x0 as Tensor, t: t as Tensor }).y,
  dstep: ([a, b, c, d], p) => {
    const y = S.step(S.ss(a as Tensor, b as Tensor, c as Tensor, d, { dt: p['dt'] as number }), {
      n: p['n'] as number,
      input: p['input'] as number,
    }).y
    return col(y)
  },
  dimpulse: ([a, b, c, d], p) => {
    const y = S.impulse(S.ss(a as Tensor, b as Tensor, c as Tensor, d, { dt: p['dt'] as number }), {
      n: p['n'] as number,
      input: p['input'] as number,
    }).y
    return col(y)
  },
  initial: ([a, b, c, d, x0, t]) =>
    S.initial(S.ss(a as Tensor, b as Tensor, c as Tensor, d), x0 as Tensor, { t: t as Tensor }).y,
  'initial-d': ([a, b, c, d, x0], p) =>
    S.initial(S.ss(a as Tensor, b as Tensor, c as Tensor, d, { dt: p['dt'] as number }), x0 as Tensor, {
      n: p['n'] as number,
    }).y,
}

describe('systems/responses against scipy.signal and python-control', () => {
  runFixture('systems/responses', FNS, it)
})
