import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { isMdmlError } from '../base/index.ts'
import { concat, reshape, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as S from './index.ts'

const w = [0.2, 1, 3]
const response = (sys: S.Lti): number[] => {
  const h = S.freqresp(sys, w).h
  return [...toFlat(h.re as Tensor), ...toFlat(h.im as Tensor)]
}
const close = (a: readonly number[], b: readonly number[], tol = 1e-10): void => {
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries())
    expect(Math.abs(v - (b[i] ?? Number.NaN))).toBeLessThan(tol * Math.max(1, Math.abs(v)))
}

const g1 = S.tf([1, 2], [1, 3, 2])
const g2 = S.tf([3], [1, 0.5])

describe('interconnections', () => {
  it('agree between transfer-function and state-space forms', () => {
    close(response(S.series(g1, g2)), response(S.series(S.toSs(g1), S.toSs(g2))))
    close(response(S.parallel(g1, g2)), response(S.parallel(S.toSs(g1), S.toSs(g2))))
    close(response(S.feedback(g1, g2)), response(S.feedback(S.toSs(g1), S.toSs(g2))))
    close(response(S.feedback(g1, g2, 1)), response(S.feedback(S.toSs(g1), S.toSs(g2), 1)))
  })

  it('keep the kind: zero-pole-gain stays zero-pole-gain, a gain is a static system', () => {
    const z = S.series(S.zpk([-1], [-2], 1), S.zpk([], [-3], 2))
    expect(z.kind).toBe('zpk')
    expect([...z.poles.re].sort((x, y) => x - y)).toEqual([-3, -2])
    expect(toFlat(S.series(g1, 2).num as Tensor)).toEqual([2, 4])
    expect(S.feedback(S.toSs(g1), 0.5).kind).toBe('ss')
    // Two-output transfer functions in parallel go through state space and come back as one.
    const simo = S.parallel(
      S.tf([[1], [2]], [1, 1]),
      S.tf(
        [
          [0, 1],
          [1, 0],
        ],
        [1, 2],
      ),
    )
    expect(simo.kind).toBe('tf')
    expect(simo.num.shape).toEqual([2, 3])
    expect(S.feedback(S.tf([[1], [2]], [1, 1]), S.ss([[-1]], [[1, 0]], [[1]])).kind).toBe('ss')
  })

  it('append places systems side by side and connect closes loops', () => {
    const both = S.append(g1, g2)
    expect(S.dimensions(both)).toEqual({ inputs: 2, outputs: 2, states: 3 })
    const loop = S.connect(
      both,
      [
        [2, 1],
        [1, -2],
      ],
      [1],
      [2],
    )
    close(response(loop), response(S.series(S.feedback(g1, g2), g2)))
  })

  it('reject mismatched sample times, shapes and ill-posed loops', () => {
    expect(() => S.series(g1, S.tf([1], [1, 0.5], { dt: 0.1 }))).toThrow()
    expect(() => S.series(g1)).toThrow()
    expect(() => S.parallel(S.append(g1, g2), g1)).toThrow()
    expect(() => S.connect(S.append(g1, g2), [[3, 1]], [1], [1])).toThrow()
    try {
      S.feedback(S.ss([[-1]], [[1]], [[1]], [[1]]), 1, 1)
      expect.unreachable()
    } catch (e) {
      expect(isMdmlError(e, 'singular')).toBe(true)
    }
  })
})

describe('interconnections are differentiable', () => {
  it('in transfer-function coefficients', () => {
    const f = (n1: Value, d2: Value): Value => {
      const r = S.feedback(S.tf(n1, [1, 3, 2]), S.tf([3], d2))
      return concat([r.num, r.den], 0)
    }
    expect(checkPrimitive(f, [tensor([1, 2]), tensor([1, 0.5])], { vmap: false }).failures).toEqual([])
  })

  it('in state-space matrices', () => {
    const b2 = tensor([[1], [0.5]])
    const f = (a: Value, d: Value): Value => {
      const r = S.feedback(S.ss(a, b2, [[1, 0]], d), S.ss([[-2]], [[1]], [[1]], [[0.3]]))
      return concat([reshape(r.a, [-1]), reshape(r.b, [-1]), reshape(r.c, [-1]), reshape(r.d, [-1])], 0)
    }
    const a = tensor([
      [-1, 0.4],
      [0.2, -2],
    ])
    expect(checkPrimitive(f, [a, tensor([[0.1]])], { vmap: false }).failures).toEqual([])
  })
})
