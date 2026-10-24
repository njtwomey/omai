import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { eigvals, inv } from '../numerics/linalg/index.ts'
import {
  add,
  item,
  matmul,
  reshape,
  sliceAxis,
  sub,
  type Tensor,
  tensor,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import * as S from './index.ts'

const maxAbs = (v: Value): number => toFlat(v as Tensor).reduce((acc, x) => Math.max(acc, Math.abs(x)), 0)

const stableArb = fc
  .array(
    fc.integer({ min: -100, max: 100 }).map((k) => k / 100),
    { minLength: 15, maxLength: 15 },
  )
  .map((v) => {
    const a = tensor([
      [(v[0] ?? 0) - 2.5, v[1] ?? 0, v[2] ?? 0],
      [v[3] ?? 0, (v[4] ?? 0) - 2.5, v[5] ?? 0],
      [v[6] ?? 0, v[7] ?? 0, (v[8] ?? 0) - 2.5],
    ])
    const b = tensor([[v[9] ?? 0], [1 + (v[10] ?? 0)], [v[11] ?? 0]])
    const c = tensor([[1 + (v[12] ?? 0), v[13] ?? 0, v[14] ?? 0]])
    return { a, b, c }
  })
  .filter(({ a }) => toFlat(eigvals(a).re).every((x) => x < -0.1))

describe('controllability, observability and Gramians', () => {
  it('Gramians solve their Lyapunov equations (continuous and discrete)', () => {
    fc.assert(
      fc.property(stableArb, ({ a, b, c }) => {
        const s = S.ss(a, b, c)
        const wc = S.gram(s, 'c')
        const residual = add(add(matmul(a, wc), matmul(wc, transpose(a))), matmul(b, transpose(b)))
        expect(maxAbs(residual)).toBeLessThan(1e-9)
        const wo = S.gram(s, 'o')
        const ro = add(add(matmul(transpose(a), wo), matmul(wo, a)), matmul(transpose(c), c))
        expect(maxAbs(ro)).toBeLessThan(1e-9)
        const d = S.cont2discrete(s, 0.2)
        const wd = S.gram(d, 'c')
        const rd = add(sub(matmul(d.a, matmul(wd, transpose(d.a))), wd), matmul(d.b, transpose(d.b)))
        expect(maxAbs(rd)).toBeLessThan(1e-9)
        const u = S.gram(s, 'cf')
        expect(maxAbs(sub(matmul(transpose(u), u), wc))).toBeLessThan(1e-9)
      }),
      { numRuns: 20 },
    )
  })

  it('detect uncontrollable and unobservable modes', () => {
    const s = S.ss(
      [
        [-1, 0],
        [0, -2],
      ],
      [[1], [0]],
      [[1, 1]],
    )
    expect(S.isControllable(s)).toBe(false)
    expect(S.isObservable(s)).toBe(true)
    expect(S.isControllable(S.tf([1], [1, 3, 2]))).toBe(true)
    expect(
      toFlat(
        S.ctrb(
          [
            [-1, 0],
            [0, -2],
          ],
          [[1], [0]],
        ) as Tensor,
      ),
    ).toEqual([1, -1, 0, 0])
    expect(() => S.gram(S.ss([[1]], [[1]], [[1]]), 'c')).toThrow()
  })

  it('the Kalman decomposition keeps the transfer function in its controllable-observable block', () => {
    const t = tensor([
      [1, 0.5, 0.2, 0],
      [0, 1, -0.3, 0.1],
      [0.4, 0, 1, 0.2],
      [0, 0.3, 0, 1],
    ])
    const a0 = tensor([
      [-1, 0, 0, 0],
      [0, -2, 0, 0],
      [0, 0, -3, 0],
      [0, 0, 0, -4],
    ])
    // Modes: -1 controllable and observable, -2 controllable only, -3 observable only, -4 neither; then hidden by T.
    const s = S.ss(
      matmul(t, matmul(a0, inv(t))),
      matmul(t, tensor([[1], [1], [0], [0]])),
      matmul(tensor([[1, 0, 1, 0]]), inv(t)),
    )
    const k = S.kalmanDecomposition(s)
    expect(k.dims).toEqual({ co: 1, cn: 1, uo: 1, un: 1 })
    const co = S.ss(
      sliceAxis(sliceAxis(k.sys.a, 0, 0, 1), 1, 0, 1),
      sliceAxis(k.sys.b, 0, 0, 1),
      sliceAxis(k.sys.c, 1, 0, 1),
    )
    const w = [0.3, 2]
    const h1 = S.freqresp(co, w).h
    const h2 = S.freqresp(s, w).h
    for (const [i, v] of toFlat(h1.re as Tensor).entries()) expect(v).toBeCloseTo(toFlat(h2.re as Tensor)[i] ?? 0, 9)
    // Structure: B has no uncontrollable rows, C no unobservable columns.
    expect(maxAbs(sliceAxis(k.sys.b, 0, 2, 4))).toBeLessThan(1e-10)
    expect(maxAbs(sliceAxis(k.sys.c, 1, 1, 2))).toBeLessThan(1e-10)
    expect(maxAbs(sliceAxis(k.sys.c, 1, 3, 4))).toBeLessThan(1e-10)
  })

  it('Gramians and ctrb are differentiable', () => {
    const a = tensor([
      [-1, 0.3],
      [0.2, -2],
    ])
    const b = tensor([[1], [0.5]])
    const f = (av: Value, bv: Value): Value => reshape(S.gram(S.ss(av, bv, [[1, 0]]), 'c'), [-1])
    expect(checkPrimitive(f, [a, b], { vmap: false }).failures).toEqual([])
    const g = (av: Value): Value => S.ctrb(av, b)
    expect(checkPrimitive(g, [a], { vmap: false }).failures).toEqual([])
    expect(item(S.h2Norm(S.ss(a, b, [[1, 0]])))).toBeGreaterThan(0)
  })
})
