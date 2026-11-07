import { describe, expect, it } from 'vitest'
import { checkPrimitive, jacfwd, jvp } from '../../autodiff/index.ts'
import { slogdet } from '../../numerics/linalg/index.ts'
import {
  concat,
  item,
  reshape,
  sliceAxis,
  sum,
  type Tensor,
  take,
  tanh,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { exponential, normal } from '../index.ts'
import * as B from './index.ts'

const closeArrays = (a: readonly number[], b: readonly number[], digits = 12): void => {
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries()) expect(v).toBeCloseTo(b[i] ?? Number.NaN, digits)
}

describe('elementwise bijectors', () => {
  const x = tensor([-1.3, -0.2, 0.4, 2.1])
  const pos = tensor([0.3, 1, 2.5])
  const cases: readonly (readonly [string, B.Bijector, Tensor])[] = [
    ['identity', B.identity(), x],
    ['exp', B.exp(), x],
    ['softplus', B.softplus(), x],
    ['sigmoid', B.sigmoid(), x],
    ['tanh', B.tanh(), x],
    ['affine', B.affine(0.5, -2), x],
    ['power', B.power(1.7), pos],
    ['normalCdf', B.normalCdf(), x],
  ]
  for (const [name, b, at] of cases) {
    it(`${name}: inverts, and its log-determinant is log |f'|`, () => {
      closeArrays(toFlat(b.inverse(b.forward(at))), toFlat(at), 10)
      const derivative = toFlat(jvp((v: Value) => b.forward(v), at, tensor(toFlat(at).map(() => 1))).tangent)
      closeArrays(
        toFlat(b.forwardLogDetJacobian(at)),
        derivative.map((d) => Math.log(Math.abs(d))),
        10,
      )
      closeArrays(
        toFlat(b.inverseLogDetJacobian(b.forward(at))),
        toFlat(b.forwardLogDetJacobian(at)).map((v) => -v),
        10,
      )
    })
  }

  it('affine knows the sign of its scale and chains compose', () => {
    expect(B.affine(0, -2).sign).toBe(-1)
    const c = B.chain([B.exp(), B.affine(1, 2), B.invert(B.exp())])
    // log(1 + 2 e^x), with log-derivative log(2 e^x / (1 + 2 e^x)).
    expect(item(c.forward(0.5))).toBeCloseTo(Math.log(1 + 2 * Math.exp(0.5)), 14)
    expect(item(c.inverse(c.forward(0.5)))).toBeCloseTo(0.5, 14)
    expect(item(c.forwardLogDetJacobian(0.5))).toBeCloseTo(Math.log((2 * Math.exp(0.5)) / (1 + 2 * Math.exp(0.5))), 14)
    expect(c.sign).toBe(1)
  })
})

// log |det J| of a map between flat coordinate vectors, by forward-mode autodiff.
const logAbsDet = (f: (v: Value) => Value, x: Tensor): number => {
  const j = jacfwd(f)(x) as Tensor
  return item(slogdet(j).logabsdet)
}

describe('structured bijectors', () => {
  it('ordered maps onto increasing vectors with the right log-determinant', () => {
    const b = B.ordered()
    const x = tensor([0.4, -1, 0.3, 2])
    const y = b.forward(x)
    expect(toFlat(y).every((v, i, a) => i === 0 || v > (a[i - 1] ?? 0))).toBe(true)
    closeArrays(toFlat(b.inverse(y)), toFlat(x))
    expect(item(b.forwardLogDetJacobian(x))).toBeCloseTo(
      logAbsDet((v) => b.forward(v), x),
      12,
    )
  })

  it('stick-breaking maps onto the simplex; its log-determinant is that of the first K - 1 coordinates', () => {
    const b = B.stickBreaking()
    const x = tensor([0.4, -1, 0.3])
    const y = b.forward(x)
    expect(item(sum(y))).toBeCloseTo(1, 14)
    closeArrays(toFlat(b.inverse(y)), toFlat(x))
    const head = (v: Value): Value => sliceAxis(b.forward(v), -1, 0, 3)
    expect(item(b.forwardLogDetJacobian(x))).toBeCloseTo(logAbsDet(head, x), 12)
    expect(b.forwardShape([5, 3])).toEqual([5, 4])
  })

  it('correlation Cholesky factors have unit rows; the log-determinant is over the strict lower triangle', () => {
    const b = B.correlationCholesky()
    const x = tensor([0.4, -1, 0.3, 0.8, -0.2, 1.1])
    const l = b.forward(x) as Tensor
    expect(l.shape).toEqual([4, 4])
    const rows = toFlat(sum(reshape(tensor(toFlat(l).map((v) => v * v)), [4, 4]), -1))
    closeArrays(rows, [1, 1, 1, 1], 14)
    closeArrays(toFlat(b.inverse(l)), toFlat(x), 10)
    const strict = [4, 8, 9, 12, 13, 14]
    const lower = (v: Value): Value => take(reshape(b.forward(v), [16]), strict, 0)
    expect(item(b.forwardLogDetJacobian(x))).toBeCloseTo(logAbsDet(lower, x), 10)
  })

  // Unconstrained values of any size, as an optimiser or a leapfrog step reaches: the forward map stays finite and the
  // inverse recovers them (the old forms went NaN from |u| = 19 and lost the value from 15).
  it('stick-breaking and correlation Cholesky round-trip at large unconstrained values', () => {
    for (const [name, bijector] of [
      ['stickBreaking', B.stickBreaking()],
      ['correlationCholesky', B.correlationCholesky()],
    ] as const) {
      for (const u of [0, 5, 15, 20, 25, 35, -20, -35]) {
        const x = tensor([u, -u, u])
        const y = bijector.forward(x)
        expect(toFlat(y).every(Number.isFinite), `${name} forward at ${u}`).toBe(true)
        expect(Number.isFinite(item(bijector.forwardLogDetJacobian(x))), `${name} log-determinant at ${u}`).toBe(true)
        closeArrays(toFlat(bijector.inverse(y)), [u, -u, u], 6)
      }
    }
  })

  it('lower Cholesky and its outer product map onto positive-definite matrices', () => {
    const lc = B.lowerCholesky()
    const x = tensor([0.2, -0.5, 0.3, 1.2, 0.1, -0.4])
    const l = lc.forward(x)
    closeArrays(toFlat(lc.inverse(l)), toFlat(x))
    const lowerIdx = [0, 3, 4, 6, 7, 8]
    expect(item(lc.forwardLogDetJacobian(x))).toBeCloseTo(
      logAbsDet((v) => take(reshape(lc.forward(v), [9]), lowerIdx, 0), x),
      12,
    )
    const pd = B.chain([lc, B.choleskyOuterProduct()])
    expect(item(pd.forwardLogDetJacobian(x))).toBeCloseTo(
      logAbsDet((v) => take(reshape(pd.forward(v), [9]), lowerIdx, 0), x),
      10,
    )
    const corr = B.chain([B.correlationCholesky(), B.choleskyOuterProduct('correlation')])
    const z = tensor([0.3, -0.6, 0.9])
    expect(item(corr.forwardLogDetJacobian(z))).toBeCloseTo(
      logAbsDet((v) => take(reshape(corr.forward(v), [9]), [3, 6, 7], 0), z),
      10,
    )
  })

  it('coupling, permutation and autoregressive layers invert with consistent log-determinants', () => {
    const conditioner = (xa: Value): B.ShiftLogScale => ({
      shift: concat([tanh(xa), sum(xa, -1, true)], -1),
      logScale: concat([sum(xa, -1, true), tanh(xa)], -1),
    })
    const coupling = B.affineCoupling(1, conditioner)
    const x = tensor([0.3, -0.7, 1.4])
    closeArrays(toFlat(coupling.inverse(coupling.forward(x))), toFlat(x))
    expect(item(coupling.forwardLogDetJacobian(x))).toBeCloseTo(
      logAbsDet((v) => coupling.forward(v), x),
      12,
    )
    const perm = B.permute([2, 0, 1])
    closeArrays(toFlat(perm.inverse(perm.forward(x))), toFlat(x))
    // An autoregressive conditioner: outputs depend only on earlier inputs.
    const ar = (y: Value): B.ShiftLogScale => {
      const y0 = sliceAxis(y, -1, 0, 1)
      const y1 = sliceAxis(y, -1, 1, 2)
      const zero = sliceAxis(tensor([0]), -1, 0, 1)
      return { shift: concat([zero, tanh(y0), y1], -1), logScale: concat([zero, y0, tanh(y1)], -1) }
    }
    const maf = B.autoregressive(ar)
    const y = maf.forward(x)
    closeArrays(toFlat(maf.inverse(y)), toFlat(x), 12)
    expect(item(maf.forwardLogDetJacobian(x))).toBeCloseTo(
      logAbsDet((v) => maf.forward(v), x),
      12,
    )
    expect(item(maf.inverseLogDetJacobian(y))).toBeCloseTo(-item(maf.forwardLogDetJacobian(x)), 12)
  })

  it('MADE masks are autoregressive', () => {
    const [m1, m2] = B.madeMasks(3, [4]) as [Tensor, Tensor]
    expect(m1.shape).toEqual([3, 4])
    // Output k can reach input i only through paths with i < k.
    const product = (i: number, k: number): number => {
      let s = 0
      for (let h = 0; h < 4; h++) s += (toFlat(m1)[i * 4 + h] ?? 0) * (toFlat(m2)[h * 3 + k] ?? 0)
      return s
    }
    for (let i = 0; i < 3; i++) for (let k = 0; k <= i; k++) expect(product(i, k)).toBe(0)
  })
})

describe('unconstraining', () => {
  it('maps the reals onto each support', () => {
    const u = tensor([-2, 0, 3])
    const cases = [
      [{ kind: 'positive' }, (v: number) => v > 0],
      [{ kind: 'unitInterval' }, (v: number) => v > 0 && v < 1],
      [{ kind: 'interval', low: tensor(-1), high: tensor(4) }, (v: number) => v > -1 && v < 4],
      [{ kind: 'greaterThan', low: tensor(2) }, (v: number) => v > 2],
      [{ kind: 'lessThan', high: tensor(2) }, (v: number) => v < 2],
      [{ kind: 'circular' }, (v: number) => Math.abs(v) < Math.PI],
    ] as const
    for (const [support, ok] of cases) {
      const b = B.bijectorForSupport(support)
      expect(toFlat(b.forward(u)).every(ok)).toBe(true)
      closeArrays(toFlat(b.inverse(b.forward(u))), toFlat(u), 10)
    }
    expect(() => B.bijectorForSupport({ kind: 'boolean' })).toThrow(/no bijection/)
  })

  it('transformed log densities integrate to one on the real line', () => {
    const { logDensity } = B.unconstrained(exponential(2))
    const grid = Array.from({ length: 4001 }, (_, i) => -20 + i * 0.01)
    const values = toFlat(logDensity(tensor(grid))).map(Math.exp)
    const mass = values.reduce((s, v) => s + v, 0) * 0.01
    expect(mass).toBeCloseTo(1, 4)
  })

  it('log densities on the unconstrained space differentiate', () => {
    const f = (m: Value, s: Value): Value => {
      const ld = B.transformLogDensity((x) => normal(m, s).logProb(x), B.bijectorForSupport({ kind: 'unitInterval' }))
      return ld(tensor([0.3, -1.2]))
    }
    expect(checkPrimitive(f, [tensor(0.4), tensor(0.7)], { rtol: 1e-5, atol: 1e-8 }).failures).toEqual([])
  })
})
