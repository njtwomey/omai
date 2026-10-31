import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toArray, toFlat } from '../tensor/index.ts'
import * as O from './index.ts'

type Out = Readonly<Record<string, unknown>>

const decision = (K: number[][], coef: readonly number[], bias: number): number[] =>
  K.map((row) => row.reduce((s, k, i) => s + k * (coef[i] ?? 0), bias))

describe('the nu and one-class duals against libsvm (scikit-learn)', () => {
  for (const c of loadFixture('optim/smo_nu').cases) {
    const out = c.output as Out
    it(c.name, () => {
      const K = toArray(decode(c.inputs[0]) as Tensor) as number[][]
      const options = { tol: 1e-10 }
      const nu = c.params['nu'] as number
      const r =
        c.fn === 'nuSvcDual'
          ? O.nuSvcDual(K, decode(c.inputs[1]) as Tensor, nu, options)
          : c.fn === 'nuSvrDual'
            ? O.nuSvrDual(K, decode(c.inputs[1]) as Tensor, c.params['C'] as number, nu, options)
            : O.oneClassDual(K, nu, options)
      expect(r.converged).toBe(true)
      expectMatches(r.coef, out['coef'], c.rtol, c.atol, `${c.name} coef`)
      expectMatches(r.bias, out['intercept'], c.rtol, c.atol, `${c.name} intercept`)
      expectMatches(tensor(decision(K, toFlat(r.coef), r.bias)), out['decision'], c.rtol, c.atol, `${c.name} decision`)
    })
  }
})

describe('SVDD', () => {
  it('equals the one-class machine on a kernel with unit diagonal, scaled by nu n', () => {
    const cases = loadFixture('optim/smo_nu').cases.filter((c) => c.fn === 'oneClassDual')
    for (const c of cases) {
      const K = toArray(decode(c.inputs[0]) as Tensor) as number[][]
      const n = K.length
      const nu = c.params['nu'] as number
      const svdd = O.svddDual(K, 1 / (nu * n), { tol: 1e-10 })
      const scaled = tensor(toFlat(svdd.coef).map((v) => v * nu * n))
      expectMatches(scaled, (c.output as Out)['coef'], 1e-6, 1e-6, `${c.name} svdd coef`)
    }
  })
})
