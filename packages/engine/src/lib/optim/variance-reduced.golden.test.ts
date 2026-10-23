import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { maxOf } from '../base/index.ts'
import { add, mul, softplus, square, sub, sum, type Tensor, tensor, toArray, type Value } from '../tensor/index.ts'
import * as O from './index.ts'

type Problem = { readonly sum: O.FiniteSum; readonly lipschitz: number; readonly prox?: O.ProximalStep }

const softThreshold =
  (lambda: number): O.ProximalStep =>
  (x, step) =>
    Float64Array.from(x, (v) => Math.sign(v) * Math.max(Math.abs(v) - lambda * step, 0))

function problem(fn: string, x: number[][], y: number[], params: Readonly<Record<string, unknown>>): Problem {
  const n = x.length
  const rows = x.map((r) => tensor(r))
  const sq = maxOf(x.map((r) => r.reduce((a, v) => a + v * v, 0)))
  const dot = (w: unknown, i: number): Value => sum(mul(rows[i] as Value, w as Value))
  if (fn === 'ridge') {
    const alpha = params['alpha'] as number
    return {
      sum: { n, term: (w, i) => add(square(sub(dot(w, i), y[i] ?? 0)), mul(sum(square(w as Value)), alpha / n)) },
      lipschitz: 2 * sq + (2 * alpha) / n,
    }
  }
  const c = params['C'] as number
  const loss = (w: unknown, i: number): Value => softplus(mul(dot(w, i), -(y[i] ?? 0)))
  if (fn === 'logisticL2') {
    return {
      sum: { n, term: (w, i) => add(loss(w, i), mul(sum(square(w as Value)), 1 / (2 * c * n))) },
      lipschitz: 0.25 * sq + 1 / (c * n),
    }
  }
  return { sum: { n, term: loss }, lipschitz: 0.25 * sq, prox: softThreshold(1 / (c * n)) }
}

describe('variance-reduced methods against scikit-learn SAG and SAGA', () => {
  for (const c of loadFixture('optim/variance_reduced').cases) {
    const x = toArray(decode(c.inputs[0]) as Tensor) as number[][]
    const y = toArray(decode(c.inputs[1]) as Tensor) as number[]
    const p = problem(c.fn, x, y, c.params)
    const options = {
      lipschitz: p.lipschitz,
      tol: 1e-10,
      maxEpochs: 3000,
      ...(p.prox === undefined ? {} : { prox: p.prox }),
    }
    const methods = p.prox === undefined ? { sag: O.sag, saga: O.saga, svrg: O.svrg } : { saga: O.saga, svrg: O.svrg }
    for (const [name, method] of Object.entries(methods)) {
      it(`${c.name} by ${name}`, () => {
        // Not from 0: the tensor softplus has a zero derivative at exactly 0 (a max(x, 0) kink), see the report.
        const r = method(p.sum, [0.1, -0.1, 0.1], options)
        expect(r.converged).toBe(true)
        expectMatches(r.x as Value, c.output, c.rtol, c.atol, `${c.name} ${name}`)
      })
    }
  }
})
