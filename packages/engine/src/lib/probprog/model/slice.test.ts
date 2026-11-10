// The first vertical slice (plan 7.2, steps 1 and 2): the probit regression written in the model builder agrees with the
// hand-written program on its log joint at random points, and NUTS through `toProgram` reproduces NUTS on the program.
import { describe, expect, it } from 'vitest'
import { bernoulli, normal } from '../../dists/index.ts'
import { ndtr } from '../../numerics/special/index.ts'
import { foldIn, key, normal as normalDraws } from '../../random/index.ts'
import { item, matmul, type Tensor, tensor, toFlat } from '../../tensor/index.ts'
import { logJoint } from '../density.ts'
import { mcmc } from '../engines/nuts.ts'
import { observe, plate, sample } from '../primitives.ts'
import * as d from './dists/index.ts'
import { defineModel, ops, toProgram } from './index.ts'

const N = 40
const D = 3

/** An intercept and two covariates; the labels come from a known weight vector through the probit link. */
function dataset(): { readonly X: number[][]; readonly y: number[] } {
  const z = toFlat(normalDraws(key(11), [N, D - 1]))
  const u = toFlat(normalDraws(key(12), [N]))
  const truth = [0.3, 1.2, -0.8]
  const X: number[][] = []
  const y: number[] = []
  for (let i = 0; i < N; i++) {
    const row = [1, z[i * 2] ?? 0, z[i * 2 + 1] ?? 0]
    X.push(row)
    const eta = row.reduce((a, v, j) => a + v * (truth[j] ?? 0), 0)
    // The label is 1 when the latent utility eta + noise is positive: a probit model.
    y.push(eta + (u[i] ?? 0) > 0 ? 1 : 0)
  }
  return { X, y }
}

const probitModel = defineModel('probit', (m) => {
  const n = m.range('row')
  const f = m.range('feature')
  const X = m.data('X', { dims: [n, f], role: 'covariate' })
  const y = m.data('y', { dims: [n], support: 'binary', role: 'response' })
  const w = m.latent('w', d.normal(0, 2), { dims: [f] })
  m.observe('y', d.bernoulli({ probs: ops.probit(X.matmul(w)) }), y)
})

const { X, y } = dataset()

/** The same model written by hand against the program layer. */
const handWritten = (): void => {
  const w = plate('feature', D, () => sample('w', normal(0, 2)))
  const probs = ndtr(matmul(tensor(X), w))
  plate('row', N, () => observe('y', bernoulli({ probs }), y))
}

describe('probit slice: the builder against the hand-written program', () => {
  const bound = probitModel.bind({ X, y })
  const program = toProgram(bound)

  it('binds with the sizes the data implies', () => {
    expect(bound.sizes).toEqual({ row: N, feature: D })
  })

  it('has the same log joint at 20 random points (1e-12)', () => {
    for (let i = 0; i < 20; i++) {
      const w = normalDraws(foldIn(key(5), i), [D]) as Tensor
      const a = item(logJoint(program, undefined, { w }).logJoint)
      const b = item(logJoint(handWritten, undefined, { w }).logJoint)
      expect(Math.abs(a - b), `point ${i}`).toBeLessThan(1e-12)
    }
  })

  it('runs NUTS through toProgram with the same draws as on the hand-written program', () => {
    const options = { key: key(3), numChains: 2, numWarmup: 60, numDraws: 40 }
    const a = mcmc(program, undefined, options)
    const b = mcmc(handWritten, undefined, options)
    const da = a.draws?.['w']
    const db = b.draws?.['w']
    expect(da?.shape).toEqual([80, D])
    expect(toFlat(da as Tensor)).toEqual(toFlat(db as Tensor))
  })
})
