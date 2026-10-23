import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { key } from '../../random/index.ts'
import { type Tensor, tensor, toArray } from '../../tensor/index.ts'
import * as P from '../index.ts'

const vec = (x: unknown): number[] => toArray(decode(x) as Tensor) as number[]
const mat = (x: unknown): number[][] => toArray(decode(x) as Tensor) as number[][]
const networkOf = (p: Readonly<Record<string, unknown>>): P.LinearGaussianBayesNet =>
  P.linearGaussianBayesNet(p['network'] as P.LinearGaussianNode[])

describe('linear-Gaussian networks against pgmpy, numpy and scipy', () => {
  for (const c of loadFixture('pgm/models/linear_gaussian').cases) {
    it(c.name, () => {
      switch (c.fn) {
        case 'joint': {
          const j = P.toJointGaussian(networkOf(c.params))
          const order = c.params['order'] as string[]
          const idx = order.map((v) => j.variables.indexOf(v))
          const values = [
            ...idx.map((i) => j.mean[i] ?? 0),
            ...idx.flatMap((i) => idx.map((k) => j.covariance[i]?.[k] ?? 0)),
          ]
          expectMatches(tensor(values), c.output, 1e-12, 1e-12, c.name)
          // The GMRF of the network has the inverse covariance as its precision.
          const field = P.gmrfFromLinearGaussian(networkOf(c.params))
          const cov = P.gmrfCovariance(field)
          for (const [i, r] of j.covariance.entries())
            for (const [k, v] of r.entries()) expect(cov[i]?.[k]).toBeCloseTo(v, 9)
          return
        }
        case 'loglik': {
          const g = networkOf(c.params)
          const rows = mat(c.inputs[0]).map((r) => Object.fromEntries(g.nodes.map((_n, i) => [`x${i + 1}`, r[i] ?? 0])))
          expectMatches(P.linearGaussianLogLikelihood(g, rows), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'posterior': {
          const y = vec(c.inputs[0])
          const observed = c.params['observed'] as string[]
          const m = P.linearGaussianPosterior(
            networkOf(c.params),
            c.params['query'] as string[],
            Object.fromEntries(observed.map((v, i) => [v, y[i] ?? 0])),
          )
          expectMatches(tensor([...m.mean, ...m.covariance.flat(), m.logMass]), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'gmrfConditional': {
          const names = c.params['names'] as string[]
          const observed = c.params['observed'] as string[]
          const field = P.gmrf(names, vec(c.inputs[1]), mat(c.inputs[0]))
          const xb = vec(c.inputs[2])
          const cond = P.gmrfConditional(field, Object.fromEntries(observed.map((v, i) => [v, xb[i] ?? 0])))
          expectMatches(tensor([...cond.mean]), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'gmrfLogpdf': {
          const names = c.params['names'] as string[]
          const field = P.gmrf(names, vec(c.inputs[1]), mat(c.inputs[0]))
          const values = mat(c.inputs[2]).map((x) =>
            P.gmrfLogDensity(field, Object.fromEntries(names.map((v, i) => [v, x[i] ?? 0]))),
          )
          expectMatches(tensor(values), c.output, c.rtol, c.atol, c.name)
          // The canonical factor of the field gives the same log density.
          const f = P.gmrfToFactor(field)
          const x = Object.fromEntries(names.map((v, i) => [v, mat(c.inputs[2])[0]?.[i] ?? 0]))
          expect(P.gaussianLogValue(f, x)).toBeCloseTo(values[0] ?? 0, 10)
          return
        }
        default:
          throw new Error(`no case ${c.fn}`)
      }
    })
  }
})

describe('linear-Gaussian sampling', () => {
  it('ancestral samples and GMRF samples have the model moments', () => {
    const g = P.linearGaussianBayesNet([
      { name: 'x', parents: [], weights: [], intercept: 1, variance: 1 },
      { name: 'y', parents: ['x'], weights: [2], intercept: -1, variance: 0.5 },
    ])
    const n = 20000
    const s = P.linearGaussianSample(g, key(1), n)
    const meanY = s.reduce((a, r) => a + (r['y'] ?? 0), 0) / n
    expect(Math.abs(meanY - 1)).toBeLessThan(4 * Math.sqrt(4.5 / n))
    const field = P.gmrfFromLinearGaussian(g)
    const t = P.gmrfSample(field, key(2), n)
    const cov = t.reduce((a, r) => a + ((r['x'] ?? 0) - 1) * ((r['y'] ?? 0) - 1), 0) / n
    expect(Math.abs(cov - 2)).toBeLessThan(0.1)
  })

  it('validates its inputs', () => {
    expect(() =>
      P.linearGaussianBayesNet([{ name: 'x', parents: [], weights: [], intercept: 0, variance: 0 }]),
    ).toThrow(/variance/)
    expect(() =>
      P.gmrf(
        ['a', 'b'],
        [0, 0],
        [
          [1, 0.5],
          [0.4, 1],
        ],
      ),
    ).toThrow(/symmetric/)
    expect(() => P.gmrf(['a'], [0], [[-1]])).toThrow(/positive definite/)
  })
})
