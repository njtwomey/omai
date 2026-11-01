import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { families } from '../../dists/index.ts'
import { key } from '../../random/index.ts'
import { asTensor, item, toFlat, type Value } from '../../tensor/index.ts'
import type { Params } from '../module/module.ts'
import { DistributionalAdditive, headFamily } from './heads.ts'
import { linearTerm, splineTerm } from './terms.ts'

const n = 24
const eta = (k: number, shift = 0): Float64Array =>
  Float64Array.from({ length: n }, (_, i) => shift + 0.3 * Math.sin(i * (k + 1)))
const ones = new Float64Array(n).fill(1)
const positive = Float64Array.from({ length: n }, (_, i) => 0.5 + ((i * 7) % 11) / 4)
const counts = Float64Array.from({ length: n }, (_, i) => (i * 5) % 7)

describe('headFamily on the library families', () => {
  const cases: readonly [string, Float64Array][] = [
    ['normal', Float64Array.from({ length: n }, (_, i) => Math.cos(i) * 2)],
    ['gamma', positive],
    ['negativeBinomial', counts],
    ['studentT', Float64Array.from({ length: n }, (_, i) => Math.cos(i) * 2)],
    ['logNormal', positive],
  ]
  for (const [name, y] of cases) {
    it(`${name}: the loss is minus the mean log-likelihood of lib/gam's family, with the same gradient`, () => {
      const head = headFamily(name as never)
      const reference = families.gamlssExtended(name as never)
      const K = head.predictors.length
      expect(head.predictors).toEqual(reference.predictors)
      const etas = Array.from({ length: K }, (_, k) => eta(k, name === 'studentT' && k === 2 ? 1.5 : 0))
      const d = reference.derivatives(y, ones, etas, [], 2)
      const loss = head.nll(
        asTensor(Array.from(y)),
        etas.map((e) => asTensor(Array.from(e))),
      )
      expect(item(loss)).toBeCloseTo(-d.logLik / n, 9)
      const flat = (v: Value[]): number[] => v.flatMap((t) => toFlat(t))
      const analytic = grad((e: Value[]) => head.nll(asTensor(Array.from(y)), e))(
        etas.map((e) => asTensor(Array.from(e))),
      ) as Value[]
      const want = d.grad.flatMap((gk) => Array.from(gk, (v) => -v / n))
      flat(analytic).forEach((v, i) => {
        expect(v).toBeCloseTo(want[i] as number, 8)
      })
    })
  }

  it('an extended family trains through its derivatives: Gaussian location-scale and zero-inflated Poisson', () => {
    for (const [family, y] of [
      [families.gaussianLocationScale(), Float64Array.from({ length: n }, (_, i) => Math.cos(i))],
      [families.zeroInflatedPoissonLogit(), counts],
    ] as const) {
      const head = headFamily(family, y)
      const K = head.predictors.length
      const etas = Array.from({ length: K }, (_, k) => eta(k, k === 1 ? -0.5 : 0.2))
      const aux = family.aux(y, ones).map((a) => a.start)
      const d = family.derivatives(y, ones, etas, aux, 2)
      const at = (e: Value[]): Value => head.nll(asTensor(Array.from(y)), e)
      expect(item(at(etas.map((e) => asTensor(Array.from(e)))))).toBeCloseTo(-d.logLik / n, 10)
      const analytic = grad(at)(etas.map((e) => asTensor(Array.from(e)))) as Value[]
      const want = d.grad.flatMap((gk) => Array.from(gk, (v) => -v / n))
      analytic
        .flatMap((t) => toFlat(t))
        .forEach((v, i) => {
          expect(v).toBeCloseTo(want[i] as number, 9)
        })
    }
  })

  it('starts the intercepts at the family starting values for a named family, and at moments for the rest', () => {
    const y = Float64Array.from({ length: n }, (_, i) => 3 + Math.sin(i))
    const named = headFamily('normal').start(y)
    expect(named[0]).toBeCloseTo(y.reduce((a, b) => a + b, 0) / n, 9)
    const weibull = headFamily(families.gamlssWeibull(), y)
    expect(weibull.start(y)).toHaveLength(2)
    expect(weibull.start(y).every(Number.isFinite)).toBe(true)
  })
})

describe('DistributionalAdditive', () => {
  const x = asTensor(Array.from({ length: 30 }, (_, i) => [((i * 7) % 30) / 30, ((i * 11) % 30) / 30]))
  const y = Float64Array.from({ length: 30 }, (_, i) => 1 + Math.sin(i))
  const net = new DistributionalAdditive({
    inFeatures: 2,
    family: headFamily('normal'),
    terms: [[splineTerm(0, { interiorKnots: 4, lambda: 1 }), linearTerm(1)], [splineTerm(1, { interiorKnots: 3 })]],
    sample: x,
    y,
    key: key(1),
  })

  it('has one additive predictor per parameter, started at the family values', () => {
    expect(net.predictors).toEqual(['mu', 'sigma'])
    expect(net.forward(x).shape).toEqual([30, 2])
    expect(Object.keys(net.params).sort()).toEqual(['p0', 'p1'])
    const start = headFamily('normal').start(y)
    expect(toFlat((net.params['p0'] as Params)['intercept'] as Value)[0]).toBeCloseTo(start[0] as number, 12)
    expect(toFlat((net.params['p1'] as Params)['intercept'] as Value)[0]).toBeCloseTo(start[1] as number, 12)
    const p = toFlat(net.naturalParameters(net.params, x))
    expect(p[1]).toBeCloseTo(Math.exp(toFlat(net.apply(net.params, x))[1] as number), 12)
  })

  it('trains every predictor: the loss has gradients in all heads, and the distribution follows the predictors', () => {
    const loss = (p: Params): Value => net.nll(net.apply(p, x), asTensor(Array.from(y)))
    const g = grad(loss)(net.params) as Params
    for (const head of ['p0', 'p1']) {
      const leaves = Object.values(g[head] as Params).flatMap((t) =>
        typeof t === 'object' && 'shape' in (t as object) ? [t as Value] : (Object.values(t as Params) as Value[]),
      )
      expect(leaves.some((t) => toFlat(t).some((v) => Math.abs(v) > 1e-9))).toBe(true)
    }
    const dist = net.distribution(net.params, x)
    expect(toFlat(dist?.mean() as Value)).toHaveLength(30)
    expect(toFlat(net.penalty(net.params, 30) as Value)[0]).toBeGreaterThanOrEqual(0)
  })
})
