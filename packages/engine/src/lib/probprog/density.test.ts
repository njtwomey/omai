import { describe, expect, it } from 'vitest'
import { valueAndGrad } from '../autodiff/index.ts'
import { logSumExpOf } from '../base/index.ts'
import { bernoulli, categorical, exponential, gamma, normal } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { item, type TensorLike, take, tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  constrainSites,
  enumerate,
  findInitialParams,
  initToMedian,
  logJoint,
  markov,
  observe,
  plate,
  potentialEnergy,
  sample,
  unconstrainSites,
} from './index.ts'

const W = [0.3, 0.7]
const MU = [-1, 2]
const Y = [0.5, -1.2, 2.2]

function mixture(y: TensorLike): void {
  const locs = sample('locs', normal(0, 3), { sampleShape: [2] })
  plate('data', 3, () => {
    const z = sample('z', categorical({ probs: W }), { infer: { enumerate: 'parallel' } })
    observe('y', normal(take(locs, z), 1), y)
  })
}

const logN = (x: number, m: number, s: number): number =>
  -0.5 * ((x - m) / s) ** 2 - Math.log(s) - 0.5 * Math.log(2 * Math.PI)

describe('logJoint with enumeration', () => {
  it('sums local discrete sites out inside a plate', () => {
    const lj = item(logJoint(mixture, Y, { locs: MU }).logJoint)
    let want = logN(MU[0] ?? 0, 0, 3) + logN(MU[1] ?? 0, 0, 3)
    for (const y of Y) want += logSumExpOf([0, 1].map((k) => Math.log(W[k] ?? 0) + logN(y, MU[k] ?? 0, 1)))
    expect(lj).toBeCloseTo(want, 12)
  })

  it('sums a global discrete site out across a plate', () => {
    const model = (y: TensorLike): void => {
      const c = sample('c', bernoulli({ probs: 0.4 }), { infer: { enumerate: 'parallel' } })
      plate('data', 3, () => observe('y', normal(take(tensor(MU), c), 1), y))
    }
    const lj = item(logJoint(model, Y, {}).logJoint)
    const want = logSumExpOf(
      [0, 1].map((k) => Math.log(k === 1 ? 0.4 : 0.6) + Y.reduce((s, y) => s + logN(y, MU[k] ?? 0, 1), 0)),
    )
    expect(lj).toBeCloseTo(want, 12)
  })

  it('sums a hidden Markov chain out, reusing dimensions beyond the history', () => {
    const T = 6
    const trans = [
      [0.9, 0.1],
      [0.2, 0.8],
    ]
    const ys = [0.1, 1.9, 2.3, -0.4, 0.2, 2.5]
    const hmm = (): void => {
      markov(
        'time',
        T,
        (t, prev: Value | undefined) => {
          const probs = prev === undefined ? tensor([0.5, 0.5]) : take(tensor(trans), prev)
          const z = sample(`z${t}`, categorical({ probs }), { infer: { enumerate: 'parallel' } })
          observe(`y${t}`, normal(take(tensor([0, 2]), z), 1), ys[t] ?? 0)
          return z
        },
        undefined as Value | undefined,
      )
    }
    const r = logJoint(hmm, undefined)
    // Brute force over all 2^6 paths.
    const paths: number[] = []
    for (let m = 0; m < 2 ** T; m++) {
      let lp = 0
      let prev = -1
      for (let t = 0; t < T; t++) {
        const z = (m >> t) & 1
        lp += Math.log(prev < 0 ? 0.5 : (trans[prev]?.[z] ?? 0)) + logN(ys[t] ?? 0, z === 1 ? 2 : 0, 1)
        prev = z
      }
      paths.push(lp)
    }
    expect(item(r.logJoint)).toBeCloseTo(logSumExpOf(paths), 10)
    const dims = new Set(
      [...r.trace.sites.values()].flatMap((s) => (s.type === 'sample' && s.enumDim !== undefined ? [s.enumDim] : [])),
    )
    expect(dims.size).toBe(2)
  })

  it('differentiates through the contraction', () => {
    const f = valueAndGrad((locs: Value) => logJoint(mixture, Y, { locs }).logJoint)
    const g = toFlat(f(tensor(MU)).grad)
    const h = 1e-6
    for (let i = 0; i < 2; i++) {
      const up = [...MU]
      const down = [...MU]
      up[i] = (up[i] ?? 0) + h
      down[i] = (down[i] ?? 0) - h
      const fd =
        (item(logJoint(mixture, Y, { locs: up }).logJoint) - item(logJoint(mixture, Y, { locs: down }).logJoint)) /
        (2 * h)
      expect(g[i]).toBeCloseTo(fd, 6)
    }
  })
})

describe('potentialEnergy', () => {
  it('adds the log-Jacobian of each support', () => {
    const model = (): void => {
      const s = sample('s', gamma(2, 3))
      observe('y', normal(0, s), 0.4)
    }
    const pe = potentialEnergy(model, undefined)
    const u = 0.3
    const s = Math.exp(u)
    const lg = Math.log(9) + Math.log(s) - 3 * s
    expect(pe.potential([u])).toBeCloseTo(-(lg + logN(0.4, 0, s) + u), 12)
    expect(item(pe.constrain([u])['s'] ?? tensor(0))).toBeCloseTo(s, 14)
    expect(Array.from(pe.unconstrain({ s }))[0]).toBeCloseTo(u, 14)
  })

  it('sums enumerated discrete sites out automatically', () => {
    const model = (): void => {
      const mu = sample('mu', normal(0, 1))
      const c = sample('c', bernoulli({ probs: 0.5 }))
      observe('y', normal(mu, take(tensor([1, 2]), c)), 0.3)
    }
    const pe = potentialEnergy(model, undefined)
    expect(pe.enumerated).toEqual(['c'])
    const want = logN(0.1, 0, 1) + logSumExpOf([Math.log(0.5) + logN(0.3, 0.1, 1), Math.log(0.5) + logN(0.3, 0.1, 2)])
    expect(pe.potential([0.1])).toBeCloseTo(-want, 12)
  })

  it('finds a valid initial point and maps sites both ways', () => {
    const model = (): void => {
      sample('rate', exponential(2))
    }
    const pe = potentialEnergy(model, undefined)
    const init = findInitialParams(pe, model, undefined, { key: key(1), strategy: initToMedian() })
    expect(init.valid).toBe(true)
    const u = unconstrainSites(model, undefined, { rate: 0.5 })
    expect(item(u['rate'] ?? tensor(0))).toBeCloseTo(Math.log(0.5), 14)
    expect(item(constrainSites(model, undefined, { rate: Math.log(0.5) })['rate'] ?? tensor(0))).toBeCloseTo(0.5, 14)
  })

  it('refuses discrete latent sites that are not enumerated', () => {
    const model = (): void => {
      sample('mu', normal(0, 1))
      sample('k', bernoulli({ probs: 0.5 }))
    }
    expect(() => potentialEnergy(model, undefined, { enumerate: false })).toThrow(/enumerate/)
    expect(potentialEnergy(enumerate(model), undefined, { enumerate: false }).dim).toBe(1)
  })
})
