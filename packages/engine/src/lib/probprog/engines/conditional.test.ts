import { describe, expect, it } from 'vitest'
import { expectPosteriorMoments } from '../../../testing/probprog.ts'
import { categorical, dirichlet, inverseGamma, normal } from '../../dists/index.ts'
import { lgamma, ndtr, normalPdf } from '../../numerics/special/index.ts'
import { key } from '../../random/index.ts'
import { item, mul, sqrt, stack, type Tensor, take, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { nullTracer, runAlgorithm, startRun, stepContext } from '../../trace/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, toProgram } from '../model/index.ts'
import { type Model, markov, observe, param, plate, sample } from '../primitives.ts'
import { chainInference } from './chain.ts'
import { em, emAlgorithm } from './em.ts'
import { enumerationPosterior } from './enumeration.ts'
import { gibbs, gibbsAlgorithm } from './gibbs.ts'

const logN = (x: number, m: number, s: number): number =>
  -0.5 * ((x - m) / s) ** 2 - Math.log(s) - 0.5 * Math.log(2 * Math.PI)

// The brute-force comparison runs a few seconds; on a loaded machine it has crossed the default five.
describe('Gibbs sampling', { timeout: 20_000 }, () => {
  it('alternates semi-conjugate conditionals to the joint posterior (against a grid)', () => {
    const y = [1.2, 0.4, 2.3, 1.7, 0.9, 1.5]
    const model: Model = () => {
      const mu = sample('mu', normal(0, 10))
      const v = sample('v', inverseGamma(2, 2))
      plate('n', y.length, () => observe('y', normal(mu, sqrt(v)), y))
    }
    // The posterior mean of mu and v on a fine grid over (mu, log v).
    let z = 0
    let mMu = 0
    let mV = 0
    for (let i = 0; i < 400; i++) {
      const mu = -2 + (6 * i) / 399
      for (let j = 0; j < 400; j++) {
        const lv = -4 + (6 * j) / 399
        const v = Math.exp(lv)
        let lp = logN(mu, 0, 10) + Math.log(4) - 3 * lv - 2 / v + lv
        for (const x of y) lp += logN(x, mu, Math.sqrt(v))
        const w = Math.exp(lp)
        z += w
        mMu += w * mu
        mV += w * v
      }
    }
    const post = gibbs(model, undefined, { key: key(0), numDraws: 3000, numWarmup: 50 })
    expectPosteriorMoments(post, 'mu', { mean: mMu / z })
    expectPosteriorMoments(post, 'v', { mean: mV / z }, { z: 5 })
  })

  it('collapses Dirichlet weights out of a mixture (against brute force)', () => {
    const y = [-1.2, 1.4, 1.9, -0.3, 1.1]
    const mus = [-1, 1.5]
    const model: Model = () => {
      const w = sample('w', dirichlet([1, 1]))
      plate('n', y.length, () => {
        const zz = sample('z', categorical({ probs: w }))
        observe('y', normal(take(tensor(mus), zz), 1), y)
      })
    }
    // p(z | y) proportional to DirMult(z; 1, 1) prod N(y_i; mu_{z_i}, 1).
    const lbeta = (a: number, b: number): number => item(lgamma(a)) + item(lgamma(b)) - item(lgamma(a + b))
    const marg = new Array<number>(y.length).fill(0)
    let total = 0
    let ew1 = 0
    for (let m = 0; m < 2 ** y.length; m++) {
      let n1 = 0
      let lp = 0
      y.forEach((x, i) => {
        const k = (m >> i) & 1
        n1 += k
        lp += logN(x, mus[k] ?? 0, 1)
      })
      lp += lbeta(1 + y.length - n1, 1 + n1) - lbeta(1, 1)
      const w = Math.exp(lp)
      total += w
      y.forEach((_, i) => {
        marg[i] = (marg[i] ?? 0) + (((m >> i) & 1) === 1 ? w : 0)
      })
      ew1 += w * ((1 + n1) / (2 + y.length))
    }
    const post = gibbs(model, undefined, { key: key(1), numDraws: 3000, numWarmup: 50 })
    const z = post.draws?.['z'] as Tensor
    const n = z.shape[0] ?? 0
    for (let i = 0; i < y.length; i++) {
      const p = (marg[i] ?? 0) / total
      const freq = toFlat(z).filter((v, j) => j % y.length === i && v === 1).length / n
      // Collapsed sweeps mix fast here; 5 binomial standard errors with an ESS of a third of the draws.
      expect(Math.abs(freq - p), `z[${i}]`).toBeLessThan(5 * Math.sqrt((p * (1 - p)) / (n / 3)))
    }
    expectPosteriorMoments(post, 'w', { mean: 1 - ew1 / total }, { z: 5 })
  })

  it('draws mixture components group by group', () => {
    const y = [-2.1, -1.8, -2.4, 2.2, 1.9, 2.5]
    const model: Model = () => {
      const locs = plate('components', 2, () => sample('locs', normal(0, 5)))
      plate('n', y.length, () => {
        const zz = sample('z', categorical({ probs: [0.5, 0.5] }))
        observe('y', normal(take(locs, zz), 0.5), y)
      })
    }
    const post = gibbs(model, undefined, { key: key(2), numDraws: 500, numWarmup: 50, init: { locs: [-1, 1] } })
    const means = toFlat(post.mean('locs'))
    expect(Math.abs((means[0] ?? 0) + 2.1)).toBeLessThan(0.3)
    expect(Math.abs((means[1] ?? 0) - 2.2)).toBeLessThan(0.3)
  })

  it('runs collapsed topic-model updates', () => {
    const docs = [
      [0, 1, 0, 1, 0],
      [2, 3, 3, 2, 3],
    ]
    // Words `[position, document]`: the words plate is dim -2, the documents plate dim -1.
    const words = Array.from({ length: 5 }, (_, n) => [docs[0]?.[n] ?? 0, docs[1]?.[n] ?? 0])
    const lda: Model = () => {
      const phi = plate('topics', 2, () => sample('phi', dirichlet([0.5, 0.5, 0.5, 0.5])))
      plate('docs', 2, () => {
        const theta = sample('theta', dirichlet([0.5, 0.5]))
        plate(
          'words',
          5,
          () => {
            const zz = sample('z', categorical({ probs: theta }))
            observe('w', categorical({ probs: take(phi, zz) }), words)
          },
          { dim: -2 },
        )
      })
    }
    const alg = gibbsAlgorithm(lda, undefined)
    const r = runAlgorithm(alg, { seed: 3, maxSteps: 40 })
    const z = toFlat(r.state.values['z'] as Value)
    // Each document's words come from one topic: within a document the assignments agree.
    const doc0 = z.filter((_, i) => i % 2 === 0)
    const doc1 = z.filter((_, i) => i % 2 === 1)
    expect(new Set(doc0).size).toBe(1)
    expect(new Set(doc1).size).toBe(1)
    expect(doc0[0]).not.toBe(doc1[0])
  })
})

const HMM_Y = [0.1, 2.2, 1.9, -0.3, 0.2, 2.4, 2.1]
const hmmWith = (trans: Value, means: Value): void => {
  markov(
    't',
    HMM_Y.length,
    (t, prev: Value | undefined) => {
      const zz = sample(`z${t}`, categorical({ probs: prev === undefined ? tensor([0.5, 0.5]) : take(trans, prev) }))
      observe(`y${t}`, normal(take(means, zz), 0.6), HMM_Y[t] ?? 0)
      return zz
    },
    undefined as Value | undefined,
  )
}

describe('discrete chains and EM', () => {
  it('forward-backward agrees with enumeration', () => {
    const model: Model = () =>
      hmmWith(
        tensor([
          [0.85, 0.15],
          [0.25, 0.75],
        ]),
        tensor([0, 2]),
      )
    const chain = chainInference(model, undefined)
    const exact = enumerationPosterior(model, undefined)
    expect(chain.logEvidence?.value).toBeCloseTo(exact.logEvidence?.value ?? 0, 10)
    for (let t = 0; t < HMM_Y.length; t++) expect(item(chain.mean(`z${t}`))).toBeCloseTo(item(exact.mean(`z${t}`)), 10)
    const d = chain.sample(key(4), 2000)
    const f = toFlat(d['z1'] ?? tensor([])).reduce((a, b) => a + b, 0) / 2000
    const p = item(exact.mean('z1'))
    expect(Math.abs(f - p)).toBeLessThan(4 * Math.sqrt((p * (1 - p)) / 2000))
  })

  it('fits an HMM by EM with closed-form transition and emission updates', () => {
    const model: Model = () => {
      const trans = param(
        'trans',
        [
          [0.6, 0.4],
          [0.4, 0.6],
        ],
        { support: { kind: 'simplex' } },
      )
      const means = param('means', [-0.5, 1])
      hmmWith(trans, means)
    }
    const fast = em(model, undefined, { key: key(5), maxIter: 400, tol: 1e-12 })
    expect(fast.diagnostics['closedForm']).toEqual(['trans', 'means'])
    const slow = em(model, undefined, { key: key(5), maxIter: 400, tol: 1e-12, mStep: 'gradient' })
    expect(fast.diagnostics['objective']).toBeCloseTo(slow.diagnostics['objective'] as number, 5)
    expect(toFlat(fast.mean('means'))[1]).toBeCloseTo(toFlat(slow.mean('means'))[1] ?? 0, 3)
  })

  it('fits a mixture of experts with gradient M-steps', () => {
    // Two linear experts gated by a logistic function of x.
    const xs = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
    const ys = [-4.1, -2.9, -2.2, -0.9, 0.4, 1.1, 1.4, 2.1]
    const moe: Model = () => {
      const gate = param('gate', [0.1, 0])
      const slopes = param('slopes', [1, 0.5])
      const s = param('noise', 0.5, { support: { kind: 'positive' } })
      plate('n', xs.length, () => {
        const x = tensor(xs)
        const logits = stack([mul(x, take(gate, 0)), mul(x, take(gate, 1))], -1)
        const k = sample('k', categorical({ logits }))
        observe('y', normal(mul(take(slopes, k), x), s), ys)
      })
    }
    const post = em(moe, undefined, { key: key(6), maxIter: 300, tol: 1e-10 })
    expect(post.status).toBe('converged')
    // The shared noise scale has a closed form; the gate and the slopes take gradient M-steps.
    expect(post.diagnostics['closedForm']).toEqual(['noise'])
    const slopes = toFlat(post.mean('slopes')).sort((a, b) => a - b)
    // The negative-x points have slope about 2, the positive ones about 1.
    expect(Math.abs((slopes[0] ?? 0) - 1)).toBeLessThan(0.3)
    expect(Math.abs((slopes[1] ?? 0) - 2)).toBeLessThan(0.3)
  })

  it('never decreases the objective', () => {
    const y = [-1.9, -2.2, 0.1, 2.1, 1.8, 2.4, -0.2]
    const model: Model = () => {
      const w = param('w', [1 / 3, 1 / 3, 1 / 3], { support: { kind: 'simplex' } })
      const mu = param('mu', [-1, 0, 1])
      plate('n', y.length, () => {
        const zz = sample('z', categorical({ probs: w }))
        observe('y', normal(take(mu, zz), 0.5), y)
      })
    }
    const objectives: number[] = []
    const alg = emAlgorithm(model, undefined, { accelerate: false })
    let s = alg.init(key(0))
    for (let i = 0; i < 20; i++) {
      s = alg.step(s, stepContext(i, key(i), startRun(nullTracer, { name: 'em', seed: 0, config: {} })))
      objectives.push(s.objective)
    }
    for (let i = 1; i < objectives.length; i++)
      expect(objectives[i] ?? 0).toBeGreaterThanOrEqual((objectives[i - 1] ?? 0) - 1e-9)
  })
})

// A conjugate or mixture-component conditional is drawn on the family's own support, so a site constrained beyond it (an
// ordered pair of means, a lower bound) has to be updated by a step on its own unconstrained scale instead.
describe('Gibbs sampling of constrained sites', { timeout: 30_000 }, () => {
  const y = [-0.5, 0.2, -0.1]
  const bounded = (constrained: boolean): Model =>
    toProgram(
      defineModel('bounded', (m) => {
        const row = m.range('row')
        const obs = m.data('y', { dims: [row] })
        const mu = m.latent('mu', d.normal(0, 1), constrained ? { constraint: { lower: 0 } } : {})
        m.observe('y', d.normal(mu, 1), obs)
      }).bind({ y }),
    )

  it('keeps a conjugate normal mean above its lower bound and samples the truncated posterior', () => {
    // Prior N(0, 1), three unit-variance observations: the posterior is N(sum(y) / 4, 1 / 4), cut at zero.
    const m = (y[0] ?? 0) / 4 + (y[1] ?? 0) / 4 + (y[2] ?? 0) / 4
    const s = 0.5
    const alpha = (0 - m) / s
    const mean = m + (s * item(normalPdf(alpha))) / (1 - item(ndtr(alpha)))
    const unconstrained = gibbs(bounded(false), undefined, { key: key(0), numDraws: 1500, numWarmup: 50 })
    expect(Math.min(...toFlat(unconstrained.draws?.['mu'] as Tensor))).toBeLessThan(0)
    for (const k of [0, 1, 2, 3]) {
      const post = gibbs(bounded(true), undefined, { key: key(k), numDraws: 2500, numWarmup: 100 })
      const mu = toFlat(post.draws?.['mu'] as Tensor)
      expect(Math.min(...mu)).toBeGreaterThanOrEqual(0)
      expectPosteriorMoments(post, 'mu', { mean }, { z: 5 })
    }
  })

  it('keeps the ordered means of a mixture ordered whichever order the start is in, with the labels exact', () => {
    const data = [-2.4, -1.7, -2.1, 1.8, 2.5, 2.2, -1.9, 2.0]
    const model = toProgram(
      defineModel('ordered-mixture', (m) => {
        const row = m.range('row')
        const k = m.range('component', { size: 2 })
        const obs = m.data('y', { dims: [row] })
        const mu = m.latent('mu', d.normal(0, 3), { dims: [k], constraint: { kind: 'ordered' } })
        const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
        m.observe('y', d.normal(mu.at(z), 1), obs)
      }).bind({ y: data }, { sizes: { component: 2 } }),
    )
    // Without the constraint the same chain stays wherever it starts; with it every draw is ordered, for starts of both orders.
    for (const k of [0, 1, 2, 3, 4, 5, 6, 7]) {
      const post = gibbs(model, undefined, { key: key(k), numDraws: 200, numWarmup: 40 })
      const mu = toFlat(post.draws?.['mu'] as Tensor)
      for (let i = 0; i < mu.length; i += 2) expect(mu[i] ?? 0, `key ${k} draw ${i / 2}`).toBeLessThan(mu[i + 1] ?? 0)
      // Four labels each side: the means are those of the two clusters, shrunk a little towards the prior mean 0.
      expectPosteriorMoments(post, 'mu', { mean: (4 * -2.025) / (4 + 1 / 9) }, { z: 6, index: 0 })
      expectPosteriorMoments(post, 'mu', { mean: (4 * 2.125) / (4 + 1 / 9) }, { z: 6, index: 1 })
    }
  })
})
