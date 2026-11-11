// A topic model under expectation propagation: a plate of Dirichlet mixtures over documents, a Dirichlet word distribution per
// topic, a categorical topic per token and the word read through the topic (`phi.at(z)`), compiled to one `gate` per token on the
// topic whose branch `k` is the conjugate `discrete` factor of the word on topic `k`. The reference is the exact posterior of the
// LDA-lite benchmark (enumeration of the 2^15 topic assignments with the Dirichlets integrated out), and a one-document model
// small enough to enumerate here.
import { describe, expect, it } from 'vitest'
import { session } from '../../trace/index.ts'
import { epSpecMethod } from '../inference/index.ts'
import { analysisOf } from '../inference/model-input.ts'
import { ldaLite } from '../models/index.ts'
import * as d from './dists/index.ts'
import { defineModel, toFactorGraph } from './index.ts'

function runEp(bound: ReturnType<typeof ldaLite.bound>) {
  const method = epSpecMethod(bound)
  expect(method.check?.(analysisOf(bound)).applies).toBe(true)
  const run = session(method, { steps: 500, mode: 'eager', config: { tol: 1e-9 } })
  const post = method.posterior?.(run.state, run.config)
  if (post === undefined) throw new Error('no posterior')
  return post
}

describe('LDA-lite under EP', () => {
  const bound = ldaLite.bound()
  const truth = ldaLite.truth(undefined, 'full')

  it('compiles to one gate per token on the topic, with Dirichlet rows as the branches', () => {
    const c = toFactorGraph(bound)
    expect(c.issues).toEqual([])
    const gates = c.graph?.factors.filter((f) => f.operator === 'gate') ?? []
    // Fifteen tokens: the observed word of each is a gate on its topic over the two topics' Dirichlet word distributions.
    expect(gates.length).toBe(15)
    expect(gates[0]?.args).toEqual(['z[0]', 'w[0]', 'phi[0]', 'phi[1]'])
    // The topic of each token is the conjugate discrete factor on its document's mixture.
    expect(c.graph?.factors.filter((f) => f.operator === 'discrete').length).toBe(15)
  })

  it('agrees with the exact posterior of the topics, the mixtures and the word distributions', () => {
    const post = runEp(bound)
    expect(post.status).toBe('converged')
    let worstMean = 0
    let worstSd = 1
    let checked = 0
    for (const r of post.summary()) {
      const t = truth.marginals[r.name as string]
      if (t === undefined) continue
      checked++
      worstMean = Math.max(worstMean, Math.abs((r.mean as number) - t.mean) / t.sd)
      worstSd = Math.min(worstSd, (r.sd as number) / t.sd)
    }
    expect(checked).toBe(Object.keys(truth.marginals).length)
    // Measured 0.26 truth sds at worst, and spreads from 0.58 of the truth's (the tokens of one word that share a topic).
    expect(worstMean).toBeLessThan(0.4)
    expect(worstSd).toBeGreaterThan(0.5)
    // The EP log evidence of a loopy graph is an approximation: within half a nat of the enumeration here.
    expect(Math.abs((post.logEvidence?.value ?? Number.NaN) - (truth.logEvidence ?? 0))).toBeLessThan(0.6)
  })

  it('is exact on one document with one token (a tree)', () => {
    // A single token: theta ~ Dir(a), phi_k ~ Dir(b_k), z ~ Cat(theta), w ~ Cat(phi_z). The posterior of the word's topic is
    // proportional to E[theta_k] * E[phi_k[w]] under the priors, which is the closed form below.
    const alpha = [0.5, 1.5]
    const beta = [
      [2, 1, 1],
      [1, 1, 3],
    ]
    const model = defineModel('one-token', (b) => {
      const doc = b.range('doc', { size: 1 })
      const token = b.range('token', { size: 1 })
      const topic = b.range('topic', { size: 2 })
      const word = b.range('word', { size: 3 })
      const w = b.data('w', { dims: [doc, token], support: 'index', indexes: word })
      const theta = b.latent('theta', d.dirichlet(alpha), { dims: [doc] })
      const phi = b.latent('phi', d.dirichlet(b.const(beta, [topic])), { dims: [topic] })
      const z = b.latent('z', d.categorical({ probs: theta }), { dims: [doc, token] })
      b.observe('w', d.categorical({ probs: phi.at(z) }), w)
    })
    const post = runEp(model.bind({ w: [[2]] }) as never)
    const weight = alpha.map(
      (t, k) => ((t / 2) * ((beta[k] as number[])[2] as number)) / (beta[k] as number[]).reduce((a, v) => a + v, 0),
    )
    const total = (weight[0] as number) + (weight[1] as number)
    const z = post.summary().find((r) => r.name === 'z[0]')
    expect(z?.mean).toBeCloseTo((weight[1] as number) / total, 8)
  })
})
