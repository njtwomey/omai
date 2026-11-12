// The multiclass observations compiled with the scores packed (`packScalars`): when the scores of an observation are a matrix times one
// Gaussian vector (the class slopes of a classifier, the coefficients that every alternative of a choice shares) the observation is one
// `multiclass-row` factor on the vector, whose message carries the correlation between the scores, so the spreads are the exact ones
// where one message per class gave 0.75 to 0.85 of them. Against the exact posterior of the vector on a grid.
import { describe, expect, it } from 'vitest'
import { runEp } from '../../../../testing/factor-graph.ts'
import { defaultOperators, gaussianMv, type MpState, marginalsOf } from '../../../message-passing/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, toFactorGraph } from '../index.ts'
import { MULTICLASS_ROW_CLASSES } from './_plan.ts'

type Marginals = Record<string, { mean: number; variance: number }>

function packedEp(bound: Parameters<typeof toFactorGraph>[0]) {
  const { c, g, r } = runEp(bound, { tol: 1e-9, damping: 0.3, maxSteps: 400, verdict: true, packScalars: true })
  expect(['converged', 'completed']).toContain(r.status)
  return { c, marginals: marginalsOf(g, r.state as MpState) as unknown as Marginals }
}

/** The mean and sd of each coordinate of the posterior of `dim` coordinates on a grid of `n` points a side over [-6, 6]. */
function gridMoments(
  dim: number,
  n: number,
  logPosterior: (w: readonly number[]) => number,
): { mean: number[]; sd: number[] } {
  const h = 12 / n
  const axis = Array.from({ length: n }, (_, i) => -6 + (i + 0.5) * h)
  let z = 0
  const m1 = new Array<number>(dim).fill(0)
  const m2 = new Array<number>(dim).fill(0)
  const w = new Array<number>(dim).fill(0)
  const walk = (k: number): void => {
    if (k === dim) {
      const p = Math.exp(logPosterior(w))
      z += p
      for (let i = 0; i < dim; i++) {
        m1[i] = (m1[i] as number) + p * (w[i] as number)
        m2[i] = (m2[i] as number) + p * (w[i] as number) ** 2
      }
      return
    }
    for (const a of axis) {
      w[k] = a
      walk(k + 1)
    }
  }
  walk(0)
  const mean = m1.map((v) => v / z)
  return { mean, sd: m2.map((v, i) => Math.sqrt(v / z - (mean[i] as number) ** 2)) }
}

const logSoftmax = (u: readonly number[], y: number): number => {
  const top = Math.max(...u)
  return (u[y] as number) - top - Math.log(u.reduce((p, v) => p + Math.exp(v - top), 0))
}

describe('multiclass observations on a packed vector: one joint site per observation', () => {
  it('class slopes: ten rows are ten multiclass-row factors on one vector, and the slopes are the exact ones', () => {
    const x = [-1.4, -0.9, -0.5, -0.1, 0.2, 0.6, 1.0, 1.5, -0.3, 0.8]
    const y = [0, 0, 0, 2, 1, 1, 1, 1, 2, 2]
    const s = 2
    const model = defineModel('softmax-slopes-packed', (m) => {
      const row = m.range('row')
      const cls = m.range('class', { size: 3 })
      const w = m.latent('w', d.normal(0, s), { dims: [cls] })
      const logits = m.data('x', { dims: [row], role: 'covariate' }).mul(w)
      m.observe('y', d.categorical({ logits }, { over: cls }), m.data('y', { dims: [row], support: 'count' }))
    })
    const { c, marginals } = packedEp(model.bind({ x, y }))
    const operators = c.graph?.factors.map((f) => f.operator) ?? []
    expect(operators.filter((o) => o === 'multiclass-row')).toHaveLength(10)
    expect(operators).not.toContain('multiclass-site')
    const exact = gridMoments(3, 31, (w) => {
      let lp = (-0.5 * w.reduce((a, v) => a + v * v, 0)) / (s * s)
      for (const [i, xi] of x.entries())
        lp += logSoftmax(
          w.map((v) => v * xi),
          y[i] as number,
        )
      return lp
    })
    for (let k = 0; k < 3; k++) {
      const got = marginals[`w[${k}]`] as { mean: number; variance: number }
      expect(Math.abs(got.mean - (exact.mean[k] as number)) / (exact.sd[k] as number), `slope ${k} mean`).toBeLessThan(
        0.05,
      )
      const ratio = Math.sqrt(got.variance) / (exact.sd[k] as number)
      expect(ratio, `slope ${k} sd`).toBeGreaterThan(0.97)
      expect(ratio, `slope ${k} sd`).toBeLessThan(1.03)
    }
  }, 60_000)

  it('eight alternatives with two shared coefficients: a rank-two row each, against the grid of the coefficients', () => {
    const alternatives = 8
    const choices = 6
    const attribute = (seed: number): number[][] =>
      Array.from({ length: choices }, (_, i) =>
        Array.from({ length: alternatives }, (_, k) => 3 * (((i * 7 + k * 13 + seed * 5) % 17) / 17) - 1.5),
      )
    const price = attribute(1)
    const quality = attribute(2)
    const chosen = [3, 0, 5, 2, 7, 1]
    const s = 1.5
    const model = defineModel('choice-packed', (m) => {
      const choice = m.range('choice')
      const alternative = m.range('alternative', { size: alternatives })
      const bp = m.latent('beta_price', d.normal(0, s))
      const bq = m.latent('beta_quality', d.normal(0, s))
      const utility = m
        .data('price', { dims: [choice, alternative], role: 'covariate' })
        .mul(bp)
        .add(m.data('quality', { dims: [choice, alternative], role: 'covariate' }).mul(bq))
      m.observe(
        'chosen',
        d.categorical({ logits: utility }, { over: alternative }),
        m.data('chosen', { dims: [choice], support: 'count' }),
      )
    })
    const { c, marginals } = packedEp(model.bind({ price, quality, chosen }))
    expect(c.graph?.factors.filter((f) => f.operator === 'multiclass-row')).toHaveLength(choices)
    const exact = gridMoments(2, 241, ([a, b]) => {
      let lp = (-0.5 * ((a as number) ** 2 + (b as number) ** 2)) / (s * s)
      for (let i = 0; i < choices; i++) {
        const u = Array.from(
          { length: alternatives },
          (_, k) =>
            ((price[i] as number[])[k] as number) * (a as number) +
            ((quality[i] as number[])[k] as number) * (b as number),
        )
        lp += logSoftmax(u, chosen[i] as number)
      }
      return lp
    })
    for (const [k, name] of ['beta_price', 'beta_quality'].entries()) {
      const got = marginals[name] as { mean: number; variance: number }
      expect(Math.abs(got.mean - (exact.mean[k] as number)) / (exact.sd[k] as number), `${name} mean`).toBeLessThan(0.1)
      const ratio = Math.sqrt(got.variance) / (exact.sd[k] as number)
      expect(ratio, `${name} sd`).toBeGreaterThan(0.9)
      expect(ratio, `${name} sd`).toBeLessThan(1.1)
    }
  }, 60_000)

  it('more classes than the product grid reaches keep one message per class (no vector is packed)', () => {
    const model = defineModel('softmax-slopes-many', (m) => {
      const row = m.range('row')
      const cls = m.range('class', { size: 8 })
      const w = m.latent('w', d.normal(0, 1), { dims: [cls] })
      const logits = m.data('x', { dims: [row], role: 'covariate' }).mul(w)
      m.observe('y', d.categorical({ logits }, { over: cls }), m.data('y', { dims: [row], support: 'count' }))
    })
    const compiled = toFactorGraph(model.bind({ x: [0.5, -1, 1.5], y: [1, 4, 7] }), { packScalars: true })
    const operators = compiled.graph?.factors.map((f) => f.operator) ?? []
    expect(operators).toContain('multiclass-site')
    expect(operators).not.toContain('multiclass-row')
  })

  it('without packScalars the scores stay scalar variables (the variational rule reads that graph)', () => {
    const model = defineModel('softmax-slopes-unpacked', (m) => {
      const row = m.range('row')
      const cls = m.range('class', { size: 3 })
      const w = m.latent('w', d.normal(0, 1), { dims: [cls] })
      const logits = m.data('x', { dims: [row], role: 'covariate' }).mul(w)
      m.observe('y', d.categorical({ logits }, { over: cls }), m.data('y', { dims: [row], support: 'count' }))
    })
    const compiled = toFactorGraph(model.bind({ x: [0.5, -1], y: [1, 2] }))
    const operators = compiled.graph?.factors.map((f) => f.operator) ?? []
    expect(operators).toContain('multiclass-site')
    expect(operators).not.toContain('multiclass-row')
  })

  it('the limit of the layout plan is the limit of the operator', () => {
    const op = defaultOperators()['multiclass-row']
    const refuses = (classes: number): string | undefined =>
      op?.check?.({ y: 0, classes, matrix: new Array<number>(classes * classes).fill(0) }, [gaussianMv(classes)])
    expect(refuses(MULTICLASS_ROW_CLASSES)).toBeUndefined()
    expect(refuses(MULTICLASS_ROW_CLASSES + 1)).toMatch(/product grid/)
  })
})
