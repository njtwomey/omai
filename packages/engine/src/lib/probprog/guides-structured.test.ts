// The structured guide against the exact Gaussian posteriors of Gaussian models, correlations included: with the model's
// dependencies it recovers the posterior covariance (explaining away between two causes, a hierarchy's mean with its group
// effects, a coefficient vector with a full block), mean-field across blocks gets the means and misses the correlation, and the
// structure is what the model's graph says.
import { describe, expect, it } from 'vitest'
import { normal } from '../dists/index.ts'
import { adam } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { add, matmul, primalOf, tensor, toFlat, type Value, zeros } from '../tensor/index.ts'
import { stlElbo } from './elbo-extra.ts'
import { gaussianCondition, linearGaussianForm } from './engines/gaussian.ts'
import { svi } from './engines/svi.ts'
import { type AutoGuide, autoMultivariateNormal, autoNormal } from './guides.ts'
import { autoStructured, guideStructure } from './guides-structured.ts'
import { seed, traceOf } from './handlers.ts'
import { type Model, observe, plate, sample } from './primitives.ts'

const decaying = (lr: number) => adam({ learningRate: (t: number) => lr / (1 + t / 400) })

const explaining: Model = () => {
  const a = sample('a', normal(0, 1))
  const b = sample('b', normal(0, 1))
  observe('y', normal(add(a, b), 0.5), 1.0)
}
const Y = [0.9, -0.4, 1.6, 0.2]
const hierarchy: Model = () => {
  const mu = sample('mu', normal(0, 2))
  plate('j', 4, () => {
    const theta = sample('theta', normal(mu, 1))
    observe('y', normal(theta, 0.5), Y)
  })
}
const X = [
  [0.5, -1],
  [1.2, 0.3],
  [-0.7, 0.8],
  [0.1, -0.4],
  [1.5, 1.1],
]
const regression: Model = () => {
  const beta = sample('beta', normal(zeros([2]), 1))
  observe('y', normal(matmul(tensor(X), beta), 0.7), [0.4, 0.9, -0.3, 0.1, 1.2])
}

/** The exact posterior of the latent coordinates of a linear-Gaussian program, with the sites' order. */
function exactOf(model: Model): { mean: number[]; cov: number[][] } {
  const form = linearGaussianForm(model, undefined, traceOf(seed(model, key(0)), undefined))
  const post = gaussianCondition(form)
  const n = toFlat(post.mean).length
  const flat = toFlat(post.covariance)
  return {
    mean: [...toFlat(post.mean)],
    cov: Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => flat[i * n + j] ?? 0)),
  }
}

/** The mean and covariance of the guide's draws, coordinates in the order of its sites. */
function momentsOf(
  guide: AutoGuide<void>,
  params: Record<string, Value>,
  names: readonly string[],
): { mean: number[]; cov: number[][] } {
  const n = 20000
  const draws = guide.posterior(params, undefined).sample(key(99), n)
  const cols: number[][] = names.flatMap((name) => {
    const d = draws[name]
    const flat = toFlat(primalOf(d as Value))
    const size = flat.length / n
    return Array.from({ length: size }, (_, j) => j).flatMap((j) => [
      Array.from({ length: n }, (_, i) => flat[i * size + j] ?? 0),
    ])
  })
  const mean = cols.map((c) => c.reduce((a, b) => a + b, 0) / n)
  const cov = cols.map((ci, i) =>
    cols.map((cj, j) => ci.reduce((a, x, k) => a + (x - (mean[i] ?? 0)) * ((cj[k] ?? 0) - (mean[j] ?? 0)), 0) / n),
  )
  return { mean, cov }
}

function fit(model: Model, guide: AutoGuide<void>, steps = 2500) {
  return svi(model, undefined, {
    key: key(1),
    guide,
    numSteps: steps,
    optimizer: decaying(0.05),
    elbo: stlElbo({ numParticles: 2 }),
  }).params as Record<string, Value>
}

describe('the structure follows the model', () => {
  it('has a block per site, depending on its parents (and, for markov, co-parents)', () => {
    expect(guideStructure(hierarchy, undefined).blocks.map((b) => [b.sites, b.within, b.parents])).toEqual([
      [['mu'], 'full', []],
      [['theta'], 'diag', [0]],
    ])
    // Two causes of one effect: no edge between them in the model, a moral edge through the child.
    expect(guideStructure(explaining, undefined, { dependencies: 'parents' }).blocks.map((b) => b.parents)).toEqual([
      [],
      [],
    ])
    expect(guideStructure(explaining, undefined).blocks.map((b) => b.parents)).toEqual([[], [0]])
    expect(guideStructure(explaining, undefined, { dependencies: 'none' }).blocks.map((b) => b.parents)).toEqual([
      [],
      [],
    ])
    expect(guideStructure(explaining, undefined, { dependencies: 'full' }).blocks.map((b) => b.parents)).toEqual([
      [],
      [0],
    ])
    // Blocks merged by the caller; fewer parameters than the full-rank guide's.
    const merged = guideStructure(hierarchy, undefined, { blocks: [['mu', 'theta']] })
    expect(merged.blocks).toHaveLength(1)
    const sparse = guideStructure(hierarchy, undefined, { dependencies: 'parents' })
    expect(sparse.parameters).toBeLessThan((5 * 6) / 2 + 5)
    expect(() => guideStructure(hierarchy, undefined, { blocks: [['nope']] })).toThrow(/does not cover/)
  })
})

describe('against the exact Gaussian posterior', () => {
  it('recovers the correlation of explaining away, which mean-field cannot', () => {
    const exact = exactOf(explaining)
    const structured = autoStructured(explaining)
    const m = momentsOf(structured, fit(explaining, structured), ['a', 'b'])
    expect(Math.abs((m.mean[0] ?? 0) - (exact.mean[0] ?? 0))).toBeLessThan(0.03)
    expect(Math.abs((m.cov[0]?.[1] ?? 0) - (exact.cov[0]?.[1] ?? 0))).toBeLessThan(0.03)
    expect(Math.abs((m.cov[1]?.[1] ?? 0) - (exact.cov[1]?.[1] ?? 0))).toBeLessThan(0.03)
    expect(exact.cov[0]?.[1] ?? 0).toBeLessThan(-0.1)
    const meanField = autoStructured(explaining, { dependencies: 'none', within: 'diag' })
    const f = momentsOf(meanField, fit(explaining, meanField), ['a', 'b'])
    // Mean-field (which minimises the reverse KL) finds the means of a Gaussian, and too small a variance, with no correlation.
    expect(Math.abs((f.mean[0] ?? 0) - (exact.mean[0] ?? 0))).toBeLessThan(0.1)
    expect(Math.abs(f.cov[0]?.[1] ?? 0)).toBeLessThan(0.02)
    expect(f.cov[0]?.[0] ?? 1).toBeLessThan(0.6 * (exact.cov[0]?.[0] ?? 0))
    // Its bound is lower by the KL the correlation costs: -log(1 - 0.8^2) / 2 = 0.51.
    const elbo = (g: AutoGuide<void>): number => {
      const r = svi(explaining, undefined, {
        key: key(1),
        guide: g,
        numSteps: 2500,
        optimizer: decaying(0.05),
        elbo: stlElbo({ numParticles: 2 }),
      })
      return r.elbos.slice(-200).reduce((a, b) => a + b, 0) / 200
    }
    expect(elbo(structured) - elbo(meanField)).toBeGreaterThan(0.35)
  }, 60_000)

  it('recovers a hierarchy: the mean with its group effects, from the parents alone', () => {
    const exact = exactOf(hierarchy)
    const guide = autoStructured(hierarchy, { dependencies: 'parents' })
    const m = momentsOf(guide, fit(hierarchy, guide), ['mu', 'theta'])
    for (let i = 0; i < 5; i++) {
      expect(Math.abs((m.mean[i] ?? 0) - (exact.mean[i] ?? 0)), `mean ${i}`).toBeLessThan(0.04)
      for (let j = 0; j <= i; j++)
        expect(Math.abs((m.cov[i]?.[j] ?? 0) - (exact.cov[i]?.[j] ?? 0)), `cov ${i},${j}`).toBeLessThan(0.04)
    }
  }, 60_000)

  it('has a full block for a coefficient vector, as the full-rank guide does', () => {
    const exact = exactOf(regression)
    const guide = autoStructured(regression)
    const m = momentsOf(guide, fit(regression, guide), ['beta'])
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 2; j++) expect(Math.abs((m.cov[i]?.[j] ?? 0) - (exact.cov[i]?.[j] ?? 0))).toBeLessThan(0.03)
    // The same family as autoMultivariateNormal here: its ELBO at convergence is the same.
    const full = svi(regression, undefined, {
      key: key(1),
      guide: autoMultivariateNormal(regression),
      numSteps: 2500,
      optimizer: decaying(0.05),
      elbo: stlElbo({ numParticles: 2 }),
    })
    const mf = svi(regression, undefined, {
      key: key(1),
      guide: autoNormal(regression),
      numSteps: 2500,
      optimizer: decaying(0.05),
      elbo: stlElbo({ numParticles: 2 }),
    })
    const st = svi(regression, undefined, {
      key: key(1),
      guide,
      numSteps: 2500,
      optimizer: decaying(0.05),
      elbo: stlElbo({ numParticles: 2 }),
    })
    const last = (r: { elbos: readonly number[] }) => r.elbos.slice(-100).reduce((a, b) => a + b, 0) / 100
    expect(Math.abs(last(st) - last(full))).toBeLessThan(0.02)
    expect(last(mf)).toBeLessThan(last(st) + 0.03)
  }, 60_000)
})
