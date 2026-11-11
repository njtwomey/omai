// The robust likelihoods (Student t, Laplace, a mixture of inlier and outlier noise) and the spike and slab prior compiled to
// factor graphs and run under EP, against the exact posterior of a two-coefficient regression on a fine two-dimensional grid.
import { describe, expect, it } from 'vitest'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  type FactorGraphSpec,
  defaultOperators as factorOperators,
  type MpState,
} from '../../message-passing/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as d from './dists/index.ts'
import { defineModel, factorGraphPosterior, factorGraphVerdict, ops, toFactorGraph } from './index.ts'

const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

const X = [
  [1, -1],
  [1, -0.4],
  [1, 0.1],
  [1, 0.6],
  [1, 1],
  [1, 0.8],
  [1, -0.8],
]
const Y = [-1.1, -0.2, 0.9, 1.9, 4.6, 2.4, -2.2]

type LogLik = (resid: number) => number
const logt =
  (s: number, df: number): LogLik =>
  (r) =>
    -((df + 1) / 2) * Math.log(1 + (r / s) ** 2 / df)
const logLaplace =
  (s: number): LogLik =>
  (r) =>
    -Math.abs(r) / s
const logMix =
  (s: number, o: number, k: number): LogLik =>
  (r) =>
    Math.log(((1 - o) * Math.exp(-0.5 * (r / s) ** 2)) / s + (o * Math.exp(-0.5 * (r / (k * s)) ** 2)) / (k * s))
const logSpikeSlab = (w: number, slab: number, spike: number, pi: number): number =>
  Math.log((pi * Math.exp(-0.5 * (w / slab) ** 2)) / slab + ((1 - pi) * Math.exp(-0.5 * (w / spike) ** 2)) / spike)

/** The exact posterior mean and sd of the two weights on a grid. */
function grid(logPrior: (w: number) => number, logLik: LogLik): { mean: number[]; sd: number[] } {
  const n = 500
  const lo = -6
  const hi = 6
  const h = (hi - lo) / n
  let z = 0
  const m = [0, 0]
  const m2 = [0, 0]
  for (let i = 0; i <= n; i++)
    for (let j = 0; j <= n; j++) {
      const w = [lo + i * h, lo + j * h]
      let lp = logPrior(w[0] as number) + logPrior(w[1] as number)
      for (const [r, row] of X.entries())
        lp += logLik((Y[r] as number) - (row[0] as number) * (w[0] as number) - (row[1] as number) * (w[1] as number))
      const p = Math.exp(lp)
      z += p
      for (const k of [0, 1]) {
        m[k] = (m[k] as number) + p * (w[k] as number)
        m2[k] = (m2[k] as number) + p * (w[k] as number) ** 2
      }
    }
  const mean = m.map((v) => v / z)
  return { mean, sd: m2.map((v, k) => Math.sqrt(v / z - (mean[k] as number) ** 2)) }
}

function ep(definition: ReturnType<typeof defineModel>, mustConverge = true): { mean: number[]; sd: number[] } {
  const c = toFactorGraph(definition.bind({ X, y: Y }))
  expect(c.issues).toEqual([])
  const g = graphOf(c.graph)
  expect(factorGraphVerdict(c, g, 'ep')).toMatchObject({ ok: true })
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-9 }), { maxSteps: 400 })
  if (mustConverge) expect(r.status).toBe('converged')
  const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
  const mean = toFlat(post.mean('w'))
  const sd = post
    .summary()
    .filter((r) => r.site === 'w')
    .map((r) => r.sd)
  return { mean: [mean[0] as number, mean[1] as number], sd: [sd[0] as number, sd[1] as number] }
}

const regression = (
  name: string,
  prior: (m: Parameters<Parameters<typeof defineModel>[1]>[0]) => unknown,
  likelihood: (mean: never, y: never, m: Parameters<Parameters<typeof defineModel>[1]>[0]) => void,
) =>
  defineModel(name, (m) => {
    const row = m.range('row')
    const feature = m.range('feature')
    const Xd = m.data('X', { dims: [row, feature] })
    const y = m.data('y', { dims: [row] })
    const w = prior(m) as ReturnType<typeof m.latent>
    likelihood(Xd.matmul(w) as never, y as never, m)
  })

describe('robust likelihoods and the spike and slab prior under expectation propagation', () => {
  it('a Student t likelihood (known scale and degrees of freedom) agrees with the exact posterior', () => {
    const def = regression(
      'student-t',
      (m) => m.latent('w', d.normal(0, 3), { dims: ['feature'] }),
      (mean, y, m) => m.observe('y', d.studentT(3, mean, 0.5), y),
    )
    const got = ep(def)
    const ref = grid((w) => -0.5 * (w / 3) ** 2, logt(0.5, 3))
    for (const k of [0, 1]) {
      expect(Math.abs((got.mean[k] as number) - (ref.mean[k] as number))).toBeLessThan(0.08)
      expect((got.sd[k] as number) / (ref.sd[k] as number)).toBeGreaterThan(0.8)
      expect((got.sd[k] as number) / (ref.sd[k] as number)).toBeLessThan(1.25)
    }
  })

  it('a Laplace likelihood agrees with the exact posterior', () => {
    const def = regression(
      'laplace',
      (m) => m.latent('w', d.normal(0, 3), { dims: ['feature'] }),
      (mean, y, m) => m.observe('y', d.laplace(mean, 0.4), y),
    )
    const got = ep(def)
    const ref = grid((w) => -0.5 * (w / 3) ** 2, logLaplace(0.4))
    for (const k of [0, 1]) {
      expect(Math.abs((got.mean[k] as number) - (ref.mean[k] as number))).toBeLessThan(0.08)
      expect((got.sd[k] as number) / (ref.sd[k] as number)).toBeGreaterThan(0.8)
      expect((got.sd[k] as number) / (ref.sd[k] as number)).toBeLessThan(1.25)
    }
  })

  it('a mixture of inlier and outlier noise agrees with the exact posterior', () => {
    const def = regression(
      'outliers',
      (m) => m.latent('w', d.normal(0, 3), { dims: ['feature'] }),
      (mean, y, m) => m.observe('y', d.normalOutliers(mean, 0.4, 0.15, 12), y),
    )
    // The mixture is not log-concave, so EP may keep making small changes after it has settled near the answer.
    const got = ep(def, false)
    const ref = grid((w) => -0.5 * (w / 3) ** 2, logMix(0.4, 0.15, 12))
    for (const k of [0, 1]) {
      expect(Math.abs((got.mean[k] as number) - (ref.mean[k] as number))).toBeLessThan(0.12)
      expect((got.sd[k] as number) / (ref.sd[k] as number)).toBeGreaterThan(0.75)
      expect((got.sd[k] as number) / (ref.sd[k] as number)).toBeLessThan(1.3)
    }
  })

  it('a spike and slab prior on the coefficients (one packed vector) agrees with the exact posterior', () => {
    const def = regression(
      'spike-slab',
      (m) => m.latent('w', d.normalSpikeSlab(0.5, 3, 0.15), { dims: ['feature'] }),
      (mean, y, m) => m.observe('y', d.normal(mean, 0.5), y),
    )
    const got = ep(def)
    const ref = grid(
      (w) => logSpikeSlab(w, 3, 0.15, 0.5),
      (r) => -0.5 * (r / 0.5) ** 2,
    )
    for (const k of [0, 1]) {
      expect(Math.abs((got.mean[k] as number) - (ref.mean[k] as number))).toBeLessThan(0.1)
      expect((got.sd[k] as number) / (ref.sd[k] as number)).toBeGreaterThan(0.75)
      expect((got.sd[k] as number) / (ref.sd[k] as number)).toBeLessThan(1.3)
    }
  })

  it('ARD (a gamma precision for each coefficient, written 1 / sqrt(alpha)) agrees with the exact posterior under the Student-t prior it implies', () => {
    const def = regression(
      'ard',
      (m) => {
        const alpha = m.latent('alpha', d.gamma(2, 0.5), { dims: ['feature'] })
        return m.latent('w', d.normal(0, ops.div(1, ops.sqrt(alpha))), { dims: ['feature'] })
      },
      (mean, y, m) => m.observe('y', d.normal(mean, 0.5), y),
    )
    const got = ep(def)
    // Integrating the precision out leaves a Student t prior on each coefficient: density proportional to (b + w^2 / 2)^-(a + 1/2).
    const ref = grid(
      (w) => -2.5 * Math.log(0.5 + (w * w) / 2),
      (r) => -0.5 * (r / 0.5) ** 2,
    )
    for (const k of [0, 1]) {
      expect(Math.abs((got.mean[k] as number) - (ref.mean[k] as number))).toBeLessThan(0.1)
      expect((got.sd[k] as number) / (ref.sd[k] as number)).toBeGreaterThan(0.75)
      expect((got.sd[k] as number) / (ref.sd[k] as number)).toBeLessThan(1.3)
    }
  })

  it('an unknown scale of a heavy-tailed likelihood is refused with the reason', () => {
    const def = defineModel('t-random', (m) => {
      const row = m.range('row')
      const feature = m.range('feature')
      const Xd = m.data('X', { dims: [row, feature] })
      const w = m.latent('w', d.normal(0, 3), { dims: [feature] })
      const s = m.latent('s', d.gamma(2, 1))
      m.observe('y', d.studentT(3, Xd.matmul(w), s), m.data('y', { dims: [row] }))
    })
    const c = toFactorGraph(def.bind({ X, y: Y }))
    expect(c.issues.map((i) => i.reason).join(' ')).toMatch(/known/)
  })
})
