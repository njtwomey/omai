// Latent bivariate vectors with standard deviations and a correlation of their own: the compiler maps a population mean that is one Gaussian
// vector variable to ONE `mvn-sd-corr-group` factor, a known mean to a factor per vector. The model is a tree around that factor (Gaussian readings,
// Gaussian prior on the mean, gamma and LKJ priors), so EP's beliefs are the moment matches of the exact marginals, which a grid over
// the three parameters with the vectors and the mean integrated out densely gives (nothing of the factor's own algebra in it).
import { describe, expect, it } from 'vitest'
import { runEp } from '../../../../testing/factor-graph.ts'
import { type MpState, marginalsOf } from '../../../message-passing/index.ts'
import { choleskyInPlace, choleskyLogDet, choleskySolveInPlace } from '../../../message-passing/numerics/_bartlett.ts'
import { lgammaOf as lgamma } from '../../../message-passing/numerics/_scalar.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops, toFactorGraph } from '../index.ts'

const GROUPS = 4
const TIMES = [-1, 0, 1, 2]
const NOISE = 0.3
const PRIOR_SD = 2
const design: { g: number; t: number; y: number }[] = []
for (let j = 0; j < GROUPS; j++)
  for (const t of TIMES) design.push({ g: j, t, y: 0.5 * j + (j - 1.5) * 0.3 * t + 0.1 * Math.sin(j * 7 + t) })

const model = (eta: number) =>
  defineModel('sd-corr-effects', (m) => {
    const group = m.range('group', { size: GROUPS })
    const coef = m.range('coef', { size: 2 })
    const obs = m.range('obs')
    const tau1 = m.latent('tau1', d.gamma(2, 2))
    const tau2 = m.latent('tau2', d.gamma(2, 2))
    const rho = m.latent('rho', d.correlation(eta))
    const mu = m.latent(
      'mu',
      d.multivariateNormal([0, 0], {
        covariance: [
          [PRIOR_SD ** 2, 0],
          [0, PRIOR_SD ** 2],
        ],
      }),
      { dims: [coef] },
    )
    const sd1 = ops.div(1, ops.sqrt(tau1))
    const sd2 = ops.div(1, ops.sqrt(tau2))
    const theta = m.latent('theta', d.bivariateNormal(mu, { scale1: sd1, scale2: sd2, rho }, { over: coef }), {
      dims: [group, coef],
    })
    const x = m.data('X', { dims: [obs, coef] })
    const g = m.data('g', { dims: [obs], indexes: 'group' })
    m.observe('y', d.normal(ops.sum(ops.mul(theta.at(g), x), coef), NOISE), m.data('y', { dims: [obs] }))
  })

const bound = (eta: number) =>
  model(eta).bind({ X: design.map((o) => [1, o.t]), g: design.map((o) => o.g), y: design.map((o) => o.y) })

/** The exact marginals by a grid over `(log tau1, log tau2, logit u)` and a dense Gaussian integral over the vectors and the mean at each node. */
function reference(eta: number) {
  const dim = 2 * (GROUPS + 1)
  const n = design.length
  const s2 = NOISE ** 2
  const c0 = PRIOR_SD ** 2
  // A: the readings read theta_j[0] + t theta_j[1]; the columns are the vectors then the mean (not read)
  const at = new Float64Array(dim)
  const ata = new Float64Array(dim * dim)
  let yy = 0
  for (const o of design) {
    const row = new Float64Array(dim)
    row[2 * o.g] = 1
    row[2 * o.g + 1] = o.t
    for (let a = 0; a < dim; a++) {
      at[a] = (at[a] as number) + (row[a] as number) * o.y
      for (let b = 0; b < dim; b++)
        ata[a * dim + b] = (ata[a * dim + b] as number) + (row[a] as number) * (row[b] as number)
    }
    yy += o.y * o.y
  }
  const N = 64
  const win = [
    [-3.5, 3.5],
    [-3.5, 3.5],
    [-9, 9],
  ] as const
  const lB = 2 * lgamma(eta) - lgamma(2 * eta)
  const P = new Float64Array(dim * dim)
  const work = new Float64Array(dim * dim)
  const sol = new Float64Array(dim)
  const h = new Float64Array(dim)
  const nodes: { lw: number; t1: number; t2: number; rho: number; mean: Float64Array; cov: Float64Array }[] = []
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++)
      for (let k = 0; k < N; k++) {
        const x1 = (win[0][0] as number) + ((i + 0.5) * (win[0][1] - win[0][0])) / N
        const x2 = (win[1][0] as number) + ((j + 0.5) * (win[1][1] - win[1][0])) / N
        const s = (win[2][0] as number) + ((k + 0.5) * (win[2][1] - win[2][0])) / N
        const [t1, t2] = [Math.exp(x1), Math.exp(x2)]
        const rho = Math.tanh(s / 2)
        const om = 1 - rho * rho
        // the covariance (1/t1, rho / sqrt(t1 t2); 1/t2) and its precision
        const [s11, s22] = [1 / t1, 1 / t2]
        const s12 = rho * Math.sqrt(s11 * s22)
        const det = s11 * s22 - s12 * s12
        const lam = [s22 / det, -s12 / det, -s12 / det, s11 / det]
        P.set(ata.map((v) => v / s2))
        const add = (r: number, c: number, m: readonly number[], sign: number): void => {
          for (let a = 0; a < 2; a++)
            for (let b = 0; b < 2; b++)
              P[(2 * r + a) * dim + 2 * c + b] =
                (P[(2 * r + a) * dim + 2 * c + b] as number) + sign * (m[a * 2 + b] as number)
        }
        for (let g = 0; g < GROUPS; g++) {
          add(g, g, lam, 1)
          add(GROUPS, GROUPS, lam, 1)
          add(g, GROUPS, lam, -1)
          add(GROUPS, g, lam, -1)
        }
        for (let a = 0; a < 2; a++)
          P[(2 * GROUPS + a) * dim + 2 * GROUPS + a] = (P[(2 * GROUPS + a) * dim + 2 * GROUPS + a] as number) + 1 / c0
        work.set(P)
        if (!choleskyInPlace(work, dim)) continue
        for (let a = 0; a < dim; a++) h[a] = (at[a] as number) / s2
        sol.set(h)
        choleskySolveInPlace(work, dim, sol)
        let quad = 0
        for (let a = 0; a < dim; a++) quad += (h[a] as number) * (sol[a] as number)
        const logDetLam = -Math.log(det)
        // log p(y, params) with the vectors and the mean integrated out
        const logLik =
          -(n / 2) * Math.log(2 * Math.PI * s2) -
          yy / (2 * s2) +
          GROUPS * (0.5 * logDetLam - Math.log(2 * Math.PI)) -
          (Math.log(2 * Math.PI) + Math.log(c0)) +
          0.5 * (dim * Math.log(2 * Math.PI) - choleskyLogDet(work, dim) + quad)
        // priors: Gamma(2, 2) on each precision (the density in the log coordinate carries tau), the correlation's beta in the logit
        const logPrior =
          2 * Math.log(2) -
          lgamma(2) +
          (2 - 1) * x1 -
          2 * t1 +
          x1 +
          (2 * Math.log(2) - lgamma(2) + (2 - 1) * x2 - 2 * t2 + x2) +
          ((eta - 1) * Math.log(om / 4) + Math.log(om / 4) - (lB - 0))
        const lw = logLik + logPrior
        const cov = new Float64Array(dim * dim)
        for (let a = 0; a < dim; a++) {
          const e = new Float64Array(dim)
          e[a] = 1
          choleskySolveInPlace(work, dim, e)
          for (let b = 0; b < dim; b++) cov[b * dim + a] = e[b] as number
        }
        nodes.push({ lw, t1, t2, rho, mean: Float64Array.from(sol), cov })
        if (lw > top) top = lw
      }
  let total = 0
  for (const nd of nodes) {
    nd.lw = Math.exp(nd.lw - top)
    total += nd.lw
  }
  const moments = (f: (nd: (typeof nodes)[number]) => number): { mean: number; sd: number } => {
    let m1 = 0
    let m2 = 0
    for (const nd of nodes) {
      const v = f(nd)
      m1 += (nd.lw / total) * v
      m2 += (nd.lw / total) * v * v
    }
    return { mean: m1, sd: Math.sqrt(m2 - m1 * m1) }
  }
  const mixture = (index: number): { mean: number; sd: number } => {
    let m1 = 0
    let m2 = 0
    for (const nd of nodes) {
      const w = nd.lw / total
      const m = nd.mean[index] as number
      m1 += w * m
      m2 += w * ((nd.cov[index * dim + index] as number) + m * m)
    }
    return { mean: m1, sd: Math.sqrt(m2 - m1 * m1) }
  }
  return {
    rho: moments((nd) => nd.rho),
    tau1: moments((nd) => nd.t1),
    tau2: moments((nd) => nd.t2),
    mu0: mixture(2 * GROUPS),
    mu1: mixture(2 * GROUPS + 1),
    theta0: mixture(0),
    theta5: mixture(5),
  }
}

describe('latent bivariate vectors with a correlation and standard deviations of their own', () => {
  it('a random population mean: one mvn-sd-corr-group factor for the vectors, and the posterior keeps the vector shape', () => {
    const { c } = runEp(bound(1), { tol: 1e-8, damping: 0.3, maxSteps: 60 })
    expect(c.graph?.factors.filter((f) => f.operator === 'mvn-sd-corr')).toHaveLength(0)
    const group = c.graph?.factors.filter((f) => f.operator === 'mvn-sd-corr-group') ?? []
    expect(group).toHaveLength(1)
    expect(group[0]?.args).toEqual(['theta[0]', 'theta[1]', 'theta[2]', 'theta[3]', 'mu', 'tau1', 'tau2', 'rho'])
    expect(group[0]?.params).toEqual({ groups: GROUPS })
  })

  it.each([1, 3])('the beliefs are the moment matches of the exact marginals (eta %d)', (eta) => {
    const { g, r } = runEp(bound(eta), { tol: 1e-9, damping: 0.3, maxSteps: 80 })
    const marginals = marginalsOf(g, r.state as MpState) as Record<string, Record<string, unknown>>
    const ref = reference(eta)
    const num = (v: unknown): number => v as number
    // the correlation: a beta on (rho + 1) / 2 with the exact mean and variance of rho
    const rho = marginals['rho'] as { a: number; b: number }
    const [pa, pb] = [rho.a, rho.b]
    const rhoMean = (2 * pa) / (pa + pb) - 1
    const rhoSd = 2 * Math.sqrt((pa * pb) / ((pa + pb) ** 2 * (pa + pb + 1)))
    expect(Math.abs(rhoMean - ref.rho.mean) / ref.rho.sd).toBeLessThan(0.06)
    expect(rhoSd / ref.rho.sd).toBeGreaterThan(0.97)
    expect(rhoSd / ref.rho.sd).toBeLessThan(1.03)
    // the precisions: gammas with the exact mean and log mean
    for (const [name, want] of [
      ['tau1', ref.tau1],
      ['tau2', ref.tau2],
    ] as const) {
      const m = marginals[name] as { shape: number; rate: number }
      expect(Math.abs(num(m.shape) / num(m.rate) - want.mean) / want.sd).toBeLessThan(0.05)
      expect(Math.sqrt(num(m.shape)) / num(m.rate) / want.sd).toBeGreaterThan(0.95)
      expect(Math.sqrt(num(m.shape)) / num(m.rate) / want.sd).toBeLessThan(1.05)
    }
    // the mean and the vectors
    const mu = marginals['mu'] as { mean: number[]; variance: number[] }
    expect(Math.abs((mu.mean[0] as number) - ref.mu0.mean) / ref.mu0.sd).toBeLessThan(0.05)
    expect(Math.sqrt(mu.variance[0] as number) / ref.mu0.sd).toBeGreaterThan(0.97)
    expect(Math.sqrt(mu.variance[0] as number) / ref.mu0.sd).toBeLessThan(1.03)
    const th0 = marginals['theta[0]'] as { mean: number[]; variance: number[] }
    expect(Math.abs((th0.mean[0] as number) - ref.theta0.mean) / ref.theta0.sd).toBeLessThan(0.05)
    expect(Math.sqrt(th0.variance[0] as number) / ref.theta0.sd).toBeGreaterThan(0.97)
    expect(Math.sqrt(th0.variance[0] as number) / ref.theta0.sd).toBeLessThan(1.03)
  })

  it('a known mean is one mvn-sd-corr factor per vector', () => {
    const known = defineModel('known-mean', (m) => {
      const group = m.range('group', { size: 3 })
      const coef = m.range('coef', { size: 2 })
      const rho = m.latent('rho', d.correlation(1))
      m.latent('theta', d.bivariateNormal([0, 0], { scale1: 1, scale2: 1, rho }, { over: coef }), {
        dims: [group, coef],
      })
    })
    const c = toFactorGraph(known.bind({}))
    expect(c.issues).toEqual([])
    const factors = c.graph?.factors.filter((f) => f.operator === 'mvn-sd-corr') ?? []
    expect(factors).toHaveLength(3)
    expect(c.graph?.factors.filter((f) => f.operator === 'mvn-sd-corr-group')).toHaveLength(0)
  })

  it('refuses a vector axis that is not two long, and a vector without an axis', () => {
    const issues = (build: Parameters<typeof defineModel>[1]): string[] =>
      toFactorGraph(defineModel('bad', build).bind({} as never)).issues.map((i) => i.reason)
    const wide = issues((m) => {
      const group = m.range('group', { size: 3 })
      const coef = m.range('coef', { size: 3 })
      const rho = m.latent('rho', d.correlation(1))
      m.latent('x', d.bivariateNormal([0, 0, 0], { scale1: 1, scale2: 1, rho }, { over: coef }), {
        dims: [group, coef],
      })
    })
    expect(wide.join(' ')).toMatch(/a bivariate vector has two/)
    const bare = issues((m) => {
      const rho = m.latent('rho', d.correlation(1))
      m.latent('x', d.bivariateNormal([0, 0], { scale1: 1, scale2: 1, rho }))
    })
    expect(bare.join(' ')).toMatch(/without an axis/)
  })
})
