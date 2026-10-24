import { describe, expect, it } from 'vitest'
import {
  bimodal,
  expectInvariant,
  expectMixing,
  expectMoments,
  gaussian2,
  standardNormal,
} from '../../../testing/sampling.ts'
import { key, normal } from '../../random/index.ts'
import { metricSeries, recorder, runTree } from '../../trace/index.ts'
import { constrainChains, parameterDraws } from '../chains.ts'
import { diagnose, divergenceReport, rhat, treeDepthReport } from '../diagnostics/index.ts'
import { banana, nealsFunnel, numericTarget } from '../target.ts'
import { nuts, nutsKernel } from './nuts.ts'

/** Exact draws of the bivariate Gaussian of gaussian2(mu, s, rho), `[n, 2]`. */
function gaussianDraws(
  n: number,
  mu: readonly [number, number],
  s: readonly [number, number],
  rho: number,
): Float64Array {
  const z = normal(key(99), [n, 2]).data
  const out = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    const a = z[2 * i] ?? 0
    const b = z[2 * i + 1] ?? 0
    out[2 * i] = mu[0] + s[0] * a
    out[2 * i + 1] = mu[1] + s[1] * (rho * a + Math.sqrt(1 - rho * rho) * b)
  }
  return out
}

describe('nutsKernel', () => {
  const mu = [1, -1] as const
  const s = [1, 2] as const
  const target = gaussian2(mu, s, 0.9)

  it('leaves a correlated Gaussian invariant (exact draws in, exact draws out)', () => {
    const k = nutsKernel(target, { stepSize: 0.3 })
    expectInvariant(k, gaussianDraws(1500, mu, s, 0.9), { mean: [...mu], sd: [...s] }, { key: key(1) })
  })

  it('stays invariant with a dense mass matrix and the McLachlan integrator', () => {
    const k = nutsKernel(target, {
      stepSize: 0.5,
      inverseMassMatrix: [
        [1, 1.8],
        [1.8, 4],
      ],
      integrator: 'mcLachlan',
    })
    expectInvariant(k, gaussianDraws(1500, mu, s, 0.9), { mean: [...mu], sd: [...s] }, { key: key(2) })
  })

  it('reports tree depth, integrator steps and acceptance', () => {
    const k = nutsKernel(standardNormal(3), { stepSize: 0.2, maxTreeDepth: 6 })
    const { info } = k.step(key(3), k.init([0.1, -0.2, 0.3]))
    expect(info.treeDepth).toBeGreaterThanOrEqual(1)
    expect(info.treeDepth).toBeLessThanOrEqual(6)
    expect(info.numSteps).toBeLessThanOrEqual(2 ** (info.treeDepth ?? 0) - 1)
    expect(info.acceptanceRate).toBeGreaterThan(0.9)
    expect(info.divergent).toBe(false)
  })

  it('caps the tree depth on a flat target (no U-turn ever happens)', () => {
    const flat = numericTarget(() => 0, 2, { grad: () => [0, 0] })
    const k = nutsKernel(flat, { stepSize: 0.1, maxTreeDepth: 5 })
    const { info } = k.step(key(4), k.init([0, 0]))
    expect(info.treeDepth).toBeLessThanOrEqual(5)
  })

  it('terminates on NaN densities and infinite step sizes, flagging divergences', () => {
    const broken = numericTarget(() => Number.NaN, 2, { grad: () => [Number.NaN, Number.NaN] })
    const k = nutsKernel(broken, { stepSize: 0.1 })
    const t = k.step(key(5), k.init([0, 0]))
    expect(t.info.divergent).toBe(true)
    expect(Array.from(t.state.position)).toEqual([0, 0])
    const huge = nutsKernel(standardNormal(2), { stepSize: Number.POSITIVE_INFINITY })
    const u = huge.step(key(6), huge.init([0.5, 0.5]))
    expect(u.info.divergent).toBe(true)
    expect(u.info.treeDepth).toBe(1)
  })

  it('rejects invalid settings', () => {
    expect(() => nutsKernel(target, { stepSize: 0 })).toThrow()
    expect(() => nutsKernel(target, { stepSize: 0.1, maxTreeDepth: 0 })).toThrow()
    expect(() =>
      nutsKernel(
        numericTarget(() => 0, 1),
        { stepSize: 0.1 },
      ),
    ).toThrow()
  })
})

describe('nuts (window adaptation and chains)', () => {
  it('samples a correlated Gaussian: moments within 4 MCSE, R-hat near 1, step size tuned', () => {
    const target = gaussian2([1, -1], [1, 2], 0.9)
    const c = nuts(target, { key: key(10), numChains: 2, numWarmup: 300, numDraws: 400 })
    expectMoments(c, { mean: [1, -1], sd: [1, 2] })
    expectMixing(c, 100)
    for (let i = 0; i < 2; i++) expect(rhat(parameterDraws(c, i))).toBeLessThan(1.05)
    const p = c.parameters[0] as { stepSize: number; inverseMassMatrix: number[] }
    expect(p.stepSize).toBeGreaterThan(0.05)
    expect(p.stepSize).toBeLessThan(2)
    // The diagonal of the adapted inverse mass matrix estimates the marginal variances 1 and 4.
    expect(p.inverseMassMatrix[0]).toBeGreaterThan(0.5)
    expect(p.inverseMassMatrix[1]).toBeGreaterThan(2)
    const mean = (c.stats.acceptanceRate?.data ?? new Float64Array()).reduce((a, b) => a + b, 0) / 800
    expect(mean).toBeGreaterThan(0.6)
  })

  it('learns a dense mass matrix close to the covariance', () => {
    const target = gaussian2([0, 0], [1, 1], 0.95)
    const c = nuts(target, { key: key(11), numChains: 1, numWarmup: 400, numDraws: 100, dense: true })
    const m = (c.parameters[0] as { inverseMassMatrix: number[] }).inverseMassMatrix
    expect(m.length).toBe(4)
    expect(m[1] ?? 0).toBeGreaterThan(0.5)
    expect(Math.abs((m[1] ?? 0) - (m[2] ?? 0))).toBeLessThan(1e-12)
  })

  it("samples Haario's banana (known moments)", () => {
    const b = banana({ curvature: 0.03 })
    const c = nuts(b, { key: key(12), numChains: 2, numWarmup: 300, numDraws: 300 })
    expectMoments(c, { mean: [0, 0], sd: [10, Math.sqrt(1 + 2 * 0.03 ** 2 * 10 ** 4)] }, 5)
  })

  it("samples Neal's funnel without divergences in the non-centred form", () => {
    const f = nealsFunnel(4, { nonCentred: true })
    const c = nuts(f, { key: key(13), numChains: 2, numWarmup: 200, numDraws: 300 })
    expect(divergenceReport(c).count).toBe(0)
    const v = constrainChains(c, f)
    expectMoments(v, { mean: [0, 0, 0, 0], sd: [3] })
  })

  it('diverges in the neck of the centred funnel, and diagnose says so', () => {
    // nealsFunnel(6) with a hand-written gradient, which is much faster than autodiff here.
    const n = 6
    const lp = (x: Float64Array): number => {
      const v = x[0] ?? 0
      let ss = 0
      for (let i = 1; i < n; i++) ss += (x[i] ?? 0) ** 2
      return (-v * v) / 18 - 0.5 * ss * Math.exp(-v) - 0.5 * (n - 1) * v
    }
    const funnel = numericTarget(lp, n, {
      grad: (x) => {
        const v = x[0] ?? 0
        let ss = 0
        for (let i = 1; i < n; i++) ss += (x[i] ?? 0) ** 2
        return [
          -v / 9 + 0.5 * ss * Math.exp(-v) - 0.5 * (n - 1),
          ...Array.from(x.subarray(1), (xi) => -xi * Math.exp(-v)),
        ]
      },
    })
    // The same density as the autodiff target, up to its normalising constant.
    const ad = nealsFunnel(n)
    const at = Float64Array.from([0.3, 0.1, -0.2, 0.5, 0.4, -1])
    const at2 = Float64Array.from([-1, 0.2, 0.2, -0.1, 0.3, 0.6])
    expect(lp(at) - lp(at2)).toBeCloseTo(ad.logDensity(at) - ad.logDensity(at2), 12)
    const c = nuts(funnel, { key: key(14), numChains: 2, numWarmup: 150, numDraws: 200, maxTreeDepth: 8 })
    const d = diagnose(c, { maxTreeDepth: 8 })
    expect(d.divergences?.count ?? 0).toBeGreaterThan(0)
    expect(d.hasErrors).toBe(true)
    expect(treeDepthReport(c, 8).meanDepth.length).toBe(2)
  })

  it('samples an equal mixture with overlapping modes', () => {
    const c = nuts(bimodal(1.2), { key: key(15), numChains: 2, numWarmup: 200, numDraws: 500 })
    expectMoments(c, { mean: [0], sd: [Math.sqrt(1 + 1.2 ** 2)] })
  })

  it('traces warm-up and sampling per chain, with per-iteration statistics', () => {
    const rec = recorder()
    nuts(standardNormal(2), { key: key(16), numChains: 2, numWarmup: 30, numDraws: 20, tracer: rec })
    const run = rec.run()
    const tree = runTree(run)
    expect(tree.length).toBe(1)
    expect(tree[0]?.children.length).toBe(4)
    const chain = tree[0]?.children[1]?.run.runId ?? ''
    expect(metricSeries(run, 'treeDepth', chain).values.length).toBe(20)
    expect(metricSeries(run, 'stepSize', chain).values.length).toBe(20)
    const warm = tree[0]?.children[0]?.run.runId ?? ''
    expect(metricSeries(run, 'stepSize', warm).values.length).toBeGreaterThan(20)
  })
})
