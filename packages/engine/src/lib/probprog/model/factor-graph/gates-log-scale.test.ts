// A mixture whose components each have a mean and a spread of their own (`y ~ N(mu[z], exp(logSd[z]))`): the mean and the spread are
// both chosen by the same discrete variable, so each reading is one `gate` on it whose branch `k` is the `gaussian-scale` factor on
// the mean and the log spread of component `k`, so the component that explains the reading is the one whose mean and spread both move
// (a select of each would lose which component the pair came from).
// The reference is the four-dimensional grid of the two means and the two log spreads with the labels summed out.
import { describe, expect, it } from 'vitest'
import { session } from '../../../trace/index.ts'
import { epSpecMethod } from '../../inference/index.ts'
import { analysisOf } from '../../inference/model-input.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops, toFactorGraph } from '../index.ts'

const Y = [-2.1, -1.7, -2.4, 2, 2.6, 3.5]
const MEAN_PRIOR = [-2, 2]
const MEAN_SD = 2
const LOG_SD = 0.5

const model = defineModel('mixture-spreads', (m) => {
  const row = m.range('row')
  const comp = m.range('comp', { size: 2 })
  const y = m.data('y', { dims: [row] })
  const mu = m.latent('mu', d.normal(m.const(MEAN_PRIOR, [comp]), MEAN_SD), { dims: [comp] })
  const logSd = m.latent('logSd', d.normal(0, LOG_SD), { dims: [comp] })
  const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
  m.observe('y', d.normal(ops.at(mu, z), ops.exp(ops.at(logSd, z))), y)
})

const norm = (x: number, m: number, s: number): number =>
  Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI))

/** The posterior moments of the two means and the two log spreads on a grid, the labels summed out. */
function grid() {
  const nm = 61
  const ns = 41
  const mus = Array.from({ length: nm }, (_, i) => -8 + (16 * i) / (nm - 1))
  const logs = Array.from({ length: ns }, (_, i) => -2 + (4 * i) / (ns - 1))
  const sum = [0, 0, 0, 0]
  const sq = [0, 0, 0, 0]
  let total = 0
  let top = Number.NEGATIVE_INFINITY
  const cells: { w: number; v: number[] }[] = []
  for (const m0 of mus)
    for (const m1 of mus)
      for (const s0 of logs)
        for (const s1 of logs) {
          let lp =
            Math.log(norm(m0, MEAN_PRIOR[0] as number, MEAN_SD)) +
            Math.log(norm(m1, MEAN_PRIOR[1] as number, MEAN_SD)) +
            Math.log(norm(s0, 0, LOG_SD)) +
            Math.log(norm(s1, 0, LOG_SD))
          for (const y of Y) lp += Math.log(0.5 * norm(y, m0, Math.exp(s0)) + 0.5 * norm(y, m1, Math.exp(s1)))
          cells.push({ w: lp, v: [m0, m1, s0, s1] })
          top = Math.max(top, lp)
        }
  for (const c of cells) {
    const w = Math.exp(c.w - top)
    total += w
    c.v.forEach((x, k) => {
      sum[k] = (sum[k] as number) + w * x
      sq[k] = (sq[k] as number) + w * x * x
    })
  }
  return sum.map((s, k) => {
    const mean = s / total
    return { mean, sd: Math.sqrt((sq[k] as number) / total - mean * mean) }
  })
}

describe('a mixture with a mean and a spread per component', () => {
  const bound = model.bind({ y: Y })

  it('compiles to one gate per reading whose branches are gaussian-scale factors', () => {
    const c = toFactorGraph(bound)
    expect(c.issues).toEqual([])
    const gates = c.graph?.factors.filter((f) => f.operator === 'gate') ?? []
    expect(gates.length).toBe(Y.length)
    const branches = ((gates[0]?.params ?? { branches: [] }) as unknown as { branches: { operator: string }[] })
      .branches
    expect(branches.map((b) => b.operator)).toEqual(['gaussian-scale', 'gaussian-scale'])
  })

  it('agrees with the grid: every mean within a quarter of a sd, the spreads between half and 1.2 times the grids', () => {
    const method = epSpecMethod(bound)
    expect(method.check?.(analysisOf(bound)).applies).toBe(true)
    const run = session(method, { steps: 400, mode: 'eager', config: { tol: 1e-9, damping: 0.5 } })
    const post = method.posterior?.(run.state, run.config)
    const ref = grid()
    const rows = post?.summary() ?? []
    const names = ['mu[0]', 'mu[1]', 'logSd[0]', 'logSd[1]']
    for (const [k, name] of names.entries()) {
      const r = rows.find((q) => q.name === name)
      const t = ref[k] as { mean: number; sd: number }
      expect(Math.abs((r?.mean ?? Number.NaN) - t.mean) / t.sd).toBeLessThan(0.25)
      // One Gaussian per variable against a posterior that is a mixture over the labels of the ambiguous readings: the spreads of the
      // means come out at 0.57 to 0.71 of the grid's, those of the log spreads at 0.97 to 1.1.
      expect((r?.sd ?? Number.NaN) / t.sd).toBeGreaterThan(0.5)
      expect((r?.sd ?? Number.NaN) / t.sd).toBeLessThan(1.2)
    }
  })
})
