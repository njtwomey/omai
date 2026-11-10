// The structure reports of the benchmark suite: every model shows the findings written down for it, every Gaussian block is
// quadratic in its variables (jointly Gaussian given the rest), the collapse plan is exactly what the handler accepts,
// collapsed NUTS agrees with the truth and is far more efficient per gradient evaluation than the full model, and the
// lifted state-space program is read from its analysis.
import { describe, expect, it } from 'vitest'
import { expectPosteriorMoments } from '../../../testing/probprog.ts'
import { key } from '../../random/index.ts'
import { fromStorage, item, toFlat, type Value } from '../../tensor/index.ts'
import { collapse } from '../collapse.ts'
import { logJoint } from '../density.ts'
import { mcmc } from '../engines/nuts.ts'
import { seed, traceOf } from '../handlers.ts'
import { analyseSpec, applyCollapse, collapsePlan, fromProgram, toProgram } from '../model/index.ts'
import { eightSchoolsCentred } from './schools.ts'
import { ar1MarkovProgram, ar1StateSpace } from './state-space.ts'
import { STRUCTURE_EXPECTED, structureCampaigns, structureFindings } from './structure-campaigns.ts'
import { benchmarkModels } from './suite.ts'

const specModels = benchmarkModels().filter((m) => m.definition !== undefined)

describe('the findings of every benchmark model', () => {
  for (const m of specModels)
    it(m.name, () => {
      expect(structureFindings(m)).toEqual(STRUCTURE_EXPECTED[m.name])
    })

  it('the campaign passes and writes the table', () => {
    const [campaign] = structureCampaigns()
    const out = campaign?.run({ key: key(0), scale: 1 })
    expect(out?.passed).toBe(true)
    expect(out?.markdown).toMatch(/\| radon \| posterior \|/)
    expect(out?.markdown).toMatch(/Structure of radon/)
  })
})

describe('the Gaussian blocks are jointly Gaussian given the rest', () => {
  for (const m of specModels) {
    const bound = m.bound()
    const blocks = analyseSpec(bound).linearGaussianBlocks
    if (blocks.length === 0) continue
    it(`${m.name}: ${blocks.map((b) => `{${b.sites.join(', ')}}`).join(' ')}`, () => {
      const program = toProgram(bound)
      const t = traceOf(seed(program, key(3)), undefined)
      const base: Record<string, Value> = {}
      for (const [n, s] of t.sites) if (s.type === 'sample' && !s.isObserved) base[n] = s.value
      for (const b of blocks) {
        // The log joint along a line in the block's variables, the rest held at a draw from the prior: its second differences are constant.
        const at = (s: number): number => {
          const values: Record<string, Value> = { ...base }
          let k = 0
          // A scan's block is the whole series of its states, the sites `<scan>[t]` of the program.
          const sites = b.sites.flatMap((site) =>
            site in base ? [site] : Object.keys(base).filter((n) => n.startsWith(`${site}[`)),
          )
          for (const site of sites) {
            const v = base[site] as Value
            const moved = Array.from(toFlat(v), (x, i) => x + 0.4 * Math.sin(1 + k + i) + s * Math.cos(2 + k + i))
            values[site] = fromStorage(Float64Array.from(moved), v.shape, 'float64')
            k += moved.length
          }
          return item(logJoint(program, undefined, values).logJoint)
        }
        const h = 0.6
        const second = (s: number): number => at(s + h) - 2 * at(s) + at(s - h)
        expect(Math.abs(second(0) - second(0.7)) + Math.abs(second(0) - second(-1.1))).toBeLessThan(1e-8)
      }
    })
  }
})

describe('collapsePlan is what the handler accepts', () => {
  for (const m of specModels)
    it(m.name, () => {
      const bound = m.bound()
      const plan = collapsePlan(bound)
      const taken = (site: string): boolean => {
        try {
          collapse(toProgram(bound), undefined, { sites: [site] })
          return true
        } catch {
          return false
        }
      }
      for (const s of plan.sites) expect(taken(s), `${m.name}: ${s} is planned but refused`).toBe(true)
      const latents = bound.spec.nodes.filter((n) => n.kind === 'latent').map((n) => n.name)
      const planned = new Set(plan.sites)
      for (const s of latents) {
        const other = new Set([...planned].filter((p) => p !== s))
        // A site the handler takes alone is in the plan, or the plan says why not.
        if (taken(s) && !planned.has(s))
          expect(
            plan.refused.some((r) => r.site === s) || other.size > 0,
            `${m.name}: ${s} is accepted by the handler but neither planned nor refused`,
          ).toBe(true)
      }
    })
})

describe('collapsed NUTS on eight schools', () => {
  const bound = eightSchoolsCentred.bound()
  const plan = collapsePlan(bound)
  const collapsed = applyCollapse(bound, plan)
  const options = { key: key(1), numChains: 2, numWarmup: 300, numDraws: 400, targetAcceptance: 0.9 }
  const full = mcmc(toProgram(bound), undefined, options)
  const reduced = mcmc(collapsed.model, undefined, options)
  const truth = eightSchoolsCentred.truth(undefined, 'fast')
  const gradients = (p: ReturnType<typeof mcmc>): number =>
    toFlat(p.chains?.stats.numSteps as Value).reduce((a, b) => a + b, 0)
  const essOf = (p: ReturnType<typeof mcmc>, name: string): number =>
    p.summary().find((r) => r.name === name)?.ess ?? Number.NaN

  it('plans theta alone, in 2 of 10 coordinates', () => {
    expect(plan).toMatchObject({ sites: ['theta'], remaining: ['mu', 'tau'], dimensionBefore: 10, dimensionAfter: 2 })
  })

  it('agrees with the truth on mu and tau, and the recovered effects agree on theta', () => {
    for (const site of ['mu', 'tau']) {
      const oracle = truth.marginals[site]
      expect(oracle).toBeDefined()
      expectPosteriorMoments(reduced, site, { mean: oracle?.mean ?? 0 }, { z: 5, minEss: 100 })
    }
    // theta given (mu, tau) from recover, one draw per draw of (mu, tau): its mean is the truth's.
    const mu = toFlat(reduced.draws?.['mu'] as Value)
    const tau = toFlat(reduced.draws?.['tau'] as Value)
    const n = Math.min(300, mu.length)
    let sum = 0
    for (let i = 0; i < n; i++) {
      const r = collapsed.recover({ mu: mu[i] as number, tau: tau[i] as number }, undefined, key(100 + i))
      sum += toFlat(r['theta'] as Value)[0] as number
    }
    const oracle = truth.marginals['theta[0]']
    expect(Math.abs(sum / n - (oracle?.mean ?? 0))).toBeLessThan((5 * (oracle?.sd ?? 1)) / Math.sqrt(100))
  })

  it('is more efficient per gradient evaluation than the full model, with no divergences', () => {
    const perGrad = (p: ReturnType<typeof mcmc>, name: string): number => essOf(p, name) / gradients(p)
    expect(perGrad(reduced, 'tau') / perGrad(full, 'tau')).toBeGreaterThan(5)
    expect(perGrad(reduced, 'mu') / perGrad(full, 'mu')).toBeGreaterThan(2)
    expect((reduced.diagnostics['divergences'] as number) ?? 0).toBe(0)
  })
})

describe('the AR(1) state space', () => {
  it('is a scan: the analysis finds the Gaussian chain that the lift of the hand-written program cannot see', () => {
    const a = analyseSpec(ar1StateSpace.bound())
    expect(a.linearGaussianBlocks).toEqual([
      expect.objectContaining({ sites: ['x'], given: [], observed: ['y'], dimension: 12, whole: true }),
    ])
    expect(a.explain()).toContain('Linear-Gaussian blocks')
  })
  it('as a markov program is lifted to its site structure with every mean opaque, and read from the program analysis', () => {
    const lifted = fromProgram(ar1MarkovProgram(ar1StateSpace.data.data['y'] as number[]), undefined)
    expect(lifted.spec.nodes.filter((n) => n.kind === 'latent')).toHaveLength(
      lifted.spec.nodes.filter((n) => n.kind === 'observed').length,
    )
    expect(lifted.opaque.length).toBeGreaterThan(0)
    // Without expressions the lifted spec claims no conjugacy and no Gaussian block: the truth is the Kalman smoother's.
    const a = analyseSpec(lifted.spec)
    expect(a.conjugacy).toEqual([])
    expect(a.linearGaussianBlocks).toEqual([])
  })
})
