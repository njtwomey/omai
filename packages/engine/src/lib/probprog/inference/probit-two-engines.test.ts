// The vertical slice of the probabilistic-programming plan (7.2, steps 3 to 5): one probit regression, written once in the model
// builder, run by NUTS through the compiled program and by EP through the compiled factor graph, against the exact grid answer
// (EP's means within 0.05 posterior standard deviations, NUTS within Monte Carlo error), with Laplace beside them in the
// moments table; the tapes of both runs; both run by name from a model registry with only JSON crossing, equal to the
// main-thread runs; and an anytime snapshot every second of an injected clock from each.
import { describe, expect, it } from 'vitest'
import { expectSnapshotCadence } from '../../../testing/methods.ts'
import { type MomentRow, momentTable, probitData, probitGrid, probitModel } from '../../../testing/probit-slice.ts'
import { toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { analyseSpec, boundFromJSON, boundToJSON, toFactorGraph } from '../model/index.ts'
import { laplaceMethod } from './laplace-method.ts'
import { inferenceMethodFor, inferenceMethods, modelRegistry } from './models.ts'
import { epSpecMethod } from './mp.ts'
import { nutsMethod } from './nuts-method.ts'

const { X, y } = probitData()
const bound = probitModel.bind({ X, y })
const grid = probitGrid(X, y, 2)
const nutsKnobs = { numChains: 2, numWarmup: 250, numDraws: 400 }

const ep = epSpecMethod(bound)
const epRun = session(ep, { config: { tol: 1e-10 }, seed: 1, steps: 300, mode: 'eager' })
const nuts = nutsMethod(bound)
const nutsRun = session(nuts, { config: nutsKnobs, seed: 1, steps: 5000, mode: 'eager' })
const laplace = laplaceMethod(bound)
const laplaceRun = session(laplace, { seed: 1, steps: 500, mode: 'eager' })
const posteriors = {
  ep: ep.posterior?.(epRun.state, epRun.config),
  nuts: nuts.posterior?.(nutsRun.state, nutsRun.config),
  laplace: laplace.posterior?.(laplaceRun.state, laplaceRun.config),
}
const table: MomentRow[] = momentTable(grid, posteriors as never)
const rows = (method: string): MomentRow[] => table.filter((r) => r.method === method)

describe('the probit regression written once, run by NUTS and by EP, against the exact grid', () => {
  it('has the exact answer on a grid to compare with, and every run ended', () => {
    expect(grid.sd.every((s) => s > 0.1 && s < 0.5)).toBe(true)
    expect(epRun.status).toBe('converged')
    expect(nutsRun.status).toBe('converged')
    expect(laplaceRun.status).toBe('converged')
  })

  it('EP means are within 0.05 posterior standard deviations of the grid, with a spread within 10 per cent', () => {
    for (const r of rows('ep')) {
      expect(Math.abs(r.meanError), r.coordinate).toBeLessThan(0.05)
      expect(r.sdRatio, r.coordinate).toBeGreaterThan(0.9)
      expect(r.sdRatio, r.coordinate).toBeLessThan(1.1)
    }
    const evidence = posteriors.ep?.logEvidence?.value ?? Number.NaN
    expect(Math.abs(evidence - grid.logEvidence)).toBeLessThan(0.05)
  })

  it('NUTS means are within three Monte Carlo standard errors of the grid', () => {
    const summary = posteriors.nuts?.summary() ?? []
    expect(summary).toHaveLength(3)
    for (const [q, row] of summary.entries()) {
      const mcse = row.sd / Math.sqrt(row.ess ?? 1)
      expect(Math.abs(row.mean - (grid.mean[q] as number)), row.name).toBeLessThan(3 * mcse)
      expect(row.rhat ?? 2, row.name).toBeLessThan(1.05)
      expect(row.sd / (grid.sd[q] as number), row.name).toBeGreaterThan(0.85)
      expect(row.sd / (grid.sd[q] as number), row.name).toBeLessThan(1.15)
    }
    // A handful of divergent transitions among 800 draws is what four hundred warm-up iterations leave on a probit posterior.
    expect(posteriors.nuts?.diagnostics['divergences'] as number).toBeLessThanOrEqual(8)
  })

  it('reports EP and Laplace with their KL from the grid and their errors in standard deviations; EP is closer on the intercept', () => {
    const kl = (method: string): number[] => rows(method).map((r) => r.kl)
    expect(kl('ep').every((v) => v >= 0 && v < 0.01)).toBe(true)
    // The expected result of the plan: moment matching beats the Laplace approximation on the intercept (a finding if not).
    expect((kl('ep')[0] as number) < (kl('laplace')[0] as number)).toBe(true)
    // Laplace is a Gaussian at the mode: it is the approximation with the larger errors on this data.
    expect(Math.max(...kl('laplace'))).toBeGreaterThan(Math.max(...kl('ep')))
    expect(table.map((r) => r.method)).toEqual([
      'ep',
      'ep',
      'ep',
      'nuts',
      'nuts',
      'nuts',
      'laplace',
      'laplace',
      'laplace',
    ])
  })
})

describe('the tape of both runs', () => {
  it('EP emits messages between the ids of the graph and the marginals by variable id, every sweep', () => {
    const s = session(ep, { config: { tol: 1e-10 }, seed: 1, steps: 300, mode: 'eager', logs: 'all' })
    const run = s.run()
    const graph = toFactorGraph(bound).graph
    const variables = new Set(graph?.variables.map((v) => v.name))
    const factors = new Set(graph?.factors.map((f) => f.id))
    const messages = run.events.filter((e) => e.type === 'message' && e.kind === 'mp/message')
    expect(messages.length).toBeGreaterThan(0)
    for (const m of messages) {
      expect(factors.has((m as { from?: string }).from ?? ''), 'factor').toBe(true)
      expect(variables.has((m as { to?: string }).to ?? ''), 'variable').toBe(true)
    }
    const marginals = run.events.filter((e) => e.type === 'message' && e.kind === 'mp/marginals')
    expect(marginals).toHaveLength(s.t + 1)
    expect(Object.keys((marginals.at(-1) as { payload: object }).payload)).toContain('w')
    // The marginal of a weight at the last step is the posterior the method reads.
    const w = (marginals.at(-1) as unknown as { payload: Record<string, { mean: number[] }> }).payload['w']?.mean
    expect(w?.[0]).toBeCloseTo(toFlat(posteriors.ep?.mean('w') as never)[0] as number, 10)
  })

  it('NUTS emits per-iteration scalars (step size, acceptance, divergences) and a snapshot of the posterior so far', () => {
    const s = session(nuts, {
      config: { numChains: 1, numWarmup: 20, numDraws: 10 },
      steps: 100,
      mode: 'eager',
      logs: 'all',
    })
    const names = new Set(
      s.run().events.flatMap((e) => ('name' in e ? [e.name] : e.type === 'message' ? [e.kind] : [])),
    )
    for (const p of ['mcmc/stepSize', 'mcmc/accept', 'mcmc/divergent'])
      expect(names.has(p), `${p} in ${[...names]}`).toBe(true)
    expect(s.probe(s.t, 'mcmc/snapshot')).toMatchObject({ engine: 'nuts' })
  })
})

describe('both run by name from a model registry, with only JSON crossing', () => {
  const models = modelRegistry({ probit: { definition: probitModel, data: { X, y } } })
  const table = new Map(inferenceMethods(models, ['nuts', 'ep']).map((m) => [m.name, m]))
  const wire = (name: string, config: unknown, seed: number, steps: number) => {
    const m = table.get(name)
    if (m === undefined) throw new Error(`no method called '${name}'`)
    return { m, s: session(m, { config: JSON.parse(JSON.stringify(config)), seed, steps, mode: 'eager' }) }
  }

  it('offers ep/probit and nuts/probit, and the bound model survives the trip through JSON', () => {
    expect([...table.keys()]).toEqual(['nuts/probit', 'ep/probit'])
    const there = boundFromJSON(JSON.parse(JSON.stringify(boundToJSON(bound))))
    expect(there.sizes).toEqual(bound.sizes)
    expect(toFactorGraph(there).graph).toEqual(toFactorGraph(bound).graph)
    expect(analyseSpec(there).features).toEqual(analyseSpec(bound).features)
  })

  it('the run by name equals the main-thread run: the same marginals for EP, the same draws for NUTS', () => {
    const remoteEp = wire('ep/probit', { tol: 1e-10 }, 1, 300)
    expect(remoteEp.s.t).toBe(epRun.t)
    expect(toFlat(remoteEp.m.posterior?.(remoteEp.s.state, remoteEp.s.config)?.mean('w') as never)).toEqual(
      toFlat(posteriors.ep?.mean('w') as never),
    )
    const remoteNuts = wire('nuts/probit', nutsKnobs, 1, 5000)
    expect(remoteNuts.s.t).toBe(nutsRun.t)
    const a = remoteNuts.m.posterior?.(remoteNuts.s.state, remoteNuts.s.config)
    expect(toFlat(a?.draws?.['w'] as never)).toEqual(toFlat(posteriors.nuts?.draws?.['w'] as never))
  })

  it('takes the data of a run from JSON: flipped labels, sent with the request, give the mirror-image EP posterior', () => {
    const flipped = JSON.parse(JSON.stringify({ y: y.map((v) => 1 - v) }))
    const other = inferenceMethodFor(models, 'ep', 'probit', flipped)
    const s = session(other, { config: { tol: 1e-10 }, seed: 1, steps: 300, mode: 'eager' })
    const w0 = toFlat(posteriors.ep?.mean('w') as never)
    const w1 = toFlat(other.posterior?.(s.state, s.config)?.mean('w') as never)
    // Probit is symmetric: flipping every label flips the sign of every weight.
    for (const [q, v] of w0.entries()) expect(w1[q]).toBeCloseTo(-v, 8)
  })
})

describe('both give an anytime snapshot every second of an injected clock', () => {
  it('EP: a snapshot a second, each a posterior that reads, the last flagged final', () => {
    const snaps = expectSnapshotCadence(ep, {
      config: { granularity: 'message', tol: 1e-12 },
      steps: 100_000,
      budgetMs: 3500,
      everyMs: 1000,
      tick: 100,
    })
    expect(snaps.length).toBeGreaterThanOrEqual(3)
  })

  it('NUTS: the same cadence, each snapshot the draws so far', () => {
    const snaps = expectSnapshotCadence(nuts, {
      config: { numChains: 1, numWarmup: 30, numDraws: 400 },
      steps: 5000,
      budgetMs: 3500,
      everyMs: 1000,
      tick: 100,
    })
    expect(snaps.length).toBeGreaterThanOrEqual(3)
  })
})
