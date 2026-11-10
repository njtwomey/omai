import { normal } from 'omai-engine/dists'
import { type ElboRun, elboRunFromTape } from 'omai-engine/exposition'
import { observe, sample } from 'omai-engine/probprog'
import { pathfinderMethod, sviMethod } from 'omai-engine/probprog/inference'
import { distributionOracle } from 'omai-engine/probprog/validation'
import { Log, session } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import { elboRunView, hasElboChart } from './elbo-run.ts'

// m ~ N(0, 1), y ~ N(m, 1) with y = 2: the posterior is N(1, 1/2).
const model = (): void => {
  const m = sample('m', normal(0, 1))
  observe('y', normal(m, 1), 2)
}
const truth = { m: distributionOracle('m', normal(1, Math.SQRT1_2)) }
const svi = session(sviMethod(model), {
  steps: 300,
  seed: 3,
  mode: 'eager',
  config: { learningRate: 0.05 },
  logs: { 'svi/elbo': Log(), 'svi/smoothed': Log(), 'svi/params/*': Log(), 'svi/snapshot': Log() },
})
const record: ElboRun = elboRunFromTape({ run: svi.run() }, { truth })

type Datum = [number, number] | { value: [number, number] }
const named = (o: { series?: unknown }, name: string): { name?: string; data?: Datum[] }[] =>
  (o.series as { name?: string; data?: Datum[] }[]).filter((s) => s.name === name)

describe('variational run charts', () => {
  it('knows which charts the record has the data for', () => {
    expect(
      ['elbo', 'gradient', 'parameters', 'beliefs', 'posterior'].map((c) => hasElboChart(record, c as never)),
    ).toEqual([true, false, true, true, true])
  })

  it('draws the ELBO raw and smoothed, with the last value and the state of the run in the title', () => {
    const o = elboRunView(record)
    expect(named(o, 'ELBO')[0]?.data?.length).toBe(record.elbo.steps.length)
    expect(named(o, 'smoothed')[0] ?? named(o, `mean of ${record.smoothing}`)[0]).toBeDefined()
    expect(JSON.stringify(o)).not.toContain('running')
    const early = elboRunView({ ...record, status: 'running' })
    expect(JSON.stringify(early)).toContain('running')
  })

  it('draws the parameters by role, the beliefs with their sd band, and the final posterior over the truth', () => {
    const p = elboRunView(record, { chart: 'parameters' })
    expect((p.grid as unknown[]).length).toBe(2)
    expect(named(p, 'm_auto_loc')).toHaveLength(1)
    const b = elboRunView(record, { chart: 'beliefs' })
    expect(named(b, 'm')[0]?.data?.length).toBe(record.beliefs[0]?.steps.length)
    expect(named(b, '__sd')).toHaveLength(1)
    const f = elboRunView(record, { chart: 'posterior' })
    expect(named(f, 'truth')[0]?.data).toHaveLength(160)
    expect(named(f, 'svi')[0]?.data).toHaveLength(160)
  })

  it('draws the gradient on log axes when it spans decades, and says what is missing', () => {
    const g: ElboRun = {
      ...record,
      gradNorm: { steps: [1, 2, 3, 4], values: [100, 10, 1, 0.01] },
      gradVariance: { steps: [2, 4], values: [5, 4] },
    }
    const o = elboRunView(g, { chart: 'gradient' })
    expect((o.grid as unknown[]).length).toBe(2)
    expect(JSON.stringify(o.yAxis)).toContain('"log"')
    expect(() => elboRunView(record, { chart: 'gradient' })).toThrow(expect.objectContaining({ code: 'value' }))
  })

  it('draws a run in progress, and one that has only snapshots', () => {
    const half = elboRunFromTape({ run: { events: svi.run().events.slice(0, 120) } })
    expect(half.status).toBe('running')
    expect(named(elboRunView(half), 'ELBO')[0]?.data?.length).toBe(half.elbo.steps.length)
    const snaps = elboRunFromTape({
      snapshots: [{ step: 5, snapshot: { summary: [{ site: 'm', name: 'm', mean: 0.1, sd: 1 }] } }],
    })
    // No bound yet: the default chart is the first with data.
    expect((elboRunView(snaps).grid as unknown[]).length).toBe(1)
    expect(() => elboRunView(snaps, { chart: 'elbo' })).toThrow(expect.objectContaining({ code: 'value' }))
  })

  it("draws Pathfinder's paths beside the best ELBO and its k-hat in the title", () => {
    const ps = session(pathfinderMethod(model), {
      steps: 60,
      mode: 'eager',
      config: { maxIterations: 10 },
      logs: {
        'pathfinder/elbo': Log(),
        'pathfinder/path/*': Log(),
        'pathfinder/paretoK': Log(),
        'pathfinder/ess': Log(),
      },
    })
    const r = elboRunFromTape({ run: ps.run() })
    const o = elboRunView(r)
    expect(named(o, 'best ELBO')).toHaveLength(1)
    expect(named(o, 'path 0')).toHaveLength(1)
    expect(JSON.stringify(o)).toContain('Pareto k')
  })
})
