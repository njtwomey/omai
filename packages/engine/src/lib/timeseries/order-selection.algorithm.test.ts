import { describe, expect, it } from 'vitest'
import { key, normal, split } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../trace/index.ts'
import { arOrderAlgorithm, arOrderContract, criterionPenalty, selectArOrder } from './arma-fit.ts'
import { cointegrationRankAlgorithm, rankContract, selectCointegrationRank } from './cointegration.ts'

const y = [
  0.1, -0.2, 0.5, 0.9, 0.3, -0.4, -0.8, -0.1, 0.6, 1.0, 0.4, -0.3, -0.7, -0.2, 0.5, 0.8, 0.2, -0.5, -0.6, 0.0, 0.7, 0.9,
  0.1, -0.4,
]

// The scores the function returned before it became a run of the algorithm.
const OLD_AIC = [36.46865924432228, 33.57959248630306, -12.30483426802597, -29.701710553056067, -27.70985244803093]
const OLD_BIC = [38.46012379143026, 36.566789306965035, -8.321905173810007, -24.723049185286115, -21.735458806706983]

function system(seed: number, n: number): number[][] {
  const [k1, k2, k3] = split(key(seed), 3)
  const e1 = toFlat(normal(k1 as never, [n]))
  const e2 = toFlat(normal(k2 as never, [n]))
  const e3 = toFlat(normal(k3 as never, [n]))
  const rows: number[][] = []
  let x = 0
  let s = 0
  let z = 0
  for (let t = 0; t < n; t++) {
    x += e1[t] ?? 0
    s = 0.4 * s + 0.5 * (e2[t] ?? 0)
    z += e3[t] ?? 0
    rows.push([x, 1.5 * x + s, z])
  }
  return rows
}

describe('arOrderAlgorithm', () => {
  it('is selectArOrder: the sweep run to the end gives the scores the function gave before', () => {
    const aic = runAlgorithm(arOrderAlgorithm(y, 4), { maxSteps: 5 }).state
    aic.scores.forEach((v, i) => {
      expect(v).toBeCloseTo(OLD_AIC[i] ?? 0, 10)
    })
    expect(aic.best).toBe(3)
    const bic = runAlgorithm(arOrderAlgorithm(y, 4, 'bic'), { maxSteps: 5 }).state
    bic.scores.forEach((v, i) => {
      expect(v).toBeCloseTo(OLD_BIC[i] ?? 0, 10)
    })
    expect(selectArOrder(y, 4, 'bic')).toEqual({ order: 3, scores: bic.scores })
  })

  it('has a state that is JSON, and a run resumes from a checkpoint to the same end', () => {
    const alg = arOrderAlgorithm(y, 4)
    const whole = runAlgorithm(alg, { maxSteps: 5 })
    const half = runAlgorithm(alg, { maxSteps: 2 })
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
    const rest = runAlgorithm(alg, { maxSteps: 3, start: { state: stored, step: 2, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
  })

  it('logs the order, its criterion and the best so far by path, and the best never gets worse', () => {
    const rec = recorder()
    const final = runAlgorithm(arOrderAlgorithm(y, 4), {
      maxSteps: 5,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, roles: ['output'], detail: 'full' },
    }).state
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('select/order')).toEqual([0, 1, 2, 3, 4])
    expect(metric('select/criterion')).toEqual([...final.scores])
    const best = metric('select/best').map((i) => final.scores[i] ?? 0)
    expect(best.every((v, i) => i === 0 || v <= (best[i - 1] ?? 0))).toBe(true)
    expect(Object.keys(arOrderContract)).toHaveLength(4)
  })

  it('the penalties rank the criteria and every criterion runs', () => {
    expect(criterionPenalty('aic', 100)).toBe(2)
    expect(criterionPenalty('bic', 100)).toBeGreaterThan(criterionPenalty('aic', 100))
    expect(criterionPenalty('hqic', 100)).toBeCloseTo(2 * Math.log(Math.log(100)), 12)
    for (const c of ['aic', 'bic', 'hqic'] as const) expect(selectArOrder(y, 4, c).order).toBe(3)
  })
})

describe('cointegrationRankAlgorithm', () => {
  const rows = system(2, 120)

  it('is selectCointegrationRank: the same rank, statistics and critical values as before', () => {
    const r = selectCointegrationRank(rows, 0, 1, 'trace', 0.1)
    expect(r.rank).toBe(1)
    expect(r.statistics[0]).toBeCloseTo(50.20057874, 7)
    expect(r.statistics[1]).toBeCloseTo(8.31370321, 7)
    expect(r.critical.slice(0, 2)).toEqual([27.0669, 13.4294])
    const m = selectCointegrationRank(rows, 0, 1, 'maxeig', 0.01)
    expect(m.rank).toBe(1)
    expect(m.critical.slice(0, 2)).toEqual([25.865, 18.52])
  })

  it('tests ranks one at a time and stops at the first it accepts; resumes from a checkpoint', () => {
    const alg = cointegrationRankAlgorithm(rows)
    const run = runAlgorithm(alg, { maxSteps: 10 })
    expect(run.steps).toBe(2)
    expect(run.state.rank).toBe(1)
    const half = runAlgorithm(alg, { maxSteps: 1 })
    expect(half.state.rank).toBeNull()
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
    const rest = runAlgorithm(alg, { maxSteps: 10, start: { state: stored, step: 1, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(run.state))
    const rec = recorder()
    runAlgorithm(alg, {
      maxSteps: 10,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('select/rank')).toEqual([0, 1])
    expect(metric('select/accepted')).toEqual([0, 1])
    expect(Object.keys(rankContract)).toHaveLength(4)
  })

  it('runs every test method and significance level; the stricter level accepts no later a rank', () => {
    for (const method of ['trace', 'maxeig'] as const) {
      const ranks = ([0.1, 0.05, 0.01] as const).map((s) => selectCointegrationRank(rows, 0, 1, method, s).rank)
      expect(ranks.every((r, i) => i === 0 || r <= (ranks[i - 1] ?? 0))).toBe(true)
    }
  })
})
