import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import { hsicTest, mmdTest, nullContract, permutationNullAlgorithm } from './kernel.ts'

const x = [0, 0.1, 0.2, 0.3, 0.4]
const y = [5, 5.1, 5.2, 5.3, 5.4]

describe('permutationNullAlgorithm', () => {
  const statistic = (perm: ArrayLike<number>): number => (perm[0] ?? 0) + (perm[1] ?? 0)

  it('keeps the numbers the tests had: the p-value of mmdTest is the one it always gave', () => {
    expect(mmdTest(x, y, { seed: 0, nResamples: 199 }).pValue).toBe(0.025)
    const xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    expect(
      hsicTest(
        xs,
        xs.map((v) => v * v),
        { null: 'gamma' },
      ).pValue,
    ).toBeLessThan(0.01)
  })

  it('has a state that is JSON, and a run resumes to the same null', () => {
    const alg = permutationNullAlgorithm('demo', 350, 8, statistic, { observed: 12 })
    const whole = runAlgorithm(alg, { key: key(1), maxSteps: 100 })
    const half = runAlgorithm(alg, { key: key(1), maxSteps: 1 }).state
    expect(half.done).toBe(100)
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
    const rest = runAlgorithm(alg, { key: key(1), maxSteps: 99, start: { state: stored, step: 1, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
    expect(whole.state.nullDistribution).toHaveLength(350)
    expect(whole.status).toBe('converged')
  })

  it('logs the last null statistic, the running p-value and the count at each step', () => {
    const rec = recorder()
    runAlgorithm(permutationNullAlgorithm('demo', 300, 8, statistic, { observed: 12, batch: 100 }), {
      key: key(1),
      maxSteps: 10,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, detail: 'full' },
    })
    const metric = (n: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === n ? [e.value] : []))
    expect(metric('null/count')).toEqual([100, 200, 300])
    expect(metric('null/pvalue')).toHaveLength(3)
    expect(metric('null/pvalue').every((p) => p > 0 && p <= 1)).toBe(true)
    expect(metric('null/statistic')).toHaveLength(3)
    expect(Object.keys(nullContract)).toEqual(['null/statistic', 'null/pvalue', 'null/count'])
  })

  it('a batch of one gives the same null count with finer steps; the final running p-value is the permutation p-value', () => {
    const rec = recorder()
    const r = runAlgorithm(permutationNullAlgorithm('demo', 40, 6, statistic, { observed: 5, batch: 1 }), {
      key: key(2),
      maxSteps: 100,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, detail: 'full' },
    })
    expect(r.steps).toBe(40)
    const p = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'null/pvalue' ? [e.value] : []))
    const nulls = r.state.nullDistribution
    const expected = (nulls.filter((v) => v >= 5 - 5e-14).length + 1) / (nulls.length + 1)
    expect(p.at(-1)).toBeCloseTo(expected, 12)
  })
})
