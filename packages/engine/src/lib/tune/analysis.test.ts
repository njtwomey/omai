import { describe, expect, it } from 'vitest'
import { choice, dict, real } from '../space/index.ts'
import {
  bestParamsByObjective,
  contourData,
  empiricalDistribution,
  functionObjective,
  hypervolumeHistory,
  intermediateValues,
  multiObjective,
  optimisationHistory,
  parallelCoordinates,
  paretoFrontData,
  sliceData,
  trialsTable,
  tune,
} from './index.ts'

const space = dict({ x: real(0, 1), c: choice(['a', 'b']), lr: real(1e-3, 1, { log: true }) })
const study = tune(
  functionObjective((p, t) => {
    const { x, c } = p as { x: number; c: string }
    t.report(0, x)
    return x + (c === 'a' ? 0 : 1)
  }),
  space,
  { nTrials: 20, seed: 0 },
).study

describe('analysis data', () => {
  it('tables, histories and distributions', () => {
    const table = trialsTable(study)
    expect(table.columns['params_c']?.kind).toBe('categorical')
    expect(table.columns['value']?.kind).toBe('numeric')
    const h = optimisationHistory(study)
    expect(h.best.every((b, i) => i === 0 || b <= (h.best[i - 1] as number))).toBe(true)
    expect(intermediateValues(study).length).toBe(20)
    const edf = empiricalDistribution(study)
    expect(edf.fraction.at(-1)).toBe(1)
    expect(() => optimisationHistory(study, 3)).toThrow()
  })
  it('parallel coordinates, slices and contours', () => {
    const pc = parallelCoordinates(study)
    expect(pc.axes.map((a) => a.path)).toEqual(['x', 'c', 'lr'])
    expect(pc.axes[1]?.labels).toEqual(['a', 'b'])
    expect(sliceData(study)[0]?.x.length).toBe(20)
    const c = contourData(study, 'x', 'c', { gridSize: 4 })
    expect(c.z.length).toBe(2)
    expect(c.z[0]?.length).toBe(4)
    expect(c.y.ticks).toEqual([0, 1])
    expect(() => contourData(study, 'x', 'nope')).toThrow()
  })
  it('fronts and hypervolume histories', () => {
    const two = tune(
      multiObjective([functionObjective((x) => x as number), functionObjective((x) => 1 - (x as number) ** 2)]),
      real(0, 1),
      { nTrials: 10, seed: 0 },
    ).study
    const data = paretoFrontData(two)
    expect(data.front.numbers.length + data.dominated.numbers.length).toBe(10)
    const hv = hypervolumeHistory(two, [1, 1])
    expect(hv.every((v, i) => i === 0 || v >= (hv[i - 1] as number))).toBe(true)
    expect(bestParamsByObjective(two).length).toBe(2)
    expect(() => hypervolumeHistory(two, [1])).toThrow()
  })
})
