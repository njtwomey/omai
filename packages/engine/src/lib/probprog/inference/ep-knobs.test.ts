// The EP knobs, declared once: per-method defaults, the same names on every method, a typo refused, and the power of fractional
// EP taken by the grid and multi-start methods as it is by plain EP.
import { describe, expect, it } from 'vitest'
import { probitData, probitModel } from '../../../testing/probit-slice.ts'
import { defaults } from '../../space/index.ts'
import { session } from '../../trace/index.ts'
import { benchmarkModel } from '../models/suite.ts'
import { epGridKnobs, epGridMethod } from './ep-grid.ts'
import { checkKnobKeys, epRunKnobs } from './ep-knobs.ts'
import { epMultiStartKnobs } from './ep-multistart.ts'
import { epSpecMethod } from './mp.ts'

const RUN = ['schedule', 'damping', 'improper', 'tol', 'maxSweeps', 'restarts', 'power']

describe('epRunKnobs', () => {
  it('has the same knob names on every method, granularity only on plain EP', () => {
    for (const m of ['ep', 'ep-grid', 'ep-multistart'] as const) {
      const keys = Object.keys(epRunKnobs(m).spaces)
      for (const k of RUN) expect(keys).toContain(k)
      expect(keys.includes('granularity')).toBe(m === 'ep')
    }
  })

  it("keeps each method's own defaults: 200 sweeps at 1e-8 for ep, 200 and 300 sweeps at 1e-9 for a grid node and a start", () => {
    const pick = (m: 'ep' | 'ep-grid' | 'ep-multistart'): unknown[] => {
      const d = defaults(epRunKnobs(m)) as Record<string, unknown>
      return [d['tol'], d['maxSweeps'], d['damping'], d['schedule'], d['improper'], d['restarts'], d['power']]
    }
    expect(pick('ep')).toEqual([1e-8, 200, 0, 'serial', 'damp', 1, 1])
    expect(pick('ep-grid')).toEqual([1e-9, 200, 0, 'serial', 'damp', 1, 1])
    expect(pick('ep-multistart')).toEqual([1e-9, 300, 0, 'serial', 'damp', 1, 1])
  })

  it('is what the three methods are configured with', () => {
    const { X, y } = probitData()
    const bound = probitModel.bind({ X, y })
    const ep = epSpecMethod(bound).config.spaces
    for (const k of RUN) {
      expect(ep[k]).toEqual(epRunKnobs('ep').spaces[k])
      expect(epGridKnobs.spaces[k]).toEqual(epRunKnobs('ep-grid').spaces[k])
      expect(epMultiStartKnobs.spaces[k]).toEqual(epRunKnobs('ep-multistart').spaces[k])
    }
  })
})

describe('checkKnobKeys', () => {
  it('refuses a misspelled knob, naming it and the knobs there are', () => {
    expect(() => checkKnobKeys('test', { dampng: 0.5 }, epRunKnobs('ep'))).toThrow(/unknown EP knob 'dampng'.*damping/)
    expect(() => checkKnobKeys('test', { damping: 0.5, power: 0.5 }, epRunKnobs('ep'))).not.toThrow()
  })
})

describe('the power of fractional EP on the grid and multi-start methods', () => {
  it('grid: takes the schedule, the improper policy and the power, and one node settles the same at the defaults', () => {
    const bound = benchmarkModel('gaussian-process-regression').bound()
    const run = (config: Record<string, number | string>): number => {
      const method = epGridMethod(bound)
      const s = session(method, { steps: 2, mode: 'eager', config: { points: 3, maxNodes: 3, ...config } })
      return s.state.nodes[0]?.logEvidence ?? Number.NaN
    }
    const plain = run({})
    expect(run({ schedule: 'serial', improper: 'damp', restarts: 1, power: 1 })).toBe(plain)
    expect(Number.isFinite(run({ schedule: 'flooding', power: 0.5 }))).toBe(true)
  })
})
