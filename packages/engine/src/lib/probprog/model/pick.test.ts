import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { seed, traceOf } from '../handlers.ts'
import * as d from './dists/index.ts'
import { defineModel, ops, toProgram } from './index.ts'

describe('ops.pick along the last named dimension', () => {
  // each group has a slope; the line of a row is the slope of its group times its x; the row's group picks the entry
  const build = (lines: 'rowFirst' | 'groupFirst') =>
    defineModel('lines', (m) => {
      const row = m.range('row')
      const group = m.range('group', { size: 2 })
      const x = m.data('x', { dims: [row] })
      const slope = m.data('slope', { dims: [group] })
      const z = m.data('z', { dims: [row], support: 'index' })
      m.derived('mean', ops.pick(lines === 'rowFirst' ? ops.mul(x, slope) : ops.mul(slope, x), z))
    })
  const data = { x: [1, 2, 3], slope: [10, 100], z: [0, 1, 1] }
  const meanOf = (lines: 'rowFirst' | 'groupFirst'): number[] => {
    const t = traceOf(seed(toProgram(build(lines).bind(data)), key(0)), undefined)
    const site = t.sites.get('mean')
    if (site?.type !== 'deterministic') throw new Error('no site mean')
    return toFlat(site.value)
  }

  it('gives the slope of the group of each row times its x when the group is the last dimension', () => {
    expect(meanOf('rowFirst')).toEqual([10, 200, 300])
  })

  it('refuses a first operand whose last dimension is one of the index, and says how to order it', () => {
    // ops.mul(slope, x) has dimensions [group, row]: the last is the row, which the index runs over
    expect(() => build('groupFirst')).toThrow(/pick.*last dimension 'row'.*index/s)
  })
})
