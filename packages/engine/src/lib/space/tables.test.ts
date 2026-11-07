import { describe, expect, it } from 'vitest'
import { choice, dict, fillKnobs, int, knobOption, plainOption, readKnobs } from './index.ts'

describe('knob helpers of the option tables', () => {
  it('fill the knobs a configuration leaves out with the defaults and build the option from them', () => {
    const space = dict({ k: int(0, 9, { default: 3 }), m: choice(['a', 'b'], { default: 'a' }) })
    expect(fillKnobs(space, {})).toEqual({ k: 3, m: 'a' })
    expect(fillKnobs(space, { k: 5, extra: 1 })).toEqual({ k: 5, m: 'a', extra: 1 })
    const k = readKnobs<number | string>(space, { m: 'b' })
    expect([k('k'), k('m'), k('missing')]).toEqual([3, 'b', undefined])
    const scaled = knobOption('twice', { tag: 1 }, space, (get) => 2 * get('k'))
    expect([scaled.make({}), scaled.make({ k: 4 })]).toEqual([6, 8])
    expect(scaled.properties).toEqual({ tag: 1 })
    expect(plainOption('plain', space, (get) => get('k') + 1).make({ k: 9 })).toBe(10)
    expect(plainOption('plain', space, () => 0).properties).toEqual({})
  })
})
