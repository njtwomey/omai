import { describe, expect, it } from 'vitest'
import { wrapTarget } from './_focus.ts'

describe('wrapTarget', () => {
  it('wraps Tab from the last item to the first and Shift+Tab from the first (or the container) to the last', () => {
    const items = ['a', 'b', 'c']
    // [active, shift, where focus must go instead of the browser's move]
    const cases: readonly [string, boolean, string | undefined][] = [
      ['c', false, 'a'],
      ['b', false, undefined],
      ['a', true, 'c'],
      ['box', true, 'c'],
      ['b', true, undefined],
    ]
    for (const [active, shift, want] of cases) expect(wrapTarget(items, active, shift, 'box')).toBe(want)
    expect(wrapTarget([], 'box', false, 'box')).toBeUndefined()
  })
})
