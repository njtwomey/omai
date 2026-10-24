import { describe, expect, it } from 'vitest'
import { cycleOption, normaliseOptions } from './options.ts'

describe('options', () => {
  it('labels bare values by their text', () => {
    expect(normaliseOptions([1, { value: 2, label: 'two' }])).toEqual([
      { value: 1, label: '1' },
      { value: 2, label: 'two' },
    ])
  })

  it('cycles through enabled options, wrapping at the ends', () => {
    const options = normaliseOptions(['a', { value: 'b', disabled: true }, 'c'])
    expect(cycleOption(options, 'a', 1)).toBe('c')
    expect(cycleOption(options, 'c', 1)).toBe('a')
    expect(cycleOption(options, 'a', -1)).toBe('c')
    expect(cycleOption(options, 'z', -1)).toBe('c')
    expect(cycleOption(normaliseOptions([{ value: 'x', disabled: true }]), 'x', 1)).toBeUndefined()
  })
})
