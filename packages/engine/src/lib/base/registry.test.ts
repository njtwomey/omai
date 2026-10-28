import { describe, expect, it } from 'vitest'
import { find, lookup, matchesText, register, registry } from './registry.ts'

interface Item {
  readonly name: string
  readonly v: number
}

describe('registries', () => {
  it('keep entries in order with unique names', () => {
    const r = registry<Item>([
      { name: 'a', v: 1 },
      { name: 'b', v: 2 },
    ])
    expect(r.entries.map((e) => e.name)).toEqual(['a', 'b'])
    expect(() =>
      registry<Item>([
        { name: 'a', v: 1 },
        { name: 'a', v: 2 },
      ]),
    ).toThrow(/registered twice/)
  })

  it('extend into a new registry, replacing by name and leaving the old one untouched', () => {
    const base = registry<Item>([{ name: 'a', v: 1 }])
    const more = register(base, [
      { name: 'b', v: 2 },
      { name: 'a', v: 3 },
    ])
    expect(more.entries).toEqual([
      { name: 'a', v: 3 },
      { name: 'b', v: 2 },
    ])
    expect(base.entries).toEqual([{ name: 'a', v: 1 }])
    expect(() =>
      register(base, [
        { name: 'c', v: 1 },
        { name: 'c', v: 2 },
      ]),
    ).toThrow(/added twice/)
  })

  it('look entries up by name', () => {
    const r = registry<Item>([{ name: 'a', v: 1 }])
    expect(lookup(r, 'a').v).toBe(1)
    expect(find(r, 'b')).toBeUndefined()
    expect(() => lookup(r, 'b')).toThrow(/nothing named 'b'/)
  })

  it('match every word of a query against the fields, ignoring case', () => {
    expect(matchesText(undefined, ['x'])).toBe(true)
    expect(matchesText('  ', ['x'])).toBe(true)
    expect(matchesText('Two moons', ['moons', 'Two moons with noise', undefined])).toBe(true)
    expect(matchesText('moons noise', ['moons', 'Two moons'])).toBe(false)
    expect(matchesText('x', [undefined])).toBe(false)
  })
})
