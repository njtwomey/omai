import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { heap } from './heap.ts'

describe('heap', () => {
  it('pops in sorted order for any input', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ noNaN: true })), (xs) => {
        const h = heap((a: number, b: number) => a - b, xs.slice(0, xs.length >> 1))
        for (const x of xs.slice(xs.length >> 1)) h.push(x)
        const out: number[] = []
        while (h.size > 0) out.push(h.pop() as number)
        // -0 and +0 compare equal, so either may come first; toEqual tells them apart.
        const plain = (v: number): number => (v === 0 ? 0 : v)
        expect(out.map(plain)).toEqual([...xs].sort((a, b) => a - b).map(plain))
      }),
    )
  })

  it('works as a max-heap with a reversed comparator', () => {
    const h = heap((a: number, b: number) => b - a, [2, 9, 4])
    expect(h.peek()).toBe(9)
    expect(h.pop()).toBe(9)
    expect(h.pop()).toBe(4)
    expect(h.size).toBe(1)
    expect(heap((a: number, b: number) => a - b).pop()).toBeUndefined()
  })
})
