import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { BinaryHeap, IndexedQueue, UnionFind } from './index.ts'

describe('graph data structures', () => {
  it('UnionFind agrees with naive set merging', () => {
    fc.assert(
      fc.property(
        fc
          .integer({ min: 1, max: 20 })
          .chain((n) => fc.tuple(fc.constant(n), fc.array(fc.tuple(fc.nat(n - 1), fc.nat(n - 1)), { maxLength: 30 }))),
        ([n, unions]) => {
          const uf = new UnionFind(n)
          const label = Array.from({ length: n }, (_, i) => i)
          for (const [a, b] of unions) {
            const merged = uf.union(a, b)
            const la = label[a] ?? 0
            const lb = label[b] ?? 0
            expect(merged).toBe(la !== lb)
            for (let i = 0; i < n; i++) if (label[i] === lb) label[i] = la
          }
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) expect(uf.connected(i, j)).toBe(label[i] === label[j])
            expect(uf.setSize(i)).toBe(label.filter((l) => l === label[i]).length)
          }
          expect(uf.count).toBe(new Set(label).size)
          expect(
            uf
              .sets()
              .flat()
              .sort((a, b) => a - b),
          ).toEqual(Array.from({ length: n }, (_, i) => i))
        },
      ),
    )
    expect(() => new UnionFind(-1)).toThrow()
  })

  it('IndexedQueue pops keys by priority (ties by key) after arbitrary updates', () => {
    fc.assert(
      fc.property(fc.array(fc.tuple(fc.nat(9), fc.integer({ min: -5, max: 5 })), { maxLength: 40 }), (ops) => {
        const q = new IndexedQueue(10)
        const want = new Map<number, number>()
        for (const [k, p] of ops) {
          if (k % 2 === 0) {
            q.push(k, p)
            want.set(k, p)
          } else {
            const old = want.get(k)
            const changed = q.decrease(k, p)
            expect(changed).toBe(old === undefined || p < old)
            if (changed) want.set(k, p)
          }
        }
        expect(q.size).toBe(want.size)
        const order = [...want.entries()].sort((a, b) => a[1] - b[1] || a[0] - b[0]).map(([k]) => k)
        if (order.length > 0) expect(q.peek()).toBe(order[0])
        const got: number[] = []
        for (let k = q.pop(); k !== undefined; k = q.pop()) got.push(k)
        expect(got).toEqual(order)
        expect(q.has(0)).toBe(false)
        expect(q.priority(0)).toBeUndefined()
      }),
    )
    expect(() => new IndexedQueue(2).push(5, 0)).toThrow()
  })

  it('BinaryHeap sorts', () => {
    fc.assert(
      fc.property(fc.array(fc.integer()), (xs) => {
        const h = new BinaryHeap((a: number, b: number) => a - b, xs)
        expect(h.toArray().length).toBe(xs.length)
        const out: number[] = []
        for (let x = h.pop(); x !== undefined; x = h.pop()) out.push(x)
        expect(out).toEqual([...xs].sort((a, b) => a - b))
      }),
    )
  })
})
