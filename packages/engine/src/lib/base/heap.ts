/**
 * Priority queues: the `Heap` interface and the `heap` constructor, a binary heap on an array with push, pop and peek
 * in logarithmic time, shared by nearest-neighbour searches, shortest paths and best-first search.
 */

/**
 * A binary heap ordered by a comparator: the smallest element (by `compare`) is at the top. The priority queue behind
 * nearest-neighbour searches, graph shortest paths and best-first search.
 */
export interface Heap<T> {
  readonly size: number
  push: (item: T) => void
  /** Remove and return the top element (undefined when empty). */
  pop: () => T | undefined
  /** The top element without removing it. */
  peek: () => T | undefined
  /** The elements in heap order (not sorted). */
  toArray: () => T[]
}

/**
 * A binary heap.
 *
 * @param compare - Negative when a should come out before b.
 * @param items - Initial elements (heapified in O(n)).
 * @returns The heap.
 * @example
 * ```ts
 * import { heap } from 'mdml-engine/base'
 * const h = heap((a: number, b: number) => a - b, [5, 1, 3])
 * h.pop() // => 1
 * ```
 */
export function heap<T>(compare: (a: T, b: T) => number, items: readonly T[] = []): Heap<T> {
  const data: T[] = [...items]
  const less = (i: number, j: number): boolean => compare(data[i] as T, data[j] as T) < 0
  const swap = (i: number, j: number): void => {
    const t = data[i] as T
    data[i] = data[j] as T
    data[j] = t
  }
  const up = (i: number): void => {
    let k = i
    while (k > 0) {
      const parent = (k - 1) >> 1
      if (!less(k, parent)) break
      swap(k, parent)
      k = parent
    }
  }
  const down = (i: number): void => {
    let k = i
    for (;;) {
      const l = 2 * k + 1
      const r = l + 1
      let m = k
      if (l < data.length && less(l, m)) m = l
      if (r < data.length && less(r, m)) m = r
      if (m === k) return
      swap(k, m)
      k = m
    }
  }
  for (let i = (data.length >> 1) - 1; i >= 0; i--) down(i)
  return {
    get size() {
      return data.length
    },
    push: (item) => {
      data.push(item)
      up(data.length - 1)
    },
    pop: () => {
      if (data.length === 0) return undefined
      const top = data[0] as T
      const last = data.pop() as T
      if (data.length > 0) {
        data[0] = last
        down(0)
      }
      return top
    },
    peek: () => data[0],
    toArray: () => [...data],
  }
}
