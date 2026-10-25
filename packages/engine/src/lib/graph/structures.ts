/**
 * Data structures shared by the graph algorithms: a binary heap class over the base heap, a union-find (disjoint-set)
 * structure with path compression and union by rank for Kruskal's algorithm and components, and an indexed priority
 * queue with decrease-key for Dijkstra and Prim.
 */

import { type Heap, heap, MdmlError } from '../base/index.ts'

/**
 * A binary heap as a class, over `heap` from `mdml-engine/base`: the element that `compare` puts first is on top.
 * The priority queue of best-first searches and of lazy-deletion Dijkstra and Prim.
 *
 * @concept priority-queues
 * @example
 * ```ts
 * import { BinaryHeap } from 'mdml-engine/graph'
 * const h = new BinaryHeap((a: number, b: number) => a - b, [4, 1, 3])
 * h.push(0)
 * [h.pop(), h.pop(), h.size] // => [0, 1, 2]
 * ```
 */
export class BinaryHeap<T> {
  readonly #heap: Heap<T>

  /**
   * Build a heap.
   *
   * @param compare - Negative when `a` should come out before `b`.
   * @param items - Initial elements (heapified in O(n)).
   * @returns The heap.
   * @example
   * ```ts
   * import { BinaryHeap } from 'mdml-engine/graph'
   * new BinaryHeap((a: number, b: number) => b - a, [1, 5, 2]).peek() // => 5
   * ```
   */
  constructor(compare: (a: T, b: T) => number, items: readonly T[] = []) {
    this.#heap = heap(compare, items)
  }

  /** The number of elements. */
  get size(): number {
    return this.#heap.size
  }

  /**
   * Add an element.
   *
   * @param item - The element.
   * @returns Nothing.
   * @example
   * ```ts
   * import { BinaryHeap } from 'mdml-engine/graph'
   * const h = new BinaryHeap((a: number, b: number) => a - b)
   * h.push(2)
   * h.size // => 1
   * ```
   */
  push(item: T): void {
    this.#heap.push(item)
  }

  /**
   * Remove and return the top element.
   *
   * @returns The top element, or `undefined` when empty.
   * @example
   * ```ts
   * import { BinaryHeap } from 'mdml-engine/graph'
   * new BinaryHeap((a: number, b: number) => a - b, [3, 2]).pop() // => 2
   * ```
   */
  pop(): T | undefined {
    return this.#heap.pop()
  }

  /**
   * The top element without removing it.
   *
   * @returns The top element, or `undefined` when empty.
   * @example
   * ```ts
   * import { BinaryHeap } from 'mdml-engine/graph'
   * new BinaryHeap((a: number, b: number) => a - b).peek() // => undefined
   * ```
   */
  peek(): T | undefined {
    return this.#heap.peek()
  }

  /**
   * The elements in heap order (not sorted).
   *
   * @returns A copy of the elements.
   * @example
   * ```ts
   * import { BinaryHeap } from 'mdml-engine/graph'
   * new BinaryHeap((a: number, b: number) => a - b, [2, 1]).toArray().length // => 2
   * ```
   */
  toArray(): T[] {
    return this.#heap.toArray()
  }
}

/**
 * Disjoint sets over `0 .. n - 1` with union by rank and path halving (near-constant amortised time per operation):
 * the forest behind Kruskal's and Boruvka's spanning trees and incremental connectivity.
 *
 * @concept union-find
 * @example
 * ```ts
 * import { UnionFind } from 'mdml-engine/graph'
 * const uf = new UnionFind(4)
 * uf.union(0, 1)
 * uf.union(2, 3)
 * [uf.connected(0, 1), uf.connected(1, 2), uf.count] // => [true, false, 2]
 * ```
 */
export class UnionFind {
  readonly #parent: Int32Array
  readonly #rank: Uint8Array
  readonly #size: Int32Array
  #count: number

  /**
   * `n` singleton sets.
   *
   * @param n - The number of elements.
   * @returns The structure.
   * @throws MdmlError with code `value` when `n` is negative or not an integer.
   * @example
   * ```ts
   * import { UnionFind } from 'mdml-engine/graph'
   * new UnionFind(3).count // => 3
   * ```
   */
  constructor(n: number) {
    if (!Number.isInteger(n) || n < 0) throw new MdmlError('value', `UnionFind: size ${n} is not valid`)
    this.#parent = Int32Array.from({ length: n }, (_, i) => i)
    this.#rank = new Uint8Array(n)
    this.#size = new Int32Array(n).fill(1)
    this.#count = n
  }

  /** The number of disjoint sets. */
  get count(): number {
    return this.#count
  }

  /**
   * The representative of an element's set.
   *
   * @param x - The element.
   * @returns Its representative.
   * @example
   * ```ts
   * import { UnionFind } from 'mdml-engine/graph'
   * const uf = new UnionFind(2)
   * uf.union(0, 1)
   * uf.find(0) === uf.find(1) // => true
   * ```
   */
  find(x: number): number {
    const p = this.#parent
    let v = x
    while ((p[v] ?? v) !== v) {
      const up = p[p[v] ?? v] ?? v
      p[v] = up
      v = up
    }
    return v
  }

  /**
   * Merge the sets of two elements.
   *
   * @param a - An element.
   * @param b - Another element.
   * @returns True when they were in different sets (and are now merged).
   * @example
   * ```ts
   * import { UnionFind } from 'mdml-engine/graph'
   * const uf = new UnionFind(2)
   * [uf.union(0, 1), uf.union(1, 0)] // => [true, false]
   * ```
   */
  union(a: number, b: number): boolean {
    let ra = this.find(a)
    let rb = this.find(b)
    if (ra === rb) return false
    if ((this.#rank[ra] ?? 0) < (this.#rank[rb] ?? 0)) [ra, rb] = [rb, ra]
    this.#parent[rb] = ra
    this.#size[ra] = (this.#size[ra] ?? 0) + (this.#size[rb] ?? 0)
    if (this.#rank[ra] === this.#rank[rb]) this.#rank[ra] = (this.#rank[ra] ?? 0) + 1
    this.#count -= 1
    return true
  }

  /**
   * Whether two elements are in the same set.
   *
   * @param a - An element.
   * @param b - Another element.
   * @returns True when connected.
   * @example
   * ```ts
   * import { UnionFind } from 'mdml-engine/graph'
   * new UnionFind(2).connected(0, 1) // => false
   * ```
   */
  connected(a: number, b: number): boolean {
    return this.find(a) === this.find(b)
  }

  /**
   * The size of an element's set.
   *
   * @param x - The element.
   * @returns The number of elements in its set.
   * @example
   * ```ts
   * import { UnionFind } from 'mdml-engine/graph'
   * const uf = new UnionFind(3)
   * uf.union(0, 2)
   * uf.setSize(2) // => 2
   * ```
   */
  setSize(x: number): number {
    return this.#size[this.find(x)] ?? 1
  }

  /**
   * The sets, each sorted, ordered by their smallest element.
   *
   * @returns The partition.
   * @example
   * ```ts
   * import { UnionFind } from 'mdml-engine/graph'
   * const uf = new UnionFind(4)
   * uf.union(3, 1)
   * uf.sets() // => [[0], [1, 3], [2]]
   * ```
   */
  sets(): number[][] {
    const byRoot = new Map<number, number[]>()
    for (let v = 0; v < this.#parent.length; v++) {
      const r = this.find(v)
      const s = byRoot.get(r)
      if (s === undefined) byRoot.set(r, [v])
      else s.push(v)
    }
    return [...byRoot.values()]
  }
}

/**
 * An indexed min-priority queue over the keys `0 .. n - 1` (Sedgewick's IndexMinPQ): each key is in the queue at most
 * once, and its priority can be lowered or changed in O(log n). The queue of eager Dijkstra and Prim; ties come out in
 * key order.
 *
 * @concept priority-queues
 * @example
 * ```ts
 * import { IndexedQueue } from 'mdml-engine/graph'
 * const q = new IndexedQueue(3)
 * q.push(0, 5)
 * q.push(2, 1)
 * q.decrease(0, 0.5)
 * [q.pop(), q.pop(), q.size] // => [0, 2, 0]
 * ```
 */
export class IndexedQueue {
  readonly #heap: Int32Array
  readonly #position: Int32Array
  readonly #priority: Float64Array
  #size = 0

  /**
   * An empty queue for keys `0 .. n - 1`.
   *
   * @param n - The number of keys.
   * @returns The queue.
   * @example
   * ```ts
   * import { IndexedQueue } from 'mdml-engine/graph'
   * new IndexedQueue(4).size // => 0
   * ```
   */
  constructor(n: number) {
    this.#heap = new Int32Array(n)
    this.#position = new Int32Array(n).fill(-1)
    this.#priority = new Float64Array(n)
  }

  /** The number of keys in the queue. */
  get size(): number {
    return this.#size
  }

  #less(i: number, j: number): boolean {
    const a = this.#heap[i] ?? 0
    const b = this.#heap[j] ?? 0
    const pa = this.#priority[a] ?? 0
    const pb = this.#priority[b] ?? 0
    return pa < pb || (pa === pb && a < b)
  }

  #swap(i: number, j: number): void {
    const a = this.#heap[i] ?? 0
    const b = this.#heap[j] ?? 0
    this.#heap[i] = b
    this.#heap[j] = a
    this.#position[b] = i
    this.#position[a] = j
  }

  #up(i: number): void {
    let k = i
    while (k > 0) {
      const parent = (k - 1) >> 1
      if (!this.#less(k, parent)) break
      this.#swap(k, parent)
      k = parent
    }
  }

  #down(i: number): void {
    let k = i
    for (;;) {
      const l = 2 * k + 1
      const r = l + 1
      let m = k
      if (l < this.#size && this.#less(l, m)) m = l
      if (r < this.#size && this.#less(r, m)) m = r
      if (m === k) return
      this.#swap(k, m)
      k = m
    }
  }

  /**
   * Whether a key is in the queue.
   *
   * @param key - The key.
   * @returns True when queued.
   * @example
   * ```ts
   * import { IndexedQueue } from 'mdml-engine/graph'
   * const q = new IndexedQueue(2)
   * q.push(1, 3)
   * [q.has(0), q.has(1)] // => [false, true]
   * ```
   */
  has(key: number): boolean {
    return (this.#position[key] ?? -1) >= 0
  }

  /**
   * The priority of a queued key.
   *
   * @param key - The key.
   * @returns Its priority, or `undefined` when it is not queued.
   * @example
   * ```ts
   * import { IndexedQueue } from 'mdml-engine/graph'
   * const q = new IndexedQueue(2)
   * q.push(1, 3)
   * q.priority(1) // => 3
   * ```
   */
  priority(key: number): number | undefined {
    return this.has(key) ? this.#priority[key] : undefined
  }

  /**
   * Insert a key, or change its priority when it is already queued.
   *
   * @param key - The key.
   * @param priority - Its priority (smaller comes out first).
   * @returns Nothing.
   * @throws MdmlError with code `value` when the key is out of range.
   * @example
   * ```ts
   * import { IndexedQueue } from 'mdml-engine/graph'
   * const q = new IndexedQueue(2)
   * q.push(0, 2)
   * q.push(0, 7)
   * q.priority(0) // => 7
   * ```
   */
  push(key: number, priority: number): void {
    if (!(key >= 0 && key < this.#position.length)) {
      throw new MdmlError('value', `IndexedQueue: key ${key} is out of range`)
    }
    const at = this.#position[key] ?? -1
    this.#priority[key] = priority
    if (at >= 0) {
      this.#up(at)
      this.#down(this.#position[key] ?? 0)
      return
    }
    this.#heap[this.#size] = key
    this.#position[key] = this.#size
    this.#size += 1
    this.#up(this.#size - 1)
  }

  /**
   * Lower a key's priority (insert it when it is not queued); a higher priority is ignored.
   *
   * @param key - The key.
   * @param priority - The new priority.
   * @returns True when the queue changed.
   * @example
   * ```ts
   * import { IndexedQueue } from 'mdml-engine/graph'
   * const q = new IndexedQueue(1)
   * q.push(0, 2)
   * [q.decrease(0, 3), q.decrease(0, 1)] // => [false, true]
   * ```
   */
  decrease(key: number, priority: number): boolean {
    const current = this.priority(key)
    if (current !== undefined && current <= priority) return false
    this.push(key, priority)
    return true
  }

  /**
   * Remove and return the key with the smallest priority (the smallest key among ties).
   *
   * @returns The key, or `undefined` when empty.
   * @example
   * ```ts
   * import { IndexedQueue } from 'mdml-engine/graph'
   * const q = new IndexedQueue(3)
   * q.push(2, 1)
   * q.push(1, 1)
   * q.pop() // => 1
   * ```
   */
  pop(): number | undefined {
    if (this.#size === 0) return undefined
    const top = this.#heap[0] ?? 0
    this.#size -= 1
    this.#swap(0, this.#size)
    this.#position[top] = -1
    if (this.#size > 0) this.#down(0)
    return top
  }

  /**
   * The key with the smallest priority, without removing it.
   *
   * @returns The key, or `undefined` when empty.
   * @example
   * ```ts
   * import { IndexedQueue } from 'mdml-engine/graph'
   * const q = new IndexedQueue(2)
   * q.push(1, 0)
   * q.peek() // => 1
   * ```
   */
  peek(): number | undefined {
    return this.#size === 0 ? undefined : this.#heap[0]
  }
}
