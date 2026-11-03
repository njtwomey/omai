/**
 * A ring buffer: a fixed capacity of the latest items, the oldest dropped as new ones are pushed, read by position
 * from the oldest still held. It keeps a live run's history without the copying that dropping from the front of an
 * array costs, and counts what it has dropped so a position can be turned into a step number.
 */

/**
 * A ring buffer of the latest `capacity` items.
 *
 * @index structure The latest n items, the oldest dropped: `new RingBuffer(2000)`, `ring.push(x)`, `ring.at(0)`
 */
export class RingBuffer<T> {
  private items: (T | undefined)[]
  private first = 0
  private count = 0
  private gone = 0

  /**
   * @param capacity - The most items held (at least 1).
   */
  constructor(readonly capacity: number) {
    this.capacity = Math.max(1, Math.floor(capacity))
    this.items = new Array<T | undefined>(Math.min(this.capacity, 64))
  }

  /** How many items are held. */
  get length(): number {
    return this.count
  }

  /** How many items have been dropped to make room (so item `i` is the `dropped + i`th ever pushed). */
  get dropped(): number {
    return this.gone
  }

  /**
   * Add an item at the newest end, dropping the oldest when full.
   *
   * @param item - The item.
   * @returns Whether the oldest was dropped.
   */
  push(item: T): boolean {
    if (this.count < this.capacity) {
      // Below capacity the items sit in order from index 0 (`first` is 0 until the first drop); grow as needed.
      this.items[this.count] = item
      this.count += 1
      return false
    }
    this.items[this.first] = item
    this.first = (this.first + 1) % this.capacity
    this.gone += 1
    return true
  }

  /**
   * The item at a position from the oldest held (0) to the newest (`length - 1`).
   *
   * @param i - The position.
   * @returns The item, or undefined outside the buffer.
   */
  at(i: number): T | undefined {
    if (!(i >= 0 && i < this.count)) return undefined
    return this.items[(this.first + i) % this.capacity]
  }

  /** The newest item, if any. */
  get last(): T | undefined {
    return this.at(this.count - 1)
  }

  /** Drop everything held (the capacity stays). */
  clear(): void {
    this.items = new Array<T | undefined>(Math.min(this.capacity, 64))
    this.first = 0
    this.count = 0
    this.gone = 0
  }
}
