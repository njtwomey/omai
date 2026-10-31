import { describe, expect, it } from 'vitest'
import { linearModel } from '../linear/index.ts'
import { key, normal, randint } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { deepFeatureSynthesis, type EntityTable, selectSynthesised } from './feature-synthesis.ts'

/** Customers (age, income), their orders (amount, qty), and a hidden target built from the orders. */
function schema(n: number, seed: number): { tables: EntityTable[]; sumAmount: number[] } {
  const age = toFlat(normal(key(seed), [n]))
  const income = toFlat(normal(key(seed + 1), [n]))
  const counts = Array.from(toFlat(randint(key(seed + 2), [n], 1, 6)))
  const customer: number[] = []
  counts.forEach((c, i) => {
    for (let k = 0; k < c; k++) customer.push(i)
  })
  const amount = Array.from(toFlat(normal(key(seed + 3), [customer.length])), (v) => 10 + 3 * v)
  const qty = Array.from(toFlat(randint(key(seed + 4), [customer.length], 1, 4)))
  const sumAmount = new Array<number>(n).fill(0)
  customer.forEach((c, i) => {
    sumAmount[c] = (sumAmount[c] ?? 0) + (amount[i] ?? 0)
  })
  return {
    tables: [
      { name: 'customers', columns: { age: Array.from(age), income: Array.from(income) } },
      { name: 'orders', columns: { customer, amount, qty }, parent: { table: 'customers', key: 'customer' } },
    ],
    sumAmount,
  }
}

describe('deepFeatureSynthesis', () => {
  const customers: EntityTable = { name: 'customers', columns: { age: [30, 40, 50] } }
  const orders: EntityTable = {
    name: 'orders',
    columns: { customer: [0, 0, 1, 2, 2, 2], amount: [5, 7, 2, 1, 2, 3], qty: [1, 2, 1, 1, 1, 4] },
    parent: { table: 'customers', key: 'customer' },
  }

  it('computes the aggregates of the child rows per parent row', () => {
    const f = deepFeatureSynthesis([customers, orders], 'customers', {
      aggregations: ['sum', 'mean', 'min', 'max', 'std'],
      transforms: [],
    })
    const by = (name: string): number[] => Array.from(f.find((x) => x.name === name)?.values ?? [])
    expect(by('COUNT(orders)')).toEqual([2, 1, 3])
    expect(by('SUM(orders.amount)')).toEqual([12, 2, 6])
    expect(by('MEAN(orders.amount)')).toEqual([6, 2, 2])
    expect(by('MIN(orders.qty)')).toEqual([1, 1, 1])
    expect(by('MAX(orders.qty)')).toEqual([2, 1, 4])
    expect(by('STD(orders.amount)')[0]).toBeCloseTo(Math.sqrt(2), 12)
    expect(by('STD(orders.amount)')[1]).toBe(0)
    expect(by('age')).toEqual([30, 40, 50])
  })

  it('counts the search space exactly at depth 1: raw columns, aggregates and transforms of the raw columns', () => {
    const { tables } = schema(10, 1)
    const f = deepFeatureSynthesis(tables, 'customers', {
      aggregations: ['sum', 'mean'],
      transforms: ['add', 'multiply'],
      maxDepth: 1,
    })
    // 2 raw + COUNT + 2 columns x 2 aggregations + (age + income, age * income).
    expect(f.length).toBe(2 + 1 + 4 + 2)
    expect(f.map((x) => x.name)).toContain('age * income')
    expect(f.filter((x) => x.depth === 0).length).toBe(2)
  })

  it('stacks primitives at depth 2, and the values are those of the definition', () => {
    const { tables } = schema(10, 2)
    const f = deepFeatureSynthesis(tables, 'customers', {
      aggregations: ['sum'],
      transforms: ['multiply', 'subtract', 'divide'],
      maxDepth: 2,
    })
    const sum = f.find((x) => x.name === 'SUM(orders.amount)')
    const age = f.find((x) => x.name === 'age')
    const product = f.find((x) => x.name === 'SUM(orders.amount) * age' || x.name === 'age * SUM(orders.amount)')
    expect(sum && age && product).toBeTruthy()
    expect(product?.depth).toBe(2)
    for (let i = 0; i < 10; i++)
      expect(product?.values[i]).toBeCloseTo((sum?.values[i] ?? 0) * (age?.values[i] ?? 0), 10)
    // Division by zero is defined as zero.
    expect(f.every((x) => x.values.every(Number.isFinite))).toBe(true)
    // Both orders of the non-symmetric transforms are present.
    expect(f.some((x) => x.name === 'age / income') && f.some((x) => x.name === 'income / age')).toBe(true)
  })

  it('aggregates through a chain of tables (items of orders of customers)', () => {
    const items: EntityTable = {
      name: 'items',
      columns: { order: [0, 0, 1, 2, 5], price: [1, 2, 4, 8, 16] },
      parent: { table: 'orders', key: 'order' },
    }
    const f = deepFeatureSynthesis([customers, orders, items], 'customers', {
      aggregations: ['sum'],
      transforms: [],
      maxDepth: 2,
    })
    const nested = f.find((x) => x.name === 'SUM(orders.SUM(items.price))')
    // Order sums of price: orders 0..5 = [3, 4, 8, 0, 0, 16]; customers own orders {0,1}, {2}, {3,4,5}.
    expect(Array.from(nested?.values ?? [])).toEqual([7, 8, 16])
    expect(nested?.depth).toBe(2)
  })

  it('caps the number of features and rejects bad schemas', () => {
    const { tables } = schema(8, 3)
    expect(deepFeatureSynthesis(tables, 'customers', { maxDepth: 2, maxFeatures: 40 }).length).toBeLessThanOrEqual(40)
    expect(() => deepFeatureSynthesis(tables, 'nope')).toThrow()
    expect(() =>
      deepFeatureSynthesis([{ name: 'c', columns: { a: [1] }, parent: { table: 'missing', key: 'a' } }], 'c'),
    ).toThrow()
    expect(() =>
      deepFeatureSynthesis(
        [customers, { ...orders, columns: { ...orders.columns, customer: [0, 0, 1, 2, 2, 9] } }],
        'customers',
      ),
    ).toThrow()
  })
})

describe('selectSynthesised', () => {
  it('finds the aggregate that generated the target and then stops, among eighteen candidates', () => {
    const { tables, sumAmount } = schema(120, 4)
    const noise = toFlat(normal(key(40), [120]))
    const y = sumAmount.map((v, i) => v + 0.5 * (noise[i] ?? 0))
    const features = deepFeatureSynthesis(tables, 'customers', { maxDepth: 1 })
    expect(features.length).toBeGreaterThan(15)
    const sel = selectSynthesised(features, y, { estimator: linearModel(), maxFeatures: 3, candidates: 15 })
    expect(sel.features[0]?.name).toBe('SUM(orders.amount)')
    expect(sel.scores[0]).toBeGreaterThan(0.95)
    // Everything after the first adds nothing worth keeping.
    expect(sel.features.length).toBe(1)
  })

  it('finds an arithmetic combination (a product of two columns) at depth 1', () => {
    const { tables } = schema(150, 5)
    const age = tables[0]?.columns['age'] as ArrayLike<number>
    const income = tables[0]?.columns['income'] as ArrayLike<number>
    const e = toFlat(normal(key(50), [150]))
    const y = Array.from({ length: 150 }, (_, i) => (age[i] ?? 0) * (income[i] ?? 0) + 0.2 * (e[i] ?? 0))
    const features = deepFeatureSynthesis(tables, 'customers', { aggregations: ['sum', 'mean'], maxDepth: 1 })
    const sel = selectSynthesised(features, y, { estimator: linearModel(), maxFeatures: 2, candidates: 10 })
    expect(sel.features[0]?.name).toBe('age * income')
    expect(sel.scores[0]).toBeGreaterThan(0.9)
  })

  it('takes a second feature when it adds to the first', () => {
    const { tables, sumAmount } = schema(150, 6)
    const age = tables[0]?.columns['age'] as ArrayLike<number>
    const e = toFlat(normal(key(60), [150]))
    const y = sumAmount.map((v, i) => v + 8 * (age[i] ?? 0) + 0.3 * (e[i] ?? 0))
    const features = deepFeatureSynthesis(tables, 'customers', { aggregations: ['sum', 'mean'], maxDepth: 1 })
    const sel = selectSynthesised(features, y, { estimator: linearModel(), maxFeatures: 3, candidates: 20 })
    const names = sel.features.map((f) => f.name)
    expect(names).toContain('SUM(orders.amount)')
    expect(names).toContain('age')
    expect(sel.scores[sel.scores.length - 1]).toBeGreaterThan(0.98)
  })
})
