import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  allPayAuction,
  combinatorialAuction,
  equilibriumBid,
  expectedRevenue,
  firstPriceAuction,
  secondPriceAuction,
  vcg,
} from './index.ts'

const bids = fc.array(fc.integer({ min: 0, max: 20 }), { minLength: 2, maxLength: 5 })

describe('auctions', () => {
  it('truthful bidding is dominant in the second-price auction', () => {
    fc.assert(
      fc.property(bids, fc.integer({ min: 0, max: 20 }), (values, deviation) => {
        const truthful = secondPriceAuction(values, { values }).utilities ?? []
        const lie = [...values]
        lie[0] = deviation
        const deviated = secondPriceAuction(lie, { values }).utilities ?? []
        expect(deviated[0] ?? 0).toBeLessThanOrEqual(truthful[0] ?? 0)
      }),
    )
  })

  it('VCG for one item is the second-price auction', () => {
    fc.assert(
      fc.property(bids, (b) => {
        const v = vcg(b.map((x, i) => b.map((_, o) => (o === i ? x : 0))))
        const sp = secondPriceAuction(b)
        expect(v.payments).toEqual(sp.payments)
        expect(b[v.outcome]).toBe(b[sp.winner])
      }),
    )
  })

  it('handles reserves and missing winners', () => {
    expect(firstPriceAuction([3, 4], { reserve: 5 }).winner).toBe(-1)
    expect(secondPriceAuction([3, 8], { reserve: 5 }).price).toBe(5)
    expect(secondPriceAuction([9]).price).toBe(0)
    expect(allPayAuction([2, 2]).winner).toBe(0)
    expect(() => firstPriceAuction([1, 2], { values: [1] })).toThrow()
  })

  it('revenue is equivalent across formats in the symmetric equilibrium (Monte Carlo)', () => {
    const n = 3
    const rounds = 20_000
    const draws = toFlat(uniform(key(5), [rounds, n]))
    const revenue = { first: 0, second: 0, allPay: 0 }
    for (let r = 0; r < rounds; r++) {
      const values = draws.slice(r * n, (r + 1) * n)
      revenue.first += firstPriceAuction(values.map((v) => equilibriumBid('first-price', v, n))).price
      revenue.second += secondPriceAuction(values.map((v) => equilibriumBid('second-price', v, n))).price
      const pay = allPayAuction(values.map((v) => equilibriumBid('all-pay', v, n))).payments
      for (const p of pay) revenue.allPay += p
    }
    for (const total of Object.values(revenue)) expect(total / rounds).toBeCloseTo(expectedRevenue(n), 2)
    expect(expectedRevenue(1, 0.5)).toBeCloseTo(0.25, 12)
    expect(() => equilibriumBid('dutch' as 'all-pay', 0.5, 2)).toThrow()
  })

  it('a combinatorial auction picks the welfare-maximising bundles', () => {
    const r = combinatorialAuction([
      { bidder: 0, bundle: [0, 1], value: 5 },
      { bidder: 1, bundle: [0], value: 2 },
      { bidder: 2, bundle: [1], value: 2 },
    ])
    expect(r.accepted).toEqual([0])
    expect(r.welfare).toBe(5)
    expect(r.payments).toEqual([4, 0, 0])
  })
})
