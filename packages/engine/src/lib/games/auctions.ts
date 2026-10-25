/**
 * Auction theory: how the rules of an auction shape who wins and what they pay. Sealed-bid first-price, second-price
 * (Vickrey) and all-pay auctions with reserve prices, the symmetric equilibrium bids and expected revenues for uniform
 * values (revenue equivalence), and the Vickrey-Clarke-Groves mechanism, also applied to combinatorial auctions with
 * bundle bids.
 */

import { MdmlError } from '../base/index.ts'

/** The outcome of a single-item auction. */
export interface AuctionOutcome {
  /** The winning bidder, or -1 when no bid meets the reserve. */
  readonly winner: number
  /** What the winner pays (0 without a winner). */
  readonly price: number
  /** What every bidder pays. */
  readonly payments: readonly number[]
  /** Every bidder's utility (value if winning, minus payment), when values were given. */
  readonly utilities?: readonly number[]
}

/** Options shared by the single-item auctions. */
export interface AuctionOptions {
  /** The reserve price: bids below it are ignored (default 0). */
  readonly reserve?: number
  /** The bidders' true values, to report utilities. */
  readonly values?: readonly number[]
}

function highest(bids: readonly number[], reserve: number): { winner: number; second: number } {
  let winner = -1
  let top = -Infinity
  let second = -Infinity
  for (const [i, b] of bids.entries()) {
    if (!(b >= reserve)) continue
    if (b > top) {
      second = top
      top = b
      winner = i
    } else if (b > second) second = b
  }
  return { winner, second }
}

function outcome(
  bids: readonly number[],
  winner: number,
  payments: number[],
  values: readonly number[] | undefined,
): AuctionOutcome {
  if (values !== undefined && values.length !== bids.length)
    throw new MdmlError('shape', 'auction: one value per bidder')
  return {
    winner,
    price: winner >= 0 ? (payments[winner] ?? 0) : 0,
    payments,
    ...(values === undefined
      ? {}
      : { utilities: bids.map((_, i) => (i === winner ? (values[i] ?? 0) : 0) - (payments[i] ?? 0)) }),
  }
}

/**
 * A sealed-bid first-price auction: the highest bid at or above the reserve wins (ties to the lowest index) and pays
 * its bid. Bidding one's value earns nothing, so bidders shade their bids ({@link equilibriumBid}).
 *
 * @param bids - Every bidder's bid.
 * @param options - Reserve price and true values (for utilities).
 * @returns The winner, price, payments and utilities.
 * @concept auctions
 * @example
 * ```ts
 * import { firstPriceAuction } from 'mdml-engine/games'
 * firstPriceAuction([3, 7, 5]).price // => 7
 * ```
 */
export function firstPriceAuction(bids: readonly number[], options: AuctionOptions = {}): AuctionOutcome {
  const { winner } = highest(bids, options.reserve ?? 0)
  return outcome(
    bids,
    winner,
    bids.map((b, i) => (i === winner ? b : 0)),
    options.values,
  )
}

/**
 * A sealed-bid second-price (Vickrey 1961) auction: the highest bid at or above the reserve wins and pays the larger
 * of the second-highest such bid and the reserve. Bidding one's true value is a dominant strategy.
 *
 * @param bids - Every bidder's bid.
 * @param options - Reserve price and true values (for utilities).
 * @returns The winner, price, payments and utilities.
 * @concept auctions
 * @example
 * ```ts
 * import { secondPriceAuction } from 'mdml-engine/games'
 * secondPriceAuction([3, 7, 5], { values: [3, 7, 5] }).utilities // => [0, 2, 0]
 * ```
 */
export function secondPriceAuction(bids: readonly number[], options: AuctionOptions = {}): AuctionOutcome {
  const reserve = options.reserve ?? 0
  const { winner, second } = highest(bids, reserve)
  const price = Math.max(reserve, Number.isFinite(second) ? second : reserve)
  return outcome(
    bids,
    winner,
    bids.map((_, i) => (i === winner ? price : 0)),
    options.values,
  )
}

/**
 * A sealed-bid all-pay auction: the highest bid at or above the reserve wins, and every bidder pays its own bid (a
 * model of contests, lobbying and wars of attrition).
 *
 * @param bids - Every bidder's bid.
 * @param options - Reserve price and true values (for utilities).
 * @returns The winner, the winner's payment, everyone's payments and utilities.
 * @concept auctions
 * @example
 * ```ts
 * import { allPayAuction } from 'mdml-engine/games'
 * allPayAuction([3, 7, 5]).payments // => [3, 7, 5]
 * ```
 */
export function allPayAuction(bids: readonly number[], options: AuctionOptions = {}): AuctionOutcome {
  const { winner } = highest(bids, options.reserve ?? 0)
  return outcome(bids, winner, [...bids], options.values)
}

/**
 * The symmetric Bayes-Nash equilibrium bid of a bidder with value v among `bidders` with independent values uniform
 * on [0, 1]: `(n - 1) v / n` in a first-price auction, `v` (truthful) in a second-price one, and `(n - 1) v^n / n` in
 * an all-pay auction.
 *
 * @param format - The auction format.
 * @param value - The bidder's value in [0, 1].
 * @param bidders - The number of bidders (at least 1).
 * @returns The equilibrium bid.
 * @throws MdmlError with code `value` for an unknown format or fewer than one bidder.
 * @concept revenue-equivalence
 * @example
 * ```ts
 * import { equilibriumBid } from 'mdml-engine/games'
 * [equilibriumBid('first-price', 0.8, 4), equilibriumBid('second-price', 0.8, 4)] // => [0.6000000000000001, 0.8]
 * ```
 */
export function equilibriumBid(
  format: 'first-price' | 'second-price' | 'all-pay',
  value: number,
  bidders: number,
): number {
  if (!(bidders >= 1)) throw new MdmlError('value', 'equilibriumBid: at least one bidder')
  const n = bidders
  if (format === 'first-price') return ((n - 1) / n) * value
  if (format === 'second-price') return value
  if (format === 'all-pay') return ((n - 1) / n) * value ** n
  throw new MdmlError('value', `equilibriumBid: unknown format '${String(format)}'`)
}

/**
 * The seller's expected revenue with `bidders` independent values uniform on [0, 1] and a reserve price r, in the
 * symmetric equilibrium of any standard auction (revenue equivalence, Myerson 1981):
 * `(n - 1) / (n + 1) + r^n - 2 n r^(n+1) / (n + 1)`, maximised at r = 1/2.
 *
 * @param bidders - The number of bidders.
 * @param reserve - The reserve price in [0, 1] (default 0).
 * @returns The expected revenue.
 * @concept revenue-equivalence
 * @example
 * ```ts
 * import { expectedRevenue } from 'mdml-engine/games'
 * [expectedRevenue(2), expectedRevenue(1, 0.5)] // => [0.3333333333333333, 0.25]
 * ```
 */
export function expectedRevenue(bidders: number, reserve = 0): number {
  const n = bidders
  const r = Math.min(1, Math.max(0, reserve))
  return (n - 1) / (n + 1) + r ** n - (2 * n * r ** (n + 1)) / (n + 1)
}

/** The outcome of {@link vcg}. */
export interface VcgOutcome {
  /** The chosen outcome: the one maximising reported welfare (ties to the first). */
  readonly outcome: number
  /** The welfare of the chosen outcome. */
  readonly welfare: number
  /** Clarke pivot payments: the harm each agent does to the others. */
  readonly payments: readonly number[]
  /** Every agent's value of the outcome minus its payment. */
  readonly utilities: readonly number[]
}

/**
 * The Vickrey-Clarke-Groves mechanism with Clarke pivot payments over a finite set of outcomes: choose the outcome
 * maximising the sum of reported values, and charge each agent the welfare the others would have without it minus
 * their welfare at the chosen outcome. Truthful reporting is a dominant strategy; for one item it is the
 * second-price auction.
 *
 * @param valuations - `valuations[i][o]`: agent i's value for outcome o.
 * @returns The outcome, welfare, payments and utilities.
 * @throws MdmlError with code `shape` when the agents value different numbers of outcomes.
 * @concept vcg
 * @example
 * ```ts
 * import { vcg } from 'mdml-engine/games'
 * // One item, three bidders: outcome i gives it to bidder i.
 * vcg([[3, 0, 0], [0, 7, 0], [0, 0, 5]]).payments // => [0, 5, 0]
 * ```
 */
export function vcg(valuations: readonly (readonly number[])[]): VcgOutcome {
  const k = valuations[0]?.length ?? 0
  if (k === 0 || valuations.some((v) => v.length !== k))
    throw new MdmlError('shape', 'vcg: every agent values the same outcomes')
  const welfare = (o: number, skip: number): number => {
    let s = 0
    for (const [i, v] of valuations.entries()) if (i !== skip) s += v[o] ?? 0
    return s
  }
  const best = (skip: number): number => {
    let arg = 0
    for (let o = 1; o < k; o++) if (welfare(o, skip) > welfare(arg, skip)) arg = o
    return arg
  }
  const chosen = best(-1)
  const payments = valuations.map((_, i) => welfare(best(i), i) - welfare(chosen, i))
  return {
    outcome: chosen,
    welfare: welfare(chosen, -1),
    payments,
    utilities: valuations.map((v, i) => (v[chosen] ?? 0) - (payments[i] ?? 0)),
  }
}

/** A bid in a combinatorial auction: a bidder's value for a bundle of items. */
export interface BundleBid {
  readonly bidder: number
  readonly bundle: readonly number[]
  readonly value: number
}

/** The outcome of {@link combinatorialAuction}. */
export interface CombinatorialOutcome extends Omit<VcgOutcome, 'outcome'> {
  /** The accepted bids (indices into the bids). */
  readonly accepted: readonly number[]
}

/**
 * A combinatorial auction with XOR bids, run as VCG: every feasible allocation (each bidder wins at most one of its
 * bids, no item sold twice) is an outcome, the welfare-maximising one is chosen and payments follow Clarke's rule.
 * The allocations are enumerated, so this is for small auctions.
 *
 * @param bids - The bundle bids.
 * @returns The accepted bids, welfare, payments and utilities (one per bidder, numbered by the largest bidder index).
 * @concept vcg
 * @example
 * ```ts
 * import { combinatorialAuction } from 'mdml-engine/games'
 * // The classic example: one bidder wants both items for 2, two others one item each for 2.
 * const r = combinatorialAuction([{ bidder: 0, bundle: [0, 1], value: 2 }, { bidder: 1, bundle: [0], value: 2 }, { bidder: 2, bundle: [1], value: 2 }])
 * [r.accepted, r.payments] // => [[1, 2], [0, 0, 0]]
 * ```
 */
export function combinatorialAuction(bids: readonly BundleBid[]): CombinatorialOutcome {
  let bidders = 0
  for (const b of bids) bidders = Math.max(bidders, b.bidder + 1)
  const allocations: number[][] = []
  const search = (i: number, taken: ReadonlySet<number>, winners: ReadonlySet<number>, chosen: number[]): void => {
    if (i === bids.length) {
      allocations.push(chosen)
      return
    }
    search(i + 1, taken, winners, chosen)
    const b = bids[i]
    if (b === undefined || winners.has(b.bidder) || b.bundle.some((item) => taken.has(item))) return
    search(i + 1, new Set([...taken, ...b.bundle]), new Set([...winners, b.bidder]), [...chosen, i])
  }
  search(0, new Set(), new Set(), [])
  const valuations = Array.from({ length: bidders }, (_, j) =>
    allocations.map((acc) => {
      let v = 0
      for (const i of acc) if (bids[i]?.bidder === j) v += bids[i]?.value ?? 0
      return v
    }),
  )
  const r = vcg(valuations)
  return { accepted: allocations[r.outcome] ?? [], welfare: r.welfare, payments: r.payments, utilities: r.utilities }
}
