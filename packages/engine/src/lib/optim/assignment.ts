/**
 * Assignment and matching problems: the Hungarian algorithm for the linear sum assignment problem (as
 * `scipy.optimize.linear_sum_assignment`), Bertsekas' auction algorithm with epsilon scaling, and Gale-Shapley deferred
 * acceptance for stable matchings with a check for blocking pairs, each also a traced algorithm whose steps can be
 * animated.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { asTensor, primalOf, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'

/** Result of {@link linearSumAssignment} and {@link auction}: matched rows and columns (scipy's row_ind, col_ind). */
export interface AssignmentResult {
  /** Assigned rows, increasing. */
  readonly rowInd: readonly number[]
  /** The column assigned to each of `rowInd`. */
  readonly colInd: readonly number[]
  /** Total cost (or benefit, when maximising) of the assignment. */
  readonly cost: number
}

function matrixOf(cost: TensorLike, name: string): { a: Float64Array; n: number; m: number } {
  const t = primalOf(asTensor(cost))
  if (t.shape.length !== 2) throw new MdmlError('shape', `${name}: the cost matrix must be 2-D`)
  const a = Float64Array.from(toFlat(t))
  if (a.some(Number.isNaN)) throw new MdmlError('value', `${name}: the cost matrix contains NaN`)
  return { a, n: t.shape[0] ?? 0, m: t.shape[1] ?? 0 }
}

/** The state of {@link hungarianAlgorithm}: potentials and the matching after each added row. */
export interface HungarianState {
  /** Row potentials u (1-based, u[0] unused). */
  readonly u: Float64Array
  /** Column potentials v (v[0] is the virtual column). */
  readonly v: Float64Array
  /** The row matched to each column (1-based rows, 0 for none; index 0 is the virtual column). */
  readonly p: Int32Array
  /** Rows added so far. */
  readonly row: number
}

/**
 * The Hungarian algorithm (Kuhn-Munkres in the shortest-augmenting-path form with potentials, O(n^2 m)) as a traced
 * algorithm: each step adds one row of an n x m cost matrix (n <= m) by a Dijkstra-like search for the cheapest
 * augmenting path, updating the dual potentials u and v so that u_i + v_j <= c_ij with equality on matched pairs.
 * Records the dual objective sum u + sum v (a lower bound that reaches the optimum) and the matching.
 *
 * @param cost - The cost matrix, at most as many rows as columns (+Infinity forbids a pair).
 * @returns The algorithm.
 * @throws MdmlError with code `value` when a row cannot be assigned (all its feasible columns taken).
 * @concept assignment-problem
 * @example
 * ```ts
 * import { hungarianAlgorithm } from 'mdml-engine/optim'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * Array.from(runAlgorithm(hungarianAlgorithm([[4, 1], [2, 3]]), { maxSteps: 10 }).state.p) // => [0, 2, 1]
 * ```
 */
export function hungarianAlgorithm(cost: TensorLike): Algorithm<HungarianState> {
  const { a, n, m } = matrixOf(cost, 'hungarianAlgorithm')
  if (n > m) throw new MdmlError('shape', 'hungarianAlgorithm: needs at most as many rows as columns')
  const c = (i: number, j: number): number => a[(i - 1) * m + (j - 1)] ?? Infinity
  return {
    name: 'hungarian',
    config: { rows: n, cols: m },
    schema: channels({
      metrics: {
        dual: { description: 'dual objective: sum of row and column potentials (a lower bound)' },
        matched: { description: 'rows assigned so far' },
      },
      tensors: {
        u: { role: 'state', description: 'row potentials' },
        v: { role: 'state', description: 'column potentials' },
      },
    }),
    init: () => ({ u: new Float64Array(n + 1), v: new Float64Array(m + 1), p: new Int32Array(m + 1), row: 0 }),
    step: (s) => {
      const u = Float64Array.from(s.u)
      const v = Float64Array.from(s.v)
      const p = Int32Array.from(s.p)
      const i = s.row + 1
      const way = new Int32Array(m + 1)
      const minv = new Float64Array(m + 1).fill(Infinity)
      const used = new Uint8Array(m + 1)
      p[0] = i
      let j0 = 0
      do {
        used[j0] = 1
        const i0 = p[j0] ?? 0
        let delta = Infinity
        let j1 = -1
        for (let j = 1; j <= m; j++) {
          if (used[j] === 1) continue
          const cur = c(i0, j) - (u[i0] ?? 0) - (v[j] ?? 0)
          if (cur < (minv[j] ?? Infinity)) {
            minv[j] = cur
            way[j] = j0
          }
          if ((minv[j] ?? Infinity) < delta) {
            delta = minv[j] ?? Infinity
            j1 = j
          }
        }
        if (j1 < 0 || !Number.isFinite(delta))
          throw new MdmlError('value', 'linearSumAssignment: the cost matrix is infeasible')
        for (let j = 0; j <= m; j++) {
          if (used[j] === 1) {
            const r = p[j] ?? 0
            u[r] = (u[r] ?? 0) + delta
            v[j] = (v[j] ?? 0) - delta
          } else minv[j] = (minv[j] ?? 0) - delta
        }
        j0 = j1
      } while ((p[j0] ?? 0) !== 0)
      do {
        const j1 = way[j0] ?? 0
        p[j0] = p[j1] ?? 0
        j0 = j1
      } while (j0 !== 0)
      p[0] = 0
      return { u, v, p, row: i }
    },
    done: (s) => (s.row >= n ? 'converged' : undefined),
    observe: (s) => {
      let dual = 0
      for (let i = 1; i <= n; i++) dual += s.u[i] ?? 0
      for (let j = 1; j <= m; j++) dual += s.v[j] ?? 0
      return {
        metrics: { dual, matched: s.row },
        tensors: { u: { role: 'state', value: s.u.subarray(1) }, v: { role: 'state', value: s.v.subarray(1) } },
      }
    },
  }
}

/**
 * Solve the linear sum assignment problem (scipy.optimize.linear_sum_assignment): match each row of a rectangular cost
 * matrix to a distinct column (or each column to a row, when there are more rows) at least total cost, by the
 * Hungarian algorithm.
 *
 * @param cost - The cost matrix (+Infinity forbids a pair).
 * @param options - `maximize` to maximise instead, and run options.
 * @returns Row and column indices, rows increasing, and the total.
 * @throws MdmlError with code `value` when no complete assignment exists.
 * @concept assignment-problem
 * @example
 * ```ts
 * import { linearSumAssignment } from 'mdml-engine/optim'
 * const r = linearSumAssignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
 * [r.colInd, r.cost] // => [[1, 0, 2], 5]
 * ```
 */
export function linearSumAssignment(
  cost: TensorLike,
  options: { readonly maximize?: boolean } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): AssignmentResult {
  const { maximize, ...run } = options
  const { a, n, m } = matrixOf(cost, 'linearSumAssignment')
  if (n === 0 || m === 0) return { rowInd: [], colInd: [], cost: 0 }
  const transposed = n > m
  const rows = transposed ? m : n
  const cols = transposed ? n : m
  const work = new Float64Array(rows * cols)
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      const v = transposed ? (a[j * m + i] ?? 0) : (a[i * m + j] ?? 0)
      work[i * cols + j] = maximize === true ? -v : v
    }
  const matrix = Array.from({ length: rows }, (_, i) => Array.from(work.subarray(i * cols, (i + 1) * cols)))
  const { state } = runAlgorithm(hungarianAlgorithm(matrix), { ...run, maxSteps: rows })
  const pairs: [number, number][] = []
  for (let j = 1; j <= cols; j++) {
    const i = state.p[j] ?? 0
    if (i > 0) pairs.push(transposed ? [j - 1, i - 1] : [i - 1, j - 1])
  }
  pairs.sort((x, y) => x[0] - y[0])
  const total = pairs.reduce((s, [i, j]) => s + (a[i * m + j] ?? 0), 0)
  return { rowInd: pairs.map((q) => q[0]), colInd: pairs.map((q) => q[1]), cost: total }
}

/** Options of {@link auctionAlgorithm} and {@link auction}. */
export interface AuctionOptions {
  /** Maximise the total (default false: minimise cost, as `linearSumAssignment`). */
  readonly maximize?: boolean
  /** Starting epsilon (default half the largest absolute benefit). */
  readonly epsilon?: number
  /** Final epsilon: the assignment is within n epsilon of optimal (default 1e-9 max(1, C) / n). */
  readonly epsilonMin?: number
  /** Epsilon is divided by this between scaling phases (default 5). */
  readonly scaling?: number
}

/** The state of {@link auctionAlgorithm}. */
export interface AuctionState {
  /** Object prices. */
  readonly prices: Float64Array
  /** The object held by each person (-1 for none). */
  readonly owner: Int32Array
  /** The person holding each object (-1 for none). */
  readonly holder: Int32Array
  readonly epsilon: number
  readonly bids: number
  readonly last?: { readonly person: number; readonly object: number; readonly increment: number }
}

/**
 * Bertsekas' auction algorithm for the assignment problem with epsilon scaling, as a traced algorithm (Gauss-Seidel:
 * each step is one bid by an unassigned person for its best object, raising the price by the margin over the second
 * best plus epsilon). When everyone is assigned, epsilon shrinks by `scaling` and the assignment restarts from the
 * current prices, until epsilon reaches `epsilonMin`; the result satisfies epsilon-complementary slackness, so it is
 * within n epsilon of optimal (exactly optimal for integer benefits once epsilon < 1/n). A rectangular matrix is padded
 * with zero-benefit dummies. Records epsilon, unassigned persons and the assigned benefit, and every bid.
 *
 * @param cost - The cost (or benefit, with `maximize`) matrix.
 * @param options - Direction and epsilon schedule.
 * @returns The algorithm (on the padded square problem).
 * @concept assignment-problem
 * @reference D. P. Bertsekas, The auction algorithm: a distributed relaxation method for the assignment problem (1988).
 * @example
 * ```ts
 * import { auctionAlgorithm } from 'mdml-engine/optim'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * Array.from(runAlgorithm(auctionAlgorithm([[4, 1], [2, 3]]), { maxSteps: 1000 }).state.owner) // => [1, 0]
 * ```
 */
export function auctionAlgorithm(cost: TensorLike, options: AuctionOptions = {}): Algorithm<AuctionState> {
  const { a, n, m } = matrixOf(cost, 'auctionAlgorithm')
  const k = Math.max(n, m)
  const sign = options.maximize === true ? 1 : -1
  const benefit = new Float64Array(k * k)
  let big = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      const v = sign * (a[i * m + j] ?? 0)
      if (!Number.isFinite(v)) throw new MdmlError('value', 'auction: benefits must be finite')
      benefit[i * k + j] = v
      big = Math.max(big, Math.abs(v))
    }
  const scaling = options.scaling ?? 5
  const epsMin = options.epsilonMin ?? (1e-9 * Math.max(1, big)) / Math.max(1, k)
  const eps0 = Math.max(options.epsilon ?? big / 2, epsMin)
  const assignedValue = (s: AuctionState): number => {
    let v = 0
    for (let i = 0; i < n; i++) {
      const j = s.owner[i] ?? -1
      if (j >= 0 && j < m) v += a[i * m + j] ?? 0
    }
    return v
  }
  return {
    name: 'auction',
    config: { maximize: options.maximize ?? false, scaling, epsilonMin: epsMin } as Json,
    schema: channels({
      metrics: {
        epsilon: { description: 'current epsilon of the scaling phase' },
        unassigned: { description: 'persons without an object' },
        value: { description: 'total of the current (partial) assignment in the original matrix' },
      },
      tensors: { prices: { role: 'state', description: 'object prices' } },
      messages: { bid: { description: 'a bid: person, object and price increment' } },
    }),
    init: () => ({
      prices: new Float64Array(k),
      owner: new Int32Array(k).fill(-1),
      holder: new Int32Array(k).fill(-1),
      epsilon: eps0,
      bids: 0,
    }),
    step: (s) => {
      const person = s.owner.indexOf(-1)
      if (person < 0) {
        // Everyone is assigned: start the next phase with a smaller epsilon and the current prices.
        return {
          prices: s.prices,
          owner: new Int32Array(k).fill(-1),
          holder: new Int32Array(k).fill(-1),
          epsilon: Math.max(epsMin, s.epsilon / scaling),
          bids: s.bids,
        }
      }
      let best = -1
      let first = -Infinity
      let second = -Infinity
      for (let j = 0; j < k; j++) {
        const value = (benefit[person * k + j] ?? 0) - (s.prices[j] ?? 0)
        if (value > first) {
          second = first
          first = value
          best = j
        } else if (value > second) second = value
      }
      const increment = (Number.isFinite(second) ? first - second : 0) + s.epsilon
      const prices = Float64Array.from(s.prices)
      prices[best] = (prices[best] ?? 0) + increment
      const owner = Int32Array.from(s.owner)
      const holder = Int32Array.from(s.holder)
      const previous = holder[best] ?? -1
      if (previous >= 0) owner[previous] = -1
      holder[best] = person
      owner[person] = best
      return { prices, owner, holder, epsilon: s.epsilon, bids: s.bids + 1, last: { person, object: best, increment } }
    },
    done: (s) => (s.epsilon <= epsMin && !s.owner.includes(-1) ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        epsilon: s.epsilon,
        unassigned: s.owner.reduce((c, j) => c + (j < 0 ? 1 : 0), 0),
        value: assignedValue(s),
      },
      tensors: { prices: { role: 'state', value: s.prices } },
      ...(s.last === undefined ? {} : { messages: [{ kind: 'bid', payload: { ...s.last } }] }),
    }),
  }
}

/**
 * Solve the assignment problem by Bertsekas' auction algorithm with epsilon scaling (the same answer as
 * {@link linearSumAssignment} up to n epsilon_min; a parallelisable, economically interpretable alternative).
 *
 * @param cost - The cost (or benefit) matrix.
 * @param options - Direction, epsilon schedule, bid cap (default 1e6) and run options.
 * @returns Row and column indices and the total.
 * @concept assignment-problem
 * @example
 * ```ts
 * import { auction } from 'mdml-engine/optim'
 * auction([[4, 1, 3], [2, 0, 5], [3, 2, 2]]).colInd // => [1, 0, 2]
 * ```
 */
export function auction(
  cost: TensorLike,
  options: AuctionOptions & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> & { readonly maxBids?: number } = {},
): AssignmentResult {
  const { maximize, epsilon, epsilonMin, scaling, maxBids, ...run } = options
  const { a, n, m } = matrixOf(cost, 'auction')
  const alg = auctionAlgorithm(cost, {
    ...(maximize === undefined ? {} : { maximize }),
    ...(epsilon === undefined ? {} : { epsilon }),
    ...(epsilonMin === undefined ? {} : { epsilonMin }),
    ...(scaling === undefined ? {} : { scaling }),
  })
  const result = runAlgorithm(alg, { ...run, maxSteps: maxBids ?? 1_000_000 })
  if (result.status !== 'converged') throw new MdmlError('convergence', 'auction: bid limit reached')
  const rowInd: number[] = []
  const colInd: number[] = []
  let total = 0
  for (let i = 0; i < n; i++) {
    const j = result.state.owner[i] ?? -1
    if (j < 0 || j >= m) continue
    rowInd.push(i)
    colInd.push(j)
    total += a[i * m + j] ?? 0
  }
  return { rowInd, colInd, cost: total }
}

/** Result of {@link stableMatching}. */
export interface StableMatchingResult {
  /** The receiver matched to each proposer (-1 when unmatched). */
  readonly proposerMatch: readonly number[]
  /** The proposer matched to each receiver (-1 when unmatched). */
  readonly receiverMatch: readonly number[]
  readonly proposals: number
}

/** The state of {@link galeShapleyAlgorithm}. */
export interface GaleShapleyState {
  readonly proposerMatch: Int32Array
  readonly receiverMatch: Int32Array
  /** The next position in each proposer's list. */
  readonly next: Int32Array
  readonly proposals: number
  readonly last?: { readonly from: number; readonly to: number; readonly accepted: boolean; readonly rejected: number }
}

/**
 * The Gale-Shapley deferred-acceptance algorithm as a traced algorithm: each step, a free proposer who still has
 * receivers to try proposes to the next one on its list, who holds the better of its current partner and the proposer
 * (by its own list) and rejects the other. The result is the proposer-optimal stable matching. Lists may be incomplete
 * (an unlisted pair is unacceptable) and the sides may differ in size.
 *
 * @param proposerPrefs - For each proposer, receivers in decreasing preference.
 * @param receiverPrefs - For each receiver, proposers in decreasing preference.
 * @returns The algorithm.
 * @concept stable-matching
 * @example
 * ```ts
 * import { galeShapleyAlgorithm } from 'mdml-engine/optim'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = galeShapleyAlgorithm([[0, 1], [0, 1]], [[1, 0], [0, 1]])
 * Array.from(runAlgorithm(alg, { maxSteps: 10 }).state.proposerMatch) // => [1, 0]
 * ```
 */
export function galeShapleyAlgorithm(
  proposerPrefs: readonly (readonly number[])[],
  receiverPrefs: readonly (readonly number[])[],
): Algorithm<GaleShapleyState> {
  const p = proposerPrefs.length
  const r = receiverPrefs.length
  // rank[receiver][proposer]: position in the receiver's list (Infinity when unacceptable).
  const rank = receiverPrefs.map((list) => {
    const out = new Float64Array(p).fill(Infinity)
    for (const [k, who] of list.entries()) out[who] = k
    return out
  })
  const free = (s: GaleShapleyState): number => {
    for (let i = 0; i < p; i++)
      if ((s.proposerMatch[i] ?? -1) < 0 && (s.next[i] ?? 0) < (proposerPrefs[i]?.length ?? 0)) return i
    return -1
  }
  return {
    name: 'galeShapley',
    config: { proposers: p, receivers: r },
    schema: channels({
      metrics: { matched: { description: 'matched pairs' }, proposals: { description: 'proposals made' } },
      messages: { proposal: { description: 'a proposal: from, to, accepted, and the rejected proposer (-1 none)' } },
    }),
    init: () => ({
      proposerMatch: new Int32Array(p).fill(-1),
      receiverMatch: new Int32Array(r).fill(-1),
      next: new Int32Array(p),
      proposals: 0,
    }),
    step: (s) => {
      const i = free(s)
      const proposerMatch = Int32Array.from(s.proposerMatch)
      const receiverMatch = Int32Array.from(s.receiverMatch)
      const next = Int32Array.from(s.next)
      const j = proposerPrefs[i]?.[next[i] ?? 0] ?? 0
      next[i] = (next[i] ?? 0) + 1
      const ranks = rank[j] as Float64Array
      const current = receiverMatch[j] ?? -1
      let accepted = false
      let rejected = -1
      if (Number.isFinite(ranks[i] ?? Infinity)) {
        if (current < 0) accepted = true
        else if ((ranks[i] ?? Infinity) < (ranks[current] ?? Infinity)) {
          accepted = true
          rejected = current
          proposerMatch[current] = -1
        }
      }
      if (accepted) {
        receiverMatch[j] = i
        proposerMatch[i] = j
      } else rejected = i
      return {
        proposerMatch,
        receiverMatch,
        next,
        proposals: s.proposals + 1,
        last: { from: i, to: j, accepted, rejected },
      }
    },
    done: (s) => (free(s) < 0 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { matched: s.proposerMatch.reduce((c, j) => c + (j >= 0 ? 1 : 0), 0), proposals: s.proposals },
      ...(s.last === undefined ? {} : { messages: [{ kind: 'proposal', payload: { ...s.last } }] }),
    }),
  }
}

/**
 * The proposer-optimal stable matching by Gale-Shapley deferred acceptance.
 *
 * @param proposerPrefs - For each proposer, receivers in decreasing preference.
 * @param receiverPrefs - For each receiver, proposers in decreasing preference.
 * @param options - Run options.
 * @returns Both sides of the matching and the number of proposals.
 * @concept stable-matching
 * @example
 * ```ts
 * import { stableMatching } from 'mdml-engine/optim'
 * const men = [[0, 1, 2], [1, 0, 2], [0, 1, 2]]
 * const women = [[1, 0, 2], [0, 1, 2], [0, 1, 2]]
 * stableMatching(men, women).proposerMatch // => [0, 1, 2]
 * ```
 */
export function stableMatching(
  proposerPrefs: readonly (readonly number[])[],
  receiverPrefs: readonly (readonly number[])[],
  options: Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): StableMatchingResult {
  const total = proposerPrefs.reduce((s, l) => s + l.length, 0)
  const { state } = runAlgorithm(galeShapleyAlgorithm(proposerPrefs, receiverPrefs), {
    ...options,
    maxSteps: total + 1,
  })
  return {
    proposerMatch: Array.from(state.proposerMatch),
    receiverMatch: Array.from(state.receiverMatch),
    proposals: state.proposals,
  }
}

/**
 * The blocking pairs of a matching: a proposer and a receiver, acceptable to each other, who both prefer each other to
 * their partners (unmatched counts as worst). A matching is stable when there are none.
 *
 * @param proposerMatch - The receiver of each proposer (-1 unmatched).
 * @param proposerPrefs - Proposers' lists.
 * @param receiverPrefs - Receivers' lists.
 * @returns The blocking pairs as `[proposer, receiver]`.
 * @concept stable-matching
 * @example
 * ```ts
 * import { blockingPairs } from 'mdml-engine/optim'
 * blockingPairs([1, 0], [[0, 1], [0, 1]], [[0, 1], [0, 1]]) // => [[0, 0]]
 * ```
 */
export function blockingPairs(
  proposerMatch: readonly number[],
  proposerPrefs: readonly (readonly number[])[],
  receiverPrefs: readonly (readonly number[])[],
): [number, number][] {
  const receiverMatch = new Array<number>(receiverPrefs.length).fill(-1)
  for (const [i, j] of proposerMatch.entries()) if (j >= 0) receiverMatch[j] = i
  const out: [number, number][] = []
  for (const [i, list] of proposerPrefs.entries()) {
    const mine = proposerMatch[i] ?? -1
    for (const j of list) {
      if (j === mine) break
      const theirs = receiverMatch[j] ?? -1
      const rl = receiverPrefs[j] ?? []
      const ri = rl.indexOf(i)
      if (ri < 0) continue
      const rc = theirs < 0 ? Infinity : rl.indexOf(theirs)
      if (ri < (rc < 0 ? Infinity : rc)) out.push([i, j])
    }
  }
  return out
}
