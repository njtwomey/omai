/**
 * Stability-based solution concepts for cooperative games. The core is the set of allocations that no coalition can
 * improve on by leaving. Here are a core-membership test, the excess of every coalition, the least core and a core
 * point by linear programming, and the nucleolus, the allocation that lexicographically minimises the largest excesses.
 */

import { MdmlError } from '../base/index.ts'
import { matrixRank } from '../numerics/linalg/index.ts'
import { type LinprogOptions, linprog } from '../optim/index.ts'
import { toFlat } from '../tensor/index.ts'
import type { CoalitionalGame } from './cooperative.ts'

/** The total payoff a coalition (bitmask) receives under an allocation. */
function share(x: readonly number[], mask: number): number {
  let s = 0
  for (let i = 0; i < x.length; i++) if ((mask >> i) & 1) s += x[i] ?? 0
  return s
}

/** The indicator row of a coalition. */
function indicator(n: number, mask: number): number[] {
  return Array.from({ length: n }, (_, i) => ((mask >> i) & 1 ? 1 : 0))
}

/**
 * The excess `v(S) - x(S)` of every coalition (by bitmask) under an allocation: how much better off S would be on
 * its own. An allocation is in the core when no excess is positive.
 *
 * @param game - The game.
 * @param x - One payoff per player.
 * @returns The excess of every coalition, indexed by bitmask (0 for the empty coalition).
 * @throws MdmlError with code `shape` for an allocation of the wrong length.
 * @concept core
 * @example
 * ```ts
 * import { excesses, gloveGame } from 'mdml-engine/games'
 * excesses(gloveGame(1, 1), [0.5, 0.5]) // => [0, -0.5, -0.5, 0]
 * ```
 */
export function excesses(game: CoalitionalGame, x: readonly number[]): number[] {
  if (x.length !== game.n) throw new MdmlError('shape', 'excesses: one payoff per player')
  return Array.from(game.values, (v, m) => (m === 0 ? 0 : v - share(x, m)))
}

/**
 * Whether an allocation is in the core: efficient (`x(N) = v(N)`) and coalitionally rational (`x(S) >= v(S)` for
 * every coalition), within `tol`.
 *
 * @param game - The game.
 * @param x - One payoff per player.
 * @param tol - Tolerance (default 1e-9).
 * @returns Whether no coalition can do better on its own.
 * @concept core
 * @example
 * ```ts
 * import { gloveGame, isInCore } from 'mdml-engine/games'
 * [isInCore(gloveGame(1, 2), [1, 0, 0]), isInCore(gloveGame(1, 2), [0.5, 0.25, 0.25])] // => [true, false]
 * ```
 */
export function isInCore(game: CoalitionalGame, x: readonly number[], tol = 1e-9): boolean {
  const e = excesses(game, x)
  const full = game.values.length - 1
  if (Math.abs(e[full] ?? 0) > tol) return false
  return e.every((v) => v <= tol)
}

/** The outcome of {@link leastCore}. */
export interface LeastCoreResult {
  /** The smallest epsilon such that some efficient allocation gives every coalition at least `v(S) - epsilon`. */
  readonly epsilon: number
  /** An allocation achieving it. */
  readonly allocation: readonly number[]
}

/**
 * The least core (Maschler, Peleg and Shapley 1979): the smallest epsilon for which the epsilon-core, the efficient
 * allocations with `x(S) >= v(S) - epsilon` for every proper coalition, is non-empty, by one linear program. The core
 * is non-empty exactly when epsilon <= 0.
 *
 * @param game - The game (at least two players).
 * @param options - Options for the linear program.
 * @returns Epsilon and an allocation in the least core.
 * @throws MdmlError with code `value` for a one-player game, `convergence` when the program fails.
 * @concept core
 * @example
 * ```ts
 * import { coalitionalGame, leastCore } from 'mdml-engine/games'
 * Math.round(leastCore(coalitionalGame(3, (s) => (s.length >= 2 ? 1 : 0))).epsilon * 1e9) / 1e9 // => 0.333333333
 * ```
 */
export function leastCore(game: CoalitionalGame, options: LinprogOptions = {}): LeastCoreResult {
  const { n, values } = game
  if (n < 2) throw new MdmlError('value', 'leastCore: at least two players are needed')
  const full = values.length - 1
  const aUb: number[][] = []
  const bUb: number[] = []
  for (let m = 1; m < full; m++) {
    aUb.push([...indicator(n, m).map((v) => -v), -1])
    bUb.push(-(values[m] ?? 0))
  }
  const r = linprog(
    {
      c: [...new Array<number>(n).fill(0), 1],
      aUb,
      bUb,
      aEq: [[...new Array<number>(n).fill(1), 0]],
      bEq: [values[full] ?? 0],
      bounds: [null, null],
    },
    options,
  )
  if (!r.success) throw new MdmlError('convergence', `leastCore: the linear program ended with '${r.status}'`)
  const z = toFlat(r.x)
  return { epsilon: z[n] ?? 0, allocation: z.slice(0, n) }
}

/**
 * A point of the core, or `null` when the core is empty (decided by the least core).
 *
 * @param game - The game.
 * @param options - Tolerance for emptiness (default 1e-9) and options for the linear program.
 * @returns An allocation in the core (a least-core point), or `null`.
 * @concept core
 * @example
 * ```ts
 * import { coalitionalGame, corePoint, gloveGame } from 'mdml-engine/games'
 * [corePoint(gloveGame(1, 2))?.map((v) => Math.round(v * 1e9) / 1e9), corePoint(coalitionalGame(3, (s) => (s.length >= 2 ? 1 : 0)))] // => [[1, 0, 0], null]
 * ```
 */
export function corePoint(
  game: CoalitionalGame,
  options: LinprogOptions & { readonly tol?: number } = {},
): number[] | null {
  if (game.n === 1) return [game.values[1] ?? 0]
  const { tol = 1e-9, ...lp } = options
  const r = leastCore(game, lp)
  return r.epsilon <= tol ? [...r.allocation] : null
}

/**
 * The nucleolus (Schmeidler 1969): the allocation that lexicographically minimises the coalitions' excesses, largest
 * first, over the imputations (or, with `pre`, over all efficient allocations: the prenucleolus). Computed by the
 * sequential linear programs of Maschler, Peleg and Shapley: maximise the smallest surplus `x(S) - v(S)` of the
 * coalitions not yet settled, settle those whose surplus no optimal solution can raise, and repeat until the settled
 * coalitions determine the allocation. It lies in the core whenever the core is non-empty; for bankruptcy games it
 * is the Talmud rule.
 *
 * @param game - The game.
 * @param options - `pre` for the prenucleolus, and options for the linear programs.
 * @returns The nucleolus.
 * @throws MdmlError with code `value` when there are no imputations (`v(N) < sum of v(i)`), `convergence` when a
 * linear program fails.
 * @concept nucleolus
 * @example
 * ```ts
 * import { bankruptcyGame, nucleolus } from 'mdml-engine/games'
 * nucleolus(bankruptcyGame(200, [100, 200, 300])).map((v) => Math.round(v * 1e6) / 1e6) // => [50, 75, 75]
 * ```
 */
export function nucleolus(game: CoalitionalGame, options: LinprogOptions & { readonly pre?: boolean } = {}): number[] {
  const { pre = false, ...lp } = options
  const { n, values } = game
  const full = values.length - 1
  if (n === 1) return [values[1] ?? 0]
  let singles = 0
  for (let i = 0; i < n; i++) singles += values[1 << i] ?? 0
  if (!pre && (values[full] ?? 0) < singles - 1e-12)
    throw new MdmlError('value', 'nucleolus: the game has no imputations (v(N) < sum of v(i))')
  const bounds = Array.from({ length: n }, (_, i) => [pre ? null : (values[1 << i] ?? 0), null] as const)
  const free = new Set<number>()
  for (let m = 1; m < full; m++) free.add(m)
  const fixed = new Map<number, number>()
  let x: number[] = []
  // Each stage settles at least one coalition, so 2^n stages bound the loop.
  for (let stage = 0; stage < values.length && free.size > 0; stage++) {
    const eqRows = [[...new Array<number>(n).fill(1)]]
    const eqRhs = [values[full] ?? 0]
    for (const [m, s] of fixed) {
      eqRows.push(indicator(n, m))
      eqRhs.push((values[m] ?? 0) + s)
    }
    const coalitions = [...free]
    // Variables (x, t): maximise t subject to x(S) - v(S) >= t for every free S.
    const r = linprog(
      {
        c: [...new Array<number>(n).fill(0), -1],
        aUb: coalitions.map((m) => [...indicator(n, m).map((v) => -v), 1]),
        bUb: coalitions.map((m) => -(values[m] ?? 0)),
        aEq: eqRows.map((row) => [...row, 0]),
        bEq: eqRhs,
        bounds: [...bounds, [null, null]],
      },
      lp,
    )
    if (!r.success) throw new MdmlError('convergence', `nucleolus: the linear program ended with '${r.status}'`)
    const z = toFlat(r.x)
    const t = z[n] ?? 0
    x = z.slice(0, n)
    const tol = 1e-9 * (1 + Math.abs(t))
    const tight = coalitions.filter((m) => share(x, m) - (values[m] ?? 0) <= t + tol)
    const settle: number[] = []
    for (const m of tight) {
      // Can some optimal solution give S more than t? Maximise x(S) subject to every free surplus >= t.
      const q = linprog(
        {
          c: indicator(n, m).map((v) => -v),
          aUb: coalitions.map((k) => indicator(n, k).map((v) => -v)),
          bUb: coalitions.map((k) => -(values[k] ?? 0) - t),
          aEq: eqRows,
          bEq: eqRhs,
          bounds,
        },
        lp,
      )
      if (!q.success || -q.fun - (values[m] ?? 0) <= t + 1e-7 * (1 + Math.abs(t))) settle.push(m)
    }
    for (const m of settle.length > 0 ? settle : tight) {
      free.delete(m)
      fixed.set(m, t)
    }
    const rows = [new Array<number>(n).fill(1), ...[...fixed.keys()].map((m) => indicator(n, m))]
    if (matrixRank(rows) >= n) break
  }
  return x
}
