/**
 * What the lab shows at a moment of the playhead, read from the finished results with no computing beyond slicing: which
 * draws of a method are visible at a time, which frame it is at, and the records the library's views take (the posterior
 * comparison of the marginals, the chain diagnostics of a sampler). The results are plain data from the workers.
 */

import {
  type ChainDiagnostics,
  chainDiagnostics,
  type DrawsLike,
  type NormalsLike,
  type PosteriorComparison,
  posteriorComparison,
} from 'omai-engine/exposition'
import { weightedOracle } from 'omai-engine/probprog/validation'
import { ess as essOf, rhat as rhatOf } from 'omai-engine/sampling'
import { tensor } from 'omai-engine/tensor'
import { METHODS } from './methods.ts'
import type { LabFrame, MethodResult } from './types.ts'

/** The index of the frame a method is at at time `t` (milliseconds): the last one taken by then, the first before any. */
export function frameAt(result: MethodResult, t: number): number {
  let best = 0
  for (const [i, f] of result.frames.entries()) {
    if (f.elapsedMs <= t) best = i
    else break
  }
  return best
}

/** The frame of a method at a time, if it has any. */
export function frameOf(result: MethodResult, t: number): LabFrame | undefined {
  return result.frames[frameAt(result, t)]
}

/** The rows of the pool shown at time `t`: a sampler's draws made by then, an approximation's draws of the current snapshot. */
export function visibleRows(result: MethodResult, t: number): number[] {
  const { at, until } = result.pool
  const out: number[] = []
  const last = result.frames.length === 0 ? 0 : (result.frames[frameAt(result, t)]?.elapsedMs ?? 0)
  const time = result.organisation === 'iterate' ? last : t
  for (let i = 0; i < at.length; i++)
    if ((at[i] ?? 0) <= time && (time < (until[i] ?? 0) || until[i] === Number.POSITIVE_INFINITY)) out.push(i)
  return out
}

/** The column `j` of the given rows of a pool. */
export function column(result: MethodResult, rows: readonly number[], j: number): number[] {
  const { values, p } = result.pool
  return rows.map((i) => values[i * p + j] ?? Number.NaN)
}

/** The predictive grid of a method at a time: `size * size` probabilities, rows by ordinate. */
export function gridAt(result: MethodResult, t: number): Float32Array {
  const n = result.gridSize ** 2
  const k = frameAt(result, t)
  return result.grids.subarray(k * n, (k + 1) * n)
}

/** The colour index of a method (stable across views). */
export const colourOf = (id: string): number =>
  Math.max(
    0,
    METHODS.findIndex((m) => m.id === id),
  )

const name = (j: number): string => `w[${j}]`

/**
 * The marginals of the chosen coefficients under each method at time `t`, as the library's posterior comparison: draws for
 * a sampler or a particle method, a normal for a Gaussian approximation. The reference, when a NUTS run is among the
 * results, is the whole of its draws.
 *
 * @param results - The methods to show.
 * @param coordinates - The coefficient indices.
 * @param t - The playhead, in milliseconds.
 * @returns The record, or `undefined` while there is nothing to compare.
 */
export function marginalsAt(
  results: readonly MethodResult[],
  coordinates: readonly number[],
  t: number,
): PosteriorComparison | undefined {
  const quantities = coordinates.map(name)
  const approximations = results.flatMap((r): (DrawsLike | NormalsLike)[] => {
    if (r.gaussian) {
      const f = frameOf(r, t)
      if (f === undefined || !Number.isFinite(f.mean[0] ?? Number.NaN)) return []
      return [
        {
          name: r.label,
          normal: Object.fromEntries(
            coordinates.map((j) => [name(j), { mean: f.mean[j] ?? 0, sd: Math.max(f.sd[j] ?? 0, 1e-9) }]),
          ),
        },
      ]
    }
    const rows = visibleRows(r, t)
    if (rows.length < 3) return []
    return [
      {
        name: r.label,
        draws: Object.fromEntries(coordinates.map((j) => [name(j), Float64Array.from(column(r, rows, j))])),
      },
    ]
  })
  if (approximations.length === 0) return undefined
  const reference = results.find((r) => r.id === 'nuts' && r.pool.at.length > 3)
  const truth =
    reference === undefined
      ? undefined
      : Object.fromEntries(
          coordinates.map((j) => [
            name(j),
            weightedOracle(name(j), column(reference, visibleRows(reference, Number.POSITIVE_INFINITY), j)),
          ]),
        )
  return posteriorComparison(approximations, {
    quantities,
    quantilePoints: 80,
    gridPoints: 120,
    ...(truth === undefined ? {} : { truth, oracle: 'the whole NUTS run' }),
  })
}

/**
 * The chain diagnostics of a sampler at time `t` for the chosen coefficients: each chain's draws made by then, with the
 * R-hat and effective sample size of what there is so far.
 *
 * @param result - A sampler's result.
 * @param coordinates - The coefficient indices.
 * @param t - The playhead, in milliseconds.
 * @returns The record, or `undefined` before there are a few draws per chain.
 */
export function chainsAt(
  result: MethodResult,
  coordinates: readonly number[],
  t: number,
): ChainDiagnostics | undefined {
  if (result.organisation !== 'chain') return undefined
  const rows = visibleRows(result, t)
  const byChain = new Map<number, number[]>()
  for (const i of rows) byChain.set(result.pool.chain[i] ?? 0, [...(byChain.get(result.pool.chain[i] ?? 0) ?? []), i])
  const chains = [...byChain.values()]
  const n = Math.min(...chains.map((c) => c.length))
  if (chains.length === 0 || !(n >= 8)) return undefined
  const draws = Object.fromEntries(
    coordinates.map((j) => [name(j), chains.map((c) => column(result, c.slice(0, n), j))]),
  )
  const summary = coordinates.map((j) => {
    const x = draws[name(j)] as number[][]
    return {
      name: name(j),
      rhat: chains.length >= 2 ? rhatOf(tensor(x)) : undefined,
      ess: chains.length >= 2 ? essOf(tensor(x)) : undefined,
    }
  })
  return chainDiagnostics(draws, { summary, acfLags: 20, ecdfPoints: 0 })
}
