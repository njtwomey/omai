/**
 * What the prose of 'Breaking a stick' says: from the prior draws (how big the first piece is and how small the later ones are),
 * from the numbers alone (their range), and from the result (which pieces hold real weight, how much of the stick they hold,
 * and how close their centres are to the centres the numbers were drawn from). Pure functions, light enough for the page's bundle.
 */

import type { Result } from './compute.ts'
import { CENTRES, SHARES } from './data.ts'

/** A piece holds real weight when it holds at least this share of the stick. */
export const REAL = 0.05

/** The numbers the prose quotes about the prior draws. */
export interface PriorReading {
  /** The draws that were made. */
  readonly draws: number
  /** The average share of the stick that the first piece and the fifth piece hold, over the draws. */
  readonly first: number
  readonly fifth: number
  /** The average share that the last four pieces hold together. */
  readonly tail: number
}

const average = (values: readonly number[]): number => values.reduce((a, b) => a + b, 0) / Math.max(1, values.length)

/**
 * The numbers about the prior draws.
 *
 * @param prior - One row of weights per draw.
 * @returns The average first piece, fifth piece and tail.
 */
export const readPrior = (prior: Result['prior']): PriorReading => ({
  draws: prior.length,
  first: average(prior.map((row) => row[0] as number)),
  fifth: average(prior.map((row) => row[4] as number)),
  tail: average(prior.map((row) => row.slice(4).reduce((a, b) => a + b, 0))),
})

/** A piece that holds real weight. */
export interface Piece {
  /** Which piece it is, counting from 1. */
  readonly number: number
  readonly weight: number
  readonly centre: number
}

/** The numbers the prose quotes after the run. */
export interface Reading {
  /** The pieces that hold real weight, from the lowest centre to the highest. */
  readonly real: readonly Piece[]
  /** The share of the stick the real pieces hold together, and the share the others hold. */
  readonly held: number
  readonly rest: number
  /** How many groups the numbers were drawn from. */
  readonly groups: number
  /** For each of those groups, how far the nearest real piece's centre is from the group's centre; and the largest of these. */
  readonly errors: readonly number[]
  readonly worst: number
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The pieces with real weight, the share they hold and how close their centres are to the groups' centres.
 */
export function readResult(result: Result): Reading {
  const real = result.weights
    .map((weight, i): Piece => ({ number: i + 1, weight, centre: result.centres[i] as number }))
    .filter((p) => p.weight >= REAL)
    .sort((a, b) => a.centre - b.centre)
  const held = real.reduce((a, p) => a + p.weight, 0)
  const errors = CENTRES.map((c) => Math.min(...real.map((p) => Math.abs(p.centre - c))))
  return { real, held, rest: 1 - held, groups: CENTRES.length, errors, worst: Math.max(...errors) }
}

/** A piece is matched to a group when its centre is within this distance of the group's centre. */
const MATCH = 1

/**
 * The share of the group each piece found, for the pieces whose centre is near the centre of a group the numbers were drawn from.
 *
 * @param result - The result.
 * @returns For each piece that holds real weight, the true share of the nearest group within {@link MATCH} of its centre, or `undefined`.
 */
export const trueShares = (result: Result): (number | undefined)[] =>
  result.centres.map((centre, k) => {
    if ((result.weights[k] as number) < REAL) return undefined
    const near = CENTRES.map((c, g) => ({ g, gap: Math.abs(c - centre) })).sort((a, b) => a.gap - b.gap)[0]
    return near !== undefined && near.gap < MATCH ? SHARES[near.g] : undefined
  })

/** How many pieces have a matching group, and for how many of them the true share is within two standard deviations of the weight. */
export function readTruth(result: Result): { readonly matched: number; readonly within: number } {
  const truth = trueShares(result)
  const matched = truth.filter((t) => t !== undefined)
  const within = truth.filter(
    (t, k) => t !== undefined && Math.abs(t - (result.weights[k] as number)) <= 2 * (result.spread[k] as number),
  )
  return { matched: matched.length, within: within.length }
}
