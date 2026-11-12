/**
 * What the brute-force tests of the factor operators share: the seeded property-test configuration, the arbitraries of Gaussian
 * messages, the operator context, and the checks that compare a message with the integral of the exact one.
 */

import fc from 'fast-check'
import { expect } from 'vitest'
import * as M from '../lib/message-passing/index.ts'
import type { BrutePort } from './message-passing.ts'

// Every property of this file runs on one fixed seed, so the file is deterministic by default (a refactoring step that
// changes nothing cannot fail by luck). OMAI_FC_SEED=<n> runs another seed (a property that depends on its seed shows on a
// few of them); OMAI_FC_SEED=random draws a fresh one each run (a stress run).
const FC_SEED = 20261111
const seedOverride = process.env['OMAI_FC_SEED']
if (seedOverride !== 'random') fc.configureGlobal({ seed: seedOverride === undefined ? FC_SEED : Number(seedOverride) })
// OMAI_FC_SCALE multiplies every property's number of runs (a stress run, 15 times the runs, found nothing on five seeds).
export const scale: number = Number(process.env['OMAI_FC_SCALE'] ?? 1)
export const runs: { readonly numRuns: number } = { numRuns: 12 * scale }
export const mean: fc.Arbitrary<number> = fc.double({ min: -2, max: 2, noNaN: true })
export const variance: fc.Arbitrary<number> = fc.double({ min: 0.3, max: 3, noNaN: true })
export const gaussianArg: fc.Arbitrary<M.Eta> = fc.tuple(mean, variance).map(([m, v]) => M.gaussianEta(m, v))

export const msg = (fam: M.Family, eta: M.Eta): M.Arg => M.messageArg(fam, eta)
export const ctx = (to: number, args: M.Arg[], params: M.Params = {}, mode: M.Mode = 'ep'): M.OperatorContext => ({
  mode,
  to,
  args,
  params,
  power: 1,
})

/** The moments of the cavity times the message: what the projection of the tilted distribution must match. */
export function implied(fam: M.Family, cavity: M.Eta, message: M.Eta): number[] {
  return [...fam.moments(fam.product(cavity, message))]
}

export const near = (got: readonly number[], want: readonly number[], tol = 1e-6): void => {
  expect(got.length).toBe(want.length)
  for (const [i, v] of got.entries())
    expect(Math.abs(v - (want[i] as number)), `entry ${i}: ${v} vs ${want[i]}`).toBeLessThan(
      tol * (1 + Math.abs(want[i] as number)),
    )
}

export const boxOf = (mw: ArrayLike<number>, cw: ArrayLike<number>): [number, number][] =>
  [0, 3].map(
    (k, i) => [(mw[i] ?? 0) - 8 * Math.sqrt(cw[k] ?? 1), (mw[i] ?? 0) + 8 * Math.sqrt(cw[k] ?? 1)] as [number, number],
  )

export const gp = (eta: M.Eta): BrutePort => ({ kind: 'gaussian', eta })
export const gaussian: M.Family = M.gaussian
