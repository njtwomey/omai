// Test helpers for the time-series domain: a seeded ARMA simulator and the check that an algorithm's state survives
// serialisation in the middle of a run. Not published.
import { expect } from 'vitest'
import { armaProcess } from '../lib/datasets/index.ts'
import { key } from '../lib/random/index.ts'
import { toFlat } from '../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../lib/trace/index.ts'

/** A seeded Gaussian ARMA series of length n. */
export function simulateArma(
  seed: number,
  n: number,
  ar: readonly number[],
  ma: readonly number[],
  mean = 0,
  sigma = 1,
): number[] {
  return toFlat(armaProcess(key(seed), { ar, ma, n, mean, sigma }).data.values)
}

/**
 * Run an algorithm `first` steps, serialise its state, restore it and run `rest` more steps; the restored run must
 * observe the same as an uninterrupted run of `first + rest` steps.
 */
export function expectResumable<S>(algorithm: Algorithm<S>, first: number, rest: number): void {
  if (algorithm.serialise === undefined || algorithm.deserialise === undefined)
    throw new Error('the algorithm does not serialise')
  const straight = runAlgorithm(algorithm, { maxSteps: first + rest, seed: 1 })
  const paused = runAlgorithm(algorithm, { maxSteps: first, seed: 1 })
  const restored = algorithm.deserialise(algorithm.serialise(paused.state))
  const resumed = runAlgorithm(algorithm, {
    maxSteps: rest,
    seed: 1,
    start: { state: restored, step: paused.steps },
  })
  expect(algorithm.observe?.(resumed.state, first + rest)).toEqual(algorithm.observe?.(straight.state, first + rest))
}
