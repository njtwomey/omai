import { describe, expect, it } from 'vitest'
import { bruteVectorSite } from '../../../../testing/vector-site.ts'
import { massOf, messageArg } from '../../core/args.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import { gaussianMv, gaussianMvEta, gaussianMvMoments } from '../../core/families-vector.ts'
import type { Arg, Params } from '../../core/types.ts'
import { ruleMessage } from '../../engine/dispatch.ts'
import { defaultOperators } from '../registry.ts'
import { likelihoodPairOperator, likelihoodPairRowOperator, pairLikelihoods } from './pairs.ts'

/*
 * The pair row (two coordinates that are linear forms of one Gaussian vector, the densities of the pair registry at them) against a
 * brute-force integral over the vector itself, and against the scalar pair site on independent coordinates.
 */

interface Case {
  readonly likelihood: string
  readonly y: number
  readonly mean: readonly number[]
  readonly covariance: readonly number[]
  readonly matrix: readonly number[]
  readonly offset?: readonly number[]
  readonly extra?: Params
  readonly power?: number
  readonly grid: number
  readonly tolerance?: number
}

function check(c: Case): void {
  const d = c.mean.length
  const power = c.power ?? 1
  const args: Arg[] = [messageArg(gaussianMv(d), gaussianMvEta(c.mean, c.covariance))]
  const params: Params = {
    likelihood: c.likelihood,
    y: c.y,
    matrix: [...c.matrix],
    ...(c.offset === undefined ? {} : { offset: [...c.offset] }),
    ...c.extra,
  }
  const density = (
    pairLikelihoods()[c.likelihood] as { density: (y: number, p: Params) => (a: number, b: number) => number }
  ).density(c.y, params)
  const brute = bruteVectorSite(
    { mean: c.mean, covariance: c.covariance, eta: args[0]?.eta as number[] },
    c.matrix,
    c.offset ?? [0, 0],
    (z) => density(z[0] as number, z[1] as number),
    power,
    c.grid,
  )
  const got = likelihoodPairRowOperator.ep?.({ mode: 'ep', to: 0, args, params, power }).message as readonly number[]
  const tolerance = c.tolerance ?? 3e-3
  for (let i = 0; i < brute.message.length; i++)
    expect(
      Math.abs((got[i] as number) - (brute.message[i] as number)),
      `message entry ${i}: ${got[i]} against ${brute.message[i]}`,
    ).toBeLessThan(tolerance * (1 + Math.abs(brute.message[i] as number)))
  const evidence = likelihoodPairRowOperator.logEvidence?.({ mode: 'ep', to: 0, args, params, power }) as number
  expect(Math.abs(evidence - (brute.logMass + massOf(args[0] as Arg)))).toBeLessThan(tolerance)
}

describe('likelihood-pair-row: the message to the vector and the evidence against a brute-force integral', () => {
  it('beta outcome with the two log shapes as the coordinates of a correlated vector', () => {
    check({
      likelihood: 'beta-shapes',
      y: 0.27,
      mean: [0.6, 1.1],
      covariance: [0.5, 0.35, 0.35, 0.45],
      matrix: [1, 0, 0, 1],
      grid: 200,
    })
  })
  it('the coordinates are linear forms of a larger vector, with an offset (rank two in three dimensions)', () => {
    check({
      likelihood: 'beta-shapes',
      y: 0.62,
      mean: [0.3, 0.9, -0.2],
      covariance: [0.6, 0.2, 0.1, 0.2, 0.5, -0.15, 0.1, -0.15, 0.4],
      matrix: [1, 0, 0.5, 0, 1, -0.5],
      offset: [0.2, -0.1],
      grid: 60,
    })
  })
  it('Student-t degrees of freedom around a location, both coordinates of the vector', () => {
    check({
      likelihood: 'student-t-df',
      y: 0.4,
      mean: [0.1, 1.2],
      covariance: [0.4, -0.1, -0.1, 0.3],
      matrix: [1, 0, 0, 1],
      extra: { scale: 0.8 },
      grid: 200,
    })
  })
  it('the power divides the message: fractional EP', () => {
    check({
      likelihood: 'beta-shapes',
      y: 0.27,
      mean: [0.6, 1.1],
      covariance: [0.5, 0.35, 0.35, 0.45],
      matrix: [1, 0, 0, 1],
      power: 0.5,
      grid: 200,
    })
  })
  it('on independent coordinates it has the tilted marginals of the scalar pair site, and adds their correlation', () => {
    const mean = [0.6, 1.1]
    const diag = [0.5, 0, 0, 0.45]
    const args: Arg[] = [messageArg(gaussianMv(2), gaussianMvEta(mean, diag))]
    const params = { likelihood: 'beta-shapes', y: 0.27, matrix: [1, 0, 0, 1] }
    const joint = likelihoodPairRowOperator.ep?.({ mode: 'ep', to: 0, args, params, power: 1 }).message as number[]
    const tiltedEta = ((args[0] as Arg).eta as number[]).map((x, i) => x + (joint[i] as number))
    const { mean: mt, covariance: ct } = gaussianMvMoments(tiltedEta, 2)
    const scalars = [0, 1].map((k) => messageArg(gaussian, gaussianEta(mean[k] as number, diag[k * 3] as number)))
    for (let k = 0; k < 2; k++) {
      const s = ruleMessage(likelihoodPairOperator, {
        mode: 'ep',
        to: k,
        args: scalars,
        params: { likelihood: 'beta-shapes', y: 0.27 },
        power: 1,
      }).message as number[]
      const cav = scalars[k]?.eta as number[]
      const precision = (cav[1] as number) + (s[1] as number)
      expect(Math.abs((mt[k] as number) - ((cav[0] as number) + (s[0] as number)) / precision)).toBeLessThan(2e-3)
      expect(Math.abs((ct[k * 3] as number) * precision - 1)).toBeLessThan(5e-3)
    }
    // The shapes are tied by the density (the data fix their ratio better than their size): the joint site has the off-diagonal.
    expect(Math.abs(ct[1] as number)).toBeGreaterThan(1e-2)
  })
  it('is in the registry and refuses a density with a kink and a matrix of the wrong size', () => {
    expect(defaultOperators()['likelihood-pair-row']).toBe(likelihoodPairRowOperator)
    expect(
      likelihoodPairRowOperator.check?.({ likelihood: 'laplace-scale', y: 0, matrix: [1, 0, 0, 1] }, [gaussianMv(2)]),
    ).toMatch(/kink/)
    expect(
      likelihoodPairRowOperator.check?.({ likelihood: 'beta-shapes', y: 0.3, matrix: [1, 0, 0] }, [gaussianMv(2)]),
    ).toMatch(/matrix/)
    expect(likelihoodPairRowOperator.noRule?.vmp).toBeDefined()
  })
})
