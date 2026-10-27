import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { hiddenMarkovModel, hmmSample } from '../models/state-space.ts'
import { baumWelch, type HmmParameters, hmmDecode, hmmFromParameters, initialHmmParameters } from './baum-welch.ts'

const truth = hiddenMarkovModel({
  states: ['a', 'b'],
  initial: [0.5, 0.5],
  transition: [
    [0.9, 0.1],
    [0.2, 0.8],
  ],
  emission: {
    kind: 'categorical',
    symbols: ['0', '1', '2'],
    probabilities: [
      [0.7, 0.2, 0.1],
      [0.1, 0.2, 0.7],
    ],
  },
})

const trueParameters: HmmParameters = {
  kind: 'categorical',
  start: [0.5, 0.5],
  transition: [
    [0.9, 0.1],
    [0.2, 0.8],
  ],
  emission: [
    [0.7, 0.2, 0.1],
    [0.1, 0.2, 0.7],
  ],
}

const nonDecreasing = (xs: readonly number[]): boolean => xs.every((v, i) => i === 0 || v >= (xs[i - 1] ?? 0) - 1e-9)

describe('Baum-Welch', () => {
  const sample = hmmSample(truth, key(0), 1500)
  const symbols = sample.observations as number[]

  it('never lowers the log likelihood and recovers the chain up to a permutation', () => {
    const fit = baumWelch([symbols], { states: 2, emission: 'categorical', key: key(1), maxIter: 200, tol: 1e-6 })
    expect(fit.logLikelihoods.length).toBeGreaterThan(5)
    expect(nonDecreasing(fit.logLikelihoods)).toBe(true)
    const p = fit.params
    if (p.kind !== 'categorical') throw new Error('expected categorical parameters')
    // The state that favours symbol 2 is the truth's second state.
    const [a, b] = (p.emission[0]?.[2] ?? 0) > (p.emission[1]?.[2] ?? 0) ? [1, 0] : [0, 1]
    expect(p.transition[a]?.[a]).toBeCloseTo(0.9, 1)
    expect(p.transition[b]?.[b]).toBeCloseTo(0.8, 1)
    expect(Math.abs((p.emission[a]?.[0] ?? 0) - 0.7)).toBeLessThan(0.1)
    expect(Math.abs((p.emission[b]?.[2] ?? 0) - 0.7)).toBeLessThan(0.1)
    // The fitted model decodes about as well as the true one, in whichever order it numbered the states.
    const accuracy = (path: readonly number[], flip: boolean): number =>
      path.filter((s, t) => s === (flip ? 1 - (sample.states[t] ?? 0) : sample.states[t])).length / path.length
    const reference = accuracy(hmmDecode(trueParameters, symbols).path, false)
    expect(accuracy(hmmDecode(p, symbols).path, a === 1)).toBeGreaterThan(reference - 0.06)
  })

  it('fits Gaussian emissions to two regimes', () => {
    const regimes = hiddenMarkovModel({
      states: ['low', 'high'],
      initial: [0.5, 0.5],
      transition: [
        [0.95, 0.05],
        [0.05, 0.95],
      ],
      emission: { kind: 'gaussian', means: [[0], [4]], covariances: [[[1]], [[1]]] },
    })
    const x = hmmSample(regimes, key(2), 600).observations as number[][]
    const fit = baumWelch([x], { states: 2, emission: 'gaussian', key: key(3), maxIter: 100, covariancePrior: 0 })
    if (fit.params.kind !== 'gaussian') throw new Error('expected Gaussian parameters')
    expect(nonDecreasing(fit.logLikelihoods)).toBe(true)
    const means = fit.params.means.map((m) => m[0] ?? 0).sort((u, v) => u - v)
    expect(means[0]).toBeCloseTo(0, 0)
    expect(means[1]).toBeCloseTo(4, 0)
  })

  it('starts from valid parameters, sticky on request', () => {
    const p = initialHmmParameters([symbols], { states: 3, emission: 'categorical', stay: 0.8 }, key(0))
    expect(p.transition[0]).toEqual([0.8, expect.closeTo(0.1, 12), expect.closeTo(0.1, 12)])
    expect(hmmFromParameters(p).states).toEqual(['s0', 's1', 's2'])
    expect(() => initialHmmParameters([[0.5]], { states: 2, emission: 'categorical' }, key(0))).toThrow(/symbols/)
  })
})
