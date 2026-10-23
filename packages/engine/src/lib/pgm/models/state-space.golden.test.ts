import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { key } from '../../random/index.ts'
import { type Tensor, tensor, toArray } from '../../tensor/index.ts'
import * as P from '../index.ts'

const arr = (x: unknown): number[][] => {
  const v = toArray(decode(x) as Tensor) as unknown[]
  return (Array.isArray(v[0]) ? v : [v]) as number[][]
}

describe('linear dynamical systems against numpy', () => {
  for (const c of loadFixture('pgm/models/state_space').cases) {
    it(c.name, () => {
      const [a, cm, q, r, m0, p0, y] = c.inputs.map(arr)
      const lds = P.linearDynamicalSystem({
        transition: a ?? [],
        observation: cm ?? [],
        transitionCovariance: q ?? [],
        observationCovariance: r ?? [],
        initialMean: m0?.[0] ?? [],
        initialCovariance: p0 ?? [],
      })
      const post = P.ldsPosterior(lds, y ?? [])
      expectMatches(
        tensor([...post.means.flat(), ...post.covariances.flat(2), post.logLikelihood]),
        c.output,
        c.rtol,
        c.atol,
        c.name,
      )
    })
  }
})

describe('state-space models', () => {
  it('samples linear dynamical systems with the model moments', () => {
    const lds = P.linearDynamicalSystem({
      transition: [[0.9]],
      observation: [[2]],
      transitionCovariance: [[0.19]],
      observationCovariance: [[0.5]],
      initialMean: [0],
      initialCovariance: [[1]],
    })
    let sum = 0
    let count = 0
    for (let s = 0; s < 400; s++) {
      const { observations } = P.ldsSample(lds, key(s), 10)
      for (const o of observations) {
        sum += (o[0] ?? 0) ** 2
        count++
      }
    }
    // Stationary: Var(x) = 1, so Var(y) = 4 + 0.5.
    expect(Math.abs(sum / count - 4.5)).toBeLessThan(0.3)
  })

  it('Gaussian HMMs score sequences the same way as their per-state densities', () => {
    const hmm = P.hiddenMarkovModel({
      states: ['a', 'b'],
      initial: [1, 0],
      transition: [
        [1, 0],
        [0, 1],
      ],
      emission: { kind: 'gaussian', means: [[0], [5]], covariances: [[[1]], [[1]]] },
    })
    const ll = P.forwardBackward(P.hmmChain(hmm, [[0.5], [-0.2]])).logLikelihood
    expect(ll).toBeCloseTo(-Math.log(2 * Math.PI) - (0.25 + 0.04) / 2, 12)
    const s = P.hmmSample(hmm, key(0), 50)
    expect(s.states.every((z) => z === 0)).toBe(true)
    expect(() => P.hmmChain(hmm, ['x'])).toThrow(/vectors/)
    expect(() => P.hmmToDbn(hmm)).toThrow(/categorical/)
    expect(() =>
      P.hiddenMarkovModel({ states: ['a'], initial: [0.5], transition: [[1]], emission: hmm.emission }),
    ).toThrow(/initial/)
  })
})
