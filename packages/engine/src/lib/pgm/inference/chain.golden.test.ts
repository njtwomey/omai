import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { key } from '../../random/index.ts'
import { type Tensor, tensor, toArray } from '../../tensor/index.ts'
import { messages, recorder, tracePolicy } from '../../trace/index.ts'
import * as P from '../index.ts'

const arr = (x: unknown): unknown => toArray(decode(x) as Tensor)

function modelOf(c: { fn: string; inputs: readonly unknown[] }): {
  hmm: P.HiddenMarkovModel
  x: (number | number[])[]
} {
  const [start, trans, third, fourth, fifth] = c.inputs
  const initial = arr(start) as number[]
  const transition = arr(trans) as number[][]
  const states = initial.map((_, k) => `s${k}`)
  if (c.fn === 'categorical') {
    const probabilities = arr(third) as number[][]
    const symbols = (probabilities[0] ?? []).map((_, m) => `o${m}`)
    const hmm = P.hiddenMarkovModel({
      states,
      initial,
      transition,
      emission: { kind: 'categorical', symbols, probabilities },
    })
    return { hmm, x: arr(fourth) as number[] }
  }
  const hmm = P.hiddenMarkovModel({
    states,
    initial,
    transition,
    emission: { kind: 'gaussian', means: arr(third) as number[][], covariances: arr(fourth) as number[][][] },
  })
  return { hmm, x: arr(fifth) as number[][] }
}

describe('chain inference against hmmlearn', () => {
  for (const c of loadFixture('pgm/inference/chain').cases) {
    it(c.name, () => {
      const { hmm, x } = modelOf(c)
      const chain = P.hmmChain(hmm, x)
      for (const method of ['scaled', 'log'] as const) {
        const fb = P.forwardBackward(chain, { method })
        expectMatches(tensor(fb.posteriors), c.output, c.rtol, c.atol, `${c.name} ${method}`)
        expectMatches(fb.logLikelihood, { dtype: 'float64', shape: [], data: [c.params['score']] }, 1e-10, 0, c.name)
        expectMatches(tensor(fb.forward), c.params['filtered'], 1e-9, 1e-12, `${c.name} filtered`)
        // Pairwise posteriors marginalise to the single-step posteriors.
        for (const [t, xi] of fb.pairwise.entries()) {
          const row = xi.map((r) => r.reduce((a, b) => a + b, 0))
          for (const [k, v] of row.entries()) expect(v).toBeCloseTo(fb.posteriors[t]?.[k] ?? 0, 12)
        }
      }
      const v = P.viterbi(chain)
      expect(v.path).toEqual(c.params['viterbi'])
      expect(v.logProbability).toBeCloseTo(c.params['viterbiLogProb'] as number, 9)
      expect(P.posteriorDecode(chain)).toEqual(c.params['map'])
      const f = P.forwardFilter(chain)
      expect(f.logLikelihood).toBeCloseTo(c.params['score'] as number, 9)
      expectMatches(tensor(f.filtered), c.params['filtered'], 1e-9, 1e-12, `${c.name} forwardFilter`)
    })
  }
})

describe('chain algorithms', () => {
  const hmm = P.hiddenMarkovModel({
    states: ['fair', 'loaded'],
    initial: [0.5, 0.5],
    transition: [
      [0.95, 0.05],
      [0.1, 0.9],
    ],
    emission: {
      kind: 'categorical',
      symbols: ['1', '2', '3', '4', '5', '6'],
      probabilities: [[1, 1, 1, 1, 1, 1].map((v) => v / 6), [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]],
    },
  })
  const rolls = ['6', '6', '1', '6', '6', '6', '2', '3', '4', '1', '5', '2']

  it('backward sampling draws paths from the smoothing posterior', () => {
    const chain = P.hmmChain(hmm, rolls)
    const exact = P.forwardBackward(chain).posteriors
    const n = 4000
    const paths = P.backwardSample(chain, key(3), n)
    for (const t of [0, 5, 11]) {
      const freq = paths.filter((p) => p[t] === 1).length / n
      const p = exact[t]?.[1] ?? 0
      expect(Math.abs(freq - p)).toBeLessThan(4 * Math.sqrt((p * (1 - p)) / n) + 1e-3)
    }
  })

  it('agrees with the unrolled network and the DBN filter', () => {
    const chain = P.hmmChain(hmm, rolls.slice(0, 5))
    const fb = P.forwardBackward(chain)
    const bn = P.unroll(P.hmmToDbn(hmm), 5)
    const evidence = Object.fromEntries(rolls.slice(0, 5).map((r, t) => [`x_${t}`, r]))
    const ve = P.variableElimination(bn, ['z_2'], { evidence })
    expect(P.factorValues(ve.posterior)[1]).toBeCloseTo(fb.posteriors[2]?.[1] ?? 0, 12)
    expect(ve.logEvidence).toBeCloseTo(fb.logLikelihood, 12)
    const filtered = P.dbnFilter(
      P.hmmToDbn(hmm),
      rolls.slice(0, 5).map((r) => ({ x: r })),
    )
    for (const [t, m] of filtered.marginals.entries()) {
      expect(P.factorValues(m['z'] as P.Factor)[1]).toBeCloseTo(fb.forward[t]?.[1] ?? 0, 12)
    }
    expect(filtered.logLikelihood).toBeCloseTo(fb.logLikelihood, 12)
  })

  it('handles inhomogeneous transitions, traces messages and rejects impossible data', () => {
    const chain: P.ChainModel = {
      logInitial: [0, -Infinity],
      logTransition: [
        [
          [-Infinity, 0],
          [0, -Infinity],
        ],
        [
          [0, -Infinity],
          [-Infinity, 0],
        ],
      ],
      logEmission: [
        [0, 0],
        [0, 0],
        [0, 0],
      ],
    }
    expect(P.viterbi(chain).path).toEqual([0, 1, 1])
    const rec = recorder()
    P.forwardBackward(chain, { tracer: rec, policy: tracePolicy({ messages: true }) })
    expect(messages(rec.run(), 'forward')).toHaveLength(3)
    expect(() =>
      P.forwardBackward({
        ...chain,
        logEmission: [
          [0, 0],
          [0, -Infinity],
          [0, 0],
        ],
      }),
    ).toThrow(/zero probability/)
  })

  it('samples sequences with the model statistics', () => {
    const s = P.hmmSample(hmm, key(5), 3000)
    const loaded = s.states.filter((z) => z === 1).length / 3000
    expect(Math.abs(loaded - 1 / 3)).toBeLessThan(0.08)
  })
})
