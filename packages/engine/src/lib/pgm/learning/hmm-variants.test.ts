import { describe, expect, it } from 'vitest'
import { key, uniformStream } from '../../random/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { baumWelchAlgorithm, type HmmParameters, hmmDecode } from './baum-welch.ts'
import { isLeftRight, leftRightParameters, viterbiTrainingAlgorithm } from './hmm-variants.ts'

/** A three-state left-right model: each state emits mostly one of three symbols. */
const truth: Extract<HmmParameters, { kind: 'categorical' }> = {
  kind: 'categorical',
  start: [1, 0, 0],
  transition: [
    [0.8, 0.2, 0],
    [0, 0.7, 0.3],
    [0, 0, 1],
  ],
  emission: [
    [0.85, 0.1, 0.05],
    [0.1, 0.8, 0.1],
    [0.05, 0.15, 0.8],
  ],
}

function sample(seed: number, length: number): { x: number[]; z: number[] } {
  const u = uniformStream(key(seed))
  const draw = (p: readonly number[]): number => {
    const r = u()
    let acc = 0
    for (let i = 0; i < p.length; i++) {
      acc += p[i] ?? 0
      if (r < acc) return i
    }
    return p.length - 1
  }
  const z = [0]
  for (let t = 1; t < length; t++) z.push(draw(truth.transition[z[t - 1] ?? 0] ?? []))
  return { x: z.map((k) => draw(truth.emission[k] ?? [])), z }
}

const drawn = Array.from({ length: 40 }, (_, i) => sample(10 + i, 30))
const sequences = drawn.map((d) => d.x)
/** The stay frequency of a state along the true paths: what a maximum-likelihood fit that saw the states would find. */
const empiricalEmission = (k: number, symbol: number): number => {
  let hit = 0
  let all = 0
  for (const { x, z } of drawn) {
    for (const [t, state] of z.entries()) {
      if (state !== k) continue
      all++
      if (x[t] === symbol) hit++
    }
  }
  return hit / all
}
const empiricalStay = (k: number): number => {
  let stay = 0
  let leave = 0
  for (const { z } of drawn) for (let t = 1; t < z.length; t++) if (z[t - 1] === k) z[t] === k ? stay++ : leave++
  return stay / (stay + leave)
}
const start: HmmParameters = leftRightParameters({
  kind: 'categorical',
  start: [1 / 3, 1 / 3, 1 / 3],
  transition: Array.from({ length: 3 }, () => [1 / 3, 1 / 3, 1 / 3]),
  emission: [
    [0.5, 0.3, 0.2],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
  ],
})

describe('left-right topology', () => {
  it('masks a full model to a banded upper-triangular chain starting in state 0', () => {
    expect(start.start).toEqual([1, 0, 0])
    expect(isLeftRight(start.transition)).toBe(true)
    expect(start.transition[2]).toEqual([0, 0, 1])
    const jump2 = leftRightParameters(
      {
        ...start,
        transition: [
          [1 / 3, 1 / 3, 1 / 3],
          [1 / 3, 1 / 3, 1 / 3],
          [1 / 3, 1 / 3, 1 / 3],
        ],
      },
      2,
    )
    expect(jump2.transition[0]?.[2] ?? 0).toBeGreaterThan(0)
  })
  it('is preserved by Baum-Welch, which recovers the planted transitions', () => {
    const run = runAlgorithm(baumWelchAlgorithm(sequences, start, { tol: 1e-6 }), { maxSteps: 200 })
    const fit = run.state.params
    expect(isLeftRight(fit.transition)).toBe(true)
    // Within sampling error of the frequencies along the true paths (the planted 0.8 and 0.7 are only the expectation).
    expect(Math.abs((fit.transition[0]?.[0] ?? 0) - empiricalStay(0))).toBeLessThan(0.05)
    expect(Math.abs((fit.transition[1]?.[1] ?? 0) - empiricalStay(1))).toBeLessThan(0.07)
    expect(fit.transition[2]?.[2]).toBe(1)
    // The decoded paths visit the states in order.
    const path = hmmDecode(fit, sequences[0] ?? []).path
    expect(path.every((s, t) => t === 0 || s >= (path[t - 1] ?? 0))).toBe(true)
  })
})

describe('Viterbi training', () => {
  it('never decreases the joint score of data and best paths and stops when the paths settle', () => {
    const alg = viterbiTrainingAlgorithm(sequences, start)
    let state = alg.init(key(0))
    let last = Number.NEGATIVE_INFINITY
    let converged = false
    for (let i = 0; i < 40 && !converged; i++) {
      state = alg.step(state, undefined as never)
      expect(state.score as number).toBeGreaterThanOrEqual(last - 1e-9)
      last = state.score as number
      converged = alg.done?.(state, i) === 'converged'
    }
    expect(converged).toBe(true)
    expect(isLeftRight(state.params.transition)).toBe(true)
  })
  it('lands near Baum-Welch and the planted model for well-separated emissions', () => {
    const run = runAlgorithm(viterbiTrainingAlgorithm(sequences, start), { maxSteps: 60 })
    const p = run.state.params
    if (p.kind !== 'categorical') throw new Error('categorical expected')
    // Hard assignment pushes the ambiguous frames at state boundaries into one state, a known bias, so the tolerance is wider than for Baum-Welch.
    expect(Math.abs((p.transition[0]?.[0] ?? 0) - empiricalStay(0))).toBeLessThan(0.1)
    expect(Math.abs((p.emission[1]?.[1] ?? 0) - empiricalEmission(1, 1))).toBeLessThan(0.15)
    expect(Math.abs((p.emission[2]?.[2] ?? 0) - empiricalEmission(2, 2))).toBeLessThan(0.15)
  })
  it('fits Gaussian emissions and serialises its state', () => {
    const g: HmmParameters = {
      kind: 'gaussian',
      start: [1, 0],
      transition: [
        [0.5, 0.5],
        [0, 1],
      ],
      means: [[-1], [1]],
      covariances: [[[1]], [[1]]],
    }
    const seq = [
      [
        ...Array.from({ length: 20 }, (_, i) => [-2 + 0.05 * (i % 3)]),
        ...Array.from({ length: 20 }, (_, i) => [3 + 0.05 * (i % 3)]),
      ],
    ]
    const alg = viterbiTrainingAlgorithm(seq, g)
    const run = runAlgorithm(alg, { maxSteps: 20 })
    const p = run.state.params
    if (p.kind !== 'gaussian') throw new Error('gaussian expected')
    expect(p.means[0]?.[0]).toBeCloseTo(-1.95, 1)
    expect(p.means[1]?.[0]).toBeCloseTo(3.05, 1)
    expect(alg.deserialise?.(alg.serialise?.(run.state) as never)).toEqual(run.state)
  })
})
