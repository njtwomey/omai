import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { grad } from '../autodiff/index.ts'
import { key, normal } from '../random/index.ts'
import { defaults, grid } from '../space/index.ts'
import { add, item, logSoftmax, matmul, mul, sliceAxis, sub, tensor, toArray, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const policy = { policyChosen: tensor([-4.1, -6.3, -2.2]), policyRejected: tensor([-5.0, -5.9, -3.1]) }
const reference = { referenceChosen: tensor([-4.5, -6.0, -2.5]), referenceRejected: tensor([-4.8, -6.2, -3.0]) }

describe('preference objectives', () => {
  it('passes the conformance check for every objective', () => {
    const all = [tensor([-4.1, -6.3, -2.2]), tensor([-5.0, -5.9, -3.1])]
    expectConformant(
      (c, r) => L.dpo({ policyChosen: c, policyRejected: r, ...reference }, { beta: 0.3, labelSmoothing: 0.1 }),
      all,
    )
    expectConformant((c, r) => L.ipo({ policyChosen: c, policyRejected: r, ...reference }, { tau: 0.4 }), all)
    expectConformant((c, r) => L.kto(c, r, [1, 0, 1], { beta: 0.5, referencePoint: 0.3, lambdaU: 1.5 }), all)
    expectConformant((c, r) => L.orpo(mul(c, 0.1), mul(r, 0.1), { lambda: 0.3 }), all)
    expectConformant((c, r) => L.simpo(c, r, { beta: 1.5, gamma: 0.4 }), all)
    expectConformant((c, r) => L.bradleyTerry(c, r, { margin: 0.2 }), all)
  })

  it('gives DPO the value log 2 at the reference and the Bradley-Terry loss of the implicit rewards elsewhere', () => {
    const atReference = { policyChosen: reference.referenceChosen, policyRejected: reference.referenceRejected }
    expect(item(L.dpo({ ...atReference, ...reference }))).toBeCloseTo(Math.LN2, 14)
    const beta = 0.2
    const rewards = L.implicitRewards({ ...policy, ...reference }, { beta })
    expect(item(L.dpo({ ...policy, ...reference }, { beta }))).toBeCloseTo(
      item(L.bradleyTerry(rewards.chosen, rewards.rejected)),
      14,
    )
    expect(item(rewards.accuracy)).toBeCloseTo(2 / 3, 14)
  })

  it('mixes the two directions in conservative DPO and drops the reference in the reference-free form', () => {
    const batch = { ...policy, ...reference }
    const plain = item(L.dpo(batch, { reduction: 'sum' }))
    const flipped = item(
      L.dpo(
        {
          ...batch,
          policyChosen: batch.policyRejected,
          policyRejected: batch.policyChosen,
          referenceChosen: batch.referenceRejected,
          referenceRejected: batch.referenceChosen,
        },
        { reduction: 'sum' },
      ),
    )
    expect(item(L.dpo(batch, { reduction: 'sum', labelSmoothing: 0.25 }))).toBeCloseTo(
      0.75 * plain + 0.25 * flipped,
      13,
    )
    const free = item(L.dpo(policy, { beta: 1 }))
    const manual = toArray(sub(policy.policyChosen, policy.policyRejected)) as number[]
    expect(free).toBeCloseTo(manual.reduce((a, d) => a + Math.log(1 + Math.exp(-d)), 0) / 3, 13)
  })

  it('has IPO minimal at the margin 1 / (2 tau) and KTO at the reference point', () => {
    const tau = 0.25
    const at = (margin: number): number => item(L.ipo({ policyChosen: [margin], policyRejected: [0] }, { tau }))
    expect(at(1 / (2 * tau))).toBe(0)
    expect(at(1.5)).toBeGreaterThan(0)
    expect(item(L.kto([-3, -4], [-3, -4], [1, 0], { referencePoint: 0 }))).toBeCloseTo(0.5, 14)
    // A desirable answer more likely than under the reference costs less; an undesirable one costs more.
    expect(item(L.kto([-2], [-3], [1], { referencePoint: 0, beta: 1 }))).toBeLessThan(0.5)
    expect(item(L.kto([-2], [-3], [0], { referencePoint: 0, beta: 1 }))).toBeGreaterThan(0.5)
  })

  it('has the ORPO odds-ratio term of two likelihoods and SimPO the margin shift', () => {
    expect(item(L.orpo([Math.log(0.5)], [Math.log(0.25)], { lambda: 1 }))).toBeCloseTo(Math.LN2 + Math.log(4 / 3), 14)
    expect(item(L.orpo([Math.log(0.5)], [Math.log(0.5)], { lambda: 1 }))).toBeCloseTo(2 * Math.LN2, 14)
    expect(item(L.simpo([-1], [-2], { beta: 2, gamma: 1 }))).toBeCloseTo(Math.log(1 + Math.exp(-1)), 14)
  })

  it('sums token log-probabilities under a mask', () => {
    expect(
      toArray(
        L.sequenceLogProb(
          [
            [-1, -2, -3],
            [-1, -1, -1],
          ],
          [
            [1, 1, 0],
            [1, 1, 1],
          ],
        ),
      ),
    ).toEqual([-3, -3])
  })

  it('recovers the reward direction of a Bradley-Terry model from sampled preferences', () => {
    // Rewards r(x) = w . x with w = (2, -1); the label is the sign of the reward gap plus Gaussian noise.
    const w = [2, -1]
    const n = 600
    const [a, b] = [normal(key(1), [n, 2]), normal(key(2), [n, 2])]
    const gap = toArray(matmul(sub(a, b), tensor(w))) as number[]
    const noise = toArray(normal(key(3), [n])) as number[]
    const rowsA = toArray(a) as number[][]
    const rowsB = toArray(b) as number[][]
    const firstWins = gap.map((g, i) => g + (noise[i] ?? 0) * 0.5 > 0)
    const chosen = tensor(rowsA.map((row, i) => (firstWins[i] ? row : (rowsB[i] ?? row))))
    const rejected = tensor(rowsA.map((row, i) => (firstWins[i] ? (rowsB[i] ?? row) : row)))
    const loss = (theta: Value): Value => L.bradleyTerry(matmul(chosen, theta), matmul(rejected, theta))
    let theta: Value = tensor([0, 0])
    for (let i = 0; i < 400; i++) theta = sub(theta, grad(loss)(theta))
    const [t0 = 0, t1 = 0] = toArray(theta) as number[]
    expect((t0 * 2 - t1) / (Math.hypot(t0, t1) * Math.hypot(2, -1))).toBeGreaterThan(0.98)
  })

  it('moves a tabular policy to the DPO optimum: beta times the log-ratio margin is the logit of the preference', () => {
    // Two responses, A preferred over B with probability 0.8: minimising the expected DPO loss gives
    // beta * (log pi(A)/pi(B) - log ref(A)/ref(B)) = logit(0.8).
    const beta = 0.5
    const ref = toArray(logSoftmax(tensor([0.4, -0.2]), -1)) as number[]
    const [refA = 0, refB = 0] = ref
    const loss = (theta: Value): Value => {
      const lp = logSoftmax(theta, -1)
      const [a, b] = [sliceAxis(lp, 0, 0, 1), sliceAxis(lp, 0, 1, 2)]
      const forward = L.dpo(
        { policyChosen: a, policyRejected: b, referenceChosen: [refA], referenceRejected: [refB] },
        { beta },
      )
      const backward = L.dpo(
        { policyChosen: b, policyRejected: a, referenceChosen: [refB], referenceRejected: [refA] },
        { beta },
      )
      return add(mul(forward, 0.8), mul(backward, 0.2))
    }
    let theta: Value = tensor([0.4, -0.2])
    for (let i = 0; i < 800; i++) theta = sub(theta, grad(loss)(theta))
    const [pa = 0, pb = 0] = toArray(logSoftmax(theta, -1)) as number[]
    expect(beta * (pa - pb - (refA - refB))).toBeCloseTo(Math.log(0.8 / 0.2), 6)
  })
})

describe('the preference table', () => {
  it('makes every objective a finite loss that falls when the policy prefers the chosen answer more', () => {
    for (const config of grid(L.preferenceLosses.space, 1)) {
      const loss = L.preferenceLosses.make(config)
      const reference = { referenceChosen: [-0.9, -0.8], referenceRejected: [-0.9, -0.8] }
      const weak = item(loss({ policyChosen: [-0.9, -0.8], policyRejected: [-0.8, -0.7], ...reference }))
      const strong = item(loss({ policyChosen: [-0.3, -0.2], policyRejected: [-2.5, -2.6], ...reference }))
      expect(Number.isFinite(weak), JSON.stringify(config)).toBe(true)
      expect(strong, JSON.stringify(config)).toBeLessThan(weak)
    }
    expect(defaults(L.preferenceLosses.space)).toMatchObject({ kind: 'dpo' })
    expect(L.preferenceLosses.where((p) => p.needsReference)).toEqual(['dpo', 'ipo', 'kto'])
    expect(L.preferenceLosses.where((p) => p.lengthNormalised)).toEqual(['orpo', 'simpo'])
    expect(L.preferenceLosses.where((p) => p.rewardModel)).toEqual(['bradleyTerry'])
  })

  it('binds the knobs of DPO', () => {
    const batch = { ...policy, ...reference }
    const dpoLoss = L.preferenceLosses.make({ kind: 'dpo', beta: 0.4, labelSmoothing: 0.1 })
    expect(item(dpoLoss(batch))).toBeCloseTo(item(L.dpo(batch, { beta: 0.4, labelSmoothing: 0.1 })), 14)
  })
})
