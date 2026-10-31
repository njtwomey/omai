import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { crossEntropy } from '../losses/index.ts'
import { Linear, type LogitsFunction, mlp, Sequential, Tanh, trainState, trainStep } from '../nn/index.ts'
import { adam, adamw } from '../optim/index.ts'
import { key, normal } from '../random/index.ts'
import { matmul, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  analyticGradientLeak,
  attributeInference,
  type ClassifierTrainer,
  confidenceLogit,
  exampleLoss,
  extractionAttack,
  gradientLeakageAlgorithm,
  likelihoodRatioAttack,
  lossThresholdAttack,
  membershipReport,
  modelInversionAlgorithm,
  shadowModelAttack,
} from './privacy-attacks.ts'

const flat = (v: unknown): number[] => toFlat(v as Value)
const rows = (t: Tensor, from: number, to: number): Tensor =>
  tensor(
    Array.from({ length: to - from }, (_, i) =>
      flat(t).slice((from + i) * (t.shape[1] ?? 1), (from + i + 1) * (t.shape[1] ?? 1)),
    ),
  )
const part = (t: Tensor, from: number, to: number): Tensor => tensor(flat(t).slice(from, to))

/** Random features with random labels: nothing generalises, so any gap between train and test loss is memorisation. */
function noise(n: number, seed: number): { x: Tensor; y: Tensor } {
  const x = normal(key(seed), [n, 10]) as Tensor
  const y = tensor(
    Array.from({ length: n }, (_, i) => ((flat(normal(key(seed + 1), [n]))[i] as number) > 0 ? 1 : 0)),
    { dtype: 'int32' },
  ) as Tensor
  return { x, y }
}

const overfit: ClassifierTrainer = (x, y, k): LogitsFunction => {
  const model = mlp([10, 64, 2], { key: k })
  const opt = adam({ learningRate: 0.02 })
  let s = trainState(model, opt, k)
  for (let i = 0; i < 300; i++) s = trainStep(model, s, x, (z) => crossEntropy(z, y), opt)
  return (v) => model.apply(s.params, v, { train: false })
}

const regularised: ClassifierTrainer = (x, y, k): LogitsFunction => {
  const model = new Linear(10, 2, { key: k })
  const opt = adamw({ learningRate: 0.05, weightDecay: 5 })
  let s = trainState(model, opt, k)
  for (let i = 0; i < 100; i++) s = trainStep(model, s, x, (z) => crossEntropy(z, y), opt)
  return (v) => model.apply(s.params, v)
}

describe('membership inference', () => {
  const data = noise(120, 10)
  const train = { x: rows(data.x, 0, 60), y: part(data.y, 0, 60) }
  const held = { x: rows(data.x, 60, 120), y: part(data.y, 60, 120) }

  it('the report is exact on separable scores and chance on identical ones', () => {
    const perfect = membershipReport([3, 4, 5], [0, 1, 2])
    expect(perfect).toMatchObject({ auc: 1, accuracy: 1, advantage: 1, tprAtFpr: 1 })
    const same = membershipReport([1, 1, 1, 1], [1, 1, 1, 1])
    expect(same.auc).toBe(0.5)
    expect(same.advantage).toBe(0)
  })

  it('the loss threshold finds the overfit model’s training set and not the regularised one’s', () => {
    const over = overfit(train.x, train.y, key(1))
    const a = lossThresholdAttack(exampleLoss(over, train.x, train.y), exampleLoss(over, held.x, held.y))
    expect(a.auc).toBeGreaterThan(0.8)
    expect(a.accuracy).toBeGreaterThan(0.72)
    const reg = regularised(train.x, train.y, key(1))
    const b = lossThresholdAttack(exampleLoss(reg, train.x, train.y), exampleLoss(reg, held.x, held.y))
    expect(Math.abs(b.auc - 0.5)).toBeLessThan(0.15)
  })

  it('shadow models calibrate a threshold that transfers to the target', () => {
    const pool = noise(120, 30)
    const over = overfit(train.x, train.y, key(2))
    const a = shadowModelAttack(over, train, held, { pool, trainer: overfit, shadows: 4, key: key(5) })
    expect(a.calibratedAccuracy).toBeGreaterThan(0.75)
    expect(a.shadowLosses.members).toBeLessThan(a.shadowLosses.nonMembers)
    const reg = regularised(train.x, train.y, key(2))
    const b = shadowModelAttack(reg, train, held, { pool, trainer: regularised, shadows: 4, key: key(5) })
    expect(Math.abs(b.calibratedAccuracy - 0.5)).toBeLessThan(0.15)
  })

  it('LiRA beats chance on the overfit model, is chance on the regularised one, and has teeth at low FPR', () => {
    // Pool: 40 queries (20 members of the target, 20 not) then 60 more rows; the target trains on its members and the extras.
    const pool = noise(100, 50)
    const memberIdx = [...Array.from({ length: 20 }, (_, i) => i), ...Array.from({ length: 60 }, (_, i) => 40 + i)]
    const targetData = {
      x: tensor(memberIdx.map((i) => flat(rows(pool.x, i, i + 1)))),
      y: tensor(
        memberIdx.map((i) => flat(pool.y)[i] as number),
        { dtype: 'int32' },
      ) as Tensor,
    }
    const isMember = Array.from({ length: 40 }, (_, i) => i < 20)
    const over = overfit(targetData.x, targetData.y, key(3))
    const a = likelihoodRatioAttack(over, 40, isMember, {
      pool,
      trainer: overfit,
      shadows: 12,
      key: key(9),
      fprLevel: 0.05,
    })
    expect(a.auc).toBeGreaterThan(0.8)
    expect(a.scores).toHaveLength(40)
    const reg = regularised(targetData.x, targetData.y, key(3))
    const b = likelihoodRatioAttack(reg, 40, isMember, { pool, trainer: regularised, shadows: 12, key: key(9) })
    expect(Math.abs(b.auc - 0.5)).toBeLessThan(0.2)
  })

  it('the confidence logit is log p/(1-p)', () => {
    expect(confidenceLogit((x) => x, [[0, Math.log(3)]], [1])[0]).toBeCloseTo(Math.log(3), 12)
    expect(confidenceLogit((x) => x, [[0, 0, 0]], [2])[0]).toBeCloseTo(Math.log(0.5), 12)
  })
})

describe('attribute inference', () => {
  it('recovers a hidden attribute the model leans on, and only matches the prior when it does not', () => {
    // Label = hidden attribute (column 0) with noise on the other column.
    const s = Array.from({ length: 200 }, (_, i) => (i % 5 === 0 ? 1 : 0))
    const x = tensor(s.map((v, i) => [v, Math.sin(i)]))
    const y = tensor(s, { dtype: 'int32' })
    const dependent: LogitsFunction = (v) =>
      matmul(
        v,
        tensor([
          [-4, 4],
          [0, 0],
        ]),
      )
    const a = attributeInference(dependent, x, y, 0, [0, 1], { truth: s })
    expect(a.accuracy).toBe(1)
    expect(a.baseline).toBeCloseTo(0.8, 12)
    const independent: LogitsFunction = (v) =>
      matmul(
        v,
        tensor([
          [0, 0],
          [1, -1],
        ]),
      )
    const b = attributeInference(independent, x, y, 0, [0, 1], { truth: s, prior: [0.8, 0.2] })
    expect(b.accuracy).toBeCloseTo(b.baseline, 12)
    expect(a.accuracy).toBeGreaterThan(b.accuracy)
  })
})

describe('model inversion', () => {
  it('ascends the class’s confidence and ends on an input the model assigns to it', () => {
    const model: LogitsFunction = (v) =>
      matmul(
        v,
        tensor([
          [2, -1, -1],
          [-1, 2, -1],
          [-1, -1, 2],
        ]),
      )
    const r = runAlgorithm(modelInversionAlgorithm(model, 2, [3], { steps: 40, learningRate: 0.2, l2: 0.05 }), {
      maxSteps: 40,
    })
    expect(r.state.confidence).toBeGreaterThan(0.9)
    const x = flat(r.state.x)
    expect(x[2]).toBeGreaterThan(x[0] as number)
    expect(x[2]).toBeGreaterThan(x[1] as number)
    expect(r.steps).toBe(40)
  })
})

describe('gradient leakage', () => {
  const net = new Sequential(new Linear(4, 6, { key: key(0) }), new Tanh(), new Linear(6, 3, { key: key(1) }))
  const truth = tensor([0.7, -1.2, 0.4, 2.0])
  const label = 1
  const leaked = grad((p: unknown) =>
    crossEntropy(net.apply(p as never, tensor([flat(truth)])), tensor([label], { dtype: 'int32' })),
  )(net.params) as Record<string, Record<string, Tensor>>

  it('the first layer leaks the input exactly', () => {
    const x = analyticGradientLeak(leaked['0']?.['weight'] as Tensor, leaked['0']?.['bias'] as Tensor)
    flat(x).forEach((v, i) => {
      expect(v).toBeCloseTo(flat(truth)[i] as number, 10)
    })
  })

  it('gradient matching reconstructs the input of a tiny MLP', () => {
    const alg = gradientLeakageAlgorithm((p, v) => net.apply(p, v), net.params, leaked as never, label, [4], {
      steps: 400,
      learningRate: 0.1,
    })
    const r = runAlgorithm(alg, { maxSteps: 400, seed: 3 })
    const err = Math.max(...flat(r.state.x).map((v, i) => Math.abs(v - (flat(truth)[i] as number))))
    expect(err).toBeLessThan(0.05)
    expect(r.state.match).toBeLessThan(1e-4)
  })
})

describe('extraction', () => {
  it('a surrogate fitted to a victim’s answers agrees with it on fresh inputs', () => {
    const victimW = tensor([
      [1.5, -1],
      [-0.5, 2],
      [1, 1],
    ])
    const victim: LogitsFunction = (v) => matmul(v, victimW)
    const fit = (x: Tensor, labels: Tensor): LogitsFunction => {
      const model = new Linear(3, 2, { key: key(4) })
      const opt = adam({ learningRate: 0.1 })
      let s = trainState(model, opt, key(4))
      for (let i = 0; i < 200; i++) s = trainStep(model, s, x, (z) => crossEntropy(z, labels), opt)
      return (v) => model.apply(s.params, v)
    }
    const r = extractionAttack(victim, normal(key(5), [300, 3]) as Tensor, fit, normal(key(6), [300, 3]) as Tensor)
    expect(r.agreement).toBeGreaterThan(0.9)
  })
})
