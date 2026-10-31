import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { crossEntropy } from '../losses/index.ts'
import { adam } from '../optim/index.ts'
import { key, normal } from '../random/index.ts'
import { argmax, item, matmul, sum, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as NN from './index.ts'

const flat = (v: unknown): number[] => toFlat(v as Value)

/** Two classes: a noisy informative feature, and a clean tiny feature that separates them perfectly. */
function planted(n: number, seed: number): { x: Tensor; y: Tensor } {
  const noise = flat(normal(key(seed), [n]))
  const labels = Array.from({ length: n }, (_, i) => i % 2)
  const rows = labels.map((l, i) => [(l === 1 ? 1 : -1) + 0.8 * (noise[i] ?? 0), l === 1 ? 0.1 : -0.1])
  return { x: tensor(rows), y: tensor(labels, { dtype: 'int32' }) as Tensor }
}

function train(
  model: NN.Module,
  x: Tensor,
  y: Tensor,
  steps: number,
  adversarial?: Parameters<typeof NN.adversarialTrainStep>[5],
): void {
  const opt = adam({ learningRate: 0.05 })
  let state = NN.trainState(model, opt, key(5))
  for (let i = 0; i < steps; i++) {
    state =
      adversarial === undefined
        ? NN.trainStep(model, state, x, (z) => crossEntropy(z, y), opt)
        : NN.adversarialTrainStep(model, state, x, y, opt, adversarial).state
  }
  model.setParams(state.params)
}

const logitsOf =
  (model: NN.Module) =>
  (x: Value): Value =>
    model.apply(model.params, x, { train: false })

describe('attack primitives', () => {
  const w = [
    [1.5, -1, 0.2],
    [-0.5, 2, -0.3],
    [0.3, 0.3, 1],
  ]
  const linear: NN.LogitsFunction = (x) => matmul(x, tensor(w))
  const x = tensor([
    [0.4, -0.2, 0.7],
    [1, 0.5, -0.3],
  ])
  const y = tensor([0, 1], { dtype: 'int32' })

  it('margin loss is the best wrong logit minus the label logit', () => {
    expect(
      flat(
        NN.marginLoss(
          [
            [2, 1, 0],
            [0, 3, 1],
          ],
          [0, 0],
        ),
      ),
    ).toEqual([-1, 3])
    expect(
      flat(
        NN.marginLoss(
          [
            [2, 1, 0],
            [0, 3, 1],
          ],
          [0, 0],
          0.5,
        ),
      ),
    ).toEqual([-1, 0.5])
  })

  it('FGSM is x + eps sign(grad) and raises the loss by eps |grad|_1 to first order', () => {
    const net = NN.mlp([3, 6, 3], { key: key(1) })
    const f = logitsOf(net)
    const loss = (v: Value): Value => sum(crossEntropy(f(v), y, { reduction: 'none' }))
    const g = flat(grad(loss)(x))
    const eps = 1e-6
    const at = (d: readonly number[]): Tensor =>
      tensor([
        flat(x)
          .slice(0, 3)
          .map((v, i) => v + (d[i] as number)),
        flat(x)
          .slice(3)
          .map((v, i) => v + (d[3 + i] as number)),
      ])
    const adv = flat(NN.fgsm(f, x, y, { epsilon: eps }))
    flat(x).forEach((v, i) => {
      expect(adv[i]).toBeCloseTo(v + eps * Math.sign(g[i] as number), 14)
    })
    const rise = item(loss(tensor([adv.slice(0, 3), adv.slice(3)]))) - item(loss(x))
    const dual = g.reduce((a, b) => a + Math.abs(b), 0)
    expect(rise / (eps * dual)).toBeCloseTo(1, 3)
    // Against a central finite difference of the loss along the sign direction.
    const s = g.map(Math.sign)
    const h = 1e-6
    const plus = loss(at(s.map((v) => h * v)))
    const minus = loss(at(s.map((v) => -h * v)))
    expect((item(plus) - item(minus)) / (2 * h)).toBeCloseTo(dual, 5)
  })

  it('the l2 step moves exactly eps along the gradient and the range clips', () => {
    const adv = NN.fgsm(linear, x, y, { epsilon: 0.3, norm: 'l2' })
    const d = flat(adv).map((v, i) => v - (flat(x)[i] as number))
    expect(Math.hypot(...d.slice(0, 3))).toBeCloseTo(0.3, 12)
    expect(Math.hypot(...d.slice(3))).toBeCloseTo(0.3, 12)
    const clipped = flat(NN.fgsm(linear, x, y, { epsilon: 5, range: [-1, 1] }))
    expect(clipped.every((v) => v >= -1 && v <= 1)).toBe(true)
  })

  it('PGD stays in the ball for both norms, with and without a range', () => {
    for (const norm of ['linf', 'l2'] as const) {
      const adv = flat(NN.pgd(linear, x, y, { epsilon: 0.2, norm, steps: 7, key: key(3), range: [-2, 2] }))
      for (let r = 0; r < 2; r++) {
        const d = adv.slice(3 * r, 3 * r + 3).map((v, i) => v - (flat(x)[3 * r + i] as number))
        const size = norm === 'linf' ? Math.max(...d.map(Math.abs)) : Math.hypot(...d)
        expect(size).toBeLessThanOrEqual(0.2 + 1e-12)
      }
    }
  })

  it('targeted PGD drives inputs to the target class and untargeted to anything else', () => {
    const target = tensor([2, 2], { dtype: 'int32' })
    const adv = NN.pgd(linear, x, target, { epsilon: 3, steps: 30, targeted: true, key: key(0) })
    expect(flat(argmax(linear(adv), -1))).toEqual([2, 2])
    const away = NN.pgd(linear, x, y, { epsilon: 3, steps: 30, key: key(0), loss: 'margin' })
    flat(argmax(linear(away), -1)).forEach((p, i) => {
      expect(p).not.toBe(flat(y)[i])
    })
  })

  it('a random start needs a key', () => {
    expect(() => NN.pgd(linear, x, y, { epsilon: 0.1 })).toThrow(/key/)
    expect(NN.pgd(linear, x, y, { epsilon: 0.1, randomStart: false }).shape).toEqual([2, 3])
  })

  it('the stepped algorithm with no random start reproduces pgd and logs its paths', () => {
    const options = { epsilon: 0.4, steps: 6, randomStart: false } as const
    const alg = NN.pgdAlgorithm(linear, x, y, options)
    const r = runAlgorithm(alg, { maxSteps: 20, seed: 0 })
    expect(r.steps).toBe(6)
    const direct = flat(NN.pgd(linear, x, y, options))
    flat(r.state.x).forEach((v, i) => {
      expect(v).toBeCloseTo(direct[i] as number, 12)
    })
    expect(r.state.distance).toBeLessThanOrEqual(0.4 + 1e-12)
    // The state survives serialisation.
    expect(alg.deserialise?.(alg.serialise?.(r.state) ?? null)?.step).toBe(6)
  })
})

describe('attacks on a trained network', () => {
  const { x, y } = planted(120, 0)
  const model = NN.mlp([2, 12, 2], { key: key(2) })
  train(model, x, y, 150)
  const f = logitsOf(model)

  it('the model is accurate before the attack', () => {
    expect(NN.accuracyOf(f, x, y)).toBeGreaterThan(0.95)
  })

  it('robust accuracy falls as epsilon grows, and PGD is at least as strong as FGSM', () => {
    const eps = [0, 0.05, 0.1, 0.2, 0.5]
    const pgdCurve = NN.robustAccuracy(f, x, y, eps, (e) => NN.pgd(f, x, y, { epsilon: e, steps: 10, key: key(1) }))
    const accs = pgdCurve.map((c) => c.accuracy)
    for (let i = 1; i < accs.length; i++) expect(accs[i] as number).toBeLessThanOrEqual((accs[i - 1] as number) + 1e-12)
    expect(accs[0]).toBeGreaterThan(0.95)
    expect(accs.at(-1) as number).toBeLessThan(0.1)
    const fg = NN.robustAccuracy(f, x, y, [0.2], (e) => NN.fgsm(f, x, y, { epsilon: e }))[0]?.accuracy as number
    expect(accs[3] as number).toBeLessThanOrEqual(fg + 1e-12)
  })

  it('transfer: inputs crafted on a second model fool a differently initialised one', () => {
    const other = NN.mlp([2, 12, 2], { key: key(77) })
    train(other, x, y, 150)
    const t = NN.transferAttack(logitsOf(other), f, x, y, { epsilon: 0.3, steps: 10, key: key(4) })
    expect(t.sourceSuccess).toBeGreaterThan(0.9)
    expect(t.transferRate).toBeGreaterThan(0.5)
    expect(t.transferRate).toBeLessThanOrEqual(1)
  })

  it('the black-box square attack fools the model within its linf budget using only logits', () => {
    const r = NN.squareAttack(f, x, y, { epsilon: 0.3, queries: 300, key: key(9) })
    const success = r.success.filter(Boolean).length / r.success.length
    expect(success).toBeGreaterThan(0.8)
    expect(Math.max(...r.distance)).toBeLessThanOrEqual(0.3 + 1e-12)
    expect(Math.max(...r.queries)).toBeLessThanOrEqual(301)
    expect(NN.accuracyOf(f, r.x, y)).toBeCloseTo(1 - success, 12)
  })
})

describe('minimal-perturbation attacks against the exact distance to a linear boundary', () => {
  // Logits (x . a, x . b): the boundary is x . (a - b) = 0, at l2 distance |x . (a - b)| / |a - b|.
  const a = [1, 2]
  const b = [-1, 0.5]
  const linear: NN.LogitsFunction = (v) =>
    matmul(
      v,
      tensor([
        [a[0] as number, b[0] as number],
        [a[1] as number, b[1] as number],
      ]),
    )
  const x = tensor([
    [1, 1],
    [0.5, 2],
  ])
  const w = [(a[0] as number) - (b[0] as number), (a[1] as number) - (b[1] as number)]
  const exact = [0, 1].map(
    (r) =>
      Math.abs((flat(x)[2 * r] as number) * (w[0] as number) + (flat(x)[2 * r + 1] as number) * (w[1] as number)) /
      Math.hypot(...w),
  )

  it('DeepFool finds the distance of a linear model in one step plus the overshoot', () => {
    const r = NN.deepFool(linear, x, { overshoot: 0.02 })
    expect(r.success).toEqual([true, true])
    r.distance.forEach((d, i) => {
      expect(d).toBeGreaterThan(exact[i] as number)
      expect(d).toBeLessThan((exact[i] as number) * 1.0201)
    })
  })

  it('Carlini-Wagner l2 finds a perturbation no smaller than the exact distance and close to it', () => {
    const labels = argmax(linear(x), -1)
    const r = NN.carliniWagnerL2(linear, x, labels, { c: 20, steps: 400, learningRate: 0.02 })
    expect(r.success).toEqual([true, true])
    r.distance.forEach((d, i) => {
      expect(d).toBeGreaterThanOrEqual((exact[i] as number) - 1e-9)
      expect(d).toBeLessThan((exact[i] as number) * 1.15)
    })
  })
})

describe('adversarial training', () => {
  const { x, y } = planted(160, 1)
  const eps = 0.15

  it('buys robustness the standard network lacks, at some cost in clean accuracy', () => {
    const standard = NN.mlp([2, 12, 2], { key: key(3) })
    train(standard, x, y, 200)
    const robust = NN.mlp([2, 12, 2], { key: key(3) })
    train(robust, x, y, 200, { attack: { epsilon: eps, steps: 5 } })
    const rob = (m: NN.Module): number => {
      const f = logitsOf(m)
      return NN.accuracyOf(f, NN.pgd(f, x, y, { epsilon: eps, steps: 10, key: key(8) }), y)
    }
    const clean = (m: NN.Module): number => NN.accuracyOf(logitsOf(m), x, y)
    expect(clean(standard)).toBeGreaterThan(0.97)
    expect(rob(standard)).toBeLessThan(0.7)
    expect(rob(robust)).toBeGreaterThan(rob(standard) + 0.15)
    expect(clean(robust)).toBeLessThanOrEqual(clean(standard))
  })

  it('TRADES takes a step and reports clean and robust losses', () => {
    const m = NN.mlp([2, 6, 2], { key: key(1) })
    const opt = adam({ learningRate: 0.05 })
    const s0 = NN.trainState(m, opt, key(0))
    const r = NN.adversarialTrainStep(m, s0, x, y, opt, { attack: { epsilon: eps, steps: 3 }, trades: { beta: 6 } })
    expect(r.state.step).toBe(1)
    expect(r.robustLoss).toBeGreaterThanOrEqual(r.cleanLoss - 1e-9)
    expect(Number.isFinite(r.state.loss)).toBe(true)
  })
})
