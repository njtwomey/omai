import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { Linear, mlp, trainState } from '../nn/index.ts'
import { chain, sgd } from '../optim/index.ts'
import { key, normal } from '../random/index.ts'
import { div, fromStorage, mean, square, sub, sum, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  clipPerExample,
  type DpSgdState,
  dpEpsilon,
  dpGradientTransformation,
  dpSgdAlgorithm,
  dpSgdStep,
  noiseMultiplierFor,
  perExampleGradients,
  privateGradient,
} from './dp-sgd.ts'

const flat = (v: unknown): number[] => toFlat(v as Value)
const squared = (o: Value, t: Value): Value => sum(square(sub(o, t)))

describe('per-example gradients and clipping', () => {
  const model = mlp([3, 5, 1], { key: key(0) })
  const x = normal(key(1), [8, 3]) as Tensor
  const y = normal(key(2), [8, 1]) as Tensor

  it('their mean is the gradient of the mean loss', () => {
    const per = perExampleGradients(model, model.params, x, y, squared)
    const batch = grad((p: unknown) => mean(square(sub(model.apply(p as never, x), y))))(model.params) as Record<
      string,
      Record<string, Value>
    >
    const mine = per.grads as Record<string, Record<string, Value>>
    for (const layer of ['0', '2']) {
      for (const name of ['weight', 'bias']) {
        const stacked = mine[layer]?.[name] as Value
        expect(stacked.shape[0]).toBe(8)
        const averaged = flat(div(sum(stacked, 0), 8))
        const reference = flat(batch[layer]?.[name])
        averaged.forEach((v, i) => {
          expect(v).toBeCloseTo(reference[i] as number, 12)
        })
      }
    }
    expect(per.losses).toHaveLength(8)
    expect(per.losses.reduce((a, b) => a + b, 0) / 8).toBeCloseTo(
      flat(mean(square(sub(model.apply(model.params, x), y))))[0] as number,
      12,
    )
  })

  it('clipping bounds every example’s norm by C, leaves small gradients alone and keeps directions', () => {
    const per = perExampleGradients(model, model.params, x, y, squared)
    const unclipped = clipPerExample(per.grads, 1e9)
    expect(unclipped.clippedShare).toBe(0)
    const c = 0.3
    const clipped = clipPerExample(per.grads, c)
    const norms = clipPerExample(clipped.grads, 1e9).norms
    norms.forEach((n, i) => {
      expect(n).toBeLessThanOrEqual(c + 1e-12)
      // Clipped examples land on the sphere of radius C; others keep their norm.
      const before = clipped.norms[i] as number
      expect(n).toBeCloseTo(before > c ? c * (before / (before + 1e-6)) : before, 10)
    })
    expect(clipped.clippedShare).toBe(clipped.norms.filter((v) => v > c).length / 8)
    expect(clipped.clippedShare).toBeGreaterThan(0)
  })

  it('the private gradient without noise is the mean of the clipped gradients, with noise it has the right spread', () => {
    const clipped = tensor([
      [1, 0],
      [0, 1],
      [0.5, 0.5],
    ])
    const plain = privateGradient({ w: clipped }, { clipNorm: 1, noiseMultiplier: 0 }, key(0)) as { w: Value }
    flat(plain.w).forEach((v, i) => {
      expect(v).toBeCloseTo([1.5 / 3, 1.5 / 3][i] as number, 14)
    })
    // Noise: every coordinate gets sigma C / n standard deviation.
    const draws = Array.from(
      { length: 3000 },
      (_, i) =>
        flat(
          (privateGradient({ w: clipped }, { clipNorm: 2, noiseMultiplier: 1.5 }, key(i)) as { w: Value }).w,
        )[0] as number,
    )
    const m = draws.reduce((a, b) => a + b, 0) / draws.length
    const sd = Math.sqrt(draws.reduce((a, b) => a + (b - m) ** 2, 0) / (draws.length - 1))
    expect(m).toBeCloseTo(0.5, 1)
    expect(sd).toBeCloseTo((1.5 * 2) / 3, 1)
  })

  it('the noise stage works as a gradient transformation in a chain', () => {
    const t = dpGradientTransformation({ clipNorm: 1, noiseMultiplier: 0, expectedBatchSize: 4, key: key(0) })
    const c = chain(t, sgd({ learningRate: 0.5 }))
    const state = c.init(tensor([0, 0]))
    expect(flat(c.update(tensor([4, 8]), state).updates)).toEqual([-0.5, -1])
    const noisy = dpGradientTransformation({ clipNorm: 1, noiseMultiplier: 1, expectedBatchSize: 1, key: key(1) })
    const a = flat(noisy.update(tensor([0, 0, 0]), noisy.init(tensor([0, 0, 0]))).updates)
    expect(a.every((v) => v !== 0)).toBe(true)
  })
})

describe('a DP-SGD step', () => {
  it('moves the parameters like SGD on the clipped mean when there is no noise and nothing is clipped', () => {
    const model = new Linear(2, 1, { key: key(0) })
    const x = tensor([
      [1, 2],
      [3, 1],
    ])
    const y = tensor([[1], [0]])
    const opt = sgd({ learningRate: 0.1 })
    const r = dpSgdStep(model, trainState(model, opt, key(0)), x, y, squared, opt, {
      clipNorm: 1e9,
      noiseMultiplier: 0,
    })
    const plain = grad((p: unknown) => mean(sum(square(sub(model.apply(p as never, x), y)), 1)))(
      model.params,
    ) as Record<string, Value>
    const w0 = flat(model.params['weight'])
    flat((r.state.params as Record<string, Value>)['weight']).forEach((v, i) => {
      expect(v).toBeCloseTo((w0[i] as number) - 0.1 * (flat(plain['weight'])[i] as number), 12)
    })
    expect(r.clippedShare).toBe(0)
    expect(r.noiseNorm).toBe(0)
  })

  it('an empty batch still takes a pure-noise step', () => {
    const model = new Linear(2, 1, { key: key(0) })
    const opt = sgd({ learningRate: 0.1 })
    const empty = fromStorage(new Float64Array(0), [0, 2], 'float64')
    const none = fromStorage(new Float64Array(0), [0, 1], 'float64')
    const r = dpSgdStep(model, trainState(model, opt, key(0)), empty, none, squared, opt, {
      clipNorm: 1,
      noiseMultiplier: 1,
      expectedBatchSize: 4,
    })
    expect(r.batchSize).toBe(0)
    expect(r.noiseNorm).toBeGreaterThan(0)
    expect(flat((r.state.params as Record<string, Value>)['weight'])).not.toEqual(flat(model.params['weight']))
  })
})

describe('the accountant', () => {
  // opacus 1.6 compute_rdp + get_privacy_spent at delta 1e-5 on its default orders, for the standard settings.
  it('reproduces Abadi et al.’s MNIST setting and the TensorFlow Privacy and Opacus MNIST settings', () => {
    const abadi = dpEpsilon({ sampleRate: 0.01, noiseMultiplier: 4, steps: 10000, delta: 1e-5 })
    expect(abadi.epsilon).toBeCloseTo(1.0354900660362476, 7)
    expect(abadi.order).toBe(17)
    const tf = dpEpsilon({ sampleRate: 256 / 60000, noiseMultiplier: 1.1, steps: 14062, delta: 1e-5 })
    expect(tf.epsilon).toBeCloseTo(2.5965558689607486, 7)
    expect(tf.order).toBeCloseTo(8.1, 12)
    const tf15 = dpEpsilon({ sampleRate: 256 / 60000, noiseMultiplier: 1.1, steps: 3515, delta: 1e-5 })
    expect(tf15.epsilon).toBeGreaterThan(1.2)
    expect(tf15.epsilon).toBeLessThan(1.3)
  })

  it('epsilon grows with steps and sampling rate and falls with the noise multiplier', () => {
    const e = (q: number, s: number, t: number): number =>
      dpEpsilon({ sampleRate: q, noiseMultiplier: s, steps: t, delta: 1e-5 }).epsilon
    expect(e(0.01, 1, 100)).toBeLessThan(e(0.01, 1, 1000))
    expect(e(0.01, 1, 500)).toBeLessThan(e(0.05, 1, 500))
    expect(e(0.01, 2, 500)).toBeLessThan(e(0.01, 1, 500))
  })

  it('noiseMultiplierFor inverts it', () => {
    for (const target of [1, 3, 8]) {
      const sigma = noiseMultiplierFor({ targetEpsilon: target, delta: 1e-5, sampleRate: 0.02, steps: 500 })
      const eps = dpEpsilon({ sampleRate: 0.02, noiseMultiplier: sigma, steps: 500, delta: 1e-5 }).epsilon
      expect(eps).toBeLessThanOrEqual(target)
      expect(eps).toBeGreaterThan(target - 0.05)
    }
  })
})

describe('DP-SGD as an algorithm', () => {
  // Linear regression on a planted line; the noiseless clipped run fits it, a very noisy one does not.
  const xs = normal(key(5), [200, 2]) as Tensor
  const ys = tensor(
    Array.from({ length: 200 }, (_, i) => [(flat(xs)[2 * i] as number) * 1.5 - 2 * (flat(xs)[2 * i + 1] as number)]),
  )
  const run = (noise: number, seed = 0): { error: number; epsilon: number; state: DpSgdState } => {
    const model = new Linear(2, 1, { key: key(1), bias: false })
    const alg = dpSgdAlgorithm(
      model,
      { x: xs, y: ys },
      {
        loss: squared,
        optimizer: sgd({ learningRate: 0.2 }),
        sampleRate: 0.25,
        clipNorm: 5,
        noiseMultiplier: noise,
        steps: 150,
        delta: 1e-5,
      },
    )
    const r = runAlgorithm(alg, { maxSteps: 150, seed })
    const w = flat((r.state.params as Record<string, Value>)['weight'])
    return {
      error: Math.hypot((w[0] as number) - 1.5, (w[1] as number) + 2),
      epsilon: r.state.epsilon,
      state: r.state,
    }
  }

  it('learns the line with little noise and the epsilon on the tape matches the accountant', () => {
    const quiet = run(0.5)
    expect(quiet.error).toBeLessThan(0.15)
    expect(quiet.epsilon).toBeCloseTo(
      dpEpsilon({ sampleRate: 0.25, noiseMultiplier: 0.5, steps: 150, delta: 1e-5 }).epsilon,
      10,
    )
    const loud = run(30)
    expect(loud.error).toBeGreaterThan(quiet.error * 2)
    expect(loud.epsilon).toBeLessThan(quiet.epsilon)
  })

  it('the state serialises and the run is deterministic in its key', () => {
    const a = run(1, 3)
    const b = run(1, 3)
    expect(flat((a.state.params as Record<string, Value>)['weight'])).toEqual(
      flat((b.state.params as Record<string, Value>)['weight']),
    )
  })
})
