import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { additiveModel, kanFunctions } from '../datasets/index.ts'
import { additiveRegressor } from '../gam/index.ts'
import { mse } from '../losses/index.ts'
import { mlp, Tanh } from '../nn/index.ts'
import { adam } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { reshape, type Tensor, toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { defineTrainer } from './bases/index.ts'
import { kanRegressor } from './kan.ts'
import { progressiveKan } from './kan-progressive.ts'

const draw = (fn: string, n: number, seed: number): { x: Tensor; y: Tensor } => {
  const g = generate(kanFunctions, { key: key(seed), knobs: { function: fn, n } })
  return { x: g.data.x as Tensor, y: g.data.y as Tensor }
}
const rmse = (p: Tensor, y: Tensor): number => {
  const [a, b] = [toFlat(p), toFlat(y)]
  return Math.sqrt(a.reduce((s, v, i) => s + (v - (b[i] as number)) ** 2, 0) / a.length)
}
const std = (y: Tensor): number => {
  const v = toFlat(y)
  const m = v.reduce((a, b) => a + b, 0) / v.length
  return Math.sqrt(v.reduce((s, u) => s + (u - m) ** 2, 0) / v.length)
}

describe('the paper’s examples on planted truths', () => {
  const train = draw('exp-sin-square', 200, 1)
  const test = draw('exp-sin-square', 600, 2)

  it('f(x, y) = exp(sin(pi x) + y^2) is learned by a [2, 5, 1] KAN to a stated test error', () => {
    const fit = kanRegressor
      .with({ hidden: [5], epochs: 250, learningRate: 0.02 })
      .fit(train, { key: key(3), explain: false })
    expect(fit.network.widths).toEqual([2, 5, 1])
    const err = rmse(fit.predict?.(test.x) as Tensor, test.y)
    // Counted: the test RMSE is a few percent of the spread of f (std 1.38); asserted with a margin.
    expect(err).toBeLessThan(0.08)
    expect(err / std(test.y)).toBeLessThan(0.06)
  }, 60_000)

  it('multiplication needs two hidden nodes: [2, 2, 1] learns x y, [2, 1, 1] cannot', () => {
    const tr = draw('multiplication', 200, 1)
    const te = draw('multiplication', 500, 2)
    const run = (hidden: number[]): number =>
      rmse(
        kanRegressor
          .with({ hidden, epochs: 250, learningRate: 0.02 })
          .fit(tr, { key: key(3), explain: false })
          .predict?.(te.x) as Tensor,
        te.y,
      )
    const two = run([2])
    const one = run([1])
    expect(two / std(te.y)).toBeLessThan(0.08)
    expect(one / std(te.y)).toBeGreaterThan(0.3)
    expect(one).toBeGreaterThan(5 * two)
  }, 60_000)

  it('has fewer parameters than a tanh MLP for the same error: counted RMSE at 30 parameters against MLPs of 25 to 33', () => {
    const kan = kanRegressor
      .with({ hidden: [1], epochs: 250, learningRate: 0.02 })
      .fit(train, { key: key(3), explain: false })
    const kanError = rmse(kan.predict?.(test.x) as Tensor, test.y)
    expect(kan.network.parameterCount()).toBe(30)
    let best = Number.POSITIVE_INFINITY
    for (const lr of [0.03, 0.1]) {
      const net = defineTrainer({
        name: 'mlp',
        kind: 'regressor',
        defaults: {},
        model: (_d, _h, k) => mlp([2, 8, 1], { activation: () => new Tanh(), key: k }),
        objective: () => (p, t) => mse(p, reshape(t, p.shape)),
        optimiser: (l) => adam({ learningRate: l }),
        settings: () => ({ learningRate: lr, epochs: 2000 }),
      })
      best = Math.min(best, rmse(net.fit(train, { key: key(3), explain: false }).predict?.(test.x) as Tensor, test.y))
    }
    // The MLP has 33 parameters, 2000 epochs and the better of two learning rates; the KAN 30 and 250 epochs.
    expect(kanError).toBeLessThan(0.8 * best)
  }, 60_000)
})

describe('the progressive KAN', () => {
  const train = draw('exp-sin-square', 200, 1)
  const test = draw('exp-sin-square', 600, 2)
  const est = progressiveKan.with({ hidden: [1], grids: [3, 5], stepsPerStage: 120 })

  it('recovers the planted symbolic structure: sin and square into one node, then exp', () => {
    const fit = est.fit(train, { key: key(4), explain: false })
    expect(fit.network.widths).toEqual([2, 1, 1])
    expect(fit.network.spec.grid).toBe(5)
    const fixed = fit.symbolic()
    expect(fixed.layers.map((l) => l.edges.flat().map((e) => e?.name))).toEqual([['sin', 'x^2'], ['exp']])
    for (const l of fixed.layers) for (const row of l.edges) for (const e of row) expect(e?.r2).toBeGreaterThan(0.995)
    expect(rmse(fit.predict?.(test.x) as Tensor, test.y) / std(test.y)).toBeLessThan(0.08)
    const f = fit.formula(['x', 'y'])
    expect(f.text[0]).toContain('sin')
    expect(f.text[0]).toContain('exp')
  }, 120_000)

  it('is a staged algorithm whose tape shows the stages, and whose state serialises', () => {
    const s = est.session(train, { key: key(4), mode: 'eager' })
    s.advance()
    expect(s.status).toBe('completed')
    // 2 stages of 120 steps, a prune and a refinement.
    expect(s.t).toBe(242)
    const stage = s.probe(s.t, 'kan/stage') as {
      phase: string
      stage: number
      widths: number[]
      grid: number
      history: { kind: string }[]
    }
    expect(stage.phase).toBe('done')
    expect(stage.stage).toBe(1)
    expect(stage.grid).toBe(5)
    expect(stage.history.map((h) => h.kind)).toContain('prune')
    expect(stage.history.map((h) => h.kind)).toContain('refine')
    expect(stage.history.filter((h) => h.kind === 'grid-update').length).toBeGreaterThan(0)
    const before = s.probe(100, 'kan/stage') as typeof stage
    expect(before.phase).toBe('train')
    expect(before.grid).toBe(3)
    const loss = metricSeries(s.run(), 'solver/loss')
    expect(loss.values.length).toBeGreaterThan(200)
    // The refinement keeps the function: the loss right after it is close to the loss right before.
    const at = (t: number): number => loss.values[loss.steps.indexOf(t)] as number
    expect(Math.abs(at(122) - at(119))).toBeLessThan(0.5 * at(119) + 0.05)
    expect(at(241)).toBeLessThan(at(119))
    // Serialising the state and reading it back gives the same network.
    const fitted = s.fittedAt(s.t)
    const again = est.revive(JSON.parse(JSON.stringify(fitted.params)))
    expect(toFlat(again.predict?.(test.x) as Tensor)).toEqual(toFlat(fitted.predict?.(test.x) as Tensor))
  }, 120_000)

  it('prunes a wider network to the planted structure: [2, 5, 1] sparsified becomes [2, 1, 1] with the same accuracy', () => {
    const wide = progressiveKan.with({
      hidden: [5],
      grids: [3, 5],
      stepsPerStage: 150,
      lambdaL1: 0.2,
      lambdaEntropy: 0.2,
      prune: { nodeThreshold: 0.05, edgeThreshold: 0.05, after: 0 },
    })
    const fit = wide.fit(train, { key: key(5), explain: false })
    // Counted at these seeds: 150 parameters before, 30 after, test RMSE 0.037 (std 1.38).
    expect(fit.network.widths).toEqual([2, 1, 1])
    expect(fit.network.parameterCount()).toBe(30)
    expect(rmse(fit.predict?.(test.x) as Tensor, test.y) / std(test.y)).toBeLessThan(0.06)
  }, 120_000)
})

describe('a [d, 1] KAN as a GAM', () => {
  it('is an additive model with the same splines: the fitted functions agree with the neural additive regressor', () => {
    const g = generate(additiveModel, { key: key(1), knobs: { n: 150, nFeatures: 2, nInformative: 2, noise: 0.1 } })
    const data = { x: g.data.x, y: g.data.y }
    const epochs = 600
    const gam = additiveRegressor
      .with({ interiorKnots: 8, degree: 3, epochs, learningRate: 0.05 })
      .fit(data, { key: key(2), explain: false })
    const kan = kanRegressor
      .with({
        hidden: [],
        grid: 9,
        degree: 3,
        base: { kind: 'none' },
        bias: true,
        basis: { kind: 'bspline', boundary: 'clamped' },
        epochs,
        learningRate: 0.05,
      })
      .fit(data, { key: key(2), explain: false })
    const a = toFlat(gam.predict?.(data.x) as Tensor)
    const b = toFlat(kan.predict?.(data.x) as Tensor)
    const y = toFlat(data.y as Tensor)
    const rms = (p: ArrayLike<number>, q: ArrayLike<number>): number =>
      Math.sqrt(Array.from(p).reduce((s, v, i) => s + (v - (q[i] as number)) ** 2, 0) / p.length)
    // Both reach the least-squares fit of the same function class, so they agree far more closely than either fits the noise.
    expect(rms(a, b)).toBeLessThan(0.05)
    expect(rms(a, y)).toBeLessThan(0.2)
    expect(rms(b, y)).toBeLessThan(0.2)
    // Each KAN edge is that feature's shape function (up to the constant the additive model centres out).
    const curve = toFlat(kan.edgeCurve(0, 0, 0, [0.2, 0.5, 0.8]))
    const shape = toFlat(gam.termCurve(0, [0.2, 0.5, 0.8]))
    const shift = (curve[0] as number) - (shape[0] as number)
    expect(
      rms(
        curve.map((v) => v - shift),
        shape,
      ),
    ).toBeLessThan(0.1)
  }, 120_000)
})
