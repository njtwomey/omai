import { expect, describe as group, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { checkPrimitive, valueAndGrad } from '../autodiff/index.ts'
import { createContext, defaultContext, setDefaultContext } from '../context/index.ts'
import { xy } from '../data/index.ts'
import { mse } from '../losses/index.ts'
import { adam, sgd } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { item, type Tensor, tensor, toFlat, treePaths, type Value } from '../tensor/index.ts'
import { metricSeries, recorder } from '../trace/index.ts'
import * as NN from './index.ts'

type Params = Readonly<Record<string, unknown>>

const torchMlp = (state: Record<string, unknown>): NN.Sequential => {
  const net = new NN.Sequential(
    new NN.Linear(3, 5),
    new NN.Tanh(),
    new NN.Linear(5, 4),
    new NN.ReLU(),
    new NN.Linear(4, 2),
  )
  return net.loadStateDict(Object.fromEntries(Object.entries(state).map(([k, v]) => [k, decode(v) as Tensor])))
}

const ACTIVATIONS: Readonly<Record<string, (x: Tensor, p: Params) => Value>> = {
  relu: (x) => NN.relu(x),
  leakyRelu: (x, p) => NN.leakyRelu(x, p['slope'] as number),
  elu: (x, p) => NN.elu(x, p['alpha'] as number),
  selu: (x) => NN.selu(x),
  gelu: (x, p) => NN.gelu(x, (p['approximate'] as 'tanh' | undefined) ?? 'none'),
  silu: (x) => NN.silu(x),
  mish: (x) => NN.mish(x),
  softplus: (x, p) => NN.softplus(x, p['beta'] as number, p['threshold'] as number),
  hardtanh: (x) => NN.hardtanh(x),
  hardsigmoid: (x) => NN.hardsigmoid(x),
  glu: (x) => NN.glu(x),
}

group('nn against torch', () => {
  for (const c of loadFixture('nn/torch_reference').cases) {
    it(c.name, () => {
      const xs = c.inputs.map((x) => decode(x) as Tensor)
      if (c.fn in ACTIVATIONS) {
        expectMatches(
          (ACTIVATIONS[c.fn] as (x: Tensor, p: Params) => Value)(xs[0] as Tensor, c.params),
          c.output,
          1e-12,
          1e-14,
          c.name,
        )
        return
      }
      const net = torchMlp(c.params['state'] as Record<string, unknown>)
      if (c.fn === 'mlpForward') {
        expectMatches(net.forward(xs[0] as Tensor), c.output, 1e-12, 1e-14, c.name)
        return
      }
      const objective = (p: Params): Value => mse(net.apply(p, xs[0] as Tensor), xs[1] as Tensor)
      const { value, grad } = valueAndGrad(objective)(net.params)
      if (c.fn === 'mlpLoss') expectMatches(value, c.output, 1e-12, 1e-14, c.name)
      else expectMatches(treePaths(grad)[c.params['parameter'] as string] as Value, c.output, 1e-11, 1e-13, c.name)
    })
  }
})

group('modules', () => {
  it('state dicts round-trip and keep the containers children in step', () => {
    const a = NN.mlp([2, 3, 1], { key: key(1) })
    const b = NN.mlp([2, 3, 1], { key: key(2) })
    b.loadStateDict(a.stateDict())
    expect(toFlat(b.forward([[1, -1]]))).toEqual(toFlat(a.forward([[1, -1]])))
    expect(toFlat((b.layers[0] as NN.Module).params['weight'] as Value)).toEqual(
      toFlat(a.stateDict()['0.weight'] as Tensor),
    )
  })

  it('initialise reproducibly from the default context, in creation order', () => {
    const previous = setDefaultContext(createContext({ seed: 5 }))
    try {
      const first = new NN.Linear(4, 2).stateDict()['weight'] as Tensor
      setDefaultContext(createContext({ seed: 5 }))
      expect(toFlat(new NN.Linear(4, 2).stateDict()['weight'] as Tensor)).toEqual(toFlat(first))
      const bound = 1 / Math.sqrt(4)
      expect(toFlat(first).every((v) => Math.abs(v) <= bound)).toBe(true)
    } finally {
      setDefaultContext(previous)
    }
    expect(defaultContext()).toBe(previous)
  })

  it('dropout scales kept units in training and is the identity in evaluation', () => {
    const d = new NN.Dropout(0.5)
    const x = tensor(new Array(1000).fill(1))
    const y = toFlat(d.forward(x, { key: key(3) }))
    expect(y.every((v) => v === 0 || v === 2)).toBe(true)
    expect(Math.abs(y.reduce((s, v) => s + v, 0) / 1000 - 1)).toBeLessThan(0.1)
    expect(toFlat(d.eval().forward(x))).toEqual(toFlat(x))
  })

  it('describe themselves as structures and summaries', () => {
    const net = NN.mlp([3, 8, 2], { key: key(0) })
    const s = NN.describe(net, [16, 3])
    expect(s.type).toBe('architecture')
    expect(s.nodes.find((n) => n.id === 'layer:2')?.badges).toEqual({ shape: '[16, 2]', params: 18 })
    expect(NN.summary(net, [16, 3])).toContain('Total params: 50')
  })

  it('activations pass the conformance check', () => {
    const x = tensor([-1.3, -0.2, 0.4, 2.1])
    for (const f of [
      NN.gelu,
      NN.silu,
      NN.mish,
      NN.selu,
      (v: Value) => NN.softplus(v, 2),
      (v: Value) => NN.elu(v, 0.5),
    ]) {
      expect(checkPrimitive((v: Value) => f(v), [x], { rtol: 1e-5, atol: 1e-8 }).failures).toEqual([])
    }
  })
})

group('training', () => {
  it('fits a network to a smooth function and records the run', () => {
    const xs = Array.from({ length: 64 }, (_, i) => [-2 + (4 * i) / 63])
    const ys = xs.map(([v]) => [Math.sin(v ?? 0)])
    const net = NN.mlp([1, 16, 1], { key: key(0), activation: () => new NN.Tanh() })
    const rec = recorder()
    const r = NN.fit(net, xy(xs, ys), {
      loss: (p, t) => mse(p, t),
      optimizer: adam({ learningRate: 0.02 }),
      epochs: 300,
      batchSize: 16,
      key: key(1),
      tracer: rec,
    })
    expect(r.status).toBe('completed')
    expect(r.steps).toBe(1200)
    expect(r.losses.at(-1) ?? 1).toBeLessThan(5e-3)
    expect(metricSeries(rec.run(), 'loss').values.length).toBeGreaterThan(100)
    // The model holds the trained parameters.
    expect(item(mse(net.forward(xs), ys))).toBeLessThan(5e-3)
  })

  it('is deterministic for a key', () => {
    const data = xy([[0], [1], [2], [3]], [[0], [1], [4], [9]])
    const run = () =>
      NN.fit(NN.mlp([1, 4, 1], { key: key(2) }), data, {
        loss: (p, t) => mse(p, t),
        optimizer: sgd({ learningRate: 0.01 }),
        epochs: 5,
        batchSize: 2,
        key: key(9),
      }).losses
    expect(run()).toEqual(run())
  })
})
