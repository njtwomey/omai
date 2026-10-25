import { learnGolden, num, str } from '../../testing/learn.ts'
import { conv2d } from '../nn/functional/index.ts'
import { add, matmul, relu, reshape, type Tensor, tanh, type Value } from '../tensor/index.ts'
import {
  type DeepLiftLayer,
  deconvolution,
  deepLift,
  gradCam,
  guidedBackprop,
  inputXGradient,
  integratedGradients,
  type LrpLayer,
  type LrpRule,
  lrp,
  type PathMethod,
  saliency,
} from './gradients.ts'
import { internalInfluence, layerConductance, layerGradientXActivation, neuronConductance } from './internal.ts'

const RULES: Readonly<Record<string, readonly [LrpRule, LrpRule]>> = {
  epsilon: [{ name: 'epsilon' }, { name: 'epsilon' }],
  'epsilon-0.1': [
    { name: 'epsilon', epsilon: 0.1 },
    { name: 'epsilon', epsilon: 0.1 },
  ],
  gamma: [{ name: 'gamma' }, { name: 'epsilon' }],
}

function layers([w1, b1, w2, b2]: readonly Tensor[], act: string): DeepLiftLayer[] {
  return [
    { fn: (v: Value) => add(matmul(v, w1 as Tensor), b1 as Tensor) },
    { fn: act === 'relu' ? relu : tanh, nonlinear: true },
    { fn: (v: Value) => add(matmul(v, w2 as Tensor), b2 as Tensor) },
  ]
}

const model = (inputs: readonly Tensor[], act: string) => {
  const ls = layers(inputs, act)
  return (v: Value): Value => ls.reduce((h, l) => l.fn(h), v)
}

learnGolden('gradient attributions against captum', 'explain/gradients', {
  saliency: (inputs, p) => {
    const f = model(inputs, str(p, 'act'))
    const target = num(p, 'target')
    return {
      saliency: saliency(f, inputs[4] as Tensor, { target }),
      inputXGradient: inputXGradient(f, inputs[4] as Tensor, { target }),
    }
  },
  integratedGradients: (inputs, p) => {
    const r = integratedGradients(model(inputs, str(p, 'act')), inputs[4] as Tensor, {
      target: 1,
      baseline: inputs[5] as Tensor,
      nSteps: num(p, 'steps'),
      method: str(p, 'method') as PathMethod,
    })
    return { attributions: r.attributions, delta: r.delta }
  },
  deepLift: (inputs, p) => ({
    attributions: deepLift(layers(inputs, str(p, 'act')), inputs[4] as Tensor, {
      target: 0,
      baseline: inputs[5] as Tensor,
    }),
  }),
  backward: (inputs, p) => {
    const f = str(p, 'rule') === 'guided' ? guidedBackprop : deconvolution
    return { attributions: f(layers(inputs, 'relu'), inputs[4] as Tensor, { target: 1 }) }
  },
  lrp: (inputs, p) => {
    const [w1, b1, w2, b2, x] = inputs as Tensor[]
    const [r1, r2] = RULES[str(p, 'rule')] ?? RULES['epsilon'] ?? []
    const net: LrpLayer[] = [
      { kind: 'linear', weight: w1 as Tensor, bias: b1 as Tensor, ...(r1 === undefined ? {} : { rule: r1 }) },
      { kind: 'elementwise', fn: relu },
      { kind: 'linear', weight: w2 as Tensor, bias: b2 as Tensor, ...(r2 === undefined ? {} : { rule: r2 }) },
    ]
    return { attributions: lrp(net, x as Tensor, { target: 0 }) }
  },
  layer: (inputs, p) => {
    const [w1, b1, w2, b2, x, base] = inputs as Tensor[]
    const act = str(p, 'act') === 'relu' ? relu : tanh
    const head = (v: Value): Value => act(add(matmul(v, w1 as Tensor), b1 as Tensor))
    const tail = (h: Value): Value => add(matmul(h, w2 as Tensor), b2 as Tensor)
    const path = { target: 1, baseline: base as Tensor, nSteps: 20 }
    return {
      conductance: layerConductance(head, tail, x as Tensor, path),
      influence: internalInfluence(head, tail, x as Tensor, path),
      gradientXActivation: layerGradientXActivation(head, tail, x as Tensor, { target: 1 }),
      neuron: neuronConductance(head, tail, x as Tensor, 2, path),
    }
  },
  gradCam: (inputs, p) => {
    const [cw, cb, lw, lb, images] = inputs as Tensor[]
    const n = images?.shape[0] ?? 0
    const head = (v: Value): Value =>
      relu(conv2d(reshape(v, [n, 1, 4, 4]), cw as Tensor, { bias: cb as Tensor, padding: 1 }))
    const tail = (a: Value): Value => add(matmul(reshape(a, [n, 32]), lw as Tensor), lb as Tensor)
    const maps = gradCam(head, tail, images as Tensor, { target: 0, relu: p['relu'] === true })
    return { maps: maps.map((m) => m.flat()) }
  },
})
