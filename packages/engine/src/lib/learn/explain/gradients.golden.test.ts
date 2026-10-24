import { learnGolden, num, str } from '../../../testing/learn.ts'
import { add, matmul, relu, type Tensor, tanh, type Value } from '../../tensor/index.ts'
import {
  type DeepLiftLayer,
  deepLift,
  inputXGradient,
  integratedGradients,
  type PathMethod,
  saliency,
} from './gradients.ts'

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

learnGolden('gradient attributions against captum', 'learn/explain/gradients', {
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
})
