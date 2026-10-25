import { ints, learnGolden, num, str } from '../../testing/learn.ts'
import { add, matmul, type Tensor, tanh, type Value } from '../tensor/index.ts'
import { featureAblation, occlusion } from './perturbation.ts'

const model =
  ([w1, b1, w2, b2]: readonly Tensor[]) =>
  (v: Value): Tensor =>
    add(matmul(tanh(add(matmul(v, w1 as Tensor), b1 as Tensor)), w2 as Tensor), b2 as Tensor) as Tensor

learnGolden('perturbation attributions against captum', 'explain/perturbation', {
  ablation: (inputs, p) => {
    const label = str(p, 'label')
    const options =
      label === 'scalar'
        ? { baseline: 0.5 }
        : label === 'rows'
          ? { baseline: inputs[5] as Tensor }
          : label === 'mask'
            ? { featureMask: ints(p['mask'] as Tensor) }
            : {}
    return { attributions: featureAblation(model(inputs), inputs[4] as Tensor, { target: 1, ...options }) }
  },
  occlusion: (inputs, p) => ({
    attributions: occlusion(model(inputs), inputs[4] as Tensor, {
      shape: p['shape'] as number[],
      window: p['window'] as number[],
      strides: p['stride'] as number[],
      target: num(p, 'target', 0),
    }),
  }),
})
