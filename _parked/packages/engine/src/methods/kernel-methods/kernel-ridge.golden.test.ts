import type { Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, num, str } from '../../testing/learn.ts'
import { type KernelRidgeHyper, kernelRidge } from './kernel-ridge.ts'

learnGolden('kernel ridge against scikit-learn', 'methods/kernel_methods/kernel_ridge', {
  kernelRidge: (inputs, p) => {
    const [x, y, q] = inputs as readonly [Tensor, Tensor, Tensor]
    const gamma = num(p, 'gamma')
    const f = kernelRidge({
      kernel: str(p, 'kernel') as KernelRidgeHyper['kernel'],
      alpha: num(p, 'alpha'),
      ...(gamma > 0 ? { gamma } : {}),
    }).fit({ x, y })
    return { dual: f.params.dualCoef, predict: f.predict(q) }
  },
})
