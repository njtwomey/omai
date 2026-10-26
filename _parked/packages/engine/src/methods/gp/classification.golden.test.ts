import { rbfKernel, scaleKernel } from '../../lib/kernels/index.ts'
import type { Tensor } from '../../lib/tensor/index.ts'
import { learnGolden } from '../../testing/learn.ts'
import { gaussianProcessClassifier } from './classification.ts'

learnGolden('Gaussian-process classification against scikit-learn', 'methods/gp/classification', {
  gpc: (inputs) => {
    const [x, y, q] = inputs as readonly [Tensor, Tensor, Tensor]
    const kernel = scaleKernel(rbfKernel({ lengthScale: 0.8 }), 2)
    const f = gaussianProcessClassifier({ kernel, optimize: false }).fit({ x, y })
    return { lml: f.params.problems[0]?.logMarginalLikelihood ?? Number.NaN, proba: f.predictProba(q) }
  },
})
