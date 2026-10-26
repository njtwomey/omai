import type { Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, str } from '../../testing/learn.ts'
import { type SvcHyper, svc } from './classifier.ts'
import { linearSvc } from './linear.ts'
import { oneClassSvm } from './one-class.ts'
import { svr } from './regressor.ts'

type In = readonly [Tensor, Tensor, Tensor]

learnGolden('support vector machines against scikit-learn', 'methods/svm/svm', {
  svc: (inputs, p) => {
    const [x, y, q] = inputs as In
    const f = svc({ C: 2, kernel: str(p, 'kernel') as SvcHyper['kernel'], tol: 1e-8 }).fit({ x, y })
    const labels = Array.from(y.data)
    const counts = [0, 0]
    for (const record of f.exposition)
      if (record.kind === 'margin')
        for (const i of record.support) counts[labels[i] ?? 0] = (counts[labels[i] ?? 0] ?? 0) + 1
    return { decision: f.decisionFunction(q), nSupport: counts }
  },
  svcMulti: (inputs) => {
    const [x, y, q] = inputs as In
    return { predict: svc({ tol: 1e-8 }).fit({ x, y }).predict(q) }
  },
  svr: (inputs) => {
    const [x, t, q] = inputs as In
    return { predict: svr({ C: 3, epsilon: 0.05, tol: 1e-8 }).fit({ x, y: t }).predict(q) }
  },
  linearSvc: (inputs) => {
    const [x, y] = inputs as In
    const f = linearSvc({ C: 0.7, tol: 1e-8, maxIter: 100000 }).fit({ x, y })
    return { coef: f.params.coef[0] ?? [], intercept: f.params.intercept }
  },
  oneClass: (inputs) => {
    const [x, q] = inputs as In
    return { decision: oneClassSvm({ nu: 0.3, tol: 1e-8 }).fit({ x }).decisionFunction(q) }
  },
})
