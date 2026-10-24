import { svc as svcMethod } from 'whitebox-engine/methods/svm'
import { dict, real } from 'whitebox-engine/space'
import type { HubEntry } from '../../entry/entry.ts'
import { tuningEntry } from './_tuning.ts'

/** Tuning an RBF support vector classifier's C and gamma by cross-validation. */
export const svcSearch: HubEntry = tuningEntry({
  name: 'svc',
  title: 'Tuning an SVC’s C and gamma',
  summary: 'Bayesian optimisation of an RBF support vector classifier’s C and gamma by cross-validated accuracy.',
  tags: ['svm', 'kernels', 'Bayesian optimisation'],
  datasets: ['classification2d'],
  estimator: svcMethod({ kernel: 'rbf' }),
  space: dict({
    C: real(1e-2, 1e3, { log: true, label: 'C' }),
    gamma: real(1e-2, 1e2, { log: true, label: 'gamma' }),
  }),
  sampler: 'gp',
  trials: 25,
  cv: 3,
  views: [
    { record: 'hyperparameterSurface', title: 'Cross-validated accuracy over C and gamma' },
    { record: 'searchTrials', title: 'Trials', area: 'side' },
    { record: 'learningCurve', title: 'Each trial and the best so far', area: 'below' },
  ],
  estimatorRef: { file: 'methods/svm/classifier.ts', name: 'svc' },
})
