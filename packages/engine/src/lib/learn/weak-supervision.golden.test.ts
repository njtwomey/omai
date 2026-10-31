import { learnGolden, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { toFlat } from '../tensor/index.ts'
import { confidentJoint, dawidSkene, estimateNoiseTransition, findLabelIssues } from './weak-supervision.ts'

learnGolden('weak supervision against numpy references written from the papers', 'learn/weak_supervision', {
  dawid_skene: ([votes], p) => {
    const f = dawidSkene({
      mode: str(p, 'mode') as 'full' | 'one-coin',
      maxIter: num(p, 'iterations'),
      tol: 0,
    }).fit({ x: votes as Tensor })
    return {
      confusions: f.confusions.map((m) => m.flat()),
      priors: [...f.priors],
      posterior: f.posterior.map((r) => [...r]),
      logLikelihood: [f.logLikelihood],
    }
  },
  confident: ([proba, labels]) => {
    const cj = confidentJoint(proba as Tensor, labels as Tensor)
    const y = Array.from(toFlat(labels as Tensor))
    return {
      thresholds: cj.thresholds,
      counts: cj.counts,
      joint: cj.joint,
      guess: cj.guess,
      issues: findLabelIssues(proba as Tensor, y, 'prune_by_noise_rate'),
      anchor: estimateNoiseTransition(proba as Tensor, y, { percentile: 97 }),
    }
  },
})
