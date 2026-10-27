import { learnGolden, num } from '../../testing/learn.ts'
import { matrixView } from '../learn/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { bayesianLinearAlgorithm, bayesianLinearFit, maximiseEvidence, posteriorPredictive } from './bayesian.ts'

const designOf = (phi: Tensor, y: Tensor): { x: ReturnType<typeof matrixView>; y: Float64Array } => ({
  x: matrixView(phi, 'golden'),
  y: Float64Array.from(toFlat(y)),
})

learnGolden('Bayesian linear regression against numpy closed forms', 'linear/bayesian', {
  posterior: (inputs, p) => {
    const [phi, y, test] = inputs as [Tensor, Tensor, Tensor]
    const design = designOf(phi, y)
    const prior = { alpha: num(p, 'alpha'), beta: num(p, 'beta') }
    // The sequential algorithm, one row per step, ends at the batch posterior.
    const rows = Array.from({ length: design.x.n }, (_, i) => i)
    const seq = runAlgorithm(bayesianLinearAlgorithm(design, prior, rows), { maxSteps: rows.length + 1 }).state
    const at = posteriorPredictive(seq, test, prior.beta)
    return {
      mean: seq.mean,
      covariance: seq.covariance,
      precision: bayesianLinearFit(design, prior).precision,
      logEvidence: seq.logEvidence,
      predictiveMean: at.mean,
      predictiveVariance: at.variance,
    }
  },
  evidence: (inputs, p) => {
    const [phi, y] = inputs as [Tensor, Tensor]
    const start = { alpha: num(p, 'alpha'), beta: num(p, 'beta') }
    const fit = maximiseEvidence(designOf(phi, y), start, { maxIter: num(p, 'iterations'), tol: 0 })
    return { alpha: fit.alpha, beta: fit.beta, gamma: fit.gamma }
  },
})
