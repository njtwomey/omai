import { ampMethod, ampProblem, ampStateEvolution } from 'omai-engine/message-passing'
import { key } from 'omai-engine/random'
import { metricSeries } from 'omai-engine/trace'
import type { AnytimeRunSpec } from '../../shared/useAnytimeRuns.ts'

/** The problem a page poses: the size, how many measurements per unknown, how sparse the signal and how noisy the measurements. */
export interface Setting {
  readonly seed: number
  readonly columns: number
  readonly delta: number
  readonly sparsity: number
  readonly noiseVariance: number
}

/** What a run reads at a snapshot: the noise level and the error of every iteration so far, and the estimate. */
export interface Kept {
  readonly tau2: readonly number[]
  readonly mse: readonly number[]
  readonly estimate: readonly number[]
}

/** The problem of a setting with the truth and the state evolution the run is to follow. */
export function problemOf(setting: Setting): {
  readonly truth: readonly number[]
  readonly predicted: readonly number[]
  readonly run: AnytimeRunSpec<Kept>
} {
  const { problem, truth, prior, noiseVariance } = ampProblem(key(setting.seed), setting)
  const method = ampMethod(problem, { noiseVariance, truth })
  return {
    truth,
    predicted: ampStateEvolution(prior, setting.delta, noiseVariance, 60),
    run: {
      id: 'amp',
      label: 'AMP',
      method,
      config: { prior: 'bernoulli-gaussian', sparsity: setting.sparsity, variance: 1, maxIterations: 60, tol: 1e-9 },
      logs: 'all',
      snapshotMs: 250,
      derive: (s, x) => {
        const run = s.run()
        return {
          tau2: metricSeries(run, 'amp/tau2').values,
          mse: metricSeries(run, 'amp/mse').values,
          estimate: Array.from((x.state as { x: ArrayLike<number> }).x),
        }
      },
    },
  }
}
