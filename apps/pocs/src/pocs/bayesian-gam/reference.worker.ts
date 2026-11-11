/** The worker that computes the exact posterior of a problem (a quadrature) off the page's thread. */
import {
  type BayesianGamFitOptions,
  type BayesianGamOptions,
  bayesianGam,
  bayesianGamReference,
} from 'omai-engine/probprog/models'
import { serveStage } from 'omai-render/producer'

export interface ReferenceInput {
  readonly x: readonly number[]
  readonly y: readonly number[]
  readonly options: BayesianGamOptions
  readonly fit: BayesianGamFitOptions
}

serveStage(self, (input: ReferenceInput) =>
  bayesianGamReference(bayesianGam([...input.x], [...input.y], input.options), input.fit),
)
