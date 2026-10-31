import { learnGolden, num } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { instrumentalVariables } from './index.ts'

const ivData = (inputs: readonly unknown[]) => {
  const [y, d, z, w] = inputs as [Tensor, Tensor, Tensor, Tensor]
  return { outcome: y, endogenous: d, instruments: z, ...(w.shape[1] === 0 ? {} : { exogenous: w }) }
}

// statsmodels orders the coefficients [endogenous, controls, constant] as this does.
learnGolden(
  'Two-stage least squares, LIML and Fuller against statsmodels and the k-class formulas',
  'causal/instrumental',
  {
    tsls: (inputs) => {
      const r = instrumentalVariables(ivData(inputs))
      return {
        coefficients: r.coefficients as number[],
        standardErrors: r.standardErrors as number[],
        firstStageF: r.firstStage[0]?.fStatistic ?? Number.NaN,
        ...(r.sargan === undefined ? {} : { sargan: [r.sargan.statistic, r.sargan.pValue] }),
      }
    },
    liml: (inputs, p) => {
      const fuller = num(p, 'fuller')
      const r = instrumentalVariables(
        ivData(inputs),
        fuller < 0 ? { estimator: 'liml' } : { estimator: 'fuller', fuller },
      )
      return { coefficients: r.coefficients as number[], standardErrors: r.standardErrors as number[], kappa: r.kappa }
    },
  },
)
