import { learnGolden } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { breuschPagan, olsInfluence, varianceInflationFactors, whiteTest } from './diagnostics.ts'

learnGolden('regression diagnostics against statsmodels', 'linear/diagnostics', {
  influence: (inputs) => {
    const [x, y, w] = inputs as [Tensor, Tensor, Tensor]
    const i = olsInfluence({ x, y, weights: w })
    return {
      leverage: i.leverage,
      internal: i.internal,
      external: i.external,
      cooks: i.cooksDistance,
      dffits: i.dffits,
      dffitsInternal: i.dffitsInternal,
      dfbeta: i.dfbeta,
      dfbetas: i.dfbetas,
      covRatio: i.covRatio,
    }
  },
  vif: (inputs) => ({ vif: varianceInflationFactors(inputs[0] as Tensor) }),
  het: (inputs) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const bp = breuschPagan({ x, y })
    const classic = breuschPagan({ x, y }, { studentize: false })
    const white = whiteTest({ x, y })
    return {
      bp: [bp.statistic, bp.pValue],
      bpClassic: [classic.statistic, classic.pValue],
      white: [white.statistic, white.pValue],
    }
  },
})
