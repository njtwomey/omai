import { metricValue } from '../../testing/metrics.ts'
import { goldenSuite } from '../../testing/suite.ts'
import * as M from './index.ts'

goldenSuite(
  'correlation and agreement against scipy, pingouin, statsmodels and krippendorff',
  'metrics/correlation_agreement',
  {
    pearsonCorrelation: metricValue(M.pearsonCorrelation as never),
    spearmanCorrelation: metricValue(M.spearmanCorrelation as never),
    kendallTau: metricValue(M.kendallTau as never),
    concordanceCorrelation: metricValue(M.concordanceCorrelation as never),
    blandAltmanLower: ([x, y]) => M.blandAltman(x, y).lower,
    blandAltmanUpper: ([x, y]) => M.blandAltman(x, y).upper,
    cramersV: metricValue(M.cramersV),
    theilsU: metricValue(M.theilsU),
    weightedCohenKappa: metricValue(M.weightedCohenKappa),
    fleissKappa: ([t], o) => M.fleissKappa(t, o).estimate.value,
    intraclassCorrelation: ([r], o) => M.intraclassCorrelation(r, o).estimate.value,
    krippendorffAlpha: ([r], o) => M.krippendorffAlpha(r, o).estimate.value,
  },
)
