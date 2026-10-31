import { goldenSuite, type Params } from '../../testing/suite.ts'
import { asTensor, type Tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const ties = (p: Params): S.CoxTies => p['ties'] as S.CoxTies
const vec = (...parts: readonly (readonly number[])[]): Tensor => asTensor(parts.flat()) as Tensor

goldenSuite('survival analysis against statsmodels and scipy', 'survival/estimators', {
  km: ([time, event]) => {
    const c = S.kaplanMeier(time, event)
    return vec(c.time, c.survival, c.standardError, c.atRisk, c.events)
  },
  logrank: ([time, event, group], p) => {
    const weight = p['weight'] as S.LogRankWeight
    const r = S.logRankTest(
      time,
      event,
      group ?? time,
      weight === 'fleming-harrington' ? { weight, p: 1, q: 0 } : { weight },
    )
    return vec([r.statistic, r.pValue])
  },
  cox: ([time, event, x], p) => {
    const fit = S.coxPh(S.coxData(time, event, x ?? time), { ties: ties(p) })
    return vec(
      fit.beta,
      fit.coefficients.map((c) => c.standardError),
      [fit.logLikelihood],
    )
  },
  coxBaseline: ([time, event, x], p) => {
    const fit = S.coxPh(S.coxData(time, event, x ?? time), { ties: ties(p) })
    const b = fit.baseline[0]
    // The reference reports H just before each time; its later entries are H(t_k) at all but the last time.
    return vec((b?.time ?? []).slice(0, -1), (b?.cumulativeHazard ?? []).slice(0, -1))
  },
  coxStrata: ([time, event, x, strata]) => {
    const fit = S.coxPh(S.coxData(time, event, x ?? time, strata === undefined ? {} : { strata }))
    return vec(
      fit.beta,
      fit.coefficients.map((c) => c.standardError),
      [fit.logLikelihood],
    )
  },
  concordance: ([time, event, score]) => {
    const c = S.concordanceIndex(time, event, score ?? time)
    return vec([c.concordance, c.concordant, c.discordant, c.tied, c.comparable])
  },
  aft: ([time, event, x], p) => {
    const fit = S.aft(S.aftData(time, event, x), { distribution: p['distribution'] as S.AftDistribution })
    return vec(
      fit.theta,
      fit.coefficients.map((c) => c.standardError),
      [fit.logLikelihood],
    )
  },
})
