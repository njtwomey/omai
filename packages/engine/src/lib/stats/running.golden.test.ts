import { goldenSuite, type Params } from '../../testing/suite.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const smoothing = (p: Params): S.EwmOptions => ({
  ...(p['alpha'] === undefined ? {} : { alpha: Number(p['alpha']) }),
  ...(p['span'] === undefined ? {} : { span: Number(p['span']) }),
  ...(p['com'] === undefined ? {} : { com: Number(p['com']) }),
  ...(p['halflife'] === undefined ? {} : { halflife: Number(p['halflife']) }),
  adjust: p['adjust'] !== false,
})

const summary = (m: S.RunningMoments): ReturnType<typeof tensor> =>
  tensor([m.mean, m.variance, m.skewness, m.kurtosis, m.min, m.max])

goldenSuite('running moments against numpy and scipy, EWM against pandas', 'stats/running', {
  moments: ([x]) => summary(S.momentsOf(S.pushAll(S.momentState(), x), { ddof: 1 })),
  momentsMerged: ([x], p) => {
    const v = toFlat(x)
    const k = Number(p['split'])
    const a = S.pushAll(S.momentState(), v.slice(0, k))
    const b = S.pushAll(S.momentState(), v.slice(k))
    return summary(S.momentsOf(S.merge(a, b), { ddof: 1 }))
  },
  ewmMean: ([x], p) => S.ewmMean(x, smoothing(p)),
  ewmVar: ([x], p) => S.ewmVar(x, { ...smoothing(p), bias: p['bias'] === true }),
  ewmStd: ([x], p) => S.ewmStd(x, smoothing(p)),
})
