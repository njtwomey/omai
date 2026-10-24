import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const num = (v: unknown, d: number): number => (typeof v === 'number' ? v : d)

goldenSuite('sliding-window statistics and MASS against STUMPY', 'signal/sliding', {
  rollingMeanStd: ([t], o) => {
    const r = S.rollingMeanStd(t, num(o['m'], 3))
    return tensor([Array.from(r.mean), Array.from(r.std)])
  },
  rollingMean: ([t], o) => tensor(Array.from(S.rollingMeanStd(t, num(o['m'], 3)).mean)),
  rollingIsConstant: ([t], o) => tensor(Array.from(S.rollingIsConstant(t, num(o['m'], 3)))),
  rollingIsFinite: ([t], o) => tensor(Array.from(S.rollingIsFinite(t, num(o['m'], 3)))),
  slidingDotProduct: ([q, t]) => tensor(Array.from(S.slidingDotProduct(q, t))),
  zNormalize: ([q]) => tensor(Array.from(S.zNormalize(q))),
  mass: ([q, t], o) => tensor(Array.from(S.mass(q, t, { normalize: o['normalize'] !== false, p: num(o['p'], 2) }))),
  massChunked: ([q, t], o) => tensor(Array.from(S.massChunked(q, t, num(o['chunk'], 4096)))),
  distanceProfile: ([t], o) => tensor(Array.from(S.distanceProfile(t, num(o['m'], 3), num(o['idx'], 0)))),
})
