import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as T from './index.ts'

const num = (v: unknown, d: number): number => (typeof v === 'number' ? v : d)

goldenSuite('semantic segmentation against STUMPY', 'methods/timeseries/matrix_profile/segmentation', {
  fluss: ([I], o) => {
    const r = T.fluss(I?.data ?? [], num(o['L'], 10), num(o['regimes'], 2))
    return tensor([...Array.from(r.cac), ...r.regimes])
  },
  arcCounts: ([I]) => tensor(Array.from(T.arcCounts(I?.data ?? []))),
  floss: ([old, stream], o) => {
    const m = num(o['m'], 10)
    let s = T.flossInit(T.stump(old, m), old, num(o['L'], 10))
    for (const v of stream?.data ?? []) s = T.flossUpdate(s, Number(v))
    const I = Array.from(s.I, (v, i) => ((s.P[i] ?? 0) === Number.POSITIVE_INFINITY ? -1 : v))
    return tensor([...Array.from(s.cac), ...Array.from(s.P), ...I])
  },
})
