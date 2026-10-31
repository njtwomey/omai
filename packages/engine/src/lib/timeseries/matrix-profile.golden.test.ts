import { goldenSuite, type Params } from '../../testing/suite.ts'
import { key } from '../random/index.ts'
import { tensor } from '../tensor/index.ts'
import * as T from './index.ts'

const num = (v: unknown, d: number): number => (typeof v === 'number' ? v : d)

const flat = (mp: T.MatrixProfile): ReturnType<typeof tensor> =>
  tensor([...Array.from(mp.P), ...Array.from(mp.I), ...Array.from(mp.IL), ...Array.from(mp.IR)])

const common = (o: Params): T.StumpOptions => ({
  k: num(o['k'], 1),
  normalize: o['normalize'] !== false,
  p: num(o['p'], 2),
  batch: 7,
})

goldenSuite('matrix-profile algorithms against STUMPY', 'methods/timeseries/matrix_profile/algorithms', {
  stump: ([t, tB], o) => flat(T.stump(t, num(o['m'], 8), { ...common(o), ...(tB === undefined ? {} : { tB }) })),
  stomp: ([t, tB], o) =>
    flat(T.stomp(t, num(o['m'], 8), { k: num(o['k'], 1), batch: 9, ...(tB === undefined ? {} : { tB }) })),
  stamp: ([t], o) => {
    const mp = T.stamp(t, num(o['m'], 8), { exclusion: num(o['exclusion'], 2), key: key(3) })
    return tensor([...Array.from(mp.P), ...Array.from(mp.I)])
  },
  prescrump: ([t], o) => {
    const mp = T.prescrump(t, num(o['m'], 8), { s: num(o['s'], 1), key: key(4) })
    return tensor([...Array.from(mp.P), ...Array.from(mp.I)])
  },
  scrump: ([t], o) =>
    flat(T.scrump(t, num(o['m'], 8), { ...common(o), percentage: num(o['percentage'], 1), key: key(5) })),
  stumpi: ([t, stream], o) => {
    let s = T.stumpiInit(t, num(o['m'], 8), { egress: o['egress'] === true })
    for (const v of stream?.data ?? []) s = T.stumpiUpdate(s, Number(v))
    return tensor([...Array.from(s.profile.P), ...Array.from(s.profile.I), ...Array.from(s.profile.IL)])
  },
})
