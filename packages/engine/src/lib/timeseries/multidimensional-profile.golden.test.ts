import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as T from './index.ts'

const num = (v: unknown, d: number): number => (typeof v === 'number' ? v : d)
const nums = (v: unknown): number[] => (Array.isArray(v) ? v.map(Number) : [])

goldenSuite('multidimensional matrix profiles against STUMPY', 'methods/timeseries/matrix_profile/multidimensional', {
  mstump: ([t], o) => {
    const r = T.mstump(t, num(o['m'], 6), {
      ...(o['include'] === undefined ? {} : { include: nums(o['include']) }),
      discords: o['discords'] === true,
      batch: 11,
    })
    return tensor([...Array.from(r.P), ...Array.from(r.I)])
  },
  multiDistanceProfile: ([t], o) =>
    tensor(T.multiDistanceProfile(t, num(o['m'], 6), num(o['idx'], 0)).flatMap((r) => Array.from(r))),
  subspace: ([t], o) =>
    tensor(T.subspace(t, num(o['m'], 6), num(o['subseqIdx'], 0), num(o['nnIdx'], 0), num(o['k'], 0))),
  mdl: ([t], o) => {
    const r = T.mdl(t, num(o['m'], 6), nums(o['subseqIdx']), nums(o['nnIdx']))
    return tensor([...Array.from(r.bits), ...r.subspaces.flat()])
  },
})
