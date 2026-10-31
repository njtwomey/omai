import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as T from './index.ts'

const num = (v: unknown, d: number): number => (typeof v === 'number' ? v : d)

goldenSuite('matrix-profile discovery against STUMPY', 'methods/timeseries/matrix_profile/discovery', {
  match: ([q, t], o) => {
    const r = T.match(q, t, {
      ...(o['maxDistance'] === undefined ? {} : { maxDistance: num(o['maxDistance'], 0) }),
      ...(o['maxMatches'] === undefined ? {} : { maxMatches: num(o['maxMatches'], 0) }),
    })
    return tensor(r.flatMap((x) => [x.distance, x.index]))
  },
  motifs: ([t, P], o) => {
    const r = T.motifs(t, P?.data ?? [], {
      ...(o['maxMotifs'] === undefined ? {} : { maxMotifs: num(o['maxMotifs'], 1) }),
      ...(o['maxDistance'] === undefined ? {} : { maxDistance: num(o['maxDistance'], 0) }),
      ...(o['cutoff'] === undefined ? {} : { cutoff: num(o['cutoff'], 0) }),
      ...(o['maxMatches'] === undefined ? {} : { maxMatches: num(o['maxMatches'], 10) }),
    })
    return tensor([...r.distances.flat(), ...r.indices.flat()])
  },
  allc: ([IL, IR]) => {
    const r = T.allc(IL?.data ?? [], IR?.data ?? [])
    const chains = [...r.chains].sort((a, b) => {
      for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return (a[i] ?? 0) - (b[i] ?? 0)
      return a.length - b.length
    })
    return tensor([chains.length, ...chains.map((c) => c.length), ...chains.flat(), ...r.longest])
  },
  atsc: ([IL, IR], o) => tensor(T.atsc(IL?.data ?? [], IR?.data ?? [], num(o['j'], 0))),
})
