import { tensor } from '../../../lib/tensor/index.ts'
import { goldenSuite } from '../../../testing/suite.ts'
import * as T from './index.ts'

const num = (v: unknown, d: number): number => (typeof v === 'number' ? v : d)

goldenSuite('MPdist, snippets and ostinato against STUMPY', 'methods/timeseries/matrix_profile/snippets', {
  mpdist: ([a, b], o) => tensor([T.mpdist(a, b, num(o['m'], 8), o['k'] === undefined ? {} : { k: num(o['k'], 1) })]),
  snippets: ([t], o) => {
    const r = T.snippets(t, num(o['m'], 10), num(o['k'], 2), o['s'] === undefined ? {} : { s: num(o['s'], 1) })
    return tensor([
      ...r.indices,
      ...r.profiles.flatMap((p) => Array.from(p)),
      ...r.fractions,
      ...r.areas,
      ...r.regimes.flat(),
    ])
  },
  ostinato: (ts, o) => {
    const r = T.ostinato(
      ts.flatMap((x) => (x === undefined ? [] : [x])),
      num(o['m'], 8),
    )
    return tensor([r.radius, r.series, r.index])
  },
})
