import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as T from './index.ts'

const num = (v: unknown, d: number): number => (typeof v === 'number' ? v : d)

goldenSuite('the pan matrix profile against STUMPY', 'methods/timeseries/matrix_profile/pan', {
  bfsIndices: (_x, o) => tensor(T.bfsIndices(num(o['n'], 1))),
  stimp: ([t], o) => {
    const alg = T.stimpAlgorithm(t, { minM: num(o['minM'], 3), maxM: num(o['maxM'], 10), percentage: 1 })
    const st = runAlgorithm(alg, { maxSteps: num(o['steps'], 1) }).state
    const img = T.panImage(st, { contrast: o['contrast'] !== false, binary: o['binary'] !== false })
    return tensor(img.flatMap((r) => Array.from(r)))
  },
})
