import { goldenSuite, type Params } from '../../../testing/suite.ts'
import type { BorderMode } from '../../numerics/convolution/index.ts'
import * as I from './index.ts'

const border = (o: Params): { mode: BorderMode; cval: number } => ({
  mode: (o['mode'] as BorderMode | undefined) ?? 'reflect',
  cval: (o['cval'] as number | undefined) ?? 0,
})

goldenSuite('ndimage correlation and convolution against scipy.ndimage', 'signal/image/correlate', {
  correlate1d: ([x, w], o) =>
    I.correlate1d(x, w, (o['axis'] as number | undefined) ?? -1, {
      ...border(o),
      origin: (o['origin'] as number) ?? 0,
    }),
  convolve1d: ([x, w], o) => I.convolve1d(x, w, -1, { ...border(o), origin: o['origin'] as number }),
  correlateNd: ([x, w], o) =>
    I.correlateNd(x, w, { ...border(o), ...(o['origin'] === undefined ? {} : { origin: o['origin'] as number[] }) }),
  convolveNd: ([x, w], o) => I.convolveNd(x, w, border(o)),
})
