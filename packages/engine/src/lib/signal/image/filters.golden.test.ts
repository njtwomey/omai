import { goldenSuite, type Params } from '../../../testing/suite.ts'
import type { BorderMode } from '../../numerics/convolution/index.ts'
import { stack } from '../../tensor/index.ts'
import * as I from './index.ts'

const border = (o: Params): { mode: BorderMode; cval: number } => ({
  mode: (o['mode'] as BorderMode | undefined) ?? 'reflect',
  cval: (o['cval'] as number | undefined) ?? 0,
})
const numOrList = (v: unknown): number | number[] => (Array.isArray(v) ? v.map(Number) : Number(v))
const opt = <K extends string>(o: Params, k: K): Partial<Record<K, number | number[]>> =>
  o[k] === undefined ? {} : ({ [k]: numOrList(o[k]) } as Record<K, number | number[]>)

goldenSuite('image filters against scipy.ndimage and scikit-image', 'signal/image/filters', {
  gaussianFilter1d: ([x], o) =>
    I.gaussianFilter1d(x, o['sigma'] as number, o['axis'] as number, { order: o['order'] as number }),
  gaussianFilter: ([x], o) =>
    I.gaussianFilter(x, numOrList(o['sigma']), {
      ...border(o),
      ...opt(o, 'order'),
      ...(o['truncate'] === undefined ? {} : { truncate: o['truncate'] as number }),
      ...(o['radius'] === undefined ? {} : { radius: o['radius'] as number }),
    }),
  uniformFilter: ([x], o) => I.uniformFilter(x, numOrList(o['size']), { ...border(o), ...opt(o, 'origin') }),
  medianFilter: ([x], o) => I.medianFilter(x, numOrList(o['size']), border(o)),
  minimumFilter: ([x], o) => I.minimumFilter(x, numOrList(o['size']), border(o)),
  maximumFilter: ([x], o) => I.maximumFilter(x, numOrList(o['size']), { ...border(o), ...opt(o, 'origin') }),
  rankFilter: ([x], o) => I.rankFilter(x, o['rank'] as number, numOrList(o['size'])),
  percentileFilter: ([x], o) => I.percentileFilter(x, o['percentile'] as number, numOrList(o['size'])),
  sobel: ([x], o) => I.sobel(x, { ...border(o), ...opt(o, 'axis') }),
  scharr: ([x], o) => I.scharr(x, { ...border(o), ...opt(o, 'axis') }),
  prewitt: ([x], o) => I.prewitt(x, { ...border(o), ...opt(o, 'axis') }),
  laplace: ([x], o) => I.laplace(x, border(o)),
  gaussianLaplace: ([x], o) => I.gaussianLaplace(x, o['sigma'] as number),
  gaussianGradientMagnitude: ([x], o) => I.gaussianGradientMagnitude(x, o['sigma'] as number),
  gaborKernel: (_x, o) => {
    const k = I.gaborKernel(o['frequency'] as number, {
      theta: o['theta'] as number,
      ...(o['bandwidth'] === undefined ? {} : { bandwidth: o['bandwidth'] as number }),
      ...(o['sigmaX'] === undefined ? {} : { sigmaX: o['sigmaX'] as number }),
      ...(o['sigmaY'] === undefined ? {} : { sigmaY: o['sigmaY'] as number }),
      ...(o['nStds'] === undefined ? {} : { nStds: o['nStds'] as number }),
      ...(o['offset'] === undefined ? {} : { offset: o['offset'] as number }),
    })
    return stack([k.real, k.imag])
  },
  gabor: ([x], o) => {
    const r = I.gabor(x, o['frequency'] as number, { theta: o['theta'] as number })
    return stack([r.real, r.imag])
  },
  bilateralFilter: ([x], o) =>
    I.bilateralFilter(x, { sigmaColor: o['sigmaColor'] as number, sigmaSpatial: o['sigmaSpatial'] as number }),
})
