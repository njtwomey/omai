import { goldenSuite, type Params } from '../../../testing/suite.ts'
import { tensor, toFlat } from '../../tensor/index.ts'
import * as I from './index.ts'

const interp = (o: Params): I.InterpolationOptions => ({
  ...(o['order'] === undefined ? {} : { order: o['order'] as 0 | 1 }),
  ...(o['mode'] === undefined ? {} : { mode: o['mode'] as I.InterpolationMode }),
  ...(o['cval'] === undefined ? {} : { cval: o['cval'] as number }),
})

goldenSuite('geometric transforms against scipy.ndimage and scikit-image', 'signal/image/geometry', {
  mapCoordinates: ([x, c], o) => I.mapCoordinates(x, c, interp(o)),
  affineTransform: ([x, m], o) =>
    I.affineTransform(x, m, {
      ...interp(o),
      ...(o['offset'] === undefined ? {} : { offset: o['offset'] as number[] }),
      ...(o['outputShape'] === undefined ? {} : { outputShape: o['outputShape'] as number[] }),
    }),
  shift: ([x], o) => I.shift(x, o['shift'] as number[], interp(o)),
  zoom: ([x], o) => I.zoom(x, o['zoom'] as number[], { ...interp(o), gridMode: o['gridMode'] === true }),
  rotate: ([x], o) => I.rotate(x, o['angle'] as number, { ...interp(o), reshape: o['reshape'] !== false }),
  resize: ([x], o) =>
    I.resize(x, o['outputShape'] as number[], {
      order: o['order'] as 0 | 1,
      ...(o['mode'] === undefined ? {} : { mode: o['mode'] as I.SkimageMode }),
      ...(o['cval'] === undefined ? {} : { cval: o['cval'] as number }),
      ...(o['antiAliasing'] === undefined ? {} : { antiAliasing: o['antiAliasing'] as boolean }),
    }),
  pyramidReduce: ([x], o) =>
    I.pyramidReduce(x, {
      ...(o['factor'] === undefined ? {} : { factor: o['factor'] as number }),
      ...(o['mode'] === undefined ? {} : { mode: o['mode'] as I.SkimageMode }),
    }),
  pyramidExpand: ([x]) => I.pyramidExpand(x),
  pyramidGaussian: ([x], o) => tensor(I.pyramidGaussian(x, o['maxLayer'] as number).flatMap((l) => toFlat(l))),
  integralImage: ([x]) => I.integralImage(x),
  warpPerspective: ([x, h], o) =>
    I.warpPerspective(x, h, {
      order: o['order'] as 0 | 1,
      mode: o['mode'] as I.SkimageMode,
      cval: o['cval'] as number,
    }),
})
