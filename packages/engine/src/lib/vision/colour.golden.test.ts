import { goldenSuite } from '../../testing/suite.ts'
import * as V from './index.ts'

goldenSuite('colour conversions against OpenCV and scikit-image', 'vision/colour', {
  rgbToGrey: ([x], o) => V.rgbToGrey(x, { standard: o['standard'] as V.LumaStandard }),
  greyToRgb: ([x]) => V.greyToRgb(x),
  rgbToHsv: ([x]) => V.rgbToHsv(x),
  hsvToRgb: ([x]) => V.hsvToRgb(x),
  rgbToXyz: ([x]) => V.rgbToXyz(x),
  xyzToRgb: ([x]) => V.xyzToRgb(x),
  rgbToLab: ([x], o) =>
    V.rgbToLab(x, o['white'] === undefined ? {} : { white: o['white'] as [number, number, number] }),
  labToRgb: ([x]) => V.labToRgb(x),
  rgbToYcrcb: ([x]) => V.rgbToYcrcb(x),
  ycrcbToRgb: ([x]) => V.ycrcbToRgb(x),
})
