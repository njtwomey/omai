import { goldenSuite } from '../../testing/suite.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import * as T from './index.ts'

const stack = (rows: readonly (readonly number[])[]) =>
  fromStorage(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0], 'float64')

goldenSuite('seasonal decomposition against statsmodels', 'timeseries/decomposition', {
  seasonalDecompose: ([y], o) => {
    const d = T.seasonalDecompose(toFlat(y), o['period'] as number, {
      model: o['model'] as 'additive',
      oneSided: o['oneSided'] as boolean,
      extrapolateTrend: o['extrapolateTrend'] as number,
    })
    return stack([d.trend, d.seasonal, d.resid])
  },
  stl: ([y], o) => {
    const d = T.stl(toFlat(y), o['options'] as T.StlOptions)
    return stack([d.trend, d.seasonal, d.resid, d.weights ?? []])
  },
  mstl: ([y], o) => {
    const windows = o['windows'] as number[]
    const d = T.mstl(toFlat(y), {
      periods: o['periods'] as number[],
      ...(windows.length === 0 ? {} : { windows }),
    })
    return stack([d.trend, ...d.seasonal, d.resid])
  },
})
