import { goldenSuite } from '../../testing/suite.ts'
import * as S from './index.ts'

goldenSuite('decibels against librosa and unwrap against numpy', 'signal/signal', {
  magnitudeDb: ([x], o) => S.magnitudeDb(x, o as S.DbOptions),
  unwrap: ([x], o) => S.unwrap(x, o as S.UnwrapOptions),
})
