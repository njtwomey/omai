import { goldenSuite, type Params } from '../../../testing/suite.ts'
import * as S from '../index.ts'

const num = (o: Params, k: string): number => o[k] as number
const axis = (o: Params): { axis: number } => ({ axis: (o['axis'] as number | undefined) ?? -1 })

goldenSuite('filter application against scipy.signal', 'signal/filters/apply', {
  lfilter: ([b, a, x], o) => S.lfilter(b, a, x as never, axis(o)),
  lfilterY: ([b, a, x, zi]) => S.lfilter(b, a, x as never, { zi: zi as never }).y,
  lfilterZf: ([b, a, x, zi]) => S.lfilter(b, a, x as never, { zi: zi as never }).zf,
  lfilterZi: ([b, a]) => S.lfilterZi(b, a),
  lfiltic: ([b, a, y, x]) => S.lfiltic(b, a, y as never, x),
  sosfilt: ([sos, x]) => S.sosfilt(sos, x),
  sosfiltY: ([sos, x, zi]) => S.sosfilt(sos, x, { zi: zi as never }).y,
  sosfiltZf: ([sos, x, zi]) => S.sosfilt(sos, x, { zi: zi as never }).zf,
  sosfiltZi: ([sos]) => S.sosfiltZi(sos),
  filtfilt: ([b, a, x], o) =>
    S.filtfilt(b, a, x as never, {
      ...axis(o),
      ...(o['padtype'] === undefined ? {} : { padtype: o['padtype'] as S.PadType }),
      ...(o['padlen'] === undefined ? {} : { padlen: num(o, 'padlen') }),
      ...(o['method'] === undefined ? {} : { method: o['method'] as 'gust' }),
      ...(o['irlen'] === undefined ? {} : { irlen: num(o, 'irlen') }),
    }),
  sosfiltfilt: ([sos, x], o) =>
    S.sosfiltfilt(sos, x, {
      ...(o['padtype'] === undefined ? {} : { padtype: o['padtype'] as S.PadType }),
      ...(o['padlen'] === undefined ? {} : { padlen: num(o, 'padlen') }),
    }),
  medfilt: ([x], o) => S.medfilt(x, o['kernel'] as number | number[]),
  savgolCoeffs: (_i, o) =>
    S.savgolCoeffs(num(o, 'window'), num(o, 'order'), {
      ...(o['deriv'] === undefined ? {} : { deriv: num(o, 'deriv') }),
      ...(o['delta'] === undefined ? {} : { delta: num(o, 'delta') }),
      ...(o['pos'] === undefined ? {} : { pos: num(o, 'pos') }),
      ...(o['use'] === undefined ? {} : { use: o['use'] as 'dot' }),
    }),
  savgolFilter: ([x], o) =>
    S.savgolFilter(x, num(o, 'window'), num(o, 'order'), {
      deriv: num(o, 'deriv'),
      mode: o['mode'] as S.SavgolOptions['mode'] & string,
      cval: (o['cval'] as number | undefined) ?? 0,
      ...axis(o),
    }),
  movingAverage: ([x], o) => S.movingAverage(x, num(o, 'window'), o['mode'] as 'valid'),
  exponentialMovingAverage: ([x], o) => S.exponentialMovingAverage(x, num(o, 'alpha'), o['adjust'] as boolean),
})
