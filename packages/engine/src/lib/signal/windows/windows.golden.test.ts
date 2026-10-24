import { goldenSuite, type Params } from '../../../testing/suite.ts'
import * as S from '../index.ts'

const m = (o: Params): number => o['m'] as number
const sym = (o: Params): { sym: boolean } => ({ sym: (o['sym'] as boolean | undefined) ?? true })
const num = (o: Params, k: string): number => o[k] as number

goldenSuite('cosine windows against scipy.signal.windows', 'signal/windows/cosine', {
  boxcar: (_i, o) => S.boxcar(m(o), sym(o)),
  hann: (_i, o) => S.hann(m(o), sym(o)),
  hamming: (_i, o) => S.hamming(m(o), sym(o)),
  blackman: (_i, o) => S.blackman(m(o), sym(o)),
  blackmanHarris: (_i, o) => S.blackmanHarris(m(o), sym(o)),
  nuttall: (_i, o) => S.nuttall(m(o), sym(o)),
  flattop: (_i, o) => S.flattop(m(o), sym(o)),
  generalCosine: (_i, o) => S.generalCosine(m(o), o['a'] as number[], sym(o)),
  generalHamming: (_i, o) => S.generalHamming(m(o), num(o, 'alpha'), sym(o)),
})

goldenSuite('other windows against scipy.signal.windows', 'signal/windows/shapes', {
  bartlett: (_i, o) => S.bartlett(m(o), sym(o)),
  triang: (_i, o) => S.triang(m(o), sym(o)),
  cosine: (_i, o) => S.cosine(m(o), sym(o)),
  parzen: (_i, o) => S.parzen(m(o), sym(o)),
  bohman: (_i, o) => S.bohman(m(o), sym(o)),
  barthann: (_i, o) => S.barthann(m(o), sym(o)),
  lanczos: (_i, o) => S.lanczos(m(o), sym(o)),
  kaiser: (_i, o) => S.kaiser(m(o), num(o, 'beta'), sym(o)),
  gaussian: (_i, o) => S.gaussian(m(o), num(o, 'std'), sym(o)),
  generalGaussian: (_i, o) => S.generalGaussian(m(o), num(o, 'p'), num(o, 'sig'), sym(o)),
  tukey: (_i, o) => S.tukey(m(o), num(o, 'alpha'), sym(o)),
  exponential: (_i, o) =>
    S.exponential(m(o), {
      ...sym(o),
      tau: num(o, 'tau'),
      ...(o['center'] === undefined ? {} : { center: num(o, 'center') }),
    }),
  kaiserBesselDerived: (_i, o) => S.kaiserBesselDerived(m(o), num(o, 'beta')),
})

goldenSuite('DPSS against scipy.signal.windows.dpss', 'signal/windows/dpss', {
  dpss: (_i, o) =>
    S.dpss(m(o), num(o, 'nw'), {
      ...sym(o),
      ...(o['kmax'] === undefined ? {} : { kmax: num(o, 'kmax') }),
      ...(o['norm'] === undefined ? {} : { norm: o['norm'] as 'approximate' | 'subsample' }),
    }),
  dpssRatios: (_i, o) => S.dpssRatios(m(o), num(o, 'nw'), num(o, 'kmax')),
})

goldenSuite('getWindow against scipy.signal.get_window', 'signal/windows/get_window', {
  getWindow: (_i, o) => S.getWindow(o['window'] as S.WindowSpec, 12, o['fftbins'] as boolean),
})
