import { goldenSuite, type Params } from '../../../testing/suite.ts'
import { tensor } from '../../tensor/index.ts'
import * as S from '../index.ts'

const n = (o: Params, k: string): number => o[k] as number
const opt = <T>(o: Params, k: string): Record<string, T> => (o[k] === undefined ? {} : { [k]: o[k] as T })

goldenSuite('FIR design against scipy.signal', 'signal/filters/fir', {
  firwin: (_i, o) =>
    S.firwin(n(o, 'numtaps'), o['cutoff'] as number | number[], {
      ...opt(o, 'passZero'),
      ...opt(o, 'window'),
      ...opt(o, 'fs'),
      ...opt(o, 'scale'),
      ...opt(o, 'width'),
    }),
  kaiserord: (_i, o) => {
    const r = S.kaiserord(n(o, 'ripple'), n(o, 'width'))
    return tensor([r.numtaps, r.beta])
  },
  kaiserBeta: () => tensor([10, 30, 70].map(S.kaiserBeta)),
  kaiserAtten: () => S.kaiserAtten(31, 0.1),
  firwin2: (_i, o) =>
    S.firwin2(n(o, 'numtaps'), o['freq'] as number[], o['gain'] as number[], {
      ...opt(o, 'antisymmetric'),
      ...opt(o, 'fs'),
      ...opt(o, 'nfreqs'),
      ...('window' in o ? { window: null } : {}),
    }),
  remez: (_i, o) =>
    S.remez(n(o, 'numtaps'), o['bands'] as number[], o['desired'] as number[], {
      ...opt(o, 'weight'),
      ...opt(o, 'type'),
      ...opt(o, 'fs'),
    }),
  minimumPhase: ([h], o) => S.minimumPhase(h, { ...opt(o, 'method'), ...opt(o, 'nFft'), ...opt(o, 'half') }),
})
