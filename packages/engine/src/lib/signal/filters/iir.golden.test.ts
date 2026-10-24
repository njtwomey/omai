import { goldenSuite, type Params } from '../../../testing/suite.ts'
import type { ComplexArray } from '../../numerics/polynomial/index.ts'
import type { ZerosPolesGainData } from '../../systems/index.ts'
import { fromStorage, type Tensor, tensor } from '../../tensor/index.ts'
import * as S from '../index.ts'

const r9 = (v: number): number => Math.round(v * 1e9) / 1e9

/** Complex values sorted by rounded real then imaginary part, as `[n, 2]` (the oracle's order). */
function roots(z: ComplexArray): Tensor {
  const pairs = z.re.map((re, i) => [re, z.im[i] ?? 0] as [number, number])
  pairs.sort((a, b) => r9(a[0]) - r9(b[0]) || r9(a[1]) - r9(b[1]))
  return pairs.length === 0 ? fromStorage(new Float64Array(0), [0, 2], 'float64') : tensor(pairs)
}

const n = (o: Params, k: string): number => o[k] as number

function zpkFns(
  name: string,
  f: (o: Params) => ZerosPolesGainData,
): Record<string, (i: unknown, o: Params) => Tensor | number> {
  return {
    [`${name}Z`]: (_i, o) => roots(f(o).zeros),
    [`${name}P`]: (_i, o) => roots(f(o).poles),
    [`${name}K`]: (_i, o) => f(o).gain,
  }
}

const c2 = S.cheb2ap(3, 25)

function design(o: Params, output: S.FilterOutput) {
  return S.iirfilter(n(o, 'n'), o['wn'] as number | number[], {
    ftype: o['ftype'] as S.IirFamily,
    btype: o['btype'] as S.BandType,
    output,
    ...(o['rp'] === undefined ? {} : { rp: n(o, 'rp') }),
    ...(o['rs'] === undefined ? {} : { rs: n(o, 'rs') }),
    ...(o['fs'] === undefined ? {} : { fs: n(o, 'fs') }),
    ...(o['analog'] === undefined ? {} : { analog: o['analog'] as boolean }),
  })
}

const spec = (o: Params): S.FilterSpec => ({
  wp: o['wp'] as number,
  ws: o['ws'] as number,
  gpass: (o['gpass'] as number | undefined) ?? 1,
  gstop: (o['gstop'] as number | undefined) ?? 45,
  analog: (o['analog'] as boolean | undefined) ?? false,
  ...(o['fs'] === undefined ? {} : { fs: n(o, 'fs') }),
})

const ordered = (r: S.FilterOrder): Tensor => tensor([r.n, ...(typeof r.wn === 'number' ? [r.wn] : r.wn)])

goldenSuite('IIR design against scipy.signal', 'signal/filters/iir', {
  ...zpkFns('buttap', (o) => S.buttap(n(o, 'n'))),
  ...zpkFns('cheb1ap', (o) => S.cheb1ap(n(o, 'n'), n(o, 'rp'))),
  ...zpkFns('cheb2ap', (o) => S.cheb2ap(n(o, 'n'), n(o, 'rs'))),
  ...zpkFns('ellipap', (o) => S.ellipap(n(o, 'n'), n(o, 'rp'), n(o, 'rs'))),
  ...zpkFns('besselap', (o) => S.besselap(n(o, 'n'), o['norm'] as S.BesselNorm)),
  ...zpkFns('lp2lp', (o) => S.lp2lpZpk(c2, n(o, 'wo'))),
  ...zpkFns('lp2hp', (o) => S.lp2hpZpk(c2, n(o, 'wo'))),
  ...zpkFns('lp2bp', (o) => S.lp2bpZpk(c2, n(o, 'wo'), n(o, 'bw'))),
  ...zpkFns('lp2bs', (o) => S.lp2bsZpk(c2, n(o, 'wo'), n(o, 'bw'))),
  ...zpkFns('bilinear', (o) => S.bilinearZpk(c2, n(o, 'fs'))),
  ...zpkFns('iirZpk', (o) => design(o, 'zpk') as S.DesignOf<'zpk'>),
  iirB: (_i, o) => (design(o, 'ba') as S.DesignOf<'ba'>).num,
  iirA: (_i, o) => (design(o, 'ba') as S.DesignOf<'ba'>).den,
  iirSos: (_i, o) => (design(o, 'sos') as S.DesignOf<'sos'>).sections,
  buttord: (_i, o) => ordered(S.buttord(spec(o))),
  cheb1ord: (_i, o) => ordered(S.cheb1ord(spec(o))),
  cheb2ord: (_i, o) => ordered(S.cheb2ord(spec(o))),
  ellipord: (_i, o) => ordered(S.ellipord(spec(o))),
  iirdesign: (_i, o) =>
    S.iirdesign(
      { wp: o['wp'] as [number, number], ws: o['ws'] as [number, number], gpass: 1, gstop: 40 },
      {
        ftype: 'cheby2',
        output: 'sos',
      },
    ).sections,
  iirnotchB: (_i, o) => S.iirnotch(n(o, 'w0'), n(o, 'q'), n(o, 'fs')).num,
  iirnotchA: (_i, o) => S.iirnotch(n(o, 'w0'), n(o, 'q'), n(o, 'fs')).den,
  iirpeakB: (_i, o) => S.iirpeak(n(o, 'w0'), n(o, 'q'), n(o, 'fs')).num,
  iirpeakA: (_i, o) => S.iirpeak(n(o, 'w0'), n(o, 'q'), n(o, 'fs')).den,
  iircomb: (_i, o) => {
    const t = S.iircomb(100, 20, { ftype: o['ftype'] as 'notch', fs: 1000, passZero: o['passZero'] as boolean })
    return tensor([Array.from(t.num.shape[0] === 0 ? [] : (t.num as Tensor).data), Array.from((t.den as Tensor).data)])
  },
})
