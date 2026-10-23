import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { jvp } from '../../autodiff/index.ts'
import { full, type Tensor, type TensorLike, type Value } from '../../tensor/index.ts'
import * as S from './index.ts'

type Params = Readonly<Record<string, unknown>>
type Fn = (xs: Value[], p: Params) => Value

const unary =
  (f: (x: TensorLike) => Value): Fn =>
  (xs) =>
    f(xs[0] as Value)
const binary =
  (f: (a: TensorLike, b: TensorLike) => Value): Fn =>
  (xs) =>
    f(xs[0] as Value, xs[1] as Value)
const ternary =
  (f: (a: TensorLike, b: TensorLike, c: TensorLike) => Value): Fn =>
  (xs) =>
    f(xs[0] as Value, xs[1] as Value, xs[2] as Value)
const order =
  (f: (n: number, x: TensorLike) => Value): Fn =>
  (xs, p) =>
    f(p['n'] as number, xs[0] as Value)

const FNS: Readonly<Record<string, Fn>> = {
  erf: unary(S.erf),
  erfc: unary(S.erfc),
  erfcx: unary(S.erfcx),
  erfinv: unary(S.erfinv),
  erfcinv: unary(S.erfcinv),
  dawsn: unary(S.dawsn),
  ndtr: unary(S.ndtr),
  logNdtr: unary(S.logNdtr),
  ndtri: unary(S.ndtri),
  normalPdf: unary(S.normalPdf),
  normalLogPdf: unary(S.normalLogPdf),
  owensT: binary(S.owensT),
  logIntervalProbability: binary(S.logIntervalProbability),
  expit: unary(S.expit),
  logit: unary(S.logit),
  logSigmoid: unary(S.logSigmoid),
  softplus: unary(S.softplus),
  log1mexp: unary(S.log1mexp),
  logaddexp: binary(S.logaddexp),
  xlogy: binary(S.xlogy),
  xlog1py: binary(S.xlog1py),
  entr: unary(S.entr),
  relEntr: binary(S.relEntr),
  klDiv: binary(S.klDiv),
  gamma: unary(S.gamma),
  lgamma: unary(S.lgamma),
  rgamma: unary(S.rgamma),
  digamma: unary(S.digamma),
  polygamma: order(S.polygamma),
  lbeta: binary(S.lbeta),
  beta: binary(S.beta),
  factorial: unary(S.factorial),
  lfactorial: unary(S.lfactorial),
  binom: binary(S.binom),
  lbinom: binary(S.lbinom),
  gammainc: binary(S.gammainc),
  gammaincc: binary(S.gammaincc),
  logGammainc: binary(S.logGammainc),
  logGammaincc: binary(S.logGammaincc),
  gammaincinv: binary(S.gammaincinv),
  gammainccinv: binary(S.gammainccinv),
  betainc: ternary(S.betainc),
  betaincinv: ternary(S.betaincinv),
  stdtr: binary(S.stdtr),
  stdtrit: binary(S.stdtrit),
  chdtr: binary(S.chdtr),
  chdtrc: binary(S.chdtrc),
  fdtr: ternary(S.fdtr),
  fdtrc: ternary(S.fdtrc),
  kolmogorov: unary(S.kolmogorov),
  smirnov: order(S.smirnov),
  zeta: unary(S.zeta),
  hurwitzZeta: binary(S.hurwitzZeta),
  lambertW: unary(S.lambertW),
  expn: order(S.expn),
  expi: unary(S.expi),
  sici: (xs, p) => S.sici(xs[0] as Value)[p['part'] as 0 | 1],
  fresnel: (xs, p) => S.fresnel(xs[0] as Value)[p['part'] as 0 | 1],
  iv: binary(S.iv),
  ive: binary(S.ive),
  logIv: binary(S.logIv),
  kv: binary(S.kv),
  kve: binary(S.kve),
  jv: binary(S.jv),
  yv: binary(S.yv),
  i0e: unary(S.i0e),
  i1e: unary(S.i1e),
  ellipk: unary(S.ellipk),
  ellipe: unary(S.ellipe),
  ellipkinc: binary(S.ellipkinc),
  ellipeinc: binary(S.ellipeinc),
  ellipj: (xs, p) => S.ellipj(xs[0] as Value, xs[1] as Value)[p['part'] as 'sn' | 'cn' | 'dn'],
}

/** The elementwise derivative in argument `wrt`: a forward-mode product with a tangent of ones. */
function partial(f: Fn, xs: Value[], p: Params, wrt: number): Value {
  const x = xs[wrt] as Value
  const ones = full(x.shape, 1)
  return jvp(
    (v: Value) =>
      f(
        xs.map((u, i) => (i === wrt ? v : u)),
        p,
      ),
    x,
    ones,
  ).tangent
}

const FILES = ['erf', 'normal', 'logistic', 'gamma', 'incomplete', 'cdf', 'zeta', 'bessel', 'elliptic']

for (const file of FILES) {
  describe(`special/${file} against mpmath`, () => {
    for (const c of loadFixture(`numerics/special/${file}`).cases) {
      it(c.name, () => {
        const f = FNS[c.fn]
        if (f === undefined) throw new Error(`no function ${c.fn}`)
        const xs = c.inputs.map((x) => decode(x) as Tensor)
        const wrt = c.params['wrt'] as number | undefined
        const got = wrt === undefined ? f(xs, c.params) : partial(f, xs, c.params, wrt)
        expectMatches(got, c.output, c.rtol, c.atol, c.name)
      })
    }
  })
}
