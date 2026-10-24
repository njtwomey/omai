import { goldenSuite, type Params } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import {
  besselProcess,
  cir,
  cirBondPrice,
  type Diffusion,
  geometricBrownianMotion,
  type MertonParams,
  mertonLogReturnDensity,
  mertonPrice,
  ornsteinUhlenbeck,
  vasicekBondPrice,
} from './diffusions.ts'

function req<T>(f: T | undefined): T {
  if (f === undefined) throw new Error('missing closed form')
  return f
}

const n = (p: Params, k: string): number => p[k] as number
const ou = (p: Params) => ornsteinUhlenbeck({ theta: n(p, 'theta'), mu: n(p, 'mu'), sigma: n(p, 'sigma') })
const gbm = (p: Params) => geometricBrownianMotion({ mu: n(p, 'mu'), sigma: n(p, 'sigma') })
const cirParams = (p: Params) => ({ kappa: n(p, 'kappa'), theta: n(p, 'theta'), sigma: n(p, 'sigma') })
const merton = (p: Params): MertonParams => ({
  mu: n(p, 'mu'),
  sigma: n(p, 'sigma'),
  rate: n(p, 'rate'),
  jumpMean: n(p, 'jumpMean'),
  jumpStd: n(p, 'jumpStd'),
})
const transition = (d: Diffusion, p: Params) => req(d.transition)(tensor(n(p, 'x')), n(p, 'dt'))
const mean = (d: Diffusion, p: Params) => req(d.mean)(n(p, 'x'), n(p, 'dt'))
const variance = (d: Diffusion, p: Params) => req(d.variance)(n(p, 'x'), n(p, 'dt'))

goldenSuite('stochastic/diffusions against scipy.stats and solve_ivp', 'stochastic/diffusions', {
  'ou-transition': ([y], p) => transition(ou(p), p).logProb(y),
  'ou-mean': (_i, p) => mean(ou(p), p),
  'ou-variance': (_i, p) => variance(ou(p), p),
  'ou-stationary': ([y], p) => ou(p).stationary().logProb(y),
  'gbm-transition': ([y], p) => transition(gbm(p), p).logProb(y),
  'gbm-mean': (_i, p) => mean(gbm(p), p),
  'gbm-variance': (_i, p) => variance(gbm(p), p),
  'cir-transition': ([y], p) => transition(cir(cirParams(p)), p).logProb(y),
  'cir-mean': (_i, p) => mean(cir(cirParams(p)), p),
  'cir-variance': (_i, p) => variance(cir(cirParams(p)), p),
  'cir-stationary': ([y], p) => cir(cirParams(p)).stationary().logProb(y),
  'cir-bond': (_i, p) => cirBondPrice(n(p, 'r0'), n(p, 'maturity'), cirParams(p)),
  'vasicek-bond': (_i, p) =>
    vasicekBondPrice(n(p, 'r0'), n(p, 'maturity'), { a: n(p, 'kappa'), b: n(p, 'theta'), sigma: n(p, 'sigma') }),
  'bessel-transition': ([y], p) => transition(besselProcess({ delta: n(p, 'delta') }), p).logProb(y),
  'merton-price': (_i, p) => mertonPrice(n(p, 'spot'), n(p, 'strike'), n(p, 'maturity'), n(p, 'r'), merton(p)),
  'merton-density': ([x], p) => mertonLogReturnDensity(x, n(p, 't'), merton(p)),
})
