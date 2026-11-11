import { it } from 'vitest'
import * as d from '../probprog/model/dists/index.ts'
import { defineModel, ops } from '../probprog/model/index.ts'
import { atomRun, atomVerdict, type AtomProblem } from './ep-atoms.ts'

const Y = [1.2, 0.4, 1.9, 1.1, 0.7, 1.5]
const C = [3, 5, 2, 4]
function show(name: string, p: AtomProblem): void {
  let v
  try {
    v = atomVerdict(p)
  } catch (e) {
    console.log(name, 'THROWS', String(e).slice(0, 200))
    return
  }
  if (!v.applies) {
    console.log(name, 'REFUSED', v.reason.slice(0, 260))
    return
  }
  const r = atomRun(p, {})
  if (r.status !== 'ran') console.log(name, r.status, (r as never as { reason: string }).reason)
  else console.log(name, 'RAN', r.sweeps, JSON.stringify(Object.fromEntries(Object.entries(r.beliefs).map(([k, b]) => [k, [+b.mean.toFixed(3), +b.sd.toFixed(3)]]))))
}
const P = (definition: never, data: never, names: string[]): AtomProblem => ({ definition, data, equations: [], quantities: names.map((n) => ({ name: n, label: n, support: 'real' as const })) })
it('survey', () => {
  show('IG on variance', P(defineModel('a', (m) => {
    const s2 = m.latent('s2', d.inverseGamma(3, 2)); const mu = m.latent('mu', d.normal(0, 3))
    m.observe('y', d.normal(mu, ops.sqrt(s2)), m.data('y', { dims: ['i'], role: 'response' }))
  }) as never, { y: Y } as never, ['mu', 's2']))
  show('NIG', P(defineModel('a', (m) => {
    const s2 = m.latent('s2', d.inverseGamma(3, 2)); const mu = m.latent('mu', d.normal(0, ops.sqrt(ops.div(s2, 0.5))))
    m.observe('y', d.normal(mu, ops.sqrt(s2)), m.data('y', { dims: ['i'], role: 'response' }))
  }) as never, { y: Y } as never, ['mu', 's2']))
  show('NG via tau', P(defineModel('a', (m) => {
    const tau = m.latent('tau', d.gamma(3, 2)); const mu = m.latent('mu', d.normal(0, ops.div(1, ops.sqrt(ops.mul(0.5, tau)))))
    m.observe('y', d.normal(mu, ops.div(1, ops.sqrt(tau))), m.data('y', { dims: ['i'], role: 'response' }))
  }) as never, { y: Y } as never, ['mu', 'tau']))
  show('exponential obs gamma rate', P(defineModel('a', (m) => {
    const r = m.latent('r', d.gamma(2, 1))
    m.observe('y', d.exponential(r), m.data('y', { dims: ['i'], role: 'response' }))
  }) as never, { y: Y } as never, ['r']))
  show('gamma obs gamma rate', P(defineModel('a', (m) => {
    const r = m.latent('r', d.gamma(2, 1))
    m.observe('y', d.gamma(3, r), m.data('y', { dims: ['i'], role: 'response' }))
  }) as never, { y: Y } as never, ['r']))
  show('gamma-poisson with gamma hyper on rate', P(defineModel('a', (m) => {
    const b = m.latent('b', d.gamma(2, 2)); const r = m.latent('r', d.gamma(2, b))
    m.observe('y', d.poisson(r), m.data('y', { dims: ['i'], support: 'count' }))
  }) as never, { y: C } as never, ['r', 'b']))
  show('poisson lognormal', P(defineModel('a', (m) => {
    const r = m.latent('r', d.logNormal(0, 1))
    m.observe('y', d.poisson(r), m.data('y', { dims: ['i'], support: 'count' }))
  }) as never, { y: C } as never, ['r']))
  show('poisson exp normal hyper', P(defineModel('a', (m) => {
    const mu = m.latent('mu', d.normal(1, 1)); const x = m.latent('x', d.normal(mu, 0.5))
    m.observe('y', d.poisson(ops.exp(x)), m.data('y', { dims: ['i'], support: 'count' }))
  }) as never, { y: C } as never, ['x', 'mu']))
  show('weibull', P(defineModel('a', (m) => {
    const x = m.latent('x', d.normal(0, 1))
    m.observe('y', d.weibull(1.5, ops.exp(x)), m.data('y', { dims: ['i'], role: 'response' }))
  }) as never, { y: Y } as never, ['x']))
  show('beta hyper gamma', P(defineModel('a', (m) => {
    const a = m.latent('a', d.gamma(2, 1)); const p = m.latent('p', d.beta(a, a))
    m.observe('y', d.bernoulli({ probs: p }), m.data('y', { dims: ['i'], support: 'binary' }))
  }) as never, { y: [1, 0, 1, 1] } as never, ['p', 'a']))
  show('beta hyper lognormal', P(defineModel('a', (m) => {
    const a = m.latent('a', d.logNormal(0, 1)); const p = m.latent('p', d.beta(a, ops.add(a, 1)))
    m.observe('y', d.bernoulli({ probs: p }), m.data('y', { dims: ['i'], support: 'binary' }))
  }) as never, { y: [1, 0, 1, 1] } as never, ['p', 'a']))
  show('dirichlet hyper gamma', P(defineModel('a', (m) => {
    const a = m.latent('a', d.gamma(2, 1)); const p = m.latent('p', d.dirichlet(ops.mul(a, m.const([1, 1, 1]))))
    m.observe('y', d.categorical({ probs: p }), m.data('y', { dims: ['i'], support: 'count' }))
  }) as never, { y: [0, 1, 1, 2] } as never, ['p[0]', 'a']))
  show('geometric beta', P(defineModel('a', (m) => {
    const p = m.latent('p', d.beta(2, 2))
    m.observe('y', d.geometric(p), m.data('y', { dims: ['i'], support: 'count' }))
  }) as never, { y: C } as never, ['p']))
  show('studentT scale half', P(defineModel('a', (m) => {
    const mu = m.latent('mu', d.normal(0, 3)); const tau = m.latent('tau', d.gamma(2, 1))
    m.observe('y', d.studentT(4, mu, ops.div(1, ops.sqrt(tau))), m.data('y', { dims: ['i'], role: 'response' }))
  }) as never, { y: Y } as never, ['mu', 'tau']))
  show('logistic normal (logit prior)', P(defineModel('a', (m) => {
    const mu = m.latent('mu', d.normal(0, 1)); const x = m.latent('x', d.normal(mu, 1))
    m.observe('y', d.bernoulli({ logits: x }), m.data('y', { dims: ['i'], support: 'binary' }))
  }) as never, { y: [1, 0, 1, 1] } as never, ['x', 'mu']))
  show('binomial beta', P(defineModel('a', (m) => {
    const p = m.latent('p', d.beta(2, 2))
    m.observe('y', d.binomial(10, p), m.data('y', { dims: ['i'], support: 'count' }))
  }) as never, { y: C } as never, ['p']))
  show('gaussian prior mean hyper', P(defineModel('a', (m) => {
    const mm = m.latent('mm', d.normal(0, 5)); const mu = m.latent('mu', d.normal(mm, 1))
    m.observe('y', d.normal(mu, 1), m.data('y', { dims: ['i'], role: 'response' }))
  }) as never, { y: Y } as never, ['mu', 'mm']))
  show('mvn', P(defineModel('a', (m) => {
    const mu = m.latent('mu', d.multivariateNormal([0, 0], { covariance: [[1, 0.5], [0.5, 1]] }), { dims: ['k'] })
    m.observe('y', d.normal(mu, 1), m.data('y', { dims: ['k'], role: 'response' }))
  }) as never, { y: [1, 2] } as never, ['mu']))
  show('uniform prior', P(defineModel('a', (m) => {
    const mu = m.latent('mu', d.uniform(-2, 2))
    m.observe('y', d.normal(mu, 1), m.data('y', { dims: ['i'], role: 'response' }))
  }) as never, { y: Y } as never, ['mu']))
  show('lognormal obs', P(defineModel('a', (m) => {
    const mu = m.latent('mu', d.normal(0, 1)); const tau = m.latent('tau', d.gamma(2, 1))
    m.observe('y', d.logNormal(mu, ops.div(1, ops.sqrt(tau))), m.data('y', { dims: ['i'], role: 'response' }))
  }) as never, { y: Y } as never, ['mu', 'tau']))
})
