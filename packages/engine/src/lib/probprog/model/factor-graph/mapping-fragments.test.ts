// What each mapping emits, as the operators of the factors of a site: one atom per mapping, compiled (no run), with the fragment its
// site makes. The accuracy of the mappings is in the tests by concern; this says which factors they stand on.
import { describe, expect, it } from 'vitest'
import { fragmentOf } from '../../../../testing/factor-graph.ts'
import { atomBound, epAtoms } from '../../ep/index.ts'
import { toFactorGraph } from '../factors.ts'

/** The distinct operators of a site's factors, sorted. */
function operatorsOfSite(atom: string, site: string): string[] {
  const a = epAtoms.find((x) => x.name === atom)
  if (a === undefined) throw new Error(`no atom ${atom}`)
  const c = toFactorGraph(atomBound(a.problem(a.defaults)))
  expect(c.issues).toEqual([])
  return [...new Set(fragmentOf(c, site))].sort()
}

describe('the factors a mapping makes for its site', () => {
  const cases: readonly (readonly [mapping: string, atom: string, site: string, operators: readonly string[]])[] = [
    ['plate', 'probit', 'y', ['probit-plate']],
    ['normal (known scale)', 'gaussian', 'y', ['gaussian']],
    ['normal-log-scale', 'log-scale', 'y', ['gaussian-log-scale', 'times-constant']],
    ['normal-random-precision', 'gamma-precision', 'y', ['gaussian-gamma-precision']],
    ['normal (shared precision latent)', 'shared-precision', 'theta', ['gaussian-gamma-precision']],
    ['normal-random-scale', 'half-cauchy', 'y', ['gaussian-scale']],
    ['projected-prior', 'half-cauchy', 'sigma', ['apply', 'prior-site']],
    ['heavy-tailed', 'student-t', 'y', ['student-t-site']],
    ['ordered-numeric', 'ordinal-probit', 'y', ['likelihood-site', 'linear']],
    ['binomial-poisson-numeric', 'poisson-log', 'y', ['likelihood-site']],
    ['multiclass-logits', 'softmax-regression', 'y', ['multiclass-site', 'times-constant']],
    ['categorical-conjugate', 'prior-dirichlet-categorical', 'y', ['discrete']],
    ['beta-random-shapes', 'prior-beta-two-shapes', 'p', ['beta-two-shapes']],
    ['bernoulli', 'prior-beta-two-shapes', 'y', ['bernoulli']],
    ['discrete (stick-breaking)', 'stick-breaking', 'z', ['stick-breaking']],
    ['normal-switched (a gate over a mean)', 'gate-random', 'y', ['gaussian', 'select']],
    ['mvn-wishart-observed', 'mvn-wishart', 'y', ['mvn-wishart']],
    ['hurdle-poisson', 'hurdle-poisson', 'y', ['bernoulli', 'likelihood-site', 'linear']],
  ]
  it.each(cases)('%s: atom %s, site %s', (_mapping, atom, site, operators) => {
    expect(operatorsOfSite(atom, site)).toEqual([...operators].sort())
  })
})
