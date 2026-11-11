// The three conjugate atoms' hand-written closed forms against the library's own conjugate table (`validation/exact.ts`):
// the same posterior by an independent path, to 1e-12 in the moments and 1e-9 in the density. (The atoms keep their own
// closed forms: taking the library's makes the check's KS distance 0 become 1e-16, which a bitwise digest of the suite notices.)
import { describe, expect, it } from 'vitest'
import { betaBernoulli, conjugateMarginal, gammaPoisson, normalNormal } from '../validation/index.ts'
import { epAtom } from './ep-atom-list.ts'
import { atomReference } from './ep-atoms.ts'

describe('the conjugate atoms against the library conjugate table', () => {
  const cases = [
    {
      name: 'gaussian',
      support: 'real',
      quantity: 'mu',
      table: () => normalNormal({ priorMean: 0, priorSd: 2, sigma: 1, y: [1.2, 0.4, 1.9] }),
    },
    {
      name: 'beta-bernoulli',
      support: 'unit',
      quantity: 'p',
      table: () => betaBernoulli({ a: 2, b: 2, y: [1, 1, 0, 1, 1, 0, 1] }),
    },
    {
      name: 'gamma-poisson',
      support: 'positive',
      quantity: 'rate',
      table: () => gammaPoisson({ shape: 2, rate: 1, counts: [3, 5, 2, 4] }),
    },
  ] as const

  it.each(cases)('$name', ({ name, support, quantity, table }) => {
    const atom = epAtom(name)
    const problem = atom.problem(atom.defaults)
    const hand = problem.exact?.[quantity]
    const lib = conjugateMarginal(table(), support)
    expect(hand?.mean).toBeCloseTo(lib.mean, 12)
    expect(hand?.sd).toBeCloseTo(lib.sd, 12)
    const ref = atomReference(atom, problem).marginals[quantity]
    for (const x of [0.2, 0.5, 0.8, 1.3, 2.5]) expect(hand?.pdf(x)).toBeCloseTo(lib.pdf(x), 9)
    expect(ref?.x.length).toBeGreaterThan(10)
  })
})
