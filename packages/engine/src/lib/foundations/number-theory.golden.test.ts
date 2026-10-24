import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as F from './index.ts'

const ints = (t: Tensor): number[] => toFlat(t)
const first = (t: Tensor): number => ints(t)[0] ?? Number.NaN

goldenSuite('number theory against sympy', 'foundations/number-theory', {
  gcd: ([ab]) => tensor([F.gcd(...ints(ab))]),
  bezout: ([ab]) => {
    const [a = 0, b = 0] = ints(ab)
    const e = F.extendedEuclid(a, b)
    return tensor([e.gcd, e.x, e.y])
  },
  lcm: ([xs]) => tensor([F.lcm(...ints(xs))]),
  modInverse: ([am]) => {
    const [a = 0, m = 1] = ints(am)
    return tensor([F.modInverse(a, m)])
  },
  modPow: ([bem]) => {
    const [b = 0, e = 0, m = 1] = ints(bem)
    return tensor([F.modPow(b, e, m).value])
  },
  sieve: ([n]) => tensor(F.sieve(first(n)).primes),
  isPrime: ([ns]) => tensor(ints(ns).map((n) => (F.isPrime(n) ? 1 : 0))),
  factorise: ([n]) => tensor(F.factorise(first(n)).flat()),
  eulerPhi: ([n]) => tensor([F.eulerPhi(first(n))]),
  divisors: ([n]) => tensor(F.divisors(first(n))),
  crt: ([r, m]) => {
    const s = F.chineseRemainder(ints(r), ints(m))
    return tensor(s === null ? [] : [s.x, s.modulus])
  },
  continuedFraction: ([pq]) => {
    const [p = 0, q = 1] = ints(pq)
    return tensor(F.continuedFractionExpansion(p, q))
  },
  convergents: ([cf]) => tensor(F.convergents(ints(cf)).flat()),
  fibonacci: ([n]) => F.fibonacci(first(n)).toString(),
  lucas: ([n]) => F.lucas(first(n)).toString(),
})
