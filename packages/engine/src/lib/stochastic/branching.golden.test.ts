import { goldenSuite } from '../../testing/suite.ts'
import {
  birthDeathStationary,
  extinctionByGeneration,
  extinctionProbability,
  linearBirthDeath,
  yulePmf,
} from './branching.ts'

goldenSuite('stochastic/branching against scipy', 'stochastic/branching', {
  extinction: ([pmf]) => extinctionProbability(pmf),
  'by-generation': ([pmf], p) => extinctionByGeneration(pmf, p['steps'] as number),
  'birth-death-stationary': (_i, p) =>
    birthDeathStationary({ birth: (n) => 2 / (n + 1), death: (n) => n }, p['n'] as number),
  'linear-bd': ([t], p) => linearBirthDeath({ lambda: p['lam'] as number, mu: p['mu'] as number }).extinctionBy(t),
  yule: ([k], p) => yulePmf(k, p['t'] as number, p['lam'] as number),
})
