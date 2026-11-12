import type { PocDir } from '../../poc.ts'

export const dir: PocDir = {
  title: 'Expectation propagation',
  summary:
    'The atomic problems a suite of expectation propagation solvers has to get right: one small model each, run by EP, checked against the exact posterior, with the factor it needs named. Problems EP cannot run yet are listed with the reason.',
  order: 20,
}
