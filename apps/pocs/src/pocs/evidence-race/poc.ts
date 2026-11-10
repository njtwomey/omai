import type { Poc } from '../../poc.ts'

export const poc: Poc = {
  title: 'Evidence race: estimating a marginal likelihood',
  summary:
    'Six estimators of the log marginal likelihood of a conjugate model, whose exact value is known: nested sampling, bridge sampling, the stepping stone, thermodynamic integration, importance sampling and tempered SMC. They run together within a time budget; watch each estimate with its error bar close in on the truth, and which ones get there first.',
  tags: [
    'marginal likelihood',
    'evidence',
    'nested sampling',
    'bridge sampling',
    'stepping stone',
    'thermodynamic integration',
    'importance sampling',
    'tempered smc',
    'model comparison',
  ],
  added: '2026-11-10',
  group: 'probabilistic-programming',
}
