import type { Poc } from '../../../poc.ts'

export const poc: Poc = {
  title: 'Parallel tempering on two modes',
  summary:
    'A target with two modes eight standard deviations apart. NUTS chains each settle in one mode and never leave it; parallel tempering runs hotter replicas that cross the barrier and swap down, and the chains share the two modes equally. Watch the traces, the weight of each mode in each chain, and the swap rate between neighbouring temperatures.',
  tags: [
    'parallel tempering',
    'replica exchange',
    'multimodal',
    'bimodal',
    'mode weights',
    'nuts',
    'swap rate',
    'mcmc',
    'chain diagnostics',
  ],
  added: '2026-11-10',
}
