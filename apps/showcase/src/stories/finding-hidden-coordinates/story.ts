import type { StoryMeta } from '../../story.ts'

export const story: StoryMeta = {
  title: 'Finding hidden coordinates',
  standfirst:
    'Twenty points are each measured five ways. Expectation propagation looks for one hidden number per point that explains all five measurements.',
  section: 'inference',
  tags: ['latent variable model', 'GPLVM', 'Gaussian process', 'expectation propagation', 'message passing'],
  date: '2026-11-12',
  hook: 'Five measurements per point, one hidden number per point, found by message passing.',
}
