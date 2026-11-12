import type { StoryMeta } from '../../story.ts'

export const story: StoryMeta = {
  title: 'Eight schools and partial pooling',
  standfirst: 'Eight noisy estimates can be read separately or pooled. The data decide how much to pool.',
  section: 'structure',
  tags: ['bayesian', 'hierarchical', 'partial pooling', 'NUTS'],
  date: '2026-11-12',
  hook: 'Eight estimates, each pulled towards the group by an amount that depends on how noisy it is.',
}
