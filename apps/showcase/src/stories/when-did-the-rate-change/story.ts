import type { StoryMeta } from '../../story.ts'

export const story: StoryMeta = {
  title: 'When did the rate change?',
  standfirst:
    'The yearly count of British coal-mining disasters fell at some point. The answer to when is a distribution over the years.',
  section: 'inference',
  tags: ['changepoint', 'discrete latent', 'Poisson', 'message passing', 'coal mining'],
  date: '2026-11-12',
  hook: 'Not a date but a distribution: when the rate changed, and how sure the counts can be.',
}
