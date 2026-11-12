import type { StoryMeta } from '../../story.ts'

export const story: StoryMeta = {
  title: 'How long until it fails',
  standfirst:
    'Some units fail during a test and some are still working when it ends. A model that uses both gives a survival curve with a band.',
  section: 'uncertainty',
  tags: ['survival', 'censoring', 'Weibull', 'reliability', 'expectation propagation', 'message passing'],
  date: '2026-11-12',
  hook: 'Forty units on test, some still working at the end, and what they tell us about how long a unit lasts.',
}
