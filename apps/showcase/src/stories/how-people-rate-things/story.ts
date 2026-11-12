import type { StoryMeta } from '../../story.ts'

export const story: StoryMeta = {
  title: 'How people rate things',
  standfirst:
    'Fifteen people give scores from 1 to 5 to twenty items. Expectation propagation separates how good each item is from how generous each rater is, and learns where the scale’s cut points lie.',
  section: 'inference',
  tags: ['ratings', 'ordinal', 'ordered probit', 'cut points', 'expectation propagation', 'message passing'],
  date: '2026-11-12',
  hook: 'Ratings from 1 to 5 become a quality for each item, a generosity for each rater and the cut points of the scale.',
}
