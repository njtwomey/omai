import type { StoryMeta } from '../../story.ts'

export const story: StoryMeta = {
  title: 'Combining two measurements',
  standfirst:
    'A room is measured twice, with two instruments. Expectation propagation combines the readings by passing messages.',
  section: 'inference',
  tags: ['expectation-propagation', 'messages', 'gaussian', 'factor-graph'],
  date: '2026-11-12',
  hook: 'Three bell curves are multiplied into one narrower curve, and the messages that do it are shown.',
}
