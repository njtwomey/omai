import type { StoryMeta } from '../../story.ts'

export const story: StoryMeta = {
  title: 'A signal in the clutter',
  standfirst:
    'Some readings measure a hidden value and some are garbage. Expectation propagation finds the value, decides which readings to ignore, and shows what a single bell curve cannot say.',
  section: 'inference',
  tags: ['clutter problem', 'outliers', 'expectation propagation', 'message passing', 'robust estimation'],
  date: '2026-11-12',
  hook: 'Twelve readings, some of them garbage, and a method that finds the value without knowing which.',
}
