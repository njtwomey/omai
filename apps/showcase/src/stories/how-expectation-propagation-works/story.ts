import type { StoryMeta } from '../../story.ts'

export const story: StoryMeta = {
  title: 'How expectation propagation works',
  standfirst:
    'One run of omai’s expectation propagation on a small model, from the factor graph to the answer, step by step. Afterwards you can say what each update does and read an EP run for yourself.',
  section: 'inference',
  tags: ['expectation propagation', 'message passing', 'factor graph', 'clutter problem', 'moment matching'],
  date: '2026-11-12',
  hook: 'Every step of one expectation propagation run, with the real numbers.',
}
