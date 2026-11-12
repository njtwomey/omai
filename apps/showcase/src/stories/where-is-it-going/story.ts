import type { StoryMeta } from '../../story.ts'

export const story: StoryMeta = {
  title: 'Where is it going?',
  standfirst:
    'A sensor reads the position of a moving object with errors. Expectation propagation estimates where the object was, how fast it was going, and where it will be next, with a band that says how sure it is.',
  section: 'structure',
  tags: ['tracking', 'Kalman', 'state space', 'expectation propagation', 'message passing'],
  date: '2026-11-12',
  hook: 'Forty noisy readings become a path, a velocity and a prediction.',
}
