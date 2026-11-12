import type { StoryMeta } from '../../story.ts'

export const story: StoryMeta = {
  title: 'Who is better?',
  standfirst:
    'Five players play forty games, and nobody says how good any of them is. Expectation propagation turns the results into a rating for each player, with its doubt.',
  section: 'inference',
  tags: ['ranking', 'skill', 'TrueSkill', 'expectation propagation', 'message passing'],
  date: '2026-11-12',
  hook: 'Forty results become five ratings, each with its own doubt.',
}
