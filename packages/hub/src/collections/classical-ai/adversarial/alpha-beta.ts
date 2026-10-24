import type { HubEntry } from '../../../entry/entry.ts'
import { gameEntry } from './_game.ts'

/** Alpha-beta pruning: minimax that skips branches which cannot change the decision. */
export const alphaBeta: HubEntry = gameEntry({
  name: 'alpha-beta',
  title: 'Alpha-beta pruning',
  summary: 'Minimax with bounds on what each player can already secure: branches outside them are pruned unseen.',
  algorithm: 'alpha-beta',
  tags: ['two-player', 'zero-sum', 'pruning'],
  method: { file: 'lib/search/adversarial.ts', name: 'alphaBeta' },
  references: ['Knuth and Moore (1975), An analysis of alpha-beta pruning.'],
})
