import type { HubEntry } from '../../../entry/entry.ts'
import { gameEntry } from './_game.ts'

/** Minimax: back up the best value for the player to move, assuming the opponent does the same. */
export const minimax: HubEntry = gameEntry({
  name: 'minimax',
  title: 'Minimax',
  summary: 'Search the whole game tree depth first, backing up the maximum at MAX nodes and the minimum at MIN nodes.',
  algorithm: 'minimax',
  tags: ['two-player', 'zero-sum'],
  method: { file: 'lib/search/adversarial.ts', name: 'minimax' },
  references: ['Russell and Norvig (2020), Artificial Intelligence: A Modern Approach, section 5.2.'],
})
