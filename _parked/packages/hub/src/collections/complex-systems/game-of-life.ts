/**
 * Hub entry for Conway's Game of Life, with gliders, oscillators and a glider gun among its starting patterns, played
 * generation by generation.
 */

import { gameOfLife, type LifePattern } from 'mdml-engine/methods/complex-systems'
import { bool, choice, dict, int, real } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'

/** Conway's Game of Life. */
export const gameOfLifeEntry: HubEntry = defineEntry({
  id: 'complex-systems/game-of-life',
  title: 'Game of Life',
  summary: 'Birth on three neighbours, survival on two or three: gliders, oscillators and a gun that fires forever.',
  collection: 'complex-systems',
  tags: ['cellular automata', 'Conway', 'emergence'],
  datasets: [],
  params: dict({
    pattern: choice(['gosper gun', 'glider', 'r-pentomino', 'pulsar', 'random'], {
      default: 'gosper gun',
      label: 'start',
    }),
    generations: int(10, 200, { default: 90, label: 'generations' }),
    wrap: bool({ default: false, label: 'wrap edges (torus)' }),
    density: real(0.05, 0.6, { default: 0.3, label: 'density (random)' }),
  }),
  run: ({ params, key }) =>
    gameOfLife({
      pattern: params['pattern'] as LifePattern,
      generations: Number(params['generations']),
      wrap: params['wrap'] === true,
      density: Number(params['density']),
      key,
    }),
  views: [{ record: 'boardSteps', title: 'The board, generation by generation' }],
  methods: [{ file: 'methods/complex-systems/cellular-automata.ts', name: 'gameOfLife' }],
  references: ['Gardner (1970), The fantastic combinations of John Conway’s new solitaire game “life”.'],
})
