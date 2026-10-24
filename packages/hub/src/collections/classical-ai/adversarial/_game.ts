// Game-tree search hub entries (private to this collection): a game, the search and its settings as controls, and
// the search played node by node over the whole tree.
import { playGameSearch, randomTree, textbookTree, ticTacToe } from 'whitebox-engine/methods/gofai'
import { bool, choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry, type MethodRef } from '../../../entry/entry.ts'

/** Tic-tac-toe positions small enough to draw their whole game tree. */
const POSITIONS: Readonly<Record<string, string>> = {
  'tic-tac-toe (O to move)': 'XOX.O..X.',
  'tic-tac-toe (X to move)': 'XO..O.X.X',
}

/** What a game-search entry declares. */
export interface GameEntrySpec {
  readonly name: string
  readonly title: string
  readonly summary: string
  readonly algorithm: 'minimax' | 'alpha-beta'
  readonly tags: readonly string[]
  readonly method: MethodRef
  readonly references?: readonly string[]
}

/**
 * The hub entry of a game-tree search.
 *
 * @param spec - The search and its texts.
 * @returns The entry.
 */
export function gameEntry(spec: GameEntrySpec): HubEntry {
  return defineEntry({
    id: `classical-ai/adversarial/${spec.name}`,
    title: spec.title,
    summary: spec.summary,
    collection: 'classical-ai/adversarial',
    tags: ['adversarial search', 'game tree', ...spec.tags],
    datasets: [],
    params: dict({
      game: choice(['textbook (AIMA 5.2)', 'random tree', ...Object.keys(POSITIONS)], {
        default: 'random tree',
        label: 'game',
      }),
      branching: int(2, 4, { default: 3, label: 'moves per node (random tree)' }),
      depth: int(2, 4, { default: 3, label: 'plies (random tree)' }),
      ...(spec.algorithm === 'alpha-beta'
        ? { ordered: bool({ default: false, label: 'order moves best first' }) }
        : {}),
    }),
    run: ({ params, key }) => {
      const name = String(params['game'])
      const options = { algorithm: spec.algorithm, ordered: params['ordered'] === true }
      if (name === 'random tree')
        return playGameSearch(randomTree(key, Number(params['branching']), Number(params['depth'])), options)
      const position = POSITIONS[name]
      if (position !== undefined) return playGameSearch(ticTacToe(position), options)
      return playGameSearch(textbookTree(), options)
    },
    views: [{ record: 'boardSteps', title: 'The game tree, node by node' }],
    methods: [{ file: 'methods/gofai/adversarial.ts', name: 'playGameSearch' }, spec.method],
    ...(spec.references === undefined ? {} : { references: spec.references }),
  })
}
