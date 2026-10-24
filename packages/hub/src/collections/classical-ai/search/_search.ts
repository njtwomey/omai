// Path-search hub entries (private to this collection): a map or a generated maze, the search and its settings as
// controls, and the search played expansion by expansion.
import {
  gridSearch,
  mazeWorld,
  type PathSearchName,
  type SearchMapName,
  searchMap,
} from 'whitebox-engine/methods/gofai'
import { choice, dict, int, real, type Space } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'

/** What a path-search entry declares. */
export interface SearchEntrySpec {
  readonly name: string
  readonly title: string
  readonly summary: string
  readonly algorithm: PathSearchName
  /** The map to start on (default the maze). */
  readonly map?: 'maze' | SearchMapName
  readonly tags: readonly string[]
  /** Extra controls of this search. */
  readonly extra?: Readonly<Record<string, Space>>
  readonly method: string
  readonly references?: readonly string[]
}

/**
 * The hub entry of a path search on a grid world.
 *
 * @param spec - The search, the map it starts on and its texts.
 * @returns The entry.
 */
export function searchEntry(spec: SearchEntrySpec): HubEntry {
  return defineEntry({
    id: `classical-ai/search/${spec.name}`,
    title: spec.title,
    summary: spec.summary,
    collection: 'classical-ai/search',
    tags: ['search', 'grid world', ...spec.tags],
    datasets: [],
    params: dict({
      map: choice(['maze', 'open', 'detour', 'river', 'rooms'], { default: spec.map ?? 'maze', label: 'map' }),
      size: int(4, 14, { default: 8, label: 'maze size (rows of cells)' }),
      braid: real(0, 1, { default: 0, label: 'maze loops (braiding)' }),
      connectivity: choice([4, 8], { default: 4, label: 'neighbours' }),
      ...spec.extra,
    }),
    run: ({ params, key }) => {
      const size = Number(params['size'])
      const map = String(params['map'])
      const w =
        map === 'maze'
          ? mazeWorld(key, { rows: size, cols: Math.round(size * 1.5), braid: Number(params['braid']) })
          : searchMap(map as SearchMapName)
      return gridSearch(w, {
        algorithm: spec.algorithm,
        connectivity: Number(params['connectivity']) === 8 ? 8 : 4,
        ...(params['weight'] === undefined ? {} : { weight: Number(params['weight']) }),
        ...(params['metric'] === undefined ? {} : { metric: params['metric'] as 'manhattan' }),
      })
    },
    views: [{ record: 'boardSteps', title: 'The search, expansion by expansion' }],
    methods: [
      { file: 'methods/gofai/search.ts', name: 'gridSearch' },
      { file: spec.method.split('#')[0] ?? '', name: spec.method.split('#')[1] ?? '' },
    ],
    ...(spec.references === undefined ? {} : { references: spec.references }),
  })
}
