// A search method on the contract, the trivial end: A* on a grid map through the core's search algorithm, its native
// expansion messages and counts renamed to paths, and the map with the current node and the path so far derived as a
// frame. No serialise: there is nothing to seek in a search you can rerun in a blink.
import { gridProblem, SEARCH_MAPS, type SearchMapName, searchMap } from 'mdml-engine/methods/gofai'
import { aStarSearchAlgorithm, type SearchNode, type SearchState } from 'mdml-engine/search'
import { choice, dict } from 'mdml-engine/space'
import { defineMethod, derived, Log, type Method, message, scalar } from 'mdml-engine/trace'
import { toScene } from 'mdml-engine/worlds'

export type AStarConfig = { readonly map: SearchMapName }
export type AStarState = SearchState<number, number>

const pathOf = (node: SearchNode<number, number> | null): number[] => {
  const out: number[] = []
  for (let n = node; n !== null; n = n.parent) out.push(n.state)
  return out.reverse()
}

export const aStarMethod: Method<AStarState, AStarConfig> = defineMethod<AStarState, AStarConfig>({
  name: 'poc/a-star',
  config: dict({ map: choice(Object.keys(SEARCH_MAPS) as SearchMapName[], { default: 'detour' }) }),
  make: (c) => aStarSearchAlgorithm(gridProblem(searchMap(c.map))),
  contract: (c) => {
    const world = searchMap(c.map)
    return {
      'search/expanded': scalar({ from: 'expanded', description: 'nodes expanded so far' }),
      'search/frontier': scalar({ from: 'frontier', description: 'nodes on the frontier' }),
      'search/explored': scalar({ from: 'explored', description: 'states in the explored set' }),
      'search/expand': message({ from: 'expand', description: 'the node expanded this step, with g and h' }),
      'search/solution': message({ from: 'solution', description: 'the path found and its cost' }),
      'world/frame': derived({
        description: 'the map with the node being expanded and the path to it',
        compute: (s) =>
          toScene(world, {
            cell: 24,
            ...(s.current === null ? {} : { agent: s.current.state }),
            path: pathOf(s.solution ?? s.current),
          }),
      }),
    }
  },
  defaults: { 'search/frontier': Log(), 'search/explored': Log() },
})
