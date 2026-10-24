import { useSyncExternalStore } from 'react'
import type { LinkState, LinkStore } from '../../figure/linking.ts'

/**
 * The state of a `LinkStore` (from `whitebox-render/figure`), re-rendering when any linked figure changes it: hover,
 * pointer, selection, brush and zoom shared by several figures.
 *
 * @param store - The store the linked figures share.
 * @returns Its current state.
 */
export function useLinkState(store: LinkStore): LinkState {
  return useSyncExternalStore(store.subscribe, store.get, store.get)
}
