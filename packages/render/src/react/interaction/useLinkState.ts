/**
 * A hook reading a figure `LinkStore`, re-rendering when any linked figure changes the shared hover, pointer,
 * selection, brush or zoom.
 */

import { useSyncExternalStore } from 'react'
import type { LinkState, LinkStore } from '../../figure/linking.ts'

/**
 * The state of a `LinkStore` (from `omai-render/figure`), re-rendering when any linked figure changes it: hover,
 * pointer, selection, brush and zoom shared by several figures.
 *
 * @param store - The store the linked figures share.
 * @returns Its current state.
 * @index function React hook: read hover, selection and brush shared by linked figures: `useLinkState(store)`
 */
export function useLinkState(store: LinkStore): LinkState {
  return useSyncExternalStore(store.subscribe, store.get, store.get)
}
