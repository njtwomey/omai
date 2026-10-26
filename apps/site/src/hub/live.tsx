/**
 * The live part of a hub entry's page, loaded when the page is opened: the entry's own module (through
 * `mdml-hub/lazy`, never the whole catalogue), its widget, and its state kept in the URL. The page's header comes
 * from the hub index before any of this loads.
 */

import { type HubEntry, type HubState, isHubEntry, stateFromQuery, stateToQuery } from 'mdml-hub/entry'
import { entryModules } from 'mdml-hub/lazy'
import { HubEntryView } from 'mdml-hub/react'
import { type ReactNode, use } from 'react'
import type { HubIndexEntry } from '../../plugins/hub-index-data.ts'
import { href } from '../shell/router.ts'

// The entry's state in the query string, replacing the history entry (so the back button leaves the page).
function keepInUrl(entry: HubEntry, state: HubState, path: string): void {
  if (window.location.pathname !== href(path)) return
  const q = stateToQuery(entry, state)
  window.history.replaceState(window.history.state, '', `${href(path)}${q === '' ? '' : `?${q}`}`)
}

const loading = new Map<string, Promise<HubEntry>>()

// The entry, from its module (once; a failed load is tried again on the next render).
function load(meta: HubIndexEntry): Promise<HubEntry> {
  let pending = loading.get(meta.id)
  if (pending === undefined) {
    const loader = entryModules[meta.module]
    pending = (loader === undefined ? Promise.reject(new Error(`no hub module '${meta.module}'`)) : loader()).then(
      (mod) => {
        const entry = Object.values(mod).find((v): v is HubEntry => isHubEntry(v) && v.id === meta.id)
        if (entry === undefined) throw new Error(`hub module '${meta.module}' has no entry '${meta.id}'`)
        return entry
      },
    )
    pending.catch(() => loading.delete(meta.id))
    loading.set(meta.id, pending)
  }
  return pending
}

/**
 * An entry's live widget, suspending until its module has loaded.
 *
 * @param props - The entry's index record and its page path.
 * @returns The widget.
 */
export function LiveEntry(props: { readonly meta: HubIndexEntry; readonly path: string }): ReactNode {
  const { meta, path } = props
  const entry = use(load(meta))
  return (
    <HubEntryView
      key={entry.id}
      entry={entry}
      module={meta.module}
      initialState={stateFromQuery(entry, window.location.search)}
      onStateChange={(state) => keepInUrl(entry, state, path)}
    />
  )
}
