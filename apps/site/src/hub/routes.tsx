import type { ReactNode } from 'react'
import { catalogue, type HubEntry, type HubState, stateFromQuery, stateToQuery } from 'whitebox-hub'
import { PrevNext } from '../shell/PrevNext.tsx'
import { href } from '../shell/router.ts'
import { TocSidebar } from '../shell/TocSidebar.tsx'
import { HUB_HOME, HUB_TOC, isCollection } from './contents.ts'
import { CollectionPage, EntryPage, HubHome } from './pages.tsx'

// The entry's state in the query string, replacing the history entry (so the back button leaves the page).
function keepInUrl(entry: HubEntry, state: HubState, path: string): void {
  if (window.location.pathname !== href(path)) return
  const q = stateToQuery(entry, state)
  window.history.replaceState(window.history.state, '', `${href(path)}${q === '' ? '' : `?${q}`}`)
}

/** The hub section: renders the page for a path under /hub (the front page, a collection or an entry). */
export function HubSection(props: { readonly path: string }): ReactNode {
  const path = props.path.replace(/\/$/, '')
  const rest = decodeURIComponent(path.replace(/^\/hub\/?/, ''))
  const nav = <PrevNext nodes={HUB_TOC} home={HUB_HOME} current={path} compact />
  const entry = catalogue.byId.get(rest)
  const known = entry !== undefined || (rest !== '' && isCollection(rest))
  const page =
    entry !== undefined ? (
      <EntryPage
        entry={entry}
        nav={nav}
        initialState={stateFromQuery(entry, window.location.search)}
        onStateChange={(state) => keepInUrl(entry, state, path)}
      />
    ) : rest !== '' && isCollection(rest) ? (
      <CollectionPage path={rest} nav={nav} />
    ) : (
      <HubHome nav={nav} />
    )
  return (
    <>
      {page}
      <PrevNext nodes={HUB_TOC} home={HUB_HOME} current={known ? path : HUB_HOME.path} />
    </>
  )
}

/** The hub's sidebar. */
export function HubToc(props: { readonly path: string }): ReactNode {
  return (
    <TocSidebar nodes={HUB_TOC} current={props.path.replace(/\/$/, '')} home={HUB_HOME} placeholder="Filter methods" />
  )
}
