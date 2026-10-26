import type { ReactNode } from 'react'
import { PrevNext } from '../shell/PrevNext.tsx'
import { TocSidebar } from '../shell/TocSidebar.tsx'
import { entryById, HUB_HOME, HUB_TOC, isCollection } from './contents.ts'
import { CollectionPage, EntryPage, HubHome, StandalonePage } from './pages.tsx'
import { pageById } from './standalone.ts'

/** The hub section: renders the page for a path under /hub (the front page, a collection or an entry). */
export function HubSection(props: { readonly path: string }): ReactNode {
  const path = props.path.replace(/\/$/, '')
  const rest = decodeURIComponent(path.replace(/^\/hub\/?/, ''))
  const nav = <PrevNext nodes={HUB_TOC} home={HUB_HOME} current={path} compact />
  const entry = entryById.get(rest)
  const standalone = pageById.get(rest)
  const known = entry !== undefined || standalone !== undefined || (rest !== '' && isCollection(rest))
  const page =
    standalone !== undefined ? (
      <StandalonePage page={standalone} nav={nav} />
    ) : entry !== undefined ? (
      <EntryPage entry={entry} path={path} nav={nav} />
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
