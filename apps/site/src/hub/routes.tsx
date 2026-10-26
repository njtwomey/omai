import type { ReactNode } from 'react'
import { Card, CardGrid, PageHeader } from '../shell/layout.tsx'
import type { PageRef } from '../shell/PrevNext.tsx'
import { TocSidebar } from '../shell/TocSidebar.tsx'
import type { TocNode } from '../shell/toc.ts'
import { ENTRIES } from './entries.ts'
import { HubPage } from './HubPage.tsx'

const HUB_HOME: PageRef = { title: 'Hub', path: '/hub' }

// The contents mirror the ids: a folder per area, a page per entry.
const HUB_TOC: readonly TocNode[] = [...new Set(ENTRIES.map((e) => e.id.split('/')[0] ?? ''))].map((area) => ({
  id: area,
  title: area,
  children: ENTRIES.filter((e) => e.id.startsWith(`${area}/`)).map((e) => ({
    id: e.id,
    title: e.title,
    path: `/hub/${e.id}`,
  })),
}))

/** The hub section: the index of entries, or one entry's page. */
export function HubSection(props: { readonly path: string }): ReactNode {
  const id = props.path.replace(/\/$/, '').replace(/^\/hub\/?/, '')
  const entry = ENTRIES.find((e) => e.id === id)
  if (entry !== undefined) return <HubPage key={entry.id} entry={entry} />
  return (
    <div className="flex flex-col gap-4">
      <PageHeader
        title="The hub"
        lede="Methods on a contract, run as sessions: pick a dataset, set the hyperparameters, play the fit and scrub its tape."
      />
      <CardGrid>
        {ENTRIES.map((e) => (
          <Card key={e.id} path={`/hub/${e.id}`} title={e.title} text={e.summary} />
        ))}
      </CardGrid>
    </div>
  )
}

/** The hub's sidebar. */
export function HubToc(props: { readonly path: string }): ReactNode {
  return (
    <TocSidebar nodes={HUB_TOC} current={props.path.replace(/\/$/, '')} home={HUB_HOME} placeholder="Filter entries" />
  )
}
