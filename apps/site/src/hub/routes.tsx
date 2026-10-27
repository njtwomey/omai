import type { ReactNode } from 'react'
import { Card, CardGrid, PageHeader, PageSection } from '../shell/layout.tsx'
import { type PageRef, PrevNext } from '../shell/PrevNext.tsx'
import { TocSidebar } from '../shell/TocSidebar.tsx'
import type { TocNode } from '../shell/toc.ts'
import { AREAS, ENTRIES, type HubArea, type HubEntry } from './entries.ts'
import { HubPage } from './HubPage.tsx'

const HUB_HOME: PageRef = { title: 'Hub', path: '/hub' }

const areaOf = (entry: HubEntry): string => entry.id.split('/')[0] ?? ''

// The areas in reading order (the listed ones first, then any other an entry names), each with its entries.
const AREA_LIST: readonly { readonly area: HubArea; readonly entries: readonly HubEntry[] }[] = [
  ...AREAS,
  ...[...new Set(ENTRIES.map(areaOf))]
    .filter((id) => !AREAS.some((a) => a.id === id))
    .map((id) => ({ id, title: id, blurb: '' })),
]
  .map((area) => ({ area, entries: ENTRIES.filter((e) => areaOf(e) === area.id) }))
  .filter((group) => group.entries.length > 0)

// The contents mirror the ids: a folder per area, a page per entry.
const HUB_TOC: readonly TocNode[] = AREA_LIST.map(({ area, entries }) => ({
  id: area.id,
  title: area.title,
  children: entries.map((e) => ({ id: e.id, title: e.title, path: `/hub/${e.id}` })),
}))

/** The hub section: the index of entries by area, or one entry's page, with previous and next links. */
export function HubSection(props: { readonly path: string }): ReactNode {
  const id = props.path.replace(/\/$/, '').replace(/^\/hub\/?/, '')
  const entry = ENTRIES.find((e) => e.id === id)
  const current = entry === undefined ? HUB_HOME.path : `/hub/${entry.id}`
  const nav = <PrevNext nodes={HUB_TOC} home={HUB_HOME} current={current} compact />
  return (
    <>
      {entry !== undefined ? <HubPage key={entry.id} entry={entry} nav={nav} /> : <HubIndex nav={nav} />}
      <PrevNext nodes={HUB_TOC} home={HUB_HOME} current={current} />
    </>
  )
}

// The front page: each area with its entries as cards.
function HubIndex(props: { readonly nav: ReactNode }): ReactNode {
  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="The hub"
        lede="Methods on a contract, run as sessions: pick a dataset or a configuration, play the run and scrub its tape."
        actions={props.nav}
      />
      {AREA_LIST.map(({ area, entries }) => (
        <PageSection
          key={area.id}
          id={area.id}
          title={area.title}
          {...(area.blurb === '' ? {} : { blurb: area.blurb })}
        >
          <CardGrid>
            {entries.map((e) => (
              <Card key={e.id} path={`/hub/${e.id}`} title={e.title} text={e.summary} />
            ))}
          </CardGrid>
        </PageSection>
      ))}
    </div>
  )
}

/** The hub's sidebar. */
export function HubToc(props: { readonly path: string }): ReactNode {
  return (
    <TocSidebar nodes={HUB_TOC} current={props.path.replace(/\/$/, '')} home={HUB_HOME} placeholder="Filter entries" />
  )
}
