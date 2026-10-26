import { lazy, type ReactNode, Suspense } from 'react'
import type { HubIndexEntry } from '../../plugins/hub-index-data.ts'
import { ErrorBoundary } from '../shell/ErrorBoundary.tsx'
import { LazyRecipeView } from '../shell/LazyRecipe.tsx'
import { Link } from '../shell/Link.tsx'
import { CardGrid, PageHeader, PageSection, Prose } from '../shell/layout.tsx'
import { collectionDescription, collectionPath, groupsUnder, type HubItem, ITEM_COUNT } from './contents.ts'
import type { HubPage } from './standalone.ts'

// An entry's widget loads with its page: its module, the hub's React bindings and the engine code it runs.
const LiveEntry = lazy(() => import('./live.tsx').then((m) => ({ default: m.LiveEntry })))

const SOURCE = 'https://github.com/njtwomey/mdml-js/blob/main/packages/engine/src/'

// An item's card: its title and the whole of its text (no clamping, so nothing is cut off).
function EntryCard(props: { readonly entry: HubItem }): ReactNode {
  const { entry } = props
  return (
    <Link
      to={entry.path}
      className="group flex flex-col gap-0.5 rounded-lg border border-line bg-bg px-3 py-2.5 transition-colors hover:border-muted"
    >
      <span className="text-sm font-medium group-hover:text-accent">{entry.title}</span>
      <span className="text-xs leading-snug text-muted">{entry.text}</span>
    </Link>
  )
}

function Groups(props: { readonly path: string; readonly skipTitle?: boolean }): ReactNode {
  return groupsUnder(props.path).map((g) => (
    <PageSection
      key={g.collection}
      id={g.collection.replaceAll('/', '-')}
      title={
        props.skipTitle === true && g.collection === props.path ? (
          'Entries'
        ) : (
          <Link to={collectionPath(g.collection)} className="hover:text-accent">
            {g.collection}
          </Link>
        )
      }
    >
      <CardGrid>
        {g.entries.map((e) => (
          <EntryCard key={e.id} entry={e} />
        ))}
      </CardGrid>
    </PageSection>
  ))
}

/** The hub's front page: every collection with its entries as cards. */
export function HubHome(props: { readonly nav?: ReactNode }): ReactNode {
  return (
    <>
      <PageHeader
        title="The method hub"
        actions={props.nav}
        lede={`${ITEM_COUNT} worked methods and demos, each live on its datasets: pick the data, change the parameters, and the method runs again with the records that explain it. Collections follow the familiar libraries' layout.`}
      />
      <Groups path="" />
    </>
  )
}

/** A collection (or a folder of collections): its entries as cards, grouped by collection. */
export function CollectionPage(props: { readonly path: string; readonly nav?: ReactNode }): ReactNode {
  const { path } = props
  const cut = path.lastIndexOf('/')
  const lede = collectionDescription(path)
  return (
    <>
      <PageHeader
        title={path}
        {...(lede === undefined ? {} : { lede })}
        eyebrow={
          cut < 0
            ? { label: 'Hub', path: '/hub' }
            : { label: path.slice(0, cut), path: collectionPath(path.slice(0, cut)) }
        }
        actions={props.nav}
      />
      <Groups path={path} skipTitle />
    </>
  )
}

/** One entry: its title and summary, links to the methods' docs and source, and the live widget (loaded lazily). */
export function EntryPage(props: {
  readonly entry: HubIndexEntry
  readonly path: string
  readonly nav?: ReactNode
}): ReactNode {
  const { entry, path } = props
  return (
    <>
      <PageHeader
        title={entry.title}
        lede={entry.summary}
        eyebrow={{ label: entry.collection, path: collectionPath(entry.collection) }}
        actions={props.nav}
      />
      <div className="-mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
        {entry.methods.map((m) => (
          <span key={`${m.file}#${m.name}`} className="flex items-center gap-2">
            <Link to={`/docs/${m.file}/${m.name}`} className="font-mono text-fg hover:text-accent">
              {m.name}
            </Link>
            <a href={`${SOURCE}${m.file}`} target="_blank" rel="noreferrer" className="hover:text-fg">
              source
            </a>
          </span>
        ))}
        {entry.tags.length > 0 ? <span>{entry.tags.join(' · ')}</span> : null}
      </div>
      <ErrorBoundary resetKey={entry.id}>
        <Suspense fallback={<p className="text-sm text-muted">Loading the method…</p>}>
          <LiveEntry key={entry.id} meta={entry} path={path} />
        </Suspense>
      </ErrorBoundary>
    </>
  )
}

/** A standalone page: its question, a sentence or two, and the live demo beside its own source. */
export function StandalonePage(props: { readonly page: HubPage; readonly nav?: ReactNode }): ReactNode {
  const { page } = props
  return (
    <>
      <PageHeader
        title={page.title}
        lede={page.question}
        eyebrow={{ label: page.collection, path: collectionPath(page.collection) }}
        actions={props.nav}
      />
      <Prose text={page.explain} className="-mt-5 max-w-3xl text-base leading-relaxed text-fg-secondary" />
      <LazyRecipeView id={page.path} file={page.file} load={page.load} key={page.path} />
    </>
  )
}
