import { type ComponentType, lazy, type ReactNode, StrictMode, Suspense, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Home } from './Home.tsx'
import { currentPath, usePath } from './shell/router.ts'
import { Shell } from './shell/Shell.tsx'
import './shell/styles.css'

type PathProps = { readonly path: string }

/**
 * A section's code, loaded when the section is first visited: the docs need none of the gallery's recipes or the hub's
 * collections, and the other way round. The first page's section is loaded before the first render, so that page
 * appears at once rather than after a Suspense fallback (which React keeps up for at least 300 ms).
 */
interface Section<M> {
  /** The module, once loaded. */
  loaded: M | undefined
  readonly preload: () => Promise<M>
}

function section<M>(load: () => Promise<M>): Section<M> {
  const s: Section<M> = {
    loaded: undefined,
    preload: () =>
      load().then((m) => {
        s.loaded = m
        return m
      }),
  }
  return s
}

const gallery = section(() => import('./gallery/routes.tsx'))
const hub = section(() => import('./hub/routes.tsx'))
const docs = section(() => import('./docs/routes.tsx'))
const HubSection = lazy(() => hub.preload().then((m) => ({ default: m.HubSection })))
const HubToc = lazy(() => hub.preload().then((m) => ({ default: m.HubToc })))
const GallerySection = lazy(() => gallery.preload().then((m) => ({ default: m.GallerySection })))
const GalleryToc = lazy(() => gallery.preload().then((m) => ({ default: m.GalleryToc })))
const DocsSection = lazy(() => docs.preload().then((m) => ({ default: m.DocsSection })))
const DocsToc = lazy(() => docs.preload().then((m) => ({ default: m.DocsToc })))

// One of a section's components: rendered directly when the section was loaded as this view mounted, otherwise through
// its `React.lazy` (kept for the view's life, so a later render does not swap the component and lose its state).
function SectionView<M>(props: {
  readonly section: Section<M>
  readonly lazy: ComponentType<PathProps>
  readonly pick: (m: M) => ComponentType<PathProps>
  readonly path: string
}): ReactNode {
  const [loaded] = useState(() => props.section.loaded)
  const View = loaded === undefined ? props.lazy : props.pick(loaded)
  return <View path={props.path} />
}

function App(): ReactNode {
  const path = usePath()
  const page = path.startsWith('/gallery') ? (
    <SectionView key="gallery" section={gallery} lazy={GallerySection} pick={(m) => m.GallerySection} path={path} />
  ) : path.startsWith('/hub') ? (
    <SectionView key="hub" section={hub} lazy={HubSection} pick={(m) => m.HubSection} path={path} />
  ) : path.startsWith('/docs') ? (
    <SectionView key="docs" section={docs} lazy={DocsSection} pick={(m) => m.DocsSection} path={path} />
  ) : (
    <Home />
  )
  const sidebar = path.startsWith('/gallery') ? (
    <SectionView key="gallery" section={gallery} lazy={GalleryToc} pick={(m) => m.GalleryToc} path={path} />
  ) : path.startsWith('/hub') ? (
    <SectionView key="hub" section={hub} lazy={HubToc} pick={(m) => m.HubToc} path={path} />
  ) : path.startsWith('/docs') ? (
    <SectionView key="docs" section={docs} lazy={DocsToc} pick={(m) => m.DocsToc} path={path} />
  ) : undefined
  return (
    <Shell path={path} {...(sidebar === undefined ? {} : { sidebar: <Suspense fallback={null}>{sidebar}</Suspense> })}>
      <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>{page}</Suspense>
    </Shell>
  )
}

const root = document.getElementById('root')
if (root === null) throw new Error('index.html has no #root element')
const first = currentPath()
const preload = first.startsWith('/gallery')
  ? gallery.preload()
  : first.startsWith('/hub')
    ? hub.preload()
    : first.startsWith('/docs')
      ? docs.preload()
      : Promise.resolve()
// A failed preload renders anyway: the section's lazy load then shows the error.
void preload
  .catch(() => undefined)
  .then(() =>
    createRoot(root).render(
      <StrictMode>
        <App />
      </StrictMode>,
    ),
  )
