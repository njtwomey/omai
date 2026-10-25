import { lazy, type ReactNode, StrictMode, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import { Home } from './Home.tsx'
import { usePath } from './shell/router.ts'
import { Shell } from './shell/Shell.tsx'
import './shell/styles.css'

// Each section loads its own code when it is first visited: the docs need none of the gallery's recipes or the hub's
// collections, and the other way round.
const gallery = (): Promise<typeof import('./gallery/routes.tsx')> => import('./gallery/routes.tsx')
const hub = (): Promise<typeof import('./hub/routes.tsx')> => import('./hub/routes.tsx')
const docs = (): Promise<typeof import('./docs/routes.tsx')> => import('./docs/routes.tsx')
const GallerySection = lazy(() => gallery().then((m) => ({ default: m.GallerySection })))
const GalleryToc = lazy(() => gallery().then((m) => ({ default: m.GalleryToc })))
const HubSection = lazy(() => hub().then((m) => ({ default: m.HubSection })))
const HubToc = lazy(() => hub().then((m) => ({ default: m.HubToc })))
const DocsSection = lazy(() => docs().then((m) => ({ default: m.DocsSection })))
const DocsToc = lazy(() => docs().then((m) => ({ default: m.DocsToc })))

function App(): ReactNode {
  const path = usePath()
  const page = path.startsWith('/gallery') ? (
    <GallerySection path={path} />
  ) : path.startsWith('/hub') ? (
    <HubSection path={path} />
  ) : path.startsWith('/docs') ? (
    <DocsSection path={path} />
  ) : (
    <Home />
  )
  const sidebar = path.startsWith('/gallery') ? (
    <GalleryToc path={path} />
  ) : path.startsWith('/hub') ? (
    <HubToc path={path} />
  ) : path.startsWith('/docs') ? (
    <DocsToc path={path} />
  ) : undefined
  return (
    <Shell path={path} {...(sidebar === undefined ? {} : { sidebar: <Suspense fallback={null}>{sidebar}</Suspense> })}>
      <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>{page}</Suspense>
    </Shell>
  )
}

const root = document.getElementById('root')
if (root === null) throw new Error('index.html has no #root element')
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
