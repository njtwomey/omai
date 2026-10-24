import { type ReactNode, StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { DocsSection, DocsToc } from './docs/routes.tsx'
import { GallerySection, GalleryToc } from './gallery/routes.tsx'
import { Home } from './Home.tsx'
import { HubSection } from './hub/routes.tsx'
import { usePath } from './shell/router.ts'
import { Shell } from './shell/Shell.tsx'
import './shell/styles.css'

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
  ) : path.startsWith('/docs') ? (
    <DocsToc path={path} />
  ) : undefined
  return (
    <Shell path={path} {...(sidebar === undefined ? {} : { sidebar })}>
      {page}
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
