import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { usePath } from '../../site/src/shell/router.ts'
import { Shell } from '../../site/src/shell/Shell.tsx'
import { Crumbs } from './Crumbs.tsx'
import { Index } from './Index.tsx'
import { PocView } from './PocView.tsx'
import './styles.css'

/** The directory path of `/dir/<a>/<b>` (or `/group/<slug>`, the old address of a top-level directory), or undefined for other paths. */
function dirPathOf(path: string): string[] | undefined {
  const m = /^\/(?:dir|group)\/(.+?)\/?$/.exec(path)
  return m === null ? undefined : (m[1] as string).split('/')
}

function App() {
  const path = usePath()
  const slug = path.startsWith('/poc/') ? path.slice('/poc/'.length).replace(/\/$/, '') : undefined
  const dir = dirPathOf(path)
  return (
    <Shell
      path={path === '/' || slug !== undefined || dir !== undefined ? '/' : path}
      sections={[]}
      showSource={false}
      brand="omai pocs"
      crumbs={<Crumbs path={dir ?? []} slug={slug} />}
    >
      {slug !== undefined ? <PocView key={slug} slug={slug} /> : <Index key={dir?.join('/') ?? ''} path={dir ?? []} />}
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
