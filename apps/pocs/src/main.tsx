import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { usePath } from '../../site/src/shell/router.ts'
import { Shell } from '../../site/src/shell/Shell.tsx'
import { Crumbs } from './Crumbs.tsx'
import { Index } from './Index.tsx'
import { PocView } from './PocView.tsx'
import './styles.css'

function App() {
  const path = usePath()
  const slug = path.startsWith('/poc/') ? path.slice('/poc/'.length).replace(/\/$/, '') : undefined
  const group = path.startsWith('/group/') ? path.slice('/group/'.length).replace(/\/$/, '') : undefined
  return (
    <Shell
      path={path === '/' || slug !== undefined || group !== undefined ? '/' : path}
      sections={[]}
      showSource={false}
      brand="omai pocs"
      crumbs={<Crumbs group={group} slug={slug} />}
    >
      {slug !== undefined ? (
        <PocView key={slug} slug={slug} />
      ) : (
        <Index key={group ?? ''} {...(group === undefined ? {} : { group })} />
      )}
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
