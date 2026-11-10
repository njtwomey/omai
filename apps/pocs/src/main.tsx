import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { usePath } from '../../site/src/shell/router.ts'
import { Shell } from '../../site/src/shell/Shell.tsx'
import { Index } from './Index.tsx'
import { PocView } from './PocView.tsx'
import './styles.css'

const SECTIONS = [{ path: '/', title: 'Proofs of concept' }] as const

function App() {
  const path = usePath()
  const slug = path.startsWith('/poc/') ? path.slice('/poc/'.length).replace(/\/$/, '') : undefined
  const group = path.startsWith('/group/') ? path.slice('/group/'.length).replace(/\/$/, '') : undefined
  return (
    <Shell
      path={path === '/' || slug !== undefined || group !== undefined ? '/' : path}
      sections={SECTIONS}
      brand="omai pocs"
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
