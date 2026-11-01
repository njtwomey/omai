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
  return (
    <Shell path={path === '/' || slug !== undefined ? '/' : path} sections={SECTIONS} brand="mdml pocs">
      {slug === undefined ? <Index /> : <PocView key={slug} slug={slug} />}
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
