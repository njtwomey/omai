import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Shell } from 'whitebox-shell'
import 'whitebox-shell/styles.css'

const root = document.getElementById('root')
if (root === null) throw new Error('index.html has no #root element')
createRoot(root).render(
  <StrictMode>
    <Shell site="docs">
      <h1 className="text-2xl font-semibold">The engine documentation</h1>
    </Shell>
  </StrictMode>,
)
