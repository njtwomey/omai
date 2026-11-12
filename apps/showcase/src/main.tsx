import { cssText } from 'omai-render/theme'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ColourToggle } from '../../site/src/shell/ColourToggle.tsx'
import { Link } from '../../site/src/shell/Link.tsx'
import { usePath } from '../../site/src/shell/router.ts'
import { Splash } from './Splash.tsx'
import { StoryRoute } from './StoryRoute.tsx'
import { storySlugOf } from './story.ts'
import './styles.css'

// omai-render's theme tokens as CSS variables, light on :root and dark under the `dark` class (as the site's shell writes them).
const THEME_CSS = `${cssText('light', ':root')}\n${cssText('dark', '.dark')}`

function App() {
  const path = usePath()
  const slug = storySlugOf(path)
  return (
    <div className="min-h-screen bg-bg text-fg">
      <style>{THEME_CSS}</style>
      <header className="flex h-12 items-center gap-3 px-4 font-sans sm:px-6">
        <Link to="/" className="flex items-center gap-2 text-sm font-semibold tracking-tight">
          <span className="size-3 rounded-sm bg-gradient-to-br from-[#1f77b4] via-[#2ca02c] to-[#ff7f0e]" />
          omai showcase
        </Link>
        <div className="ml-auto">
          <ColourToggle />
        </div>
      </header>
      <main>{slug === undefined ? <Splash /> : <StoryRoute key={slug} slug={slug} />}</main>
    </div>
  )
}

const root = document.getElementById('root')
if (root === null) throw new Error('index.html has no #root element')
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
