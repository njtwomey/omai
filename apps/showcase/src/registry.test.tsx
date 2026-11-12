import type { ComponentType } from 'react'
import { renderToString } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { StoryContext } from './kit/StoryPage.tsx'
import { mismatched, pageOf, STORIES, storyOf } from './registry.ts'
import { SECTIONS } from './sections.ts'

const PAGES = import.meta.glob<{ readonly default: ComponentType }>('./stories/*/Story.tsx')

// A story must not start a worker, or compute anything, by being imported or rendered: only the play button does.
const made: unknown[] = []

beforeEach(() => {
  made.length = 0
  vi.stubGlobal(
    'Worker',
    class {
      constructor(...args: unknown[]) {
        made.push(args)
      }
    },
  )
})
afterEach(() => vi.unstubAllGlobals())

describe('the registry of stories', () => {
  it('lists the pilot, with metadata a card needs', () => {
    const entry = storyOf('bayesian-linear-regression')
    expect(entry?.title).toBe('A line, and how sure we are')
    expect(entry?.standfirst.length).toBeGreaterThan(10)
    expect(SECTIONS.map((s) => s.id)).toContain(entry?.section)
    expect(entry?.date).toMatch(/^\d{4}-\d{2}-\d{2}$/)
  })

  it('has a story.ts and a Story.tsx for every folder, with unique slugs and known sections', () => {
    expect(mismatched()).toEqual([])
    expect(new Set(STORIES.map((s) => s.slug)).size).toBe(STORIES.length)
    for (const s of STORIES) expect(SECTIONS.map((x) => x.id)).toContain(s.section)
    expect(pageOf('no-such-story')).toBeUndefined()
  })

  it('makes no worker by being imported', () => {
    expect(STORIES.length).toBeGreaterThan(0)
    expect(made).toEqual([])
  })

  it('renders the story with the play button idle, no worker made and no result drawn', async () => {
    const { default: Story } = await import('./stories/bayesian-linear-regression/Story.tsx')
    const entry = storyOf('bayesian-linear-regression')
    if (entry === undefined) throw new Error('the pilot is not registered')
    const html = renderToString(
      <StoryContext.Provider value={{ entry, prev: undefined, next: undefined }}>
        <Story />
      </StoryContext.Provider>,
    )
    expect(html).toContain('data-phase="idle"')
    expect(html).toContain('Draw the lines')
    // The page reads straight down: the results block comes right after the play button, holds only its placeholder, and nothing
    // follows it (no code, no closing prose) until a result exists.
    const play = html.indexOf('data-testid="play"')
    const results = html.indexOf('data-testid="results"')
    expect(play).toBeGreaterThan(0)
    expect(results).toBeGreaterThan(play)
    expect(html).toContain('data-state="empty"')
    expect(html).toContain('The result will appear here')
    expect(html).not.toContain('What to take from it')
    expect(html).not.toContain('How it is written')
    expect(made).toEqual([])
    // The first import of the story pulls in the chart library and the engine's tensors.
  }, 60_000)
  it('lays every story out to read straight down: the results block follows the play button, empty before the run', async () => {
    for (const entry of STORIES) {
      const load = PAGES[`./stories/${entry.slug}/Story.tsx`]
      if (load === undefined) throw new Error(`${entry.slug} has no page`)
      const { default: Story } = await load()
      const html = renderToString(
        <StoryContext.Provider value={{ entry, prev: undefined, next: undefined }}>
          <Story />
        </StoryContext.Provider>,
      )
      const play = html.indexOf('data-testid="play"')
      const results = html.indexOf('data-testid="results"')
      expect(play, `${entry.slug}: a play button`).toBeGreaterThan(0)
      expect(results, `${entry.slug}: a results block after the play button`).toBeGreaterThan(play)
      expect(html, `${entry.slug}: nothing drawn before the run`).toContain('data-state="empty"')
      expect(html, `${entry.slug}: code only after the result`).not.toContain('How it is written')
    }
    expect(made).toEqual([])
  }, 120_000)
})
