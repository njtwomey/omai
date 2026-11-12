import { renderToString } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { StoryContext } from './kit/StoryPage.tsx'
import { mismatched, pageOf, STORIES, storyOf } from './registry.ts'
import { SECTIONS } from './sections.ts'

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
    expect(html).toContain('Draw the crowd')
    expect(html).toContain('Nothing has been computed yet')
    expect(html).not.toContain('What came back')
    expect(made).toEqual([])
    // The first import of the story pulls in the chart library and the engine's tensors.
  }, 60_000)
})
