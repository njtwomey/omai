import type { ReactNode } from 'react'
import { Card, CardGrid, PageHeader, PageSection, PrevNext, Prose } from '../shell/layout.tsx'
import { RecipeView } from '../shell/RecipeView.tsx'
import { SECTIONS } from './recipe.ts'
import { ENTRIES, type Entry, entriesOf } from './registry.ts'

/** The gallery's front page: each section with its recipes as cards. */
export function GalleryHome(): ReactNode {
  return (
    <>
      <PageHeader
        title="The render gallery"
        lede={`${ENTRIES.length} small live recipes for whitebox-render: what it draws, how figures are laid out, and how its controls, interactions and players work. Each page shows the figure beside the code that makes it.`}
      />
      {SECTIONS.map((s) => {
        const items = entriesOf(s.id)
        if (items.length === 0) return null
        return (
          <PageSection key={s.id} id={s.id} title={s.title} blurb={s.blurb}>
            <CardGrid>
              {items.map((e) => (
                <Card key={e.path} path={e.path} title={e.title} text={e.question} />
              ))}
            </CardGrid>
          </PageSection>
        )
      })}
    </>
  )
}

/** One section: its recipes. */
export function SectionPage(props: { readonly section: (typeof SECTIONS)[number] }): ReactNode {
  const { section } = props
  const items = entriesOf(section.id)
  return (
    <>
      <PageHeader title={section.title} lede={section.blurb} eyebrow={{ label: 'Gallery', path: '/gallery' }} />
      {items.length === 0 ? (
        <p className="text-sm text-muted">No recipes here yet.</p>
      ) : (
        <CardGrid>
          {items.map((e) => (
            <Card key={e.path} path={e.path} title={e.title} text={e.question} />
          ))}
        </CardGrid>
      )}
    </>
  )
}

/** One recipe: the question, a sentence or two, the live example beside its own source, and the neighbours. */
export function EntryPage(props: { readonly entry: Entry }): ReactNode {
  const { entry } = props
  const section = SECTIONS.find((s) => s.id === entry.section)
  const at = ENTRIES.indexOf(entry)
  const prev = ENTRIES[at - 1]
  const next = ENTRIES[at + 1]
  const { Example } = entry
  return (
    <>
      <PageHeader
        title={entry.question}
        eyebrow={{ label: section?.title ?? entry.section, path: `/gallery/${entry.section}` }}
      />
      <Prose text={entry.explain} className="-mt-5 max-w-3xl text-base leading-relaxed text-fg-secondary" />
      <RecipeView code={entry.snippet} file={entry.file} key={entry.path}>
        <Example />
      </RecipeView>
      <PrevNext
        {...(prev === undefined ? {} : { prev: { path: prev.path, title: prev.title } })}
        {...(next === undefined ? {} : { next: { path: next.path, title: next.title } })}
      />
    </>
  )
}
