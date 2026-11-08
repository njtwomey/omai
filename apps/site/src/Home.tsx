import type { ReactNode } from 'react'
import { docsPath, useDocs } from './docs/data.ts'
import { ENTRIES, SECTIONS as GALLERY_SECTIONS } from './gallery/registry.ts'
import { Carousel, type Slide } from './home/Carousel.tsx'
import { EXAMPLE_SLIDES } from './home/examples.tsx'
import { HeroGp } from './home/HeroGp.tsx'
import { Thumbnail } from './home/Thumbnail.tsx'
import { Link } from './shell/Link.tsx'

export const REPOSITORY = 'https://github.com/njtwomey/omai'

const BUILT_ON = ['React', 'Apache ECharts', 'Tailwind CSS', 'KaTeX', 'CodeMirror']

/** The carousel's slides: figures made for this page, each moving on its own while it is shown. */
const SLIDES: readonly Slide[] = EXAMPLE_SLIDES

/** A section's heading: a kicker, the title and a line under it; the title links to the section's id. */
function SectionHead(props: {
  readonly id: string
  readonly kicker: string
  readonly title: string
  readonly children: ReactNode
}): ReactNode {
  return (
    <div id={props.id} className="max-w-2xl scroll-mt-20 space-y-2">
      <div className="text-xs font-medium tracking-widest text-muted uppercase">{props.kicker}</div>
      <h2 className="text-3xl font-semibold tracking-tight">{props.title}</h2>
      <p className="text-muted">{props.children}</p>
    </div>
  )
}

const BUTTON = 'inline-flex items-center gap-1.5 rounded-md px-4 py-2 text-sm font-medium transition-colors'

/** The landing page: a live figure, the library in numbers, more live figures, the packages and the gallery. */
export function Home(): ReactNode {
  return (
    <div className="mx-auto flex w-full max-w-[1200px] flex-col gap-20 px-1 pt-4 pb-12 md:px-4">
      <Hero />
      <Stats />
      {SLIDES.length === 0 ? null : <Showcase />}
      <Packages />
      <GallerySections />
      <footer className="border-t border-line pt-6 text-sm text-muted">
        Built on {BUILT_ON.join(', ')}. Source on{' '}
        <a href={REPOSITORY} className="text-fg underline underline-offset-4 hover:no-underline">
          GitHub
        </a>
        .
      </footer>
    </div>
  )
}

function Hero(): ReactNode {
  return (
    <section className="relative grid grid-cols-1 items-start gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
      <div className="space-y-6">
        <div className="inline-flex items-center gap-2 rounded-full border border-line px-3 py-1 text-xs text-muted">
          <span className="size-1.5 animate-pulse rounded-full bg-fg" />
          Everything on this site runs in your browser
        </div>
        <h1 className="text-4xl leading-[1.05] font-semibold tracking-tight md:text-6xl">
          Observable methods in machine learning.
        </h1>
        <p className="max-w-prose text-base text-muted md:text-lg">
          omai is a machine learning library in which each method records what it does, step by step, so a run can be
          watched, replayed and drawn. The figures here run the library itself: drag a point and the model refits.
          Nothing is computed on a server.
        </p>
        <div className="flex flex-wrap gap-2">
          <Link to="/gallery" className={`${BUTTON} bg-fg text-bg hover:opacity-90`}>
            Browse the gallery <span aria-hidden="true">→</span>
          </Link>
          <Link to="/docs" className={`${BUTTON} border border-line hover:bg-surface`}>
            Read the docs
          </Link>
        </div>
      </div>
      <div className="rounded-xl border border-line bg-bg p-3 shadow-lg shadow-black/5 dark:shadow-black/40">
        <HeroGp />
      </div>
    </section>
  )
}

function Stats(): ReactNode {
  const docs = useDocs()
  const exports = docs === null ? undefined : docs.files.reduce((n, f) => n + f.symbols.length, 0)
  const domains = docs?.folder('lib')?.folders.length
  const stats = [
    { value: ENTRIES.length, label: 'gallery recipes' },
    { value: exports, label: 'documented exports' },
    { value: domains, label: 'engine domains' },
    { value: 0, label: 'servers involved' },
  ]
  return (
    <section className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line md:grid-cols-4">
      {stats.map((s) => (
        <div key={s.label} className="bg-bg px-5 py-5">
          <div className="font-mono text-3xl font-semibold tracking-tight tabular-nums md:text-4xl">
            {s.value === undefined ? '…' : s.value.toLocaleString()}
          </div>
          <div className="text-sm text-muted">{s.label}</div>
        </div>
      ))}
    </section>
  )
}

function Showcase(): ReactNode {
  return (
    <section className="space-y-6">
      <SectionHead id="examples" kicker="Examples" title="Programs that draw their own figures">
        Each figure here is made by a short program in the repository&rsquo;s examples folder, which also keeps the tape
        of the run it was drawn from.
      </SectionHead>
      <Carousel slides={SLIDES} />
    </section>
  )
}

const CARDS = [
  {
    name: 'omai-engine',
    role: 'Compute',
    text: 'Tensors with autodiff, numerics, probability, learning, neural networks, reinforcement learning, logic and datasets. Methods that work in steps are algorithms that record a trace.',
    names: 'cholesky',
    module: 'omai-engine/numerics/linalg',
    cta: 'Engine reference',
    to: '/docs',
  },
  {
    name: 'omai-render',
    role: 'Figures',
    text: 'ECharts options, layouts, scene and structure drawings, SVG, GIF and video export, and React components including the session player.',
    names: 'cartesian, lines',
    module: 'omai-render/charts',
    cta: 'Render gallery',
    to: '/gallery',
  },
  {
    name: 'omai-meta',
    role: 'Documentation',
    text: 'Reads the descriptions in the code, runs the examples in them and keeps the list of concepts. This site’s reference is generated by it.',
    names: 'extract',
    module: 'omai-meta',
    cta: 'Read the docs',
    to: '/docs',
  },
] as const

function Packages(): ReactNode {
  const docs = useDocs()
  const domains = docs?.folder('lib')?.folders ?? []
  const chips = (name: string): readonly { readonly label: string; readonly to: string }[] =>
    name === 'omai-engine'
      ? domains.map((path) => ({ label: path.split('/').at(-1) ?? path, to: docsPath(path) }))
      : name === 'omai-render'
        ? GALLERY_SECTIONS.map((s) => ({ label: s.title, to: `/gallery/${s.id}` }))
        : []
  return (
    <section className="space-y-6">
      <SectionHead id="three-packages" kicker="The packages" title="Three packages">
        The compute, the figures that draw it and the tooling that documents it. Each is documented from its own source.
      </SectionHead>
      <div className="grid gap-4 md:grid-cols-3">
        {CARDS.map((p) => (
          <div key={p.name} className="flex flex-col gap-4 rounded-xl border border-line bg-bg p-6">
            <div>
              <div className="text-xs font-medium tracking-wide text-muted uppercase">{p.role}</div>
              <div className="font-mono text-sm font-semibold">{p.name}</div>
            </div>
            <p className="text-sm text-muted">{p.text}</p>
            <code className="block rounded-md bg-surface px-3 py-2 font-mono text-xs break-words">
              import {'{'} {p.names} {'}'} from &apos;{p.module}&apos;
            </code>
            <div className="flex max-h-40 flex-1 flex-wrap content-start gap-1.5 overflow-y-auto">
              {chips(p.name).map((c) => (
                <Link
                  key={c.to}
                  to={c.to}
                  className="rounded-full border border-line px-2.5 py-0.5 text-xs text-muted transition-colors hover:text-fg"
                >
                  {c.label}
                </Link>
              ))}
            </div>
            <Link to={p.to} className="flex items-center gap-1 pt-1 text-sm font-medium">
              {p.cta} <span aria-hidden="true">→</span>
            </Link>
          </div>
        ))}
      </div>
    </section>
  )
}

function GallerySections(): ReactNode {
  // One strong picture per section: the first recipe of each, with the sections' names under it.
  const picks = GALLERY_SECTIONS.map((sec) => ENTRIES.find((e) => e.section === sec.id)).filter((e) => e !== undefined)
  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <SectionHead id="gallery" kicker="Render gallery" title={`${ENTRIES.length} recipes, one question each`}>
          Every recipe answers one question about drawing, laying out or controlling a figure, with a live example and
          the code that makes it.
        </SectionHead>
        <Link to="/gallery" className={`${BUTTON} border border-line hover:bg-surface`}>
          See them all <span aria-hidden="true">→</span>
        </Link>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {picks.map((e) => (
          <Link
            key={e.path}
            to={e.path}
            className="group relative overflow-hidden rounded-lg border border-line bg-bg transition hover:-translate-y-0.5 hover:border-muted hover:shadow-md"
          >
            <div className="flex aspect-[4/3] items-center justify-center overflow-hidden">
              <Thumbnail section={e.section} slug={e.slug} />
            </div>
            <div className="border-t border-line px-3 py-2 text-sm font-medium">
              {GALLERY_SECTIONS.find((sec) => sec.id === e.section)?.title}
            </div>
          </Link>
        ))}
      </div>
    </section>
  )
}
