import { type PlaybackController, Player, usePlayback, useTicker } from 'omai-render/react'
import { type ReactNode, useCallback, useEffect, useRef, useState } from 'react'
import type { ExampleModule, Live } from '../../../../examples/_shared/contract.ts'
import index from '../../../../examples/index.json' with { type: 'json' }
import { useColourMode } from '../shell/colour-mode.ts'
import type { Slide } from './Carousel.tsx'
import { useOnScreen } from './live.ts'

/** One example of `examples/`, as `examples/index.json` lists it (written by `make readme-examples`). */
interface ExampleEntry {
  readonly name: string
  readonly title: string
  readonly summary: string
  readonly alt: string
  readonly hasReadme: boolean
  readonly tapes: readonly string[]
  readonly home: { readonly label: string; readonly seconds?: number; readonly fps?: number } | null
}

const REPOSITORY = 'https://github.com/njtwomey/omai'

// The examples' programs, loaded when a slide is first on screen: the site runs them in the browser, so the figure is
// the example's own `View`, played through the steps of its own run.
const programs = import.meta.glob<ExampleModule>('../../../../examples/*/main.tsx')

const loaderOf = (name: string): (() => Promise<ExampleModule>) | undefined =>
  Object.entries(programs).find(([path]) => path.endsWith(`/examples/${name}/main.tsx`))?.[1]

/** The links under a figure: the example's README, its source and the tape of its run. */
function Links(props: { readonly entry: ExampleEntry }): ReactNode {
  const { entry } = props
  const base = `${REPOSITORY}/tree/main/examples/${entry.name}`
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 px-1 text-xs text-muted">
      <span className="font-medium text-fg">{entry.title}</span>
      {entry.hasReadme ? (
        <a href={`${base}/README.md`} className="underline underline-offset-4 hover:text-fg">
          README
        </a>
      ) : null}
      <a href={`${base}/main.tsx`} className="underline underline-offset-4 hover:text-fg">
        Source
      </a>
      {entry.tapes.length > 0 ? (
        <a href={`${base}/${entry.tapes[0]}`} className="underline underline-offset-4 hover:text-fg">
          Tape of the run
        </a>
      ) : null}
    </div>
  )
}

/** The frame the figures sit in: a lighter dotted ground, in the page's colours. */
const FIGURE_BOX =
  'dotted min-h-64 overflow-hidden rounded-lg p-1 [&>svg]:mx-auto [&>svg]:h-auto [&>svg]:max-h-[560px] [&>svg]:w-full'

/** An example computed as it is watched: each tick takes a step of its live run and draws what it has so far. */
function Stepped(props: {
  readonly entry: ExampleEntry
  readonly program: ExampleModule
  readonly seen: boolean
}): ReactNode {
  const { entry, program, seen } = props
  const mode = useColourMode()
  const run = useRef<Live<unknown> | undefined>(undefined)
  if (run.current === undefined) run.current = program.live?.()
  const [, redraw] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [fps, setFps] = useState(entry.home?.fps ?? 10)
  const [steps, setSteps] = useState(0)
  const [over, setOver] = useState(false)
  const advance = useCallback((n: number): void => {
    const live = run.current
    if (live === undefined) return
    let more = true
    for (let i = 0; i < n && more; i++) {
      more = live.step()
      setSteps((k) => k + 1)
    }
    setOver(!more)
    redraw((k) => k + 1)
  }, [])
  const restart = useCallback((): void => {
    run.current = program.live?.()
    setSteps(0)
    setOver(false)
    redraw((k) => k + 1)
  }, [program])
  // The library's ticker: the steps due at the rate, from the display's animation frames.
  useTicker(seen && playing && !over, fps, advance)
  // A finished run rests a moment, then starts again.
  useEffect(() => {
    if (!over || !seen) return
    const id = setTimeout(restart, 2500)
    return () => clearTimeout(id)
  }, [over, seen, restart])
  const controller: PlaybackController = {
    frame: steps,
    count: steps + 1,
    playing: playing && !over,
    fps,
    open: !over,
    play: () => setPlaying(true),
    pause: () => setPlaying(false),
    toggle: () => setPlaying((p) => !p),
    step: (dir) => {
      setPlaying(false)
      if (dir === 1) advance(1)
    },
    seek: () => undefined,
    setFps,
    reset: restart,
  }
  const result = run.current?.result()
  const frames = program.frames?.(result) ?? 0
  return (
    <>
      <div className={FIGURE_BOX}>
        {result === undefined ? null : (
          <program.View result={result} mode={mode} {...(program.frames === undefined ? {} : { at: frames })} />
        )}
      </div>
      <Player controller={controller} stamp={(f) => `step ${f}`} className="px-1" />
    </>
  )
}

/** An example run to the end and then played back frame by frame, for those without a live run (their runs are quick). */
function Replayed(props: {
  readonly entry: ExampleEntry
  readonly program: ExampleModule
  readonly seen: boolean
}): ReactNode {
  const { entry, program, seen } = props
  const mode = useColourMode()
  const [result, setResult] = useState<{ readonly value: unknown } | null>(null)
  const [failed, setFailed] = useState(false)
  const started = useRef(false)
  useEffect(() => {
    if (!seen || started.current) return
    started.current = true
    // After a tick, so the slide shows its note before the computation takes the thread.
    setTimeout(() => {
      Promise.resolve(program.run()).then(
        (out) => setResult({ value: out.result }),
        () => setFailed(true),
      )
    }, 30)
  }, [seen, program])
  const frames = result === null ? 0 : (program.frames?.(result.value) ?? 0)
  // The library's own playback: the run's frames at a rate the reader can change, looping, with play, step and scrub.
  const playback = usePlayback(frames > 0 ? frames + 1 : 0, { loop: true })
  const controls = useRef(playback)
  controls.current = playback
  const fps = entry.home?.fps ?? 10
  const { setFps } = playback
  useEffect(() => {
    if (frames > 0) setFps(fps)
  }, [frames, fps, setFps])
  // Each time the slide comes on screen it starts again from the first frame; off screen it pauses.
  useEffect(() => {
    if (frames === 0) return
    if (seen) {
      controls.current.seek(0)
      controls.current.play()
    } else controls.current.pause()
  }, [seen, frames])
  return (
    <>
      <div className={FIGURE_BOX}>
        {result === null ? (
          <p className="p-6 text-sm text-muted">{failed ? 'This example did not run.' : 'Running the example…'}</p>
        ) : (
          <program.View result={result.value} mode={mode} {...(frames > 0 ? { at: playback.frame } : {})} />
        )}
      </div>
      {frames > 0 ? <Player controller={playback} stamp={(f) => `${f} / ${frames}`} className="px-1" /> : null}
    </>
  )
}

/** One slide: loads the example when it is first on screen, then shows it live or replayed. */
function LiveExample(props: { readonly entry: ExampleEntry }): ReactNode {
  const { entry } = props
  const box = useRef<HTMLDivElement>(null)
  const seen = useOnScreen(box)
  const [program, setProgram] = useState<ExampleModule | null>(null)
  const started = useRef(false)
  useEffect(() => {
    const load = loaderOf(entry.name)
    if (!seen || started.current || load === undefined) return
    started.current = true
    void load().then(setProgram)
  }, [seen, entry.name])
  return (
    <div ref={box} className="space-y-2">
      {program === null ? (
        <div className={FIGURE_BOX}>
          <p className="p-6 text-sm text-muted">Loading the example…</p>
        </div>
      ) : program.live === undefined ? (
        <Replayed entry={entry} program={program} seen={seen} />
      ) : (
        <Stepped entry={entry} program={program} seen={seen} />
      )}
      <Links entry={entry} />
    </div>
  )
}

/** The carousel's slides: the examples that name a tab in their `meta.home`, in the order of the index. */
export const EXAMPLE_SLIDES: readonly Slide[] = (index as readonly ExampleEntry[])
  .filter((e) => e.home !== null)
  .map((entry) => ({
    label: entry.home?.label ?? entry.name,
    blurb: entry.summary,
    duration: (entry.home?.seconds ?? 10) * 1000,
    render: () => <LiveExample entry={entry} />,
  }))
