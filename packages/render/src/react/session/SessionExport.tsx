/**
 * The export menu of a method page: the whole page (every figure at the viewed step, under its title and over its
 * caption) as one PNG or SVG, the page played from step a to step b as a GIF or a WebM, and the tape as a JSON file
 * (the recorded events and the probes the layout reads) that the same menu loads back, so a page can be reproduced from
 * a file with nothing recomputed. The menu sits in the session player; the figures it captures are the ones inside the
 * element it is given.
 */

import { type ReactElement, type RefObject, useRef, useState } from 'react'
import { encodeGif } from '../../animation/gif.ts'
import { canEncodeVideo, encodeWebm } from '../../animation/video.ts'
import { canProbeAsync } from '../../producer/remote.ts'
import { parseTape, type TapeFile, tapeOf } from '../../producer/tape.ts'
import { svgDataUrl, svgToPixels, svgToPng } from '../../svg/export.ts'
import { type ColourMode, tokens } from '../../theme/tokens.ts'
import { cx, iconButtonClass, menuClass, menuItemClass } from '../controls/classes.ts'
import { Popover } from '../controls/Popover.tsx'
import { figureSvg } from '../FigureFrame.tsx'
import { useColourMode } from '../theme.tsx'
import { framesBetween, type PageFrame, pageSvg, probeRequestsOf } from './export.ts'
import type { Narrative, Panel } from './layout.ts'
import { type LiveSession, type LiveSessions, variantsOf } from './useSession.ts'

/** Props of {@link SessionExport}. */
export interface SessionExportProps {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any> | LiveSessions<any, any>
  readonly at: number | null
  readonly onScrub: (t: number | null) => void
  /** The element holding the page's figures (the panels). */
  readonly root: RefObject<HTMLElement | null>
  /** The page's title and caption, set above and under the figures. */
  readonly title?: string
  readonly caption?: string
  /** The name exported files start with (default `page`). */
  readonly filename?: string
  /** The layout (or narrative) the page shows, so a tape file stores the probes it reads. */
  readonly layout?: readonly Panel[] | Narrative
  /** Called with a tape file chosen from disk; the page plays it back (see `useTape`). Without it, no load item. */
  readonly onLoadTape?: (file: TapeFile) => void
}

/** The most frames of a recording. */
const MAX_FRAMES = 200

const nextPaint = (): Promise<void> =>
  new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(resolve, 30)))
  })

function download(url: string, name: string): void {
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
}

function downloadBlob(bytes: BlobPart, type: string, name: string): void {
  const url = URL.createObjectURL(new Blob([bytes], { type }))
  download(url, name)
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

// The page as it stands: the figures in the root, framed by the title and caption. `undefined` when nothing to export.
function capture(
  root: HTMLElement,
  mode: ColourMode,
  frame: Omit<PageFrame, 'mode'>,
): { readonly svg: string; readonly width: number; readonly height: number } | undefined {
  const bg = tokens(mode).bg
  const inner = figureSvg(root, { background: bg, mode })
  if (inner === undefined) return undefined
  return pageSvg(inner, root.offsetWidth, root.offsetHeight, { ...frame, mode })
}

/**
 * The export menu of a method page.
 *
 * @param props - The session, the cursor, the element to capture, the title and caption, and what to do with a loaded
 *   tape file.
 * @returns The menu button.
 * @index function React component: page export menu: PNG, SVG, GIF, WebM, tape JSON: `<SessionExport live={live} ... />`
 */
export function SessionExport(props: SessionExportProps): ReactElement {
  const { live, at, onScrub, root } = props
  const mode = useColourMode()
  const anchor = useRef<HTMLButtonElement>(null)
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const head = live.head
  const viewed = at === null ? head : Math.min(at, head)
  const [from, setFrom] = useState(0)
  const [to, setTo] = useState<number | null>(null)
  const last = to ?? head
  const name = props.filename ?? 'page'
  const frame = {
    ...(props.title === undefined ? {} : { title: props.title }),
    ...(props.caption === undefined ? {} : { caption: props.caption }),
  }
  const requests = props.layout === undefined ? [] : probeRequestsOf(props.layout)

  const still = (format: 'png' | 'svg'): void => {
    const el = root.current
    if (el === null) return
    const page = capture(el, mode, frame)
    if (page === undefined) {
      setNote('nothing to export yet')
      return
    }
    setOpen(false)
    if (format === 'svg') {
      download(svgDataUrl(page.svg), `${name}.svg`)
      return
    }
    svgToPng(page.svg, page.width, page.height, { background: tokens(mode).bg })
      .then((url) => download(url, `${name}.png`))
      .catch(() => setNote('the browser could not rasterise the page'))
  }

  // A recording seeks the cursor through the steps and rasterises the page at each. Sessions in a worker are asked for
  // the step's probes first, so the page draws complete when it is captured.
  const record = async (format: 'gif' | 'webm'): Promise<void> => {
    const el = root.current
    if (el === null || busy !== null) return
    const steps = framesBetween(from, last, MAX_FRAMES)
    const previous = at
    setOpen(false)
    setNote(null)
    try {
      const bg = tokens(mode).bg
      const pixels = []
      for (const [i, step] of steps.entries()) {
        setBusy(`recording ${i + 1} / ${steps.length}`)
        await Promise.all(
          variantsOf(live).flatMap((v) => {
            const s = v.live.session
            if (!canProbeAsync(s)) return []
            return requests
              .filter((r) => s.contract[r.path]?.kind === 'derived')
              .map((r) => s.probeAsync(step, r.path, r.args).catch(() => undefined))
          }),
        )
        onScrub(step)
        await nextPaint()
        const page = capture(el, mode, frame)
        if (page === undefined) break
        pixels.push(await svgToPixels(page.svg, page.width, page.height, { background: bg }))
      }
      setBusy('encoding')
      const fps = Math.min(30, Math.max(4, Math.round(steps.length / 6)))
      if (format === 'gif') downloadBlob(encodeGif(pixels, { fps }) as BlobPart, 'image/gif', `${name}.gif`)
      else downloadBlob((await encodeWebm(pixels, { fps })) as BlobPart, 'video/webm', `${name}.webm`)
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e))
    } finally {
      onScrub(previous)
      setBusy(null)
    }
  }

  const saveTape = async (): Promise<void> => {
    const sessions = variantsOf(live).map((v) => ({ label: v.label, session: v.live.session }))
    const first = sessions[0]
    if (first === undefined || busy !== null) return
    setOpen(false)
    setNote(null)
    setBusy('writing the tape')
    try {
      const file = await tapeOf(first.session.method.name, sessions, {
        requests,
        at: viewed,
        ...(props.title === undefined ? {} : { title: props.title }),
        ...(props.layout === undefined ? {} : { layout: props.layout as never }),
      })
      downloadBlob(JSON.stringify(file), 'application/json', `${name}.tape.json`)
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const loadTape = (file: File | undefined): void => {
    if (file === undefined || props.onLoadTape === undefined) return
    setOpen(false)
    file
      .text()
      .then((text) => {
        props.onLoadTape?.(parseTape(text))
        setNote(null)
      })
      .catch((e: unknown) => setNote(e instanceof Error ? e.message : String(e)))
  }

  const small = 'h-7 w-16 rounded border border-(--wb-grid) bg-(--wb-bg) px-1 text-xs tabular-nums text-(--wb-fg)'
  return (
    <div className="flex items-center gap-2">
      <button
        ref={anchor}
        type="button"
        aria-label="Export the page"
        aria-haspopup="dialog"
        aria-expanded={open}
        className={cx(iconButtonClass, 'w-auto gap-1 rounded-md px-2 text-xs')}
        disabled={busy !== null}
        onClick={() => setOpen((o) => !o)}
      >
        export
      </button>
      {busy !== null ? <span className="font-mono text-xs text-(--wb-muted)">{busy}</span> : null}
      {note !== null ? <span className="text-xs text-(--wb-danger,#cf222e)">{note}</span> : null}
      <Popover anchor={anchor} open={open} onClose={() => setOpen(false)} align="start" role="dialog" label="Export">
        <div className={cx(menuClass, 'flex w-64 flex-col gap-1 p-2')} data-testid="export-menu">
          <div className="px-1 text-[11px] font-medium text-(--wb-muted)">the page at step {viewed}</div>
          <div className="flex gap-1">
            <button
              type="button"
              className={cx(menuItemClass, 'justify-center rounded border border-(--wb-grid)')}
              onClick={() => still('png')}
            >
              PNG
            </button>
            <button
              type="button"
              className={cx(menuItemClass, 'justify-center rounded border border-(--wb-grid)')}
              onClick={() => still('svg')}
            >
              SVG
            </button>
          </div>
          <div className="mt-1 px-1 text-[11px] font-medium text-(--wb-muted)">the page played from a to b</div>
          <div className="flex items-center gap-1 px-1 text-xs">
            <input
              aria-label="from step"
              type="number"
              className={small}
              min={0}
              max={head}
              value={Math.min(from, head)}
              onChange={(e) => setFrom(Math.max(0, Math.round(Number(e.target.value))))}
            />
            to
            <input
              aria-label="to step"
              type="number"
              className={small}
              min={0}
              max={head}
              value={Math.min(last, head)}
              onChange={(e) => setTo(Math.max(0, Math.round(Number(e.target.value))))}
            />
          </div>
          <div className="flex gap-1">
            <button
              type="button"
              className={cx(menuItemClass, 'justify-center rounded border border-(--wb-grid)')}
              onClick={() => void record('gif')}
            >
              GIF
            </button>
            {canEncodeVideo() ? (
              <button
                type="button"
                className={cx(menuItemClass, 'justify-center rounded border border-(--wb-grid)')}
                onClick={() => void record('webm')}
              >
                WebM
              </button>
            ) : null}
          </div>
          <div className="mt-1 px-1 text-[11px] font-medium text-(--wb-muted)">the tape</div>
          <div className="flex gap-1">
            <button
              type="button"
              className={cx(menuItemClass, 'justify-center rounded border border-(--wb-grid)')}
              onClick={() => void saveTape()}
            >
              save JSON
            </button>
            {props.onLoadTape === undefined ? null : (
              <label className={cx(menuItemClass, 'justify-center rounded border border-(--wb-grid)')}>
                load JSON
                <input
                  type="file"
                  accept=".json,application/json"
                  className="sr-only"
                  data-testid="load-tape"
                  onChange={(e) => {
                    loadTape(e.target.files?.[0])
                    e.target.value = ''
                  }}
                />
              </label>
            )}
          </div>
        </div>
      </Popover>
    </div>
  )
}
