import { defaults } from 'mdml-engine/space'
import type { Config } from 'mdml-engine/trace'
import type { TapeFile } from 'mdml-render/producer'
import {
  ControlPanel,
  type Panel,
  SessionPanels,
  SessionPlayer,
  SpaceControls,
  useSession,
  useSessionPreferences,
  useTape,
} from 'mdml-render/react'
import { type ReactNode, useRef, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { ORBIT_STEPS, orbit } from '../../toys/orbit.ts'
import { toyRegistry } from '../../toys/registry.ts'

export const recipe: Recipe = {
  title: 'Export and tape',
  question: 'How do I save a method page as an image or a recording, and reproduce it later from a file?',
  explain:
    'The export menu of `SessionPlayer` (give it `pageExport`) captures the whole page, not one figure: every panel at the viewed step under the title and over the caption, as one PNG or SVG; or the page played from step a to step b as a GIF or a WebM. "save JSON" writes the tape: the recorded events and the probes the layout reads, at a spread of steps, with the method\'s name (`tapeOf`). "load JSON" reads such a file back (`parseTape`), and `useTape(file, registry)` returns the live handle the player and the panels already take, over sessions replayed from the file: nothing is recomputed, the registry only supplies the method\'s name, its configuration space and projections. Save the tape, change a knob, then load the file: the page returns exactly as it was.',
}

const LAYOUT: readonly Panel[] = [
  {
    kind: 'curves',
    paths: ['radius'],
    events: [{ path: 'laps', label: 'lap' }],
    title: 'radius, with the laps marked (click to scrub)',
  },
  { kind: 'probe', path: 'plane/distance', title: 'the plane around the point' },
  { kind: 'trajectory', paths: ['position'], title: 'the path walked' },
]

const TITLE = 'A point walking a wobbly circle'
const CAPTION = 'Radius against the step, the distance over the plane at the viewed step, and the path so far.'

export default function ExportAndTapeRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(orbit.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const [file, setFile] = useState<TapeFile | undefined>(undefined)
  const prefs = useSessionPreferences()
  const live = useSession(orbit, { config: params, mode: prefs.mode, steps: ORBIT_STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  const played = useTape(file, toyRegistry)
  const shown = played ?? live
  const root = useRef<HTMLDivElement>(null)
  return (
    <div className="flex flex-col gap-3">
      {file === undefined ? (
        <ControlPanel columns={2}>
          <SpaceControls space={orbit.config} value={params} onChange={(p) => setParams(p as Config)} />
        </ControlPanel>
      ) : (
        <div
          className="flex flex-wrap items-center gap-3 rounded border border-(--wb-accent) px-3 py-2 text-sm"
          data-testid="from-file"
        >
          <span>
            Playing back <strong>{file.title ?? file.method}</strong> from a tape file:{' '}
            {file.variants[0]?.run.events.length} events and {file.variants[0]?.probes.length} stored probes, nothing
            recomputed.
          </span>
          <button
            type="button"
            className="rounded border border-(--wb-grid) px-2 py-0.5 text-xs hover:bg-(--wb-surface)"
            onClick={() => {
              setFile(undefined)
              setAt(null)
            }}
          >
            back to the live session
          </button>
        </div>
      )}
      <SessionPlayer
        live={shown}
        at={at}
        onScrub={setAt}
        pageExport={{
          root,
          title: TITLE,
          caption: CAPTION,
          filename: 'orbit',
          layout: LAYOUT,
          onLoadTape: (f) => {
            setFile(f)
            setAt(f.at ?? null)
          },
        }}
      />
      <div ref={root} className="bg-(--wb-bg)">
        <SessionPanels id="gallery/export-and-tape" live={shown} at={at} onScrub={setAt} layout={LAYOUT} />
      </div>
    </div>
  )
}
