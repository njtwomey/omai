import { defaults } from 'mdml-engine/space'
import type { Config } from 'mdml-engine/trace'
import {
  ControlPanel,
  type Narrative,
  NarrativeView,
  SessionPlayer,
  SpaceControls,
  useSession,
  useSessionPreferences,
} from 'mdml-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { ORBIT_STEPS, orbit } from '../../toys/orbit.ts'

export const recipe: Recipe = {
  title: 'Narrative',
  question: 'How do I tell one run in several pages, each showing its own part of the same tape?',
  explain:
    "A `Narrative` is an ordered list of pages; each page has a text block, a layout of panels and the moments it is about (`at` steps, or the firings of an event of the contract). The session is made once and recorded fully; every page subscribes to its own paths, so the page switcher moves between views of one tape and keeps the cursor. The step-through (the arrows, or Alt+Left and Alt+Right) moves the cursor to the page's marked steps and on to the next page's when a page has none left. Here a point walking a circle told in three pages: where it is, how its radius moves, and its laps.",
}

const NARRATIVE: Narrative = {
  title: 'A walk in three pages',
  pages: [
    {
      id: 'where',
      title: 'Where it is',
      text: 'A point walks a circle whose radius **wobbles**. The first plot is the path so far; the second is the distance from the point over the plane around it, at the step the cursor is on.\n\nPress the arrows to visit the start and a step in the first lap. The cursor belongs to the run, not to the page: switch pages and it stays where it is.',
      layout: [
        { kind: 'trajectory', paths: ['position'], title: 'the path walked so far' },
        { kind: 'probe', path: 'plane/distance', title: 'the plane around the point' },
      ],
      moments: [
        { at: 'start', label: 'the start' },
        { at: 30, label: 'a quarter into the first lap' },
      ],
    },
    {
      id: 'radius',
      title: 'How the radius moves',
      text: 'The radius is `1 + wobble * sin(5 angle)`: five wobbles per lap, whatever the number of laps. The curve is the same tape as the first page, read at its `radius` path; click it to move the cursor.',
      layout: [
        { kind: 'curves', paths: ['radius'], title: 'radius (click to scrub)' },
        { kind: 'sweep', path: 'radius', over: 'angle', title: 'the radius against the angle (each point a step)' },
      ],
      moments: [{ event: 'lap/end', label: 'the end of a lap' }],
    },
    {
      id: 'laps',
      title: 'The laps',
      text: 'Each time the point comes round, the walk emits `lap/end`. The marks on the curve are those firings; the step-through goes lap by lap, and `end` is the last step once the run is over.',
      layout: [
        {
          kind: 'curves',
          paths: ['radius'],
          events: [{ path: 'laps', label: 'lap' }],
          title: 'radius, with the laps marked',
        },
      ],
      moments: [
        { event: 'lap/end', nth: 0, label: 'the first lap' },
        { event: 'lap/end', nth: 'last', label: 'the last lap' },
        { at: 'end', label: 'the end of the run' },
      ],
    },
  ],
}

export default function NarrativeRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(orbit.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(orbit, { config: params, mode: prefs.mode, steps: ORBIT_STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={orbit.config} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setParams(defaults(orbit.config) as Config)} />
      <NarrativeView id="gallery/narrative" narrative={NARRATIVE} live={live} at={at} onScrub={setAt} />
    </div>
  )
}
