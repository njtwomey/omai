/**
 * The React view of a reinforcement-learning run: returns with their rolling mean, episode lengths, and a value and
 * policy snapshot (a map for grid worlds).
 */

import { type ReactElement, useCallback } from 'react'
import type { EpisodeRecord } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { type EpisodeViewOptions, episodeView } from '../../views/episodes.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const EPISODE_FIGURE = { width: 720, aspect: 1.9 } as const

/**
 * A reinforcement-learning run: returns with their rolling mean, episode lengths, and a value and policy snapshot (a
 * map for grid worlds).
 *
 * @param props - The record, the snapshot, the grid's shape, action glyphs and window, and the figure settings.
 * @returns The figure.
 */
export function EpisodeView(
  props: RecordViewProps<EpisodeRecord> & Omit<EpisodeViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, EPISODE_FIGURE)
  const { record, snapshot, gridShape, actionGlyphs, window } = props
  const build = useCallback(
    (fig: Figure) =>
      episodeView(record, {
        figure: fig,
        ...(snapshot === undefined ? {} : { snapshot }),
        ...(gridShape === undefined ? {} : { gridShape }),
        ...(actionGlyphs === undefined ? {} : { actionGlyphs }),
        ...(window === undefined ? {} : { window }),
      }),
    [record, snapshot, gridShape, actionGlyphs, window],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'episodes'} />
}
