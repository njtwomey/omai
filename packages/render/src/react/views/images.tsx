/**
 * The React player of image steps: a row of small images per group (data, reconstructions, samples) at each step,
 * beside the objective.
 */

import type { ImageSteps } from 'omai-engine/exposition'
import { type ReactElement, useCallback, useMemo } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { imageAspect, imageFrames, imageGridFigure, imageGridView } from '../../views/images.ts'
import { FramesPlayer, type PlayerViewProps, RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

/**
 * Small images played step by step: a row of images per group (data, reconstructions, samples) beside the objective.
 *
 * @param props - The record, the images per group, the playback rate, and the figure settings.
 * @returns The figure with its player.
 */
export function ImagePlayer(
  props: RecordViewProps<ImageSteps> & PlayerViewProps & { readonly columns?: number },
): ReactElement {
  const { record, columns } = props
  const defaults = useMemo(() => ({ aspect: imageAspect(record, columns) }), [record, columns])
  const figure = useViewFigure(props.figure, defaults)
  const frames = useCallback(
    (fig: Figure) => imageFrames(record, { figure: fig, ...(columns === undefined ? {} : { columns }) }),
    [record, columns],
  )
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'images'} />
}

/** Props of {@link ImageGridView}: the record, the tape step to show, the images per group and the frame settings. */
export interface ImageGridViewProps extends RecordViewProps<ImageSteps> {
  /** The tape step to show: the latest recorded one at or before it (default the last recorded). */
  readonly step?: number
  readonly columns?: number
}

/**
 * Many small images at a step as a grid: a row per group with its name, the images' captions beneath them; a still, with
 * no player and no objective.
 *
 * @param props - The record, the step, the images per group, and the figure and frame settings.
 * @returns The figure.
 */
export function ImageGridView(props: ImageGridViewProps): ReactElement {
  const { record, columns, step } = props
  const given = useMemo(() => props.figure ?? imageGridFigure(record, 440, columns), [props.figure, record, columns])
  const figure = useViewFigure(given, {})
  const build = useCallback(
    (fig: Figure) => imageGridView(record, step, { figure: fig, ...(columns === undefined ? {} : { columns }) }),
    [record, step, columns],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'images'} />
}
