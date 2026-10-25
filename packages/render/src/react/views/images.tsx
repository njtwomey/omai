/**
 * The React player of image steps: a row of small images per group (data, reconstructions, samples) at each step,
 * beside the objective.
 */

import type { ImageSteps } from 'mdml-engine/exposition'
import { type ReactElement, useCallback, useMemo } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { imageAspect, imageFrames } from '../../views/images.ts'
import { FramesPlayer, type PlayerViewProps, type RecordViewProps, useViewFigure } from './_frame.tsx'

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
