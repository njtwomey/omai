import { type ReactElement, useCallback } from 'react'
import type { ImageSteps } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { imageFrames } from '../../views/images.ts'
import { FramesPlayer, type PlayerViewProps, type RecordViewProps, useViewFigure } from './_frame.tsx'

const IMAGE_FIGURE = { aspect: 2.2 } as const

/**
 * Small images played step by step: a row of images per group (data, reconstructions, samples) beside the objective.
 *
 * @param props - The record, the images per group, the playback rate, and the figure settings.
 * @returns The figure with its player.
 */
export function ImagePlayer(
  props: RecordViewProps<ImageSteps> & PlayerViewProps & { readonly columns?: number },
): ReactElement {
  const figure = useViewFigure(props.figure, IMAGE_FIGURE)
  const { record, columns } = props
  const frames = useCallback(
    (fig: Figure) => imageFrames(record, { figure: fig, ...(columns === undefined ? {} : { columns }) }),
    [record, columns],
  )
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'images'} />
}
