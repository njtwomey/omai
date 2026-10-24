import { type ReactElement, useCallback, useMemo } from 'react'
import type { EnsembleStages, PipelineStages } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { type EnsembleViewOptions, ensembleFrames, pipelineScene } from '../../views/stages.ts'
import { FramesPlayer, type PlayerViewProps, type RecordViewProps, SceneFigure, useViewFigure } from './_frame.tsx'

const ENSEMBLE_FIGURE = { aspect: 1.75 } as const

/**
 * An ensemble stage by stage: training and validation scores, member weights, and what each member was fitted to
 * (AdaBoost's sample weights as sizes, boosting residuals as colours) over the training rows, played stage by stage.
 *
 * @param props - The record, the training rows, labels and columns, the score label, the playback rate, and the
 * figure settings.
 * @returns The figure with its player.
 */
export function EnsembleView(
  props: RecordViewProps<EnsembleStages> & Omit<EnsembleViewOptions, 'mode' | 'figure'> & PlayerViewProps,
): ReactElement {
  const figure = useViewFigure(props.figure, ENSEMBLE_FIGURE)
  const { record, points, labels, columns, scoreLabel } = props
  const frames = useCallback(
    (fig: Figure) =>
      ensembleFrames(record, {
        figure: fig,
        ...(points === undefined ? {} : { points }),
        ...(labels === undefined ? {} : { labels }),
        ...(columns === undefined ? {} : { columns }),
        ...(scoreLabel === undefined ? {} : { scoreLabel }),
      }),
    [record, points, labels, columns, scoreLabel],
  )
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'ensemble'} />
}

/**
 * A fitted pipeline: one card per stage with the shapes it takes and gives, what it learned and the first rows of its
 * output.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function PipelineView(props: RecordViewProps<PipelineStages>): ReactElement {
  const { record } = props
  const scene = useMemo(() => pipelineScene(record), [record])
  const defaults = useMemo(
    () => ({ width: 720, height: Math.round((720 * scene.height) / Math.max(scene.width, 360)) + 8 }),
    [scene],
  )
  const figure = useViewFigure(props.figure, defaults)
  return <SceneFigure {...props} figure={figure} scene={scene} filename={props.filename ?? 'pipeline'} />
}
