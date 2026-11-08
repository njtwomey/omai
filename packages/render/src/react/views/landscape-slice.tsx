/**
 * The React view of a loss-landscape slice: the curve or the contours with the optimiser's iterates, and for a plane a
 * button switching to a surface to orbit.
 */

import type { LandscapeSlice } from 'omai-engine/exposition'
import { fitCamera, camera as makeCamera } from 'omai-engine/scene'
import { type ReactElement, useCallback, useMemo, useState } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { landscapeSliceScene3d, landscapeSliceView, sliceMinimum } from '../../views/landscape-slice.ts'
import { FigureFrame } from '../FigureFrame.tsx'
import { Scene3DView } from '../Scene3DView.tsx'
import { ToolButton } from '../session/parts.tsx'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

/** Props of {@link LandscapeSliceView}: the record, the frame settings and whether a plane starts as a surface. */
export interface LandscapeSliceViewProps extends RecordViewProps<LandscapeSlice> {
  /** Draw a plane as a surface to orbit rather than as contours (default false); the buttons switch. */
  readonly surface?: boolean
}

const DEFAULTS = { aspect: 1.3 } as const

/**
 * A slice through a loss landscape. The caption says in words where the lowest loss on the slice is and how many
 * iterates are drawn; a plane has buttons for contours and for a surface.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function LandscapeSliceView(props: LandscapeSliceViewProps): ReactElement {
  const { record } = props
  const figure = useViewFigure(props.figure, DEFAULTS)
  const [asSurface, setAsSurface] = useState(props.surface === true)
  const build = useCallback((fig: Figure) => landscapeSliceView(record, { figure: fig }), [record])
  const scene3 = useMemo(
    () => (record.dims === 2 && asSurface ? landscapeSliceScene3d(record, { mode: figure.mode }) : undefined),
    [record, asSurface, figure.mode],
  )
  const min = sliceMinimum(record)
  const caption =
    props.caption ??
    `${record.caption === undefined ? '' : `${record.caption}. `}Lowest on the slice: ${min.loss.toPrecision(3)}.`
  const toggle =
    record.dims === 2 ? (
      <div className="flex items-center gap-1" role="toolbar" aria-label="How the plane is drawn">
        <ToolButton
          label="contours"
          title="Draw the plane as contours"
          pressed={!asSurface}
          onClick={() => setAsSurface(false)}
        />
        <ToolButton
          label="surface"
          title="Draw the plane as a surface to orbit"
          pressed={asSurface}
          onClick={() => setAsSurface(true)}
        />
      </div>
    ) : null
  if (scene3 === undefined)
    return (
      <RecordFigure
        {...props}
        figure={figure}
        build={build}
        filename={props.filename ?? 'landscape'}
        caption={
          <div className="flex flex-col gap-1">
            {toggle}
            <span>{caption}</span>
          </div>
        }
      />
    )
  return (
    <FigureFrame
      width={figure.width}
      height={figure.height}
      filename={props.filename ?? 'landscape'}
      {...(props.title === undefined ? {} : { title: props.title })}
      caption={
        <div className="flex flex-col gap-1">
          {toggle}
          <span>{caption} Drag to orbit.</span>
        </div>
      }
    >
      <Scene3DView
        scene={scene3}
        camera={fitCamera(makeCamera({ azimuth: -50, elevation: 35 }), scene3)}
        width={figure.width}
        height={figure.height}
      />
    </FigureFrame>
  )
}
