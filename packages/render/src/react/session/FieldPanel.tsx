/**
 * The field panel of a session: a vector field probed at the viewed step, drawn as arrows or streamlines with its
 * nullclines and classified fixed points, and tensor paths of the tape drawn over it as a trajectory ending in its
 * current point.
 */

import type { VectorField } from 'mdml-engine/exposition'
import { type ReactNode, useMemo } from 'react'
import { cartesian, type Layer } from '../../charts/plot.ts'
import { vectorFieldLayers } from '../../views/fields.ts'
import { Chart } from '../Chart.tsx'
import { FigureFrame } from '../FigureFrame.tsx'
import { useColourMode } from '../theme.tsx'
import type { Panel } from './layout.ts'
import { FIG, Note, useProbe, useRun } from './RowsPanel.tsx'
import { trajectoryLayers } from './trajectory.ts'
import type { LiveSession } from './useSession.ts'

/**
 * A vector field at the viewed step with, when asked, trajectories of the tape over it. The axes keep the field's
 * domain, so the picture holds still while the field and the paths move; a probe that is not a `vectorField` says why
 * in the frame's caption.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function FieldPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: Extract<Panel, { kind: 'field' }>
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { live, panel, t } = props
  const run = useRun(live)
  const mode = useColourMode()
  const probed = useProbe(live, t, panel.path, panel.args)
  const record =
    (probed.value as { kind?: unknown } | undefined)?.kind === 'vectorField' ? (probed.value as VectorField) : undefined
  const option = useMemo(() => {
    if (record === undefined) return undefined
    const over = panel.trajectory
    const layers: Layer[] = [
      ...vectorFieldLayers(record, {
        mode,
        style: panel.style ?? 'arrows',
        // Under a trajectory the field recedes.
        ...(over === undefined ? {} : { colour: 'grid' }),
      }),
      ...(over === undefined
        ? []
        : trajectoryLayers(run, t, {
            paths: over.paths,
            mode,
            ...(over.dims === undefined ? {} : { dims: over.dims }),
            ...(over.trail === undefined ? {} : { trail: over.trail }),
            ...(over.points === undefined ? {} : { points: over.points }),
          })),
    ]
    const names = record.features ?? ['x', 'y']
    const x0 = record.x[0] ?? 0
    const y0 = record.y[0] ?? 0
    return {
      ...cartesian(layers, {
        mode,
        figure: FIG,
        x: { label: names[0], limits: [x0, record.x[record.x.length - 1] ?? x0 + 1] },
        y: { label: names[1], limits: [y0, record.y[record.y.length - 1] ?? y0 + 1] },
        legend: (record.nullclines?.length ?? 0) + (record.fixedPoints?.length ?? 0) > 0,
      }),
      animation: false,
    }
  }, [record, panel.style, panel.trajectory, run, t, mode])
  const problem =
    probed.error ?? (record === undefined && probed.value !== undefined ? 'the probe is not a vector field' : undefined)
  return (
    <FigureFrame
      figure={FIG}
      title={props.title}
      filename={props.filename}
      {...(problem === undefined ? {} : { caption: <Note>unavailable: {problem}</Note> })}
    >
      {option === undefined ? <div className="h-full" /> : <Chart figure={FIG} option={option} />}
    </FigureFrame>
  )
}
