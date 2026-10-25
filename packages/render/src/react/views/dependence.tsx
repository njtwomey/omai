/**
 * React views of the dependence record: one feature's PDP with ICE curves or ALE (or small multiples), and a pair's
 * joint effect as a heatmap.
 */

import type { Dependence } from 'mdml-engine/exposition'
import { type ReactElement, useCallback, useMemo } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { dependenceSurfaceView, dependenceView } from '../../views/dependence.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const SURFACE_FIGURE = { aspect: 1.25, padding: { right: 96 } } as const

/**
 * How a model's output depends on a feature: the averaged curve (partial dependence or ALE) over every row's ICE
 * curve and the data's rug; with several features and no `curve`, small multiples.
 *
 * @param props - The record, the curve (or `all`), how many ICE curves, and the figure settings.
 * @returns The figure.
 */
export function DependenceView(
  props: RecordViewProps<Dependence> & { readonly curve?: number; readonly all?: boolean; readonly maxIce?: number },
): ReactElement {
  const { record, curve, all, maxIce } = props
  const many = (all ?? (curve === undefined && record.curves.length > 1)) && record.curves.length > 1
  const defaults = useMemo(() => {
    if (record.curves.length === 0) return SURFACE_FIGURE
    if (!many) return { aspect: 1.5 }
    const rows = Math.ceil(Math.min(9, record.curves.length) / 3)
    return { width: 720, height: Math.max(240, 40 + rows * 210) }
  }, [record, many])
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback(
    (fig: Figure) =>
      dependenceView(record, {
        figure: fig,
        ...(curve === undefined ? {} : { curve }),
        ...(all === undefined ? {} : { all }),
        ...(maxIce === undefined ? {} : { maxIce }),
      }),
    [record, curve, all, maxIce],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'dependence'} />
}

/**
 * A pair of features' joint effect as a heatmap: two-way partial dependence, or the second-order ALE interaction
 * centred on zero.
 *
 * @param props - The record, which surface, and the figure settings.
 * @returns The figure.
 */
export function DependenceSurfaceView(
  props: RecordViewProps<Dependence> & { readonly surface?: number },
): ReactElement {
  const { record, surface } = props
  const figure = useViewFigure(props.figure, SURFACE_FIGURE)
  const build = useCallback(
    (fig: Figure) => dependenceSurfaceView(record, { figure: fig, ...(surface === undefined ? {} : { surface }) }),
    [record, surface],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'interaction'} />
}
