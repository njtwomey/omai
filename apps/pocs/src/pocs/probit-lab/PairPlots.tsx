import { type PairDraws, posteriorPairs } from 'omai-engine/exposition'
import { PosteriorPairsView } from 'omai-render/react'
import { type ReactNode, useMemo } from 'react'
import { column, frameAt, visibleRows } from './shown.ts'
import type { MethodResult } from './types.ts'

const FRAMES = 40
const MAX_POINTS = 300

/**
 * What the corner plot is given of one method: a sampler's draws with the time each was made (so the plot fills in as the
 * playhead moves), an approximation's draws as of its last snapshot (they appear when the run has them: the earlier
 * snapshots are on the decision boundary and the marginals, not here).
 */
function drawsOf(result: MethodResult, coords: readonly number[]): PairDraws {
  const last = result.frames[frameAt(result, Number.POSITIVE_INFINITY)]?.elapsedMs ?? result.elapsedMs
  const rows = visibleRows(result, result.organisation === 'iterate' ? last : Number.POSITIVE_INFINITY)
  const at = result.organisation === 'iterate' ? rows.map(() => last) : rows.map((i) => result.pool.at[i] ?? 0)
  return {
    name: result.label,
    draws: Object.fromEntries(coords.map((j) => [`w${j}`, column(result, rows, j)])),
    steps: at,
    chain: rows.map((i) => result.pool.chain[i] ?? 0),
  }
}

/**
 * The corner plot of the chosen coefficients with the methods overlaid, drawn by the library's `PosteriorPairsView` and
 * following the page's playhead: the scatter below the diagonal fills in with the draws made by `cursor` (milliseconds of
 * the run), the densities on the diagonal and the correlations above it follow. The page's own checkboxes choose the
 * coefficients and methods, so the view's switches are off.
 *
 * @param props - The methods to overlay, the coefficient indices, their labels and the playhead in milliseconds.
 * @returns The figure, or a note while no method has a draw.
 */
export function PairPlots(props: {
  readonly results: readonly MethodResult[]
  readonly coordinates: readonly number[]
  readonly cursor: number
}): ReactNode {
  const { results, coordinates, cursor } = props
  const record = useMemo(() => {
    const runs = results.map((r) => drawsOf(r, coordinates)).filter((r) => (r.steps?.length ?? 0) > 1)
    if (runs.length === 0 || coordinates.length < 2) return undefined
    return posteriorPairs(runs, {
      coordinates: coordinates.map((j) => `w${j}`),
      frames: FRAMES,
      maxPoints: MAX_POINTS,
    })
  }, [results, coordinates])
  if (record === undefined) return <p className="text-sm text-muted">No method has draws to show yet.</p>
  return <PosteriorPairsView record={record} cursor={cursor} controls={false} filename="probit-lab-pairs" />
}
