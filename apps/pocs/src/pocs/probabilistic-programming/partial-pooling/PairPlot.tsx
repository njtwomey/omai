import { type PairDraws, posteriorPairs } from 'omai-engine/exposition'
import { PosteriorPairsView } from 'omai-render/react'
import { type ReactNode, useMemo } from 'react'
import { GENERATING, type GroupData, type PoolingResult } from './compute.ts'

/** The name of a group's effect on the plot's axes. */
export const effectName = (data: GroupData, j: number): string => `theta${j} (n=${data.size[j]})`

/**
 * The corner plot of the group mean, the log of the spread and the effects of two chosen groups, from the draws of one
 * run, at the playhead: the draws made by then, the divergent transitions marked apart, the generating values as the
 * reference. The library's `PosteriorPairsView` draws it; the run's iterations are its steps.
 *
 * @param props - The groups, the run, the two groups chosen and the playhead in steps.
 * @returns The figure, or a note while the run has no draws.
 */
export function PairPlot(props: {
  readonly data: GroupData
  readonly result: PoolingResult
  readonly groups: readonly [number, number]
  readonly cursor: number
}): ReactNode {
  const { data, result, groups, cursor } = props
  const record = useMemo(() => {
    if (result.at.length < 2) return undefined
    const names = groups.map((j) => effectName(data, j))
    const draws: PairDraws = {
      name: result.method,
      draws: {
        mu: result.mu,
        'log tau': result.logTau,
        [names[0] as string]: result.theta[groups[0]] ?? [],
        [names[1] as string]: result.theta[groups[1]] ?? [],
      },
      steps: result.at,
      chain: result.chain,
      divergent: result.divergent,
    }
    return posteriorPairs([draws], {
      coordinates: ['mu', 'log tau', ...(names as string[])],
      frames: 40,
      maxPoints: 400,
      truth: {
        mu: GENERATING.mu,
        'log tau': Math.log(data.settings.tau),
        [names[0] as string]: data.truth[groups[0]] ?? Number.NaN,
        [names[1] as string]: data.truth[groups[1]] ?? Number.NaN,
      },
      referenceLabel: 'generating value',
    })
  }, [data, result, groups])
  if (record === undefined) return <p className="text-sm text-muted">The run has no draws.</p>
  return (
    <PosteriorPairsView
      record={record}
      cursor={cursor}
      controls={false}
      filename={`partial-pooling-${result.method}`}
    />
  )
}
