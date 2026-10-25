/**
 * The standard view of a threshold sweep: the threshold explorer over the record, its threshold the reader's to drag.
 */

import type { ReactElement } from 'react'
import type { ThresholdSweep } from 'whitebox-engine/exposition'
import { ThresholdExplorer, type ThresholdExplorerProps } from '../ThresholdExplorer.tsx'
import type { RecordViewProps } from './_frame.tsx'

/** Props of {@link ThresholdSweepView}. */
export interface ThresholdSweepViewProps
  extends RecordViewProps<ThresholdSweep>,
    Pick<ThresholdExplorerProps, 'threshold' | 'onThreshold' | 'followCheapest' | 'figures' | 'table' | 'range'> {}

/**
 * A threshold sweep: each class's scores and the cost against the threshold (lines to drag), ROC and precision-recall
 * with the operating point, and the contingency table at the threshold (it follows the cheapest when the sweep has
 * costs, until the reader moves it).
 *
 * @param props - The record, the threshold (controlled or not) and which figures to show.
 * @returns The figures and the table.
 */
export function ThresholdSweepView(props: ThresholdSweepViewProps): ReactElement {
  const { record, figure, filename, title: _title, caption: _caption, ...rest } = props
  return (
    <ThresholdExplorer
      record={record}
      followCheapest={props.followCheapest ?? record.cost !== undefined}
      {...rest}
      {...(figure === undefined ? {} : { figure })}
      filename={filename ?? 'threshold-sweep'}
    />
  )
}
