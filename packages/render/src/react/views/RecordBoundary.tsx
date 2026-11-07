/**
 * An error boundary for one record view: a view that throws on a malformed record (a record read from a tape file, or
 * from a newer engine) shows a one-line note in its place and the rest of the page keeps drawing.
 */

import type { Exposition } from 'mdml-engine/exposition'
import { Component, type ReactElement, type ReactNode } from 'react'

/** Props of {@link RecordBoundary}. */
export interface RecordBoundaryProps {
  /** The record being drawn: a new record clears a failure, so the next one gets a try. */
  readonly record: Exposition
  readonly children: ReactNode
}

interface RecordBoundaryState {
  readonly error: Error | null
}

/**
 * Catches an error thrown while a record's view renders and draws "this record could not be drawn" with the message
 * instead. A different `record` clears the failure.
 *
 * @index function React component: error boundary around a record view: `<RecordBoundary record={r}>`
 */
export class RecordBoundary extends Component<RecordBoundaryProps, RecordBoundaryState> {
  override state: RecordBoundaryState = { error: null }

  static getDerivedStateFromError(error: unknown): RecordBoundaryState {
    return { error: error instanceof Error ? error : new Error(String(error)) }
  }

  override componentDidUpdate(prev: RecordBoundaryProps): void {
    if (this.state.error !== null && prev.record !== this.props.record) this.setState({ error: null })
  }

  override render(): ReactElement {
    const { error } = this.state
    if (error === null) return <>{this.props.children}</>
    return (
      <p role="alert" data-record-kind={this.props.record.kind} className="text-xs text-(--wb-danger,#cf222e)">
        this record could not be drawn: {error.message}
      </p>
    )
  }
}
