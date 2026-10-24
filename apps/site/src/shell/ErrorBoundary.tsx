import { Component, type ErrorInfo, type ReactNode } from 'react'

interface State {
  readonly error: Error | null
}

/**
 * Keeps one live example's failure to itself: the example's area shows the error (and a retry) instead of the whole
 * page going blank. `resetKey` clears it when the example changes.
 */
export class ErrorBoundary extends Component<{ readonly children: ReactNode; readonly resetKey?: string }, State> {
  override state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  override componentDidUpdate(previous: { readonly resetKey?: string }): void {
    if (previous.resetKey !== this.props.resetKey && this.state.error !== null) this.setState({ error: null })
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(error, info.componentStack)
  }

  override render(): ReactNode {
    const { error } = this.state
    if (error === null) return this.props.children
    return (
      <div role="alert" className="flex flex-col gap-2 rounded-lg border border-danger/50 p-4 text-sm">
        <span className="font-medium text-danger">This example failed: {error.message}</span>
        <button
          type="button"
          onClick={() => this.setState({ error: null })}
          className="w-fit rounded-md border border-line px-2 py-1 text-xs hover:bg-surface"
        >
          Try again
        </button>
      </div>
    )
  }
}
