// Two small hooks that give every factory call the same loading / error /
// data states, so no tab can forget one of them.
//
// Both ignore a response that arrives after a newer request was started (or
// after the component unmounted): with a slow service, switching quickly
// between reference profiles must not let an old answer overwrite the new one.

import { useCallback, useEffect, useRef, useState } from "react"

export interface RequestState<T> {
  data: T | null
  error: unknown
  loading: boolean
}

export type QueryResult<T> = RequestState<T> & { reload: () => void }

/**
 * Fetch whenever ``key`` changes; ``key === null`` means "do not fetch".
 * ``fetcher`` must be stable (a module-level function), since it is an
 * effect dependency.
 *
 * Data belongs to the key it was fetched for: after the key changes the old
 * data is no longer returned, so a view never shows one classifier's card
 * under another's name. A reload of the same key keeps the data until the new
 * answer arrives.
 */
export function useFactoryQuery<K, T>(key: K | null, fetcher: (key: K) => Promise<T>): QueryResult<T> {
  const [state, setState] = useState<RequestState<T> & { key: K | null }>({
    data: null,
    error: null,
    loading: key !== null,
    key,
  })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    if (key === null) return
    let current = true
    setState((s) => ({ data: s.key === key ? s.data : null, error: null, loading: true, key }))
    fetcher(key).then(
      (data) => current && setState({ data, error: null, loading: false, key }),
      (error) => current && setState({ data: null, error, loading: false, key }),
    )
    return () => {
      current = false
    }
  }, [key, fetcher, attempt])

  const reload = useCallback(() => setAttempt((n) => n + 1), [])
  if (key === null) return { data: null, error: null, loading: false, reload }
  // Between a key change and the effect that follows it, report a fetch in
  // progress rather than the previous key's result.
  if (state.key !== key) return { data: null, error: null, loading: true, reload }
  return { data: state.data, error: state.error, loading: state.loading, reload }
}

/** A call made on demand (a form submit). ``run`` resolves to the data, or null on failure. */
export function useFactoryAction<A, T>(action: (args: A) => Promise<T>) {
  const [state, setState] = useState<RequestState<T>>({ data: null, error: null, loading: false })
  const latest = useRef(0)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  const run = useCallback(
    async (args: A): Promise<T | null> => {
      const id = ++latest.current
      setState((s) => ({ data: s.data, error: null, loading: true }))
      try {
        const data = await action(args)
        if (mounted.current && id === latest.current) setState({ data, error: null, loading: false })
        return data
      } catch (error) {
        if (mounted.current && id === latest.current) setState({ data: null, error, loading: false })
        return null
      }
    },
    [action],
  )

  const reset = useCallback(() => {
    latest.current++
    setState({ data: null, error: null, loading: false })
  }, [])

  return { ...state, run, reset }
}
