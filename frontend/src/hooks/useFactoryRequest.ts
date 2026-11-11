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

/**
 * Fetch whenever ``key`` changes; ``key === null`` means "do not fetch".
 * ``fetcher`` must be stable (a module-level function), since it is an
 * effect dependency.
 */
export function useFactoryQuery<K, T>(key: K | null, fetcher: (key: K) => Promise<T>) {
  const [state, setState] = useState<RequestState<T>>({ data: null, error: null, loading: key !== null })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    if (key === null) return
    let current = true
    setState((s) => ({ data: s.data, error: null, loading: true }))
    fetcher(key).then(
      (data) => current && setState({ data, error: null, loading: false }),
      (error) => current && setState({ data: null, error, loading: false }),
    )
    return () => {
      current = false
    }
  }, [key, fetcher, attempt])

  const reload = useCallback(() => setAttempt((n) => n + 1), [])
  return { ...state, reload }
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
