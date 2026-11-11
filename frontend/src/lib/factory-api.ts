// Typed client for the factory service (src/microservices/factory_service).
//
// The rest of the dashboard talks to the API gateway on port 8000 (see
// services/api.ts), and the gateway routes /api/v1/factory to the factory
// service, so that is the default here too. VITE_FACTORY_API_BASE overrides
// the whole base, for example to reach the service directly during
// development.
//
// fetch is used rather than axios so the error path can keep the HTTP status
// and FastAPI's "detail" separately: the UI shows both, because "the service
// is down" (no status, or 502/503/504 from the gateway) and "the service
// rejected this input" (422) call for different actions by the user.

import type {
  ClassifierCardResponse,
  ClassifierListResponse,
  ErrorBody,
  HardwareResponse,
  PlanRequest,
  PlanResponse,
  PredictRequest,
  PredictResponse,
  ReviewCallRequest,
  ReviewCallResponse,
  TrainRequest,
  TrainResponse,
  ValidationIssue,
} from "@/types/factory"

const GATEWAY = import.meta.env.VITE_API_BASE || "http://localhost:8000"
export const FACTORY_BASE: string = (import.meta.env.VITE_FACTORY_API_BASE || `${GATEWAY}/api/v1/factory`).replace(
  /\/+$/,
  "",
)

/** A failed call. ``status`` is null when no HTTP response arrived at all. */
export class FactoryApiError extends Error {
  readonly status: number | null
  readonly path: string

  constructor(status: number | null, message: string, path: string) {
    super(message)
    this.name = "FactoryApiError"
    this.status = status
    this.path = path
  }
}

function describeIssue(issue: ValidationIssue): string {
  // loc starts with where the value came from ("body", "query"); the rest
  // names the field, which is what the user can act on.
  const where = issue.loc.filter((part, i) => !(i === 0 && (part === "body" || part === "query")))
  return where.length ? `${where.join(".")}: ${issue.msg}` : issue.msg
}

function messageFrom(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as ErrorBody).detail
    if (typeof detail === "string") return detail
    if (Array.isArray(detail)) return detail.map(describeIssue).join("; ")
  }
  return fallback
}

async function request<T>(path: string, init?: { method?: string; body?: unknown; query?: Record<string, string> }) {
  const query = init?.query && Object.keys(init.query).length ? `?${new URLSearchParams(init.query)}` : ""
  const url = `${FACTORY_BASE}${path}${query}`
  let response: Response
  try {
    response = await fetch(url, {
      method: init?.method ?? "GET",
      headers: init?.body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: init?.body !== undefined ? JSON.stringify(init.body) : undefined,
    })
  } catch (err) {
    // fetch rejects only when no response arrived: the gateway is not
    // running, the host is unreachable, or CORS blocked the request.
    const reason = err instanceof Error ? err.message : String(err)
    throw new FactoryApiError(null, `could not reach ${FACTORY_BASE} (${reason})`, path)
  }
  const text = await response.text()
  let body: unknown = undefined
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = undefined // a proxy's HTML error page, say; reported below by status
    }
  }
  if (!response.ok) {
    const fallback = text && body === undefined ? text.slice(0, 300) : response.statusText || "request failed"
    throw new FactoryApiError(response.status, messageFrom(body, fallback), path)
  }
  if (body === undefined) {
    throw new FactoryApiError(response.status, "the response was not JSON", path)
  }
  return body as T
}

const enc = encodeURIComponent

export const factoryApi = {
  /** The service's own machine, or a named reference profile. */
  hardware: (reference?: string | null) =>
    request<HardwareResponse>("/hardware", { query: reference ? { reference } : undefined }),

  plan: (req: PlanRequest) => request<PlanResponse>("/plan", { method: "POST", body: req }),

  listClassifiers: () => request<ClassifierListResponse>("/classifiers"),

  /** Synchronous: the response arrives when training has finished. */
  trainClassifier: (req: TrainRequest) => request<TrainResponse>("/classifiers", { method: "POST", body: req }),

  classifierCard: (name: string) => request<ClassifierCardResponse>(`/classifiers/${enc(name)}`),

  predict: (name: string, req: PredictRequest) =>
    request<PredictResponse>(`/classifiers/${enc(name)}/predict`, { method: "POST", body: req }),

  reviewCall: (req: ReviewCallRequest) => request<ReviewCallResponse>("/review-call", { method: "POST", body: req }),
}

/** A one-line, human-readable account of a failure, status first. */
export function describeError(err: unknown): { title: string; message: string; hint?: string } {
  if (err instanceof FactoryApiError) {
    if (err.status === null) {
      return {
        title: "Factory service unreachable",
        message: err.message,
        hint: "Is the API gateway running, and is the factory service (port 8006) up?",
      }
    }
    const title = `HTTP ${err.status}`
    switch (err.status) {
      case 502:
      case 503:
        return { title, message: err.message, hint: "The gateway answered but could not reach the factory service." }
      case 504:
        return {
          title,
          message: err.message,
          hint:
            "The gateway stopped waiting for the service (it allows 30 seconds). A training run may still " +
            "finish on the server; refresh the classifier list in a minute.",
        }
      case 404:
        // The service's own 404s name what is missing ("no classifier named
        // 'x'"). FastAPI's bare "Not Found" means no route matched the path at
        // all, which for a fixed path like /hardware means the request reached
        // a server that does not serve it under this path. The known cause is
        // the API gateway stripping "/api/v1/factory" before forwarding, while
        // the factory service only serves paths that keep it.
        if (err.message === "Not Found") {
          return {
            title,
            message: `${err.message} (${FACTORY_BASE}${err.path})`,
            hint:
              "No route matched this path. If the request went through the API gateway, the gateway may be " +
              "forwarding it without the /api/v1/factory prefix the factory service expects. Setting " +
              "VITE_FACTORY_API_BASE to the service's own address (port 8006, /api/v1/factory) bypasses the " +
              "gateway, provided the service allows this page's origin (CORS).",
          }
        }
        return { title, message: err.message }
      case 500:
        return {
          title,
          message: err.message,
          hint:
            "The service failed while handling the request; its log has the traceback. When training, a base " +
            "model that the chosen student cannot load is a common cause.",
        }
      case 413:
        return { title, message: err.message, hint: "The request is larger than the service accepts; split it." }
      case 422:
        return { title, message: err.message, hint: "The service rejected the input." }
      default:
        return { title, message: err.message }
    }
  }
  return { title: "Unexpected error", message: err instanceof Error ? err.message : String(err) }
}
