// Browser-side helpers for the Factory page: CSV parsing, the cheap input
// checks that mirror the factory service's own, and small formatters.
//
// The checks here only save a round trip. The service repeats every one of
// them and remains the authority; where it can be configured differently
// (row and length limits are environment variables there) nothing is
// enforced here, so a server with raised limits is never second-guessed.

/** The service's classifier-name rule (main.py ``_NAME``, matched in full). */
export const CLASSIFIER_NAME = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/

/** Defaults of the service's environment-configurable limits, for hints only. */
export const DEFAULT_LIMITS = {
  maxTrainRows: 50000,
  maxPredictRows: 10000,
  maxTextChars: 20000,
}

/** The repo ids the service allows by default (FACTORY_ALLOWED_MODELS). */
export const DEFAULT_ALLOWED_MODELS = [
  "minishlab/potion-base-32M",
  "minishlab/potion-base-8M",
  "BAAI/bge-small-en-v1.5",
  "sentence-transformers/all-MiniLM-L6-v2",
  "intfloat/multilingual-e5-small",
  "answerdotai/ModernBERT-base",
]

/**
 * Python's round(): halves go to the even neighbour. The service sizes each
 * class's hold-out with it, so the minimum computed below must use the same
 * rule or it would disagree at exact halves (2.5 rounds to 2, not 3).
 */
export function pyRound(x: number): number {
  const floor = Math.floor(x)
  const diff = x - floor
  if (diff > 0.5) return floor + 1
  if (diff < 0.5) return floor
  return floor % 2 === 0 ? floor : floor + 1
}

/**
 * Examples each label needs for a given hold-out share: the same search the
 * service does in train_classifier. Every class gives up
 * max(2, round(n * holdout)) examples for calibration and test and must keep
 * at least one to train on; 4 is the floor (at the default 0.2 it is also the
 * answer).
 */
export function minExamplesPerLabel(holdout: number): number {
  for (let n = 4; n < 100; n++) {
    if (n - Math.max(2, pyRound(n * holdout)) >= 1) return n
  }
  return 100
}

export interface ParsedCsv {
  texts: string[]
  labels: string[]
  /** Problems that stop the data being sent as it is. */
  errors: string[]
  /** Rows skipped (blank text or label), reported but not fatal. */
  skipped: number
}

/**
 * Split CSV text into rows of fields (RFC 4180: quoted fields may hold
 * commas, newlines and doubled quotes). Handles \n and \r\n line ends.
 */
export function splitCsv(source: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ""
  let quoted = false
  const text = source.replace(/^﻿/, "") // a byte-order mark from spreadsheet exports
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"'
          i++
        } else {
          quoted = false
        }
      } else {
        field += ch
      }
    } else if (ch === '"' && field === "") {
      quoted = true
    } else if (ch === ",") {
      row.push(field)
      field = ""
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++
      row.push(field)
      rows.push(row)
      row = []
      field = ""
    } else {
      field += ch
    }
  }
  if (field !== "" || row.length) {
    row.push(field)
    rows.push(row)
  }
  // Blank lines become a single empty field; they carry nothing.
  return rows.filter((r) => !(r.length === 1 && r[0].trim() === ""))
}

/** Parse labelled data: a header row naming a ``text`` and a ``label`` column. */
export function parseLabelledCsv(source: string): ParsedCsv {
  const rows = splitCsv(source)
  const out: ParsedCsv = { texts: [], labels: [], errors: [], skipped: 0 }
  if (!rows.length) {
    out.errors.push("no rows found")
    return out
  }
  const header = rows[0].map((h) => h.trim().toLowerCase())
  const ti = header.indexOf("text")
  const li = header.indexOf("label")
  if (ti < 0 || li < 0) {
    out.errors.push(
      `the first row must name a "text" and a "label" column; found ${header.map((h) => `"${h}"`).join(", ") || "nothing"}`,
    )
    return out
  }
  for (const row of rows.slice(1)) {
    const text = (row[ti] ?? "").trim()
    const label = (row[li] ?? "").trim()
    if (!text || !label) {
      out.skipped++
      continue
    }
    out.texts.push(text)
    out.labels.push(label)
  }
  if (!out.texts.length) out.errors.push("the header was found but no row has both a text and a label")
  return out
}

/** Count examples per label, most frequent first. */
export function labelCounts(labels: string[]): [string, number][] {
  const counts = new Map<string, number>()
  for (const l of labels) counts.set(l, (counts.get(l) ?? 0) + 1)
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
}

/** The service's training-data checks that can be made before sending. */
export function trainingDataProblems(labels: string[], holdout: number): string[] {
  const problems: string[] = []
  const counts = labelCounts(labels)
  if (counts.length < 2) problems.push("at least two distinct labels are needed")
  const needed = minExamplesPerLabel(holdout)
  const short = counts.filter(([, n]) => n < needed)
  if (short.length) {
    problems.push(
      `each label needs at least ${needed} examples with a hold-out share of ${holdout}; too few: ` +
        short.map(([l, n]) => `"${l}" (${n})`).join(", "),
    )
  }
  return problems
}

/** Non-blank lines of a text area, trimmed. */
export function nonBlankLines(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
}

/**
 * Where ``span`` sits inside ``text``. The service's rule flags carry the
 * regex match verbatim, so an exact search finds it; a case-insensitive
 * search is the fallback. Returns null when it cannot be located, and the
 * caller then shows the span separately rather than guessing.
 */
export function locateSpan(text: string, span: string): [number, number] | null {
  if (!span) return null
  let at = text.indexOf(span)
  if (at < 0) at = text.toLowerCase().indexOf(span.toLowerCase())
  return at < 0 ? null : [at, at + span.length]
}

/** "returns_not_guaranteed" -> "returns not guaranteed". */
export function humanize(key: string): string {
  return key.replace(/_/g, " ")
}

/** A metric value for display; tolerates the strings the service uses for NaN. */
export function fmtNumber(value: unknown, digits = 3): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value)
    if (value !== 0 && Math.abs(value) < 10 ** -digits) return value.toExponential(1)
    return value.toFixed(digits)
  }
  return value === null || value === undefined ? "n/a" : String(value)
}

export function fmtPercent(value: unknown, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : fmtNumber(value)
}
