// Small building blocks shared by the Factory tabs: the loading, empty and
// error states every call needs, the evidence-level badge, and a labelled
// form field. Kept here so the four tabs say the same thing the same way.

import type { ReactNode } from "react"
import { AlertTriangle, Inbox, Loader2, RefreshCw, ServerCrash } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { describeError } from "@/lib/factory-api"
import { cn } from "@/lib/utils"
import type { EvidenceLevel } from "@/types/factory"

export function LoadingState({ label }: { label: string }) {
  // role="status" makes screen readers announce the wait without moving focus.
  return (
    <div role="status" className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
      {label}
    </div>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed p-8 text-center">
      <Inbox className="h-6 w-6 text-muted-foreground" aria-hidden="true" />
      <p className="text-sm font-medium">{title}</p>
      {children && <div className="text-sm text-muted-foreground">{children}</div>}
    </div>
  )
}

/**
 * A failed call, said plainly: the HTTP status (or that no response came
 * back at all), the service's own message, and what is likely wrong.
 */
export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const { title, message, hint } = describeError(error)
  return (
    <Alert variant="destructive">
      <ServerCrash aria-hidden="true" />
      <AlertTitle className="text-destructive dark:text-destructive-foreground">{title}</AlertTitle>
      <AlertDescription>
        <p className="break-words text-destructive dark:text-destructive-foreground">{message}</p>
        {hint && <p className="text-muted-foreground">{hint}</p>}
        {onRetry && (
          <Button variant="outline" size="sm" className="mt-2" onClick={onRetry}>
            <RefreshCw className="mr-1 h-3 w-3" aria-hidden="true" />
            Try again
          </Button>
        )}
      </AlertDescription>
    </Alert>
  )
}

/** Problems found in the form before anything was sent. */
export function InputProblems({ problems }: { problems: string[] }) {
  if (!problems.length) return null
  return (
    <Alert variant="destructive">
      <AlertTriangle aria-hidden="true" />
      <AlertTitle className="text-destructive dark:text-destructive-foreground">Fix before sending</AlertTitle>
      <AlertDescription>
        <ul className="list-disc space-y-1 pl-4 text-destructive dark:text-destructive-foreground">
          {problems.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      </AlertDescription>
    </Alert>
  )
}

// Strongest evidence gets the most confident colour; an assumption is drawn
// as a plain outline so it never looks like a finding.
const EVIDENCE_STYLE: Record<EvidenceLevel, string> = {
  measured: "bg-chart-2/15 text-chart-2 border-chart-2/30",
  published: "bg-chart-1/15 text-chart-1 border-chart-1/30",
  reported: "bg-chart-3/15 text-chart-3 border-chart-3/30",
  assumption: "bg-transparent text-muted-foreground border-border",
}

const EVIDENCE_MEANING: Record<EvidenceLevel, string> = {
  measured: "reproduced in this repository",
  published: "read in a primary source",
  reported: "seen only in a secondary write-up",
  assumption: "a reasoned default with no direct evidence",
}

export function EvidenceBadge({ level }: { level: EvidenceLevel }) {
  return (
    <Badge variant="outline" className={cn(EVIDENCE_STYLE[level] ?? "")} title={EVIDENCE_MEANING[level]}>
      {level}
    </Badge>
  )
}

export function EvidenceLegend() {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-2 text-xs text-muted-foreground">
      {(Object.keys(EVIDENCE_MEANING) as EvidenceLevel[]).map((level) => (
        <li key={level} className="flex items-center gap-1.5">
          <EvidenceBadge level={level} />
          {EVIDENCE_MEANING[level]}
        </li>
      ))}
    </ul>
  )
}

/** A label above its control, with an optional hint tied to it by aria-describedby. */
export function Field({
  id,
  label,
  hint,
  className,
  children,
}: {
  id: string
  label: string
  hint?: string
  className?: string
  children: ReactNode
}) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <Label htmlFor={id}>{label}</Label>
      {children}
      {hint && (
        <p id={`${id}-hint`} className="text-xs text-muted-foreground">
          {hint}
        </p>
      )}
    </div>
  )
}

/** A term and its value, for the read-only summaries. */
export function Stat({ label, value, className }: { label: string; value: ReactNode; className?: string }) {
  return (
    <div className={cn("space-y-0.5", className)}>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="text-sm font-medium break-words">{value}</dd>
    </div>
  )
}
