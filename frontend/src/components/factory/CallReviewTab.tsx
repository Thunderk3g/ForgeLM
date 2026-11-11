// Call review tab: screen the agent's side of a sales call (POST
// /review-call) with the BFSI pack's three tiers: deterministic rules for
// required disclosures and red-flag phrases, an optional trained classifier,
// and a list of utterances routed to a person.
//
// Nothing here decides anything. The service says so in its response and the
// page repeats it above the results: a flag is a prompt for human review,
// never a finding of a violation. Each flag is shown with the exact text that
// triggered it, highlighted inside its utterance, so the reviewer can judge
// the evidence rather than the label.

import { Fragment, useId, useState, type FormEvent } from "react"
import { CheckCircle2, CircleAlert, Flag, ShieldAlert, Users, XCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { factoryApi } from "@/lib/factory-api"
import { DEFAULT_LIMITS, fmtNumber, humanize, locateSpan, nonBlankLines } from "@/lib/factory-utils"
import { useFactoryAction, useFactoryQuery } from "@/hooks/useFactoryRequest"
import type { ReviewCallResponse, ReviewFlag } from "@/types/factory"
import { EmptyState, ErrorState, Field, InputProblems, LoadingState } from "./factory-shared"

const NO_CLASSIFIER = "__rules_only__"

// Illustrative agent lines, written for this page; not real call data.
const SAMPLE_CALL = [
  "Good morning sir, this call is being recorded for quality and training purposes.",
  "This plan is just like a fixed deposit, your money will double in 7 years.",
  "Guaranteed 12 percent return every year, no risk at all.",
  "You pay the premium for 5 years and the policy term is 10 years.",
  "You also get a free-look period of 30 days to cancel.",
  "The offer is valid today only, so please decide now.",
].join("\n")

/**
 * Split an utterance into plain and highlighted runs. Ranges may overlap (two
 * rules can match the same words), so characters are marked first and runs
 * grouped afterwards.
 */
function segments(text: string, ranges: [number, number][]): { text: string; marked: boolean }[] {
  if (!ranges.length) return [{ text, marked: false }]
  const mask = new Array<boolean>(text.length).fill(false)
  for (const [a, b] of ranges) for (let i = a; i < b; i++) mask[i] = true
  const out: { text: string; marked: boolean }[] = []
  for (let i = 0; i < text.length; i++) {
    const last = out[out.length - 1]
    if (last && last.marked === mask[i]) last.text += text[i]
    else out.push({ text: text[i], marked: mask[i] })
  }
  return out
}

function HighlightedUtterance({ text, flags }: { text: string; flags: ReviewFlag[] }) {
  const located = flags.map((f) => locateSpan(text, f.span))
  const ranges = located.filter((r): r is [number, number] => r !== null)
  const unlocated = flags.filter((_, i) => located[i] === null)
  return (
    <>
      <p className="leading-relaxed">
        {segments(text, ranges).map((s, i) =>
          s.marked ? (
            <mark key={i} className="rounded-sm bg-chart-3/30 px-0.5 text-foreground">
              {s.text}
            </mark>
          ) : (
            <Fragment key={i}>{s.text}</Fragment>
          ),
        )}
      </p>
      {unlocated.map((f, i) => (
        <p key={i} className="text-xs text-muted-foreground">
          Matched span (not found verbatim in the text): <q>{f.span}</q>
        </p>
      ))}
    </>
  )
}

function FlagBadge({ flag }: { flag: ReviewFlag }) {
  return (
    <li className="flex flex-wrap items-center gap-1.5 text-xs">
      <Badge variant="outline" className="border-chart-3/40 text-chart-3">
        <Flag aria-hidden="true" />
        {humanize(flag.category)}
      </Badge>
      <Badge variant="secondary">{flag.tier === "rule" ? "rule" : "classifier"}</Badge>
      {flag.confidence !== null && <span className="text-muted-foreground">confidence {fmtNumber(flag.confidence, 2)}</span>}
      {flag.tier === "rule" && (
        <span className="text-muted-foreground">
          on <q>{flag.span}</q>
        </span>
      )}
    </li>
  )
}

function ReviewResult({ result, utterances }: { result: ReviewCallResponse; utterances: string[] }) {
  const uid = useId()
  const [onlyRouted, setOnlyRouted] = useState(false)
  const disclosures = Object.entries(result.disclosures)
  const found = disclosures.filter(([, span]) => span !== null)
  const routed = new Set(result.needs_review)
  const byUtterance = new Map<number, ReviewFlag[]>()
  for (const f of result.flags) byUtterance.set(f.utterance_index, [...(byUtterance.get(f.utterance_index) ?? []), f])
  const shown = utterances.map((text, i) => ({ text, i })).filter(({ i }) => !onlyRouted || routed.has(i))

  return (
    <div className="space-y-6">
      <Alert className="border-chart-3/50 bg-chart-3/10">
        <ShieldAlert aria-hidden="true" />
        <AlertTitle className="line-clamp-none">Flags are prompts for human review, not findings</AlertTitle>
        <AlertDescription>
          <p className="font-medium text-foreground">Service: "{result.disclaimer}".</p>
          <p>
            The disclosure checklist and patterns are illustrative defaults, not a statement of regulatory
            requirements. A person decides.
          </p>
        </AlertDescription>
      </Alert>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card className="py-4">
          <CardContent>
            <p className="text-xs text-muted-foreground">Disclosures found</p>
            <p className="text-2xl font-bold">
              {found.length} / {disclosures.length}
            </p>
          </CardContent>
        </Card>
        <Card className="py-4">
          <CardContent>
            <p className="text-xs text-muted-foreground">Flags</p>
            <p className="text-2xl font-bold">{result.flags.length}</p>
          </CardContent>
        </Card>
        <Card className="py-4">
          <CardContent>
            <p className="text-xs text-muted-foreground">Utterances routed for review</p>
            <p className="text-2xl font-bold">{result.needs_review.length}</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Required disclosures</CardTitle>
          <CardDescription>Each should be said at least once in the call. The first matching words are shown.</CardDescription>
        </CardHeader>
        <CardContent>
          {disclosures.length === 0 ? (
            <p className="text-sm text-muted-foreground">The service checked no disclosures.</p>
          ) : (
            <ul className="grid gap-2 sm:grid-cols-2">
              {disclosures.map(([check, span]) => (
                <li
                  key={check}
                  className={`flex items-start gap-2 rounded-md border p-3 text-sm ${span === null ? "border-destructive/40" : ""}`}
                >
                  {span === null ? (
                    <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" aria-hidden="true" />
                  ) : (
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-chart-2" aria-hidden="true" />
                  )}
                  <div className="min-w-0">
                    <p className="font-medium">
                      {humanize(check)}
                      <span className="sr-only">{span === null ? ": missing" : ": found"}</span>
                    </p>
                    {span === null ? (
                      <p className="text-xs text-destructive">missing</p>
                    ) : (
                      <p className="break-words text-xs text-muted-foreground">
                        <q>{span}</q>
                      </p>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-1.5">
            <CardTitle className="text-base">Utterances</CardTitle>
            <CardDescription>Highlighted words triggered a flag. Routed utterances go to a reviewer.</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Switch id={`${uid}-only`} checked={onlyRouted} onCheckedChange={setOnlyRouted} />
            <Label htmlFor={`${uid}-only`} className="font-normal">
              Only routed
            </Label>
          </div>
        </CardHeader>
        <CardContent>
          {shown.length === 0 ? (
            <EmptyState title={onlyRouted ? "Nothing was routed for review" : "No utterances"} />
          ) : (
            <ol className="space-y-2">
              {shown.map(({ text, i }) => {
                const flags = byUtterance.get(i) ?? []
                const isRouted = routed.has(i)
                return (
                  <li
                    key={i}
                    className={`rounded-md border p-3 text-sm ${isRouted ? "border-chart-3/50 bg-chart-3/5" : ""}`}
                  >
                    <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <span className="font-mono">#{i + 1}</span>
                      {isRouted && (
                        <Badge variant="outline" className="border-chart-3/50 text-chart-3">
                          <Users aria-hidden="true" />
                          routed for review
                        </Badge>
                      )}
                      {isRouted && flags.length === 0 && <span>the classifier was unsure</span>}
                    </div>
                    <HighlightedUtterance text={text} flags={flags} />
                    {flags.length > 0 && (
                      <ul className="mt-2 space-y-1" aria-label={`Flags on utterance ${i + 1}`}>
                        {flags.map((f, k) => (
                          <FlagBadge key={k} flag={f} />
                        ))}
                      </ul>
                    )}
                  </li>
                )
              })}
            </ol>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export function CallReviewTab() {
  const uid = useId()
  const [text, setText] = useState("")
  const [classifier, setClassifier] = useState(NO_CLASSIFIER)
  const [flagThreshold, setFlagThreshold] = useState("0.6")
  const [unsureBelow, setUnsureBelow] = useState("0.5")
  const [problems, setProblems] = useState<string[]>([])
  const [sent, setSent] = useState<string[]>([])
  const review = useFactoryAction(factoryApi.reviewCall)
  const classifiers = useFactoryQuery("all", factoryApi.listClassifiers)

  const lines = nonBlankLines(text)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    const found: string[] = []
    if (!lines.length) found.push("Paste at least one utterance, one per line.")
    const ft = Number(flagThreshold)
    const ub = Number(unsureBelow)
    const useClassifier = classifier !== NO_CLASSIFIER
    if (useClassifier && (flagThreshold.trim() === "" || !(ft >= 0 && ft <= 1)))
      found.push("Flag threshold must be between 0 and 1.")
    if (useClassifier && (unsureBelow.trim() === "" || !(ub >= 0 && ub <= 1)))
      found.push("Unsure-below must be between 0 and 1.")
    setProblems(found)
    if (found.length) return
    setSent(lines)
    await review.run({
      utterances: lines,
      classifier: useClassifier ? classifier : null,
      ...(useClassifier ? { flag_threshold: ft, unsure_below: ub } : {}),
    })
  }

  return (
    <div className="grid gap-6 xl:grid-cols-5">
      <Card className="h-fit xl:col-span-2">
        <CardHeader>
          <CardTitle>Screen a sales call</CardTitle>
          <CardDescription>
            The agent's side only, one utterance per line. Rules check required disclosures and red-flag phrases; a
            classifier trained on the BFSI taxonomy can add its own flags.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="space-y-5" noValidate>
            <div className="space-y-1.5">
              <div className="flex items-end justify-between gap-2">
                <Label htmlFor={`${uid}-utts`}>Agent utterances</Label>
                <Button type="button" variant="outline" size="sm" onClick={() => setText(SAMPLE_CALL)}>
                  Insert example
                </Button>
              </div>
              <Textarea
                id={`${uid}-utts`}
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="min-h-48"
                placeholder="One utterance per line"
                aria-describedby={`${uid}-utts-count`}
              />
              <p id={`${uid}-utts-count`} className="text-xs text-muted-foreground">
                {lines.length} utterance{lines.length === 1 ? "" : "s"}; blank lines are ignored.
                {lines.length > DEFAULT_LIMITS.maxPredictRows &&
                  ` The service accepts ${DEFAULT_LIMITS.maxPredictRows} per request by default.`}
              </p>
            </div>

            <Field
              id={`${uid}-clf`}
              label="Classifier (optional)"
              hint="Only a classifier trained on the BFSI sales-call labels adds flags; any other only marks unsure lines."
            >
              <Select value={classifier} onValueChange={setClassifier}>
                <SelectTrigger id={`${uid}-clf`} className="w-full" aria-describedby={`${uid}-clf-hint`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_CLASSIFIER}>None: rules only</SelectItem>
                  {(classifiers.data?.classifiers ?? []).map((c) => (
                    <SelectItem key={c.name} value={c.name}>
                      {c.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            {classifiers.loading && <p className="text-xs text-muted-foreground">Loading classifiers...</p>}
            {classifiers.error !== null && (
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">
                  The classifier list could not be loaded; rules-only screening may still work.
                </p>
                <ErrorState error={classifiers.error} onRetry={classifiers.reload} />
              </div>
            )}
            {classifiers.data && classifiers.data.classifiers.length === 0 && (
              <p className="text-xs text-muted-foreground">No trained classifiers yet; screening uses rules only.</p>
            )}

            {classifier !== NO_CLASSIFIER && (
              <div className="grid grid-cols-2 gap-3">
                <Field id={`${uid}-ft`} label="Flag threshold" hint="Flag a risk label at or above this confidence.">
                  <Input
                    id={`${uid}-ft`}
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    inputMode="decimal"
                    value={flagThreshold}
                    onChange={(e) => setFlagThreshold(e.target.value)}
                    aria-describedby={`${uid}-ft-hint`}
                  />
                </Field>
                <Field id={`${uid}-ub`} label="Unsure below" hint="Route to a person below this confidence.">
                  <Input
                    id={`${uid}-ub`}
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    inputMode="decimal"
                    value={unsureBelow}
                    onChange={(e) => setUnsureBelow(e.target.value)}
                    aria-describedby={`${uid}-ub-hint`}
                  />
                </Field>
              </div>
            )}

            <InputProblems problems={problems} />
            <Button type="submit" className="w-full" disabled={review.loading}>
              <CircleAlert className="mr-2 h-4 w-4" aria-hidden="true" />
              {review.loading ? "Screening..." : "Screen the call"}
            </Button>
          </form>
        </CardContent>
      </Card>

      <div className="space-y-6 xl:col-span-3">
        {review.loading && <LoadingState label="Screening the call..." />}
        {review.error !== null && <ErrorState error={review.error} />}
        {!review.loading && !review.error && !review.data && (
          <Card>
            <CardContent className="space-y-2 py-10 text-center text-sm text-muted-foreground">
              <p>Paste the agent's utterances and press Screen the call.</p>
              <p>Flags are prompts for human review, not findings of a violation.</p>
            </CardContent>
          </Card>
        )}
        {review.data && !review.loading && <ReviewResult result={review.data} utterances={sent} />}
      </div>
    </div>
  )
}
