// Plan tab: describe the task and the constraints, get back the planner's
// approach and its ordered candidates (POST /plan).
//
// The planner's value is as much in *why* and *how sure* as in *what*, so
// every candidate shows its evidence level, the claim and source behind it
// on demand, and plainly says when it cannot run on the chosen hardware.

import { useId, useState, type FormEvent } from "react"
import { AlertTriangle, ChevronDown, ClipboardList, Lightbulb, Rocket, ShieldCheck, XCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { factoryApi } from "@/lib/factory-api"
import { humanize } from "@/lib/factory-utils"
import { useFactoryAction } from "@/hooks/useFactoryRequest"
import { TASK_KINDS, type EvidenceEntry, type PlanRequest, type PlanStep, type TaskKind } from "@/types/factory"
import { EvidenceBadge, EvidenceLegend, ErrorState, Field, InputProblems, LoadingState, Stat } from "./factory-shared"

const THIS_MACHINE = "__probe__"
const SAME_AS_TRAINING = "__same__"

const TASK_LABEL: Record<TaskKind, string> = {
  text_classification: "Text classification",
  tabular_classification: "Tabular classification",
  generation: "Generation (fine-tune an LLM)",
  document_qa: "Document Q&A",
}

// Numeric inputs are kept as the text the user typed, so a field can be
// empty while it is being edited; they are parsed and checked on submit.
interface FormState {
  task: TaskKind
  n_classes: string
  labelled_per_class: string
  n_unlabelled: string
  avg_tokens: string
  n_rows: string
  n_features: string
  model_params_billion: string
  max_latency_ms: string
  multilingual: boolean
  needs_world_knowledge: boolean
  has_text_columns: boolean
  temporal_or_grouped: boolean
  knowledge_changes_often: boolean
  llm_at_inference: boolean
  teacher_available: boolean
  commercial_use: boolean
  needs_inherent_explainability: boolean
  data_must_stay_on_prem: boolean
  hardware: string
  deploy: string
}

// The service's defaults (PlanRequest), except the sizes a user must think
// about, which start at values that make a meaningful first plan.
const INITIAL: FormState = {
  task: "text_classification",
  n_classes: "10",
  labelled_per_class: "8",
  n_unlabelled: "0",
  avg_tokens: "32",
  n_rows: "10000",
  n_features: "20",
  model_params_billion: "3",
  max_latency_ms: "",
  multilingual: false,
  needs_world_knowledge: false,
  has_text_columns: false,
  temporal_or_grouped: false,
  knowledge_changes_often: false,
  llm_at_inference: false,
  teacher_available: false,
  commercial_use: true,
  needs_inherent_explainability: false,
  data_must_stay_on_prem: false,
  hardware: THIS_MACHINE,
  deploy: SAME_AS_TRAINING,
}

type NumberKey =
  | "n_classes"
  | "labelled_per_class"
  | "n_unlabelled"
  | "avg_tokens"
  | "n_rows"
  | "n_features"
  | "model_params_billion"
  | "max_latency_ms"
type BoolKey = {
  [K in keyof FormState]: FormState[K] extends boolean ? K : never
}[keyof FormState]

// The bounds of PlanRequest's Field(...) declarations.
const NUMBER_RULES: Record<NumberKey, { label: string; integer: boolean; min: number; minExclusive?: boolean; max?: number }> = {
  n_classes: { label: "Classes", integer: true, min: 1 },
  labelled_per_class: { label: "Labelled examples per class", integer: false, min: 0 },
  n_unlabelled: { label: "Unlabelled examples", integer: true, min: 0 },
  avg_tokens: { label: "Average tokens per text", integer: true, min: 1 },
  n_rows: { label: "Rows", integer: true, min: 0 },
  n_features: { label: "Features", integer: true, min: 0 },
  model_params_billion: { label: "Model size (billion parameters)", integer: false, min: 0, minExclusive: true, max: 2000 },
  max_latency_ms: { label: "Latency budget (ms per item)", integer: false, min: 0, minExclusive: true },
}

// Which inputs each task kind reads (see the _plan_* functions in
// forgelm/planner.py). Inputs a kind ignores are hidden and not sent.
const TASK_NUMBERS: Record<TaskKind, NumberKey[]> = {
  text_classification: ["n_classes", "labelled_per_class", "n_unlabelled", "avg_tokens"],
  tabular_classification: ["n_rows", "n_features", "n_classes"],
  generation: ["model_params_billion"],
  document_qa: [],
}
const TASK_FLAGS: Record<TaskKind, { key: BoolKey; label: string }[]> = {
  text_classification: [
    { key: "multilingual", label: "Texts are in more than one language" },
    { key: "needs_world_knowledge", label: "Labels need world knowledge the texts do not contain" },
  ],
  tabular_classification: [
    { key: "has_text_columns", label: "Some columns hold free text" },
    { key: "temporal_or_grouped", label: "Rows are ordered in time or grouped (split by time or group)" },
  ],
  generation: [{ key: "knowledge_changes_often", label: "The knowledge it must use changes often" }],
  document_qa: [],
}
const CONSTRAINTS: { key: BoolKey; label: string }[] = [
  { key: "llm_at_inference", label: "The deployed system may call an LLM" },
  { key: "teacher_available", label: "An LLM is available at build time (as a teacher)" },
  { key: "commercial_use", label: "Commercial use (licences must allow it)" },
  { key: "needs_inherent_explainability", label: "Predictions must be explainable by the model itself" },
  { key: "data_must_stay_on_prem", label: "Data must stay on premises" },
]

function buildRequest(form: FormState): { request: PlanRequest; problems: string[] } {
  const problems: string[] = []
  const request: PlanRequest = { task: form.task }
  const keys: NumberKey[] = [...TASK_NUMBERS[form.task], "max_latency_ms"]
  for (const key of keys) {
    const rule = NUMBER_RULES[key]
    const raw = form[key].trim()
    if (raw === "") {
      if (key === "max_latency_ms") continue // optional: no budget
      problems.push(`${rule.label} is required`)
      continue
    }
    const value = Number(raw)
    if (!Number.isFinite(value)) problems.push(`${rule.label} must be a number`)
    else if (rule.integer && !Number.isInteger(value)) problems.push(`${rule.label} must be a whole number`)
    else if (rule.minExclusive ? value <= rule.min : value < rule.min)
      problems.push(`${rule.label} must be ${rule.minExclusive ? "greater than" : "at least"} ${rule.min}`)
    else if (rule.max !== undefined && value > rule.max) problems.push(`${rule.label} must be at most ${rule.max}`)
    else request[key] = value
  }
  for (const { key } of [...TASK_FLAGS[form.task], ...CONSTRAINTS]) request[key] = form[key]
  request.hardware = form.hardware === THIS_MACHINE ? null : form.hardware
  request.deploy = form.deploy === SAME_AS_TRAINING ? null : form.deploy
  return { request, problems }
}

function SourceText({ source }: { source: string }) {
  // Sources are a repository path or URLs joined by " ; ". URLs become links.
  const parts = source.split(/\s+;\s+/)
  return (
    <span className="break-all">
      {parts.map((part, i) => (
        <span key={i}>
          {i > 0 && "; "}
          {/^https?:\/\//.test(part) ? (
            <a href={part} target="_blank" rel="noreferrer noopener" className="underline underline-offset-2">
              {part}
            </a>
          ) : (
            <code className="text-xs">{part}</code>
          )}
        </span>
      ))}
    </span>
  )
}

function EvidenceList({ entries }: { entries: EvidenceEntry[] }) {
  if (!entries.length) {
    return (
      <p className="text-sm text-muted-foreground">
        No evidence entry is cited: this step is an assumption, a reasoned default to check on your own data.
      </p>
    )
  }
  return (
    <ul className="space-y-3">
      {entries.map((e) => (
        <li key={e.key} className="rounded-md border bg-muted/30 p-3 text-sm">
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <EvidenceBadge level={e.level} />
            <code className="text-xs text-muted-foreground">{e.key}</code>
          </div>
          <p>{e.claim}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Source: <SourceText source={e.source} />
          </p>
        </li>
      ))}
    </ul>
  )
}

function detailValue(value: unknown): string {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean"
    ? String(value)
    : JSON.stringify(value)
}

function StepCard({ step, index }: { step: PlanStep; index: number }) {
  const [open, setOpen] = useState(false)
  const details = Object.entries(step.detail ?? {})
  return (
    <li
      className={`rounded-lg border p-4 ${step.feasible ? "" : "border-dashed bg-muted/20"}`}
      aria-label={`Candidate ${index + 1}: ${step.name}`}
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-3">
          <span
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary"
            aria-hidden="true"
          >
            {index + 1}
          </span>
          <h4 className="pt-0.5 font-semibold">{step.name}</h4>
        </div>
        <div className="flex flex-wrap gap-2 pl-10 sm:pl-0">
          <EvidenceBadge level={step.evidence_level} />
          {!step.feasible && (
            <Badge variant="outline" className="border-destructive/40 text-destructive">
              not runnable here
            </Badge>
          )}
        </div>
      </div>
      <dl className="mt-3 space-y-2 pl-10 text-sm">
        <div>
          <dt className="text-xs uppercase tracking-wide text-muted-foreground">Do</dt>
          <dd className="break-words">{step.action}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-muted-foreground">Why</dt>
          <dd className="text-muted-foreground">{step.why}</dd>
        </div>
        {step.missing.length > 0 && (
          <div>
            <dt className="text-xs uppercase tracking-wide text-muted-foreground">Missing packages</dt>
            <dd>
              <code className="rounded bg-muted px-1 py-0.5 text-xs">pip install {step.missing.join(" ")}</code>
            </dd>
          </div>
        )}
        {details.length > 0 && (
          <div>
            <dt className="text-xs uppercase tracking-wide text-muted-foreground">Detail</dt>
            <dd className="flex flex-wrap gap-1.5 pt-1">
              {details.map(([k, v]) => (
                <Badge key={k} variant="secondary" className="max-w-full whitespace-normal font-normal">
                  {humanize(k)}: {detailValue(v)}
                </Badge>
              ))}
            </dd>
          </div>
        )}
      </dl>
      <Collapsible open={open} onOpenChange={setOpen} className="mt-3 pl-10">
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="-ml-3">
            <ChevronDown className={`h-4 w-4 transition-transform ${open ? "rotate-180" : ""}`} aria-hidden="true" />
            {open ? "Hide" : "Show"} evidence ({step.evidence.length})
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="pt-2">
          <EvidenceList entries={step.evidence} />
        </CollapsibleContent>
      </Collapsible>
    </li>
  )
}

function BulletCard({
  title,
  icon: Icon,
  items,
  description,
}: {
  title: string
  icon: typeof Rocket
  items: string[]
  description?: string
}) {
  if (!items.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Icon className="h-4 w-4 text-primary" aria-hidden="true" />
          {title}
        </CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        <ul className="list-disc space-y-1.5 pl-5 text-sm">
          {items.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </CardContent>
    </Card>
  )
}

export function PlanTab({ referenceProfiles }: { referenceProfiles: string[] }) {
  const [form, setForm] = useState<FormState>(INITIAL)
  const [problems, setProblems] = useState<string[]>([])
  const planCall = useFactoryAction(factoryApi.plan)
  const uid = useId()

  const set = <K extends keyof FormState>(key: K, value: FormState[K]) => setForm((f) => ({ ...f, [key]: value }))

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const { request, problems } = buildRequest(form)
    setProblems(problems)
    if (!problems.length) void planCall.run(request)
  }

  const numberInput = (key: NumberKey, hint?: string) => {
    const rule = NUMBER_RULES[key]
    const id = `${uid}-${key}`
    return (
      <Field key={key} id={id} label={rule.label} hint={hint}>
        <Input
          id={id}
          type="number"
          inputMode={rule.integer ? "numeric" : "decimal"}
          step={rule.integer ? 1 : "any"}
          min={rule.minExclusive ? undefined : rule.min}
          max={rule.max}
          value={form[key]}
          onChange={(e) => set(key, e.target.value)}
          aria-describedby={hint ? `${id}-hint` : undefined}
        />
      </Field>
    )
  }

  const switchRow = ({ key, label }: { key: BoolKey; label: string }) => {
    const id = `${uid}-${key}`
    return (
      <div key={key} className="flex items-start gap-3">
        <Switch id={id} checked={form[key]} onCheckedChange={(v) => set(key, v)} className="mt-0.5" />
        <Label htmlFor={id} className="font-normal leading-snug">
          {label}
        </Label>
      </div>
    )
  }

  const plan = planCall.data?.plan

  return (
    <div className="grid gap-6 xl:grid-cols-5">
      <Card className="xl:col-span-2 h-fit">
        <CardHeader>
          <CardTitle>Describe the task</CardTitle>
          <CardDescription>The planner returns the cheapest approach the evidence supports, and why.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="space-y-6" noValidate>
            <Field id={`${uid}-task`} label="Task">
              <Select value={form.task} onValueChange={(v) => set("task", v as TaskKind)}>
                <SelectTrigger id={`${uid}-task`} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TASK_KINDS.map((k) => (
                    <SelectItem key={k} value={k}>
                      {TASK_LABEL[k]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            {TASK_NUMBERS[form.task].length > 0 ? (
              <div className="grid gap-4 sm:grid-cols-2">
                {TASK_NUMBERS[form.task].map((key) =>
                  numberInput(
                    key,
                    key === "labelled_per_class"
                      ? "Human-labelled; 0 if none yet."
                      : key === "model_params_billion"
                        ? "Between 0 and 2000."
                        : undefined,
                  ),
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                This task kind takes no size inputs; the plan depends on the constraints and hardware below.
              </p>
            )}
            {TASK_FLAGS[form.task].length > 0 && <div className="space-y-3">{TASK_FLAGS[form.task].map(switchRow)}</div>}

            <fieldset className="space-y-3">
              <legend className="mb-2 text-sm font-semibold">Constraints</legend>
              {CONSTRAINTS.map(switchRow)}
              {numberInput("max_latency_ms", "Optional; per item, on the deployment hardware.")}
            </fieldset>

            <fieldset className="grid gap-4 sm:grid-cols-2">
              <legend className="mb-2 text-sm font-semibold">Hardware</legend>
              <Field id={`${uid}-hw`} label="Train on">
                <Select value={form.hardware} onValueChange={(v) => set("hardware", v)}>
                  <SelectTrigger id={`${uid}-hw`} className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={THIS_MACHINE}>The service's machine</SelectItem>
                    {referenceProfiles.map((p) => (
                      <SelectItem key={p} value={p}>
                        {p}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              <Field id={`${uid}-deploy`} label="Deploy on">
                <Select value={form.deploy} onValueChange={(v) => set("deploy", v)}>
                  <SelectTrigger id={`${uid}-deploy`} className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={SAME_AS_TRAINING}>Same as training</SelectItem>
                    {referenceProfiles.map((p) => (
                      <SelectItem key={p} value={p}>
                        {p}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              {referenceProfiles.length === 0 && (
                <p className="text-xs text-muted-foreground sm:col-span-2">
                  Reference profiles appear once the hardware endpoint answers.
                </p>
              )}
            </fieldset>

            <InputProblems problems={problems} />
            <Button type="submit" className="w-full" disabled={planCall.loading}>
              <ClipboardList className="mr-2 h-4 w-4" aria-hidden="true" />
              {planCall.loading ? "Planning..." : "Make a plan"}
            </Button>
          </form>
        </CardContent>
      </Card>

      <div className="space-y-6 xl:col-span-3">
        {planCall.loading && <LoadingState label="Asking the planner..." />}
        {planCall.error !== null && <ErrorState error={planCall.error} />}
        {!planCall.loading && !planCall.error && !plan && (
          <Card>
            <CardContent className="py-10 text-center text-sm text-muted-foreground">
              Describe the task and press <span className="font-medium text-foreground">Make a plan</span>. Nothing
              is trained; the planner only reads the hardware profile and its evidence table.
            </CardContent>
          </Card>
        )}
        {plan && !planCall.loading && (
          <>
            {plan.warnings.length > 0 && (
              <Alert>
                <AlertTriangle aria-hidden="true" />
                <AlertTitle>Warnings</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc space-y-1 pl-4">
                    {plan.warnings.map((w) => (
                      <li key={w}>{w}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            <Card>
              <CardHeader>
                <CardDescription>Approach</CardDescription>
                <CardTitle className="text-2xl">{plan.approach}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground">{plan.approach_why}</p>
                <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                  <Stat label="Task" value={TASK_LABEL[plan.task.kind] ?? plan.task.kind} />
                  <Stat label="Train on" value={plan.hardware_class} />
                  <Stat label="Deploy on" value={plan.deploy_class} />
                </dl>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Lightbulb className="h-4 w-4 text-primary" aria-hidden="true" />
                  Candidates, cheapest first
                </CardTitle>
                <CardDescription>
                  Each badge is the weakest evidence behind that step. "Reported" and "assumption" are hypotheses to
                  test on your own data.
                </CardDescription>
                <EvidenceLegend />
              </CardHeader>
              <CardContent>
                {plan.steps.length ? (
                  <ol className="space-y-3">
                    {plan.steps.map((s, i) => (
                      <StepCard key={`${i}-${s.name}`} step={s} index={i} />
                    ))}
                  </ol>
                ) : (
                  <p className="text-sm text-muted-foreground">The planner returned no candidates.</p>
                )}
              </CardContent>
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              <BulletCard title="Deployment" icon={Rocket} items={plan.deployment} />
              <BulletCard
                title="Guardrails"
                icon={ShieldCheck}
                items={plan.guardrails}
                description="Not optional: these are what make the numbers trustworthy."
              />
            </div>

            {plan.not_recommended.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <XCircle className="h-4 w-4 text-destructive" aria-hidden="true" />
                    Not recommended
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 text-sm">
                    {plan.not_recommended.map((n) => (
                      <li key={n.what}>
                        <span className="font-medium">{n.what}</span>
                        <span className="text-muted-foreground">: {n.why}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}

            {planCall.data?.text && (
              <details className="rounded-lg border p-4">
                <summary className="cursor-pointer text-sm font-medium">Plan as plain text</summary>
                <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs">{planCall.data.text}</pre>
              </details>
            )}
          </>
        )}
      </div>
    </div>
  )
}
