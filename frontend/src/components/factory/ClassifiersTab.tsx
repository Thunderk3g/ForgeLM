// Classifiers tab: the LLM-free students the factory service has trained,
// training a new one from labelled data, trying it, and reading its model
// card.
//
// Training is synchronous on the service: the request stays open until the
// student is fitted, calibrated and scored. The result is shown exactly as
// returned, including the calibration temperature and the evaluation
// contract's verdict, because those say whether the confidence numbers in
// "Try it" can be trusted.

import { useEffect, useId, useMemo, useRef, useState, type ChangeEvent, type FormEvent } from "react"
import { CheckCircle2, FileUp, FlaskConical, Play, RefreshCw, ScrollText, XCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"
import { factoryApi } from "@/lib/factory-api"
import {
  CLASSIFIER_NAME,
  DEFAULT_ALLOWED_MODELS,
  DEFAULT_LIMITS,
  fmtNumber,
  fmtPercent,
  labelCounts,
  modelsForStudent,
  nonBlankLines,
  pairPredictions,
  parseLabelledCsv,
  trainingDataProblems,
} from "@/lib/factory-utils"
import { useFactoryAction, useFactoryQuery, type QueryResult } from "@/hooks/useFactoryRequest"
import {
  STUDENT_KINDS,
  type ClassifierCardResponse,
  type ClassifierListResponse,
  type ContractVerdict,
  type Prediction,
  type PredictRequest,
  type StudentKind,
  type TrainRequest,
  type TrainResponse,
} from "@/types/factory"
import { EmptyState, ErrorState, Field, InputProblems, LoadingState, Stat } from "./factory-shared"

const STUDENT_INFO: Record<StudentKind, string> = {
  tfidf: "TF-IDF + logistic regression. No model download; trains in seconds.",
  static: "Static embeddings + linear head. Fast on any CPU; downloads a small model once.",
  encoder: "Frozen sentence encoder + linear head. Most accurate with few labels in this repo's tests.",
  finetuned: "Encoder fine-tuned end to end. Slow on a CPU; may exceed the gateway's 30-second wait.",
}
const DEFAULT_MODEL = "__default__"
// What make_student (forgelm/students.py) loads when no model is named, so the
// "default" choice in the form says which repo it means.
const STUDENT_DEFAULT_MODEL: Record<StudentKind, string | null> = {
  tfidf: null,
  static: "minishlab/potion-base-32M",
  encoder: "BAAI/bge-small-en-v1.5",
  finetuned: "sentence-transformers/all-MiniLM-L6-v2",
}
const MODEL_HINT: Record<StudentKind, string> = {
  tfidf: "TF-IDF has no base model.",
  static: "Static students load model2vec models only (the potion-* repos on the allow-list).",
  encoder: "Encoder students load sentence-transformers models; potion-* static models need the static student.",
  finetuned: "Fine-tuned students load transformer models; potion-* static models need the static student.",
}

const SAMPLE_CSV = `text,label
my card was blocked at the atm,card
card declined at the shop,card
how do I unblock my debit card,card
I lost my credit card,card
loan emi deducted twice,loan
what is the interest on my home loan,loan
can I prepay my personal loan,loan
loan statement not received,loan
`

// The texts travel inside the action's result, next to the predictions the
// service returned for them. The table is drawn from that one object, so it
// can never pair the texts of a new request with the answers to an old one.
interface PredictRun {
  texts: string[]
  threshold: number | undefined // undefined: the service's default was used
  predictions: Prediction[]
}
const predictAction = async (a: { name: string; req: PredictRequest }): Promise<PredictRun> => {
  const response = await factoryApi.predict(a.name, a.req)
  return { texts: a.req.texts, threshold: a.req.threshold, predictions: response.predictions }
}

function ContractBadge({ passed }: { passed: boolean | null | undefined }) {
  if (passed === true)
    return (
      <Badge variant="outline" className="border-chart-2/30 bg-chart-2/15 text-chart-2">
        <CheckCircle2 aria-hidden="true" /> contract passed
      </Badge>
    )
  if (passed === false)
    return (
      <Badge variant="outline" className="border-destructive/40 text-destructive dark:text-destructive-foreground">
        <XCircle aria-hidden="true" /> contract failed
      </Badge>
    )
  return <Badge variant="outline">no contract</Badge>
}

function ContractVerdictView({ verdict }: { verdict: ContractVerdict }) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <ContractBadge passed={verdict.passed} />
        <code className="text-xs text-muted-foreground">{verdict.contract}</code>
      </div>
      {verdict.failures.length > 0 && (
        <ul className="list-disc space-y-1 pl-5 text-sm text-destructive dark:text-destructive-foreground">
          {verdict.failures.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

function TrainResult({ result }: { result: TrainResponse }) {
  const m = result.metrics
  const rows: [string, string, string][] = [
    ["Accuracy", fmtPercent(m.accuracy), "share of test texts labelled correctly"],
    ["Macro F1", fmtNumber(m.macro_f1), "F1 averaged over labels, so rare labels count equally"],
    ["Macro precision", fmtNumber(m.macro_precision), ""],
    ["Macro recall", fmtNumber(m.macro_recall), ""],
    ["ECE", fmtNumber(m.ece), "calibration error: how far confidence is from accuracy (lower is better)"],
    ["Brier score", fmtNumber(m.brier), "lower is better"],
    ["Log loss (NLL)", fmtNumber(m.nll), "lower is better"],
  ]
  return (
    <Card className="border-primary/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FlaskConical className="h-5 w-5 text-primary" aria-hidden="true" />
          Trained {result.name}
        </CardTitle>
        <CardDescription>
          Scored on {result.n_test} held-out test texts that were not used for training or calibration.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {result.n_test < 30 && (
          <Alert>
            <AlertTitle>Small test set</AlertTitle>
            <AlertDescription>
              With {result.n_test} test texts each one moves accuracy by {fmtPercent(1 / Math.max(result.n_test, 1), 0)}
              . Treat these figures as a smoke test, not a measurement.
            </AlertDescription>
          </Alert>
        )}
        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Train / calibration / test" value={`${result.n_train} / ${result.n_calibration} / ${result.n_test}`} />
          <Stat label="Labels" value={result.labels.length} />
          <Stat
            label="Calibration temperature"
            value={<span title="softmax temperature fitted on the calibration split">{fmtNumber(result.temperature)}</span>}
          />
          <Stat label="Serving needs" value={result.serve_requires.join(", ") || "nothing extra"} />
        </dl>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Metric</TableHead>
                <TableHead className="text-right">Value</TableHead>
                <TableHead className="hidden sm:table-cell">Meaning</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map(([name, value, meaning]) => (
                <TableRow key={name}>
                  <TableCell>{name}</TableCell>
                  <TableCell className="text-right font-mono">{value}</TableCell>
                  <TableCell className="hidden whitespace-normal text-muted-foreground sm:table-cell">{meaning}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <section aria-label="Evaluation contract" className="space-y-2">
          <h3 className="text-sm font-semibold">Evaluation contract</h3>
          <ContractVerdictView verdict={result.contract} />
        </section>
        <div>
          <h3 className="mb-1 text-sm font-semibold">Labels</h3>
          <ul className="flex flex-wrap gap-1.5">
            {result.labels.map((l) => (
              <li key={l}>
                <Badge variant="secondary">{l}</Badge>
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}

function TrainForm({ onTrained }: { onTrained: (result: TrainResponse) => void }) {
  const uid = useId()
  const [name, setName] = useState("")
  const [student, setStudent] = useState<StudentKind>("static")
  const [model, setModel] = useState(DEFAULT_MODEL)
  const [holdout, setHoldout] = useState("0.2")
  const [minAccuracy, setMinAccuracy] = useState("")
  const [seed, setSeed] = useState("0")
  const [csv, setCsv] = useState("")
  const [fileNote, setFileNote] = useState<string | null>(null)
  const [problems, setProblems] = useState<string[]>([])
  const train = useFactoryAction(factoryApi.trainClassifier)
  const fileInput = useRef<HTMLInputElement>(null)

  const parsed = useMemo(() => (csv.trim() ? parseLabelledCsv(csv) : null), [csv])
  const counts = useMemo(() => (parsed ? labelCounts(parsed.labels) : []), [parsed])
  const models = modelsForStudent(student, DEFAULT_ALLOWED_MODELS)

  const chooseStudent = (kind: StudentKind) => {
    setStudent(kind)
    // A repo picked for the previous student kind may be one the new kind
    // cannot load; fall back to the new kind's default rather than send it.
    if (model !== DEFAULT_MODEL && !modelsForStudent(kind, DEFAULT_ALLOWED_MODELS).includes(model)) {
      setModel(DEFAULT_MODEL)
    }
  }

  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    // Read in the browser and put the text in the box, so what will be sent
    // is visible and editable before training.
    file.text().then(
      (text) => {
        setCsv(text)
        setFileNote(`Loaded ${file.name} (${(file.size / 1024).toFixed(1)} KiB).`)
      },
      (err: unknown) => setFileNote(`Could not read ${file.name}: ${err instanceof Error ? err.message : String(err)}`),
    )
    e.target.value = "" // so choosing the same file again re-reads it
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    const found: string[] = []
    if (!CLASSIFIER_NAME.test(name)) found.push("Name must be 1-64 characters: letters, digits, '-' or '_', not starting with '-' or '_'.")
    const h = Number(holdout)
    if (!(h > 0 && h < 0.9)) found.push("Hold-out share must be greater than 0 and less than 0.9.")
    let minAcc: number | null = null
    if (minAccuracy.trim()) {
      minAcc = Number(minAccuracy)
      if (!(minAcc > 0 && minAcc <= 1)) found.push("Minimum accuracy must be a fraction between 0 and 1, or empty.")
    }
    const s = Number(seed)
    if (!Number.isInteger(s) || s < 0) found.push("Seed must be a whole number of at least 0.")
    if (student !== "tfidf" && model !== DEFAULT_MODEL && !models.includes(model)) {
      found.push(`The ${student} student cannot load ${model}. ${MODEL_HINT[student]}`)
    }
    if (!parsed) found.push("Paste or upload labelled data first.")
    else {
      found.push(...parsed.errors)
      if (!parsed.errors.length && h > 0 && h < 0.9) found.push(...trainingDataProblems(parsed.labels, h))
    }
    setProblems(found)
    if (found.length || !parsed) return
    const req: TrainRequest = {
      name,
      texts: parsed.texts,
      labels: parsed.labels,
      student,
      model: student === "tfidf" || model === DEFAULT_MODEL ? null : model,
      holdout: h,
      min_accuracy: minAcc,
      seed: s,
    }
    const result = await train.run(req)
    if (result) onTrained(result)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Train a classifier</CardTitle>
        <CardDescription>
          Labelled data as CSV with a <code>text</code> and a <code>label</code> column. Each label is split into
          training, calibration and test parts on the service.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={submit} className="space-y-5" noValidate>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field id={`${uid}-name`} label="Name" hint="Letters, digits, '-' and '_'; up to 64 characters.">
              <Input
                id={`${uid}-name`}
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="intent-router-v1"
                autoComplete="off"
                maxLength={64}
                aria-invalid={name !== "" && !CLASSIFIER_NAME.test(name)}
                aria-describedby={`${uid}-name-hint`}
              />
            </Field>
            <Field id={`${uid}-student`} label="Student">
              <Select value={student} onValueChange={(v) => chooseStudent(v as StudentKind)}>
                <SelectTrigger id={`${uid}-student`} className="w-full" aria-describedby={`${uid}-student-info`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {STUDENT_KINDS.map((k) => (
                    <SelectItem key={k} value={k}>
                      {k}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p id={`${uid}-student-info`} className="text-xs text-muted-foreground">
                {STUDENT_INFO[student]}
              </p>
            </Field>
          </div>

          <div className="space-y-4">
            <Field
              id={`${uid}-model`}
              label="Base model"
              hint={MODEL_HINT[student]}
            >
              <Select value={student === "tfidf" ? DEFAULT_MODEL : model} onValueChange={setModel} disabled={student === "tfidf"}>
                <SelectTrigger id={`${uid}-model`} className="w-full" aria-describedby={`${uid}-model-hint`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={DEFAULT_MODEL}>
                    {STUDENT_DEFAULT_MODEL[student] ? `Default (${STUDENT_DEFAULT_MODEL[student]})` : "None"}
                  </SelectItem>
                  {models.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <div className="grid grid-cols-3 gap-3">
              <Field id={`${uid}-holdout`} label="Hold-out share">
                <Input
                  id={`${uid}-holdout`}
                  type="number"
                  step="0.05"
                  min="0.05"
                  max="0.85"
                  inputMode="decimal"
                  value={holdout}
                  onChange={(e) => setHoldout(e.target.value)}
                />
              </Field>
              <Field id={`${uid}-minacc`} label="Min. accuracy">
                <Input
                  id={`${uid}-minacc`}
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  inputMode="decimal"
                  placeholder="none"
                  value={minAccuracy}
                  onChange={(e) => setMinAccuracy(e.target.value)}
                />
              </Field>
              <Field id={`${uid}-seed`} label="Seed">
                <Input
                  id={`${uid}-seed`}
                  type="number"
                  step="1"
                  min="0"
                  inputMode="numeric"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                />
              </Field>
            </div>
          </div>
          <p className="-mt-2 text-xs text-muted-foreground">
            Hold-out is the share of each label kept back for calibration and test. Minimum accuracy, if set, becomes
            the evaluation contract's threshold.
          </p>

          <div className="space-y-2">
            <div className="flex flex-wrap items-end justify-between gap-2">
              <label htmlFor={`${uid}-csv`} className="text-sm font-medium">
                Labelled data (CSV)
              </label>
              <div className="flex flex-wrap gap-2">
                <Button type="button" variant="outline" size="sm" onClick={() => setCsv(SAMPLE_CSV)}>
                  Insert example
                </Button>
                <Button type="button" variant="outline" size="sm" onClick={() => fileInput.current?.click()}>
                  <FileUp className="h-3.5 w-3.5" aria-hidden="true" />
                  Upload CSV
                </Button>
                {/* Hidden: the button above is the keyboard-reachable way to open it. */}
                <input
                  ref={fileInput}
                  type="file"
                  accept=".csv,text/csv,text/plain"
                  className="hidden"
                  tabIndex={-1}
                  aria-hidden="true"
                  onChange={onFile}
                />
              </div>
            </div>
            <Textarea
              id={`${uid}-csv`}
              value={csv}
              onChange={(e) => setCsv(e.target.value)}
              placeholder={'text,label\n"my card was blocked",card\n...'}
              className="min-h-40 font-mono text-xs"
              spellCheck={false}
              aria-describedby={`${uid}-csv-summary`}
            />
            <div id={`${uid}-csv-summary`} className="space-y-1 text-xs text-muted-foreground">
              {fileNote && <p>{fileNote}</p>}
              {parsed && !parsed.errors.length && (
                <>
                  <p>
                    {parsed.texts.length} rows, {counts.length} labels
                    {parsed.skipped > 0 && `; ${parsed.skipped} rows without both a text and a label were skipped`}.
                    {parsed.texts.length > DEFAULT_LIMITS.maxTrainRows &&
                      ` The service accepts ${DEFAULT_LIMITS.maxTrainRows} rows per request by default.`}
                  </p>
                  <ul className="flex flex-wrap gap-1.5">
                    {counts.slice(0, 40).map(([label, n]) => (
                      <li key={label}>
                        <Badge variant="outline">
                          {label}: {n}
                        </Badge>
                      </li>
                    ))}
                    {counts.length > 40 && <li>and {counts.length - 40} more</li>}
                  </ul>
                </>
              )}
              {parsed && parsed.errors.length > 0 && <p className="text-destructive dark:text-destructive-foreground">{parsed.errors.join(" ")}</p>}
            </div>
          </div>

          <InputProblems problems={problems} />
          {train.error !== null && <ErrorState error={train.error} />}
          {train.loading && (
            <LoadingState label="Training. The request stays open until the student is fitted, calibrated and scored; a first run may also download its base model." />
          )}
          <Button type="submit" disabled={train.loading} className="w-full">
            <Play className="mr-2 h-4 w-4" aria-hidden="true" />
            {train.loading ? "Training..." : "Train"}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

function ClassifierList({
  list,
  selected,
  onSelect,
}: {
  list: QueryResult<ClassifierListResponse>
  selected: string | null
  onSelect: (name: string) => void
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-2">
        <div className="space-y-1.5">
          <CardTitle>Trained classifiers</CardTitle>
          <CardDescription>Select one to try it or read its model card.</CardDescription>
        </div>
        <Button variant="outline" size="sm" onClick={list.reload} disabled={list.loading} aria-label="Refresh the list">
          <RefreshCw className={`h-3.5 w-3.5 ${list.loading ? "animate-spin" : ""}`} aria-hidden="true" />
        </Button>
      </CardHeader>
      <CardContent>
        {list.loading && !list.data && <LoadingState label="Loading classifiers..." />}
        {list.error !== null && <ErrorState error={list.error} onRetry={list.reload} />}
        {list.data && list.data.classifiers.length === 0 && (
          <EmptyState title="No classifiers yet">Train one below.</EmptyState>
        )}
        {list.data && list.data.classifiers.length > 0 && (
          <ul className="space-y-2">
            {list.data.classifiers.map((c) => (
              <li key={c.name}>
                <button
                  type="button"
                  onClick={() => onSelect(c.name)}
                  aria-pressed={selected === c.name}
                  className={`w-full rounded-lg border p-3 text-left transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                    selected === c.name ? "border-primary bg-accent" : ""
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-medium break-all">{c.name}</span>
                    <ContractBadge passed={c.contract_passed} />
                  </div>
                  <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                    <span>student: {c.student ?? "unknown"}</span>
                    <span>accuracy: {c.accuracy === null ? "n/a" : fmtPercent(c.accuracy)}</span>
                    {c.created_utc && <span>{c.created_utc}</span>}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}

function TryIt({ name }: { name: string }) {
  const uid = useId()
  const [texts, setTexts] = useState("")
  const [threshold, setThreshold] = useState("0.5")
  const [problems, setProblems] = useState<string[]>([])
  const predict = useFactoryAction(predictAction)
  const { reset } = predict

  // A different classifier: the previous answers belong to the old one.
  useEffect(() => {
    reset()
  }, [name, reset])

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    const lines = nonBlankLines(texts)
    const t = Number(threshold)
    const found: string[] = []
    if (!lines.length) found.push("Enter at least one text, one per line.")
    if (threshold.trim() === "" || !(t >= 0 && t <= 1)) found.push("Abstention threshold must be between 0 and 1.")
    setProblems(found)
    if (found.length) return
    await predict.run({ name, req: { texts: lines, threshold: t } })
  }

  // While a new request is out, the previous answers are hidden rather than
  // left next to a text box that no longer holds what they answered.
  const shown = predict.loading ? null : predict.data
  const paired = shown ? pairPredictions(shown.texts, shown.predictions) : null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5 text-primary" aria-hidden="true" />
          Try {name}
        </CardTitle>
        <CardDescription>
          Below the threshold the classifier abstains instead of guessing; abstentions go to a person.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={submit} className="space-y-4" noValidate>
          <Field id={`${uid}-texts`} label="Texts, one per line">
            <Textarea
              id={`${uid}-texts`}
              value={texts}
              onChange={(e) => setTexts(e.target.value)}
              className="min-h-28"
              placeholder={"my card got blocked\nloan emi deducted twice"}
            />
          </Field>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
            <Field id={`${uid}-thr`} label="Abstention threshold" hint="0 never abstains; 1 always does." className="sm:w-56">
              <Input
                id={`${uid}-thr`}
                type="number"
                step="0.05"
                min="0"
                max="1"
                inputMode="decimal"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
                aria-describedby={`${uid}-thr-hint`}
              />
            </Field>
            <Button type="submit" disabled={predict.loading} className="sm:mb-5">
              {predict.loading ? "Predicting..." : "Predict"}
            </Button>
          </div>
          <InputProblems problems={problems} />
        </form>
        {predict.loading && <LoadingState label="Predicting..." />}
        {predict.error !== null && <ErrorState error={predict.error} />}
        {paired && paired.rows.length === 0 && <EmptyState title="No predictions returned" />}
        {shown && paired && !paired.complete && (
          <Alert variant="destructive">
            <AlertTitle>Unexpected answer</AlertTitle>
            <AlertDescription>
              {shown.texts.length} texts were sent but {shown.predictions.length} predictions came back; only the
              first {paired.rows.length} are shown.
            </AlertDescription>
          </Alert>
        )}
        {shown && paired && paired.rows.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground">
              {shown.texts.length} {shown.texts.length === 1 ? "text" : "texts"}
              {shown.threshold !== undefined && ` at an abstention threshold of ${shown.threshold}`}.
            </p>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Text</TableHead>
                    <TableHead>Label</TableHead>
                    <TableHead className="text-right">Confidence</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paired.rows.map(({ text, prediction: p }, i) => (
                    <TableRow key={i}>
                      <TableCell className="max-w-xs whitespace-normal break-words">{text}</TableCell>
                      <TableCell>
                        {p.abstained ? (
                          <Badge variant="outline" className="border-chart-3/40 text-chart-3">
                            abstained
                          </Badge>
                        ) : (
                          <Badge variant="secondary">{p.label}</Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <div className="hidden h-1.5 w-16 overflow-hidden rounded bg-muted sm:block" aria-hidden="true">
                            <div className="h-full bg-primary" style={{ width: `${Math.round(p.confidence * 100)}%` }} />
                          </div>
                          <span className="font-mono text-xs">{fmtNumber(p.confidence, 3)}</span>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function ModelCard({ card }: { card: ClassifierCardResponse }) {
  const r = card.run_record
  const recipe = r.recipe ?? {}
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ScrollText className="h-5 w-5 text-primary" aria-hidden="true" />
          Model card
        </CardTitle>
        <CardDescription>
          Run <code>{r.run_id}</code>, created {r.created_utc}. Generated from the run record; it states what was
          measured and does not establish fitness for any regulated purpose.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Student" value={String(recipe.student ?? "unknown")} />
          <Stat label="Base model" value={recipe.model ? String(recipe.model) : "none / default"} />
          <Stat label="Accuracy" value={fmtPercent(r.metrics?.accuracy)} />
          <Stat label="Temperature" value={fmtNumber(recipe.temperature)} />
        </dl>
        <div className="flex flex-wrap gap-2 text-xs">
          {Object.entries(r.data ?? {}).map(([split, info]) => (
            <Badge key={split} variant="outline">
              {split}: {info?.n ?? "?"} examples
            </Badge>
          ))}
        </div>
        {r.contract ? <ContractVerdictView verdict={r.contract} /> : <ContractBadge passed={null} />}
        <details className="rounded-lg border p-3" open>
          <summary className="cursor-pointer text-sm font-medium">MODEL_CARD.md</summary>
          <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap text-xs">{card.model_card}</pre>
        </details>
        <details className="rounded-lg border p-3">
          <summary className="cursor-pointer text-sm font-medium">Run record (JSON)</summary>
          <pre className="mt-3 max-h-96 overflow-auto text-xs">{JSON.stringify(r, null, 2)}</pre>
        </details>
      </CardContent>
    </Card>
  )
}

export function ClassifiersTab({ list }: { list: QueryResult<ClassifierListResponse> }) {
  const [selected, setSelected] = useState<string | null>(null)
  const card = useFactoryQuery(selected, factoryApi.classifierCard)
  const [lastTrained, setLastTrained] = useState<TrainResponse | null>(null)
  const detailRef = useRef<HTMLDivElement>(null)
  const { reload: reloadList } = list
  const { reload: reloadCard } = card

  const select = (name: string) => {
    setSelected(name)
    // Below the xl breakpoint the details sit under the list and the form;
    // bring them into view. Side by side, scrolling would only jump the page.
    if (window.matchMedia("(max-width: 1279px)").matches) {
      detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
    }
  }

  const onTrained = (result: TrainResponse) => {
    setLastTrained(result)
    reloadList()
    // Retraining an existing name replaces it; its card must be read again.
    if (selected === result.name) reloadCard()
    else setSelected(result.name)
  }

  return (
    <div className="grid gap-6 xl:grid-cols-5">
      <div className="min-w-0 space-y-6 xl:col-span-2">
        <ClassifierList list={list} selected={selected} onSelect={select} />
        <TrainForm onTrained={onTrained} />
      </div>
      <div ref={detailRef} className="min-w-0 scroll-mt-20 space-y-6 xl:col-span-3">
        {lastTrained && <TrainResult result={lastTrained} />}
        {selected === null ? (
          <Card>
            <CardContent className="py-10 text-center text-sm text-muted-foreground">
              Select a classifier to try it and read its model card, or train a new one.
            </CardContent>
          </Card>
        ) : (
          <>
            <TryIt name={selected} />
            {card.loading && !card.data && <LoadingState label="Loading the model card..." />}
            {card.error !== null && <ErrorState error={card.error} onRetry={reloadCard} />}
            {card.data && !card.error && <ModelCard card={card.data} />}
          </>
        )}
      </div>
    </div>
  )
}
