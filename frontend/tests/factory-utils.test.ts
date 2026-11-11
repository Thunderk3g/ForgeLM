// Tests for the Factory page's browser-side helpers (src/lib/factory-utils.ts).
//
// Run with "npm test". They use Node's own test runner and its built-in
// TypeScript type stripping (Node 22.18 or later), so they need no test
// framework in package.json. factory-utils.ts has no runtime imports for
// exactly this reason.
//
// The expected values for the service's rules (name pattern, per-label
// minimum, Python rounding) were computed with the Python code in
// src/microservices/factory_service/src/main.py; if that code changes, these
// numbers must be recomputed from it, not edited to make the test pass.

import assert from "node:assert/strict"
import { describe, it } from "node:test"
import {
  CLASSIFIER_NAME,
  DEFAULT_ALLOWED_MODELS,
  isStaticModel,
  minExamplesPerLabel,
  modelsForStudent,
  pairPredictions,
  parseLabelledCsv,
  pyRound,
  trainingDataProblems,
} from "../src/lib/factory-utils.ts"

describe("service rules mirrored in the browser", () => {
  it("rounds halves to even, as Python's round() does", () => {
    assert.deepEqual([0.5, 1.5, 2.5, 3.5, 2.4, 2.6].map(pyRound), [0, 2, 2, 4, 2, 3])
  })

  it("matches the service's per-label minimum", () => {
    // Python: the smallest n >= 4 with n - max(2, round(n * h)) >= 1.
    assert.equal(minExamplesPerLabel(0.2), 4)
    assert.equal(minExamplesPerLabel(0.5), 4)
    assert.equal(minExamplesPerLabel(0.85), 4)
    assert.equal(minExamplesPerLabel(0.89), 5)
  })

  it("applies the classifier-name pattern to the whole string", () => {
    const ok = ["router-v1", "A", "a_b-9", "x".repeat(64)]
    const bad = ["", "-a", "_a", "abc\n", "has space", "x".repeat(65)]
    for (const name of ok) assert.ok(CLASSIFIER_NAME.test(name), name)
    for (const name of bad) assert.ok(!CLASSIFIER_NAME.test(name), JSON.stringify(name))
  })

  it("asks for two labels and enough examples of each", () => {
    assert.deepEqual(trainingDataProblems(["a", "a", "a", "a"], 0.2), ["at least two distinct labels are needed"])
    const problems = trainingDataProblems(["a", "a", "a", "a", "b", "b", "b"], 0.2)
    assert.equal(problems.length, 1)
    assert.match(problems[0], /"b" \(3\)/)
  })
})

describe("CSV parsing", () => {
  it("handles a byte-order mark, CRLF, quoted commas, newlines and doubled quotes", () => {
    const csv = '﻿text,label\r\n"a, b",x\r\n"line one\nline two",y\r\n"say ""hi""",x\r\n,y\r\n'
    const parsed = parseLabelledCsv(csv)
    assert.deepEqual(parsed.errors, [])
    assert.deepEqual(parsed.texts, ["a, b", "line one\nline two", 'say "hi"'])
    assert.deepEqual(parsed.labels, ["x", "y", "x"])
    assert.equal(parsed.skipped, 1)
  })

  it("requires a header naming text and label", () => {
    assert.equal(parseLabelledCsv("sentence,class\nhello,x\n").errors.length, 1)
  })
})

describe("base models offered per student kind", () => {
  // Reviewer's case: student "static" with BAAI/bge-small-en-v1.5 was
  // accepted by the form and failed on the service with an HTTP 500.
  it("offers only model2vec repos to the static student", () => {
    const models = modelsForStudent("static", DEFAULT_ALLOWED_MODELS)
    assert.deepEqual(models, ["minishlab/potion-base-32M", "minishlab/potion-base-8M"])
    assert.ok(!models.includes("BAAI/bge-small-en-v1.5"))
  })

  it("offers only transformer repos to the encoder and fine-tuned students", () => {
    for (const kind of ["encoder", "finetuned"] as const) {
      const models = modelsForStudent(kind, DEFAULT_ALLOWED_MODELS)
      assert.ok(models.length > 0, kind)
      assert.ok(models.every((m) => !isStaticModel(m)), kind)
      assert.ok(models.includes("BAAI/bge-small-en-v1.5"), kind)
    }
  })

  it("offers nothing to tfidf, which has no base model", () => {
    assert.deepEqual(modelsForStudent("tfidf", DEFAULT_ALLOWED_MODELS), [])
  })

  it("splits the default allow-list with nothing left over", () => {
    const static_ = modelsForStudent("static", DEFAULT_ALLOWED_MODELS)
    const encoder = modelsForStudent("encoder", DEFAULT_ALLOWED_MODELS)
    assert.equal(static_.length + encoder.length, DEFAULT_ALLOWED_MODELS.length)
  })

  it("uses the same name test as forgelm's Embedder", () => {
    // students.py: "potion" in lowered or "model2vec" in lowered or "m2v" in lowered
    assert.ok(isStaticModel("minishlab/POTION-base-8M"))
    assert.ok(isStaticModel("someone/m2v-small"))
    assert.ok(isStaticModel("someone/model2vec-thing"))
    assert.ok(!isStaticModel("answerdotai/ModernBERT-base"))
  })
})

describe("pairing texts with predictions", () => {
  it("keeps request order", () => {
    const { rows, complete } = pairPredictions(["a", "b"], [{ label: "x" }, { label: "y" }])
    assert.ok(complete)
    assert.deepEqual(rows, [
      { text: "a", prediction: { label: "x" } },
      { text: "b", prediction: { label: "y" } },
    ])
  })

  it("reports a count mismatch instead of leaving rows blank", () => {
    const { rows, complete } = pairPredictions(["a", "b", "c"], [{ label: "x" }])
    assert.ok(!complete)
    assert.equal(rows.length, 1)
  })
})
