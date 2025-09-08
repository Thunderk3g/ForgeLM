"use client"

import type React from "react"
import { useState } from "react"
import { Play, Settings } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { FineTuneRequest, ModelInfo } from "../lib/types"

interface TrainingFormProps {
  onStartTraining: (request: FineTuneRequest) => void
  isLoading: boolean
  selectedModel: string | null
  selectedModelInfo?: ModelInfo | null
}

export const TrainingForm: React.FC<TrainingFormProps> = ({
  onStartTraining,
  isLoading,
  selectedModel,
  selectedModelInfo,
}) => {
  const [ftType, setFtType] = useState<string>("instruction")
  const [peftMethod, setPeftMethod] = useState<string>("qlora")
  const [alignment, setAlignment] = useState<string>("none")
  const [epochs, setEpochs] = useState(1)
  const [batchSize, setBatchSize] = useState(20)
  const [learningRate, setLearningRate] = useState(0.0001)
  const [gradientAccumulation, setGradientAccumulation] = useState(4)
  const [maxSeqLength, setMaxSeqLength] = useState(256)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedModel) {
      alert("Please select a model first!")
      return
    }

    onStartTraining({
      base_model_dir: selectedModelInfo?.path || "/app/models/base",
      dataset_dir: "/app/data/prepared",
      output_dir: "/app/models/fine_tuned",
      ft_type: ftType,
      peft_method: peftMethod,
      alignment: alignment,
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      gradient_accumulation_steps: gradientAccumulation,
      max_seq_length: maxSeqLength,
    })
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center space-x-3">
          <div className="rounded-md bg-primary/10 p-2">
            <Settings className="h-5 w-5 text-primary" />
          </div>
          <div>
            <CardTitle className="text-lg">Fine-tuning Configuration</CardTitle>
            <CardDescription>Configure training parameters for optimal results</CardDescription>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {selectedModel ? (
          <div className="rounded-md border border-primary/20 bg-primary/5 p-4">
            <div className="flex items-center space-x-2">
              <div className="h-2 w-2 rounded-full bg-primary" />
              <p className="text-sm font-medium text-primary">Selected Model: {selectedModel}</p>
            </div>
          </div>
        ) : (
          <div className="rounded-md border border-secondary/20 bg-secondary/5 p-4">
            <div className="flex items-center space-x-2">
              <div className="h-2 w-2 rounded-full bg-secondary" />
              <p className="text-sm font-medium text-secondary">
                Please select a model first before configuring fine-tuning
              </p>
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-3">
            <Label>Training Type</Label>
            <RadioGroup value={ftType} onValueChange={setFtType}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="instruction" id="ft-instruction" />
                <Label htmlFor="ft-instruction" className="cursor-pointer">
                  <div>
                    <p className="font-medium">Instruction Tuning</p>
                    <p className="text-sm text-muted-foreground">Train on instruction-response pairs</p>
                  </div>
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="sft" id="ft-sft" />
                <Label htmlFor="ft-sft" className="cursor-pointer">
                  <div>
                    <p className="font-medium">Supervised Fine-tuning</p>
                    <p className="text-sm text-muted-foreground">Standard supervised fine-tuning</p>
                  </div>
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="unsupervised" id="ft-unsupervised" />
                <Label htmlFor="ft-unsupervised" className="cursor-pointer">
                  <div>
                    <p className="font-medium">Unsupervised Tuning</p>
                    <p className="text-sm text-muted-foreground">Train on unlabeled text data</p>
                  </div>
                </Label>
              </div>
            </RadioGroup>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="peft-method">PEFT Method</Label>
              <Select value={peftMethod} onValueChange={setPeftMethod}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="qlora">QLoRA - 4-bit quantized</SelectItem>
                  <SelectItem value="lora">LoRA - Standard precision</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">Parameter-efficient fine-tuning method</p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="alignment">Alignment Method</Label>
              <Select value={alignment} onValueChange={setAlignment}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None</SelectItem>
                  <SelectItem value="dpo">DPO - Direct Preference</SelectItem>
                  <SelectItem value="ipo">IPO - Identity Preference</SelectItem>
                  <SelectItem value="kto">KTO - Kahneman-Tversky</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">Human preference alignment technique</p>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="epochs">Epochs</Label>
              <Input
                id="epochs"
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(Number.parseInt(e.target.value))}
                min="1"
                max="20"
              />
              <p className="text-xs text-muted-foreground">Number of training passes through dataset</p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="batch-size">Batch Size</Label>
              <Input
                id="batch-size"
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(Number.parseInt(e.target.value))}
                min="1"
                max="32"
              />
              <p className="text-xs text-muted-foreground">Samples processed per training step</p>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="learning-rate">Learning Rate</Label>
            <Input
              id="learning-rate"
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(Number.parseFloat(e.target.value))}
              min="0.00001"
              max="0.01"
              step="0.00001"
            />
            <p className="text-xs text-muted-foreground">Controls the step size during optimization (0.00001 - 0.01)</p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="gradient-accumulation">Gradient Accumulation Steps</Label>
              <Input
                id="gradient-accumulation"
                type="number"
                value={gradientAccumulation}
                onChange={(e) => setGradientAccumulation(Number.parseInt(e.target.value))}
                min="1"
                max="32"
              />
              <p className="text-xs text-muted-foreground">Accumulate gradients for effective larger batch size</p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="max-seq-length">Max Sequence Length</Label>
              <Input
                id="max-seq-length"
                type="number"
                value={maxSeqLength}
                onChange={(e) => setMaxSeqLength(Number.parseInt(e.target.value))}
                min="128"
                max="4096"
                step="128"
              />
              <p className="text-xs text-muted-foreground">Maximum input token length (128 - 4096)</p>
            </div>
          </div>

          <Button type="submit" disabled={isLoading || !selectedModel} className="w-full">
            <Play className="mr-2 h-4 w-4" />
            {isLoading ? "Starting Training..." : "Start Fine-tuning"}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
