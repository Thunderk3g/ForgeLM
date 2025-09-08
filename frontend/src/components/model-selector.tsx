"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { Download, Cpu, Check, AlertCircle, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { getAvailableModels } from "../lib/api"
import type { SelectModelRequest, ModelInfo } from "../lib/types"

interface ModelSelectorProps {
  onSelectModel: (request: SelectModelRequest) => void
  onModelSelected: (modelId: string) => void
  isLoading: boolean
  selectedModel?: string
}

const POPULAR_MODELS = [
  { id: "unsloth/llama-3.1-8b-unsloth-bnb-4bit", name: "Llama 3.1 8B (Unsloth)", size: "8B" },
  { id: "unsloth/llama-3.1-70b-unsloth-bnb-4bit", name: "Llama 3.1 70B (Unsloth)", size: "70B" },
  { id: "unsloth/mistral-7b-unsloth-bnb-4bit", name: "Mistral 7B (Unsloth)", size: "7B" },
  { id: "unsloth/codellama-34b-unsloth-bnb-4bit", name: "CodeLlama 34B (Unsloth)", size: "34B" },
]

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  onSelectModel,
  onModelSelected,
  isLoading,
  selectedModel,
}) => {
  const [localSelectedModel, setLocalSelectedModel] = useState(POPULAR_MODELS[0].id)
  const [customModel, setCustomModel] = useState("")
  const [useCustom, setUseCustom] = useState(false)
  const [quantization, setQuantization] = useState<string>("4bit")
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [viewMode, setViewMode] = useState<"popular" | "available">("available")

  useEffect(() => {
    loadAvailableModels()
  }, [])

  const loadAvailableModels = async () => {
    setLoadingModels(true)
    try {
      const response = await getAvailableModels()
      if (response.status === "success") {
        setAvailableModels(response.models)
      }
    } catch (error) {
      console.error("Failed to load available models:", error)
    } finally {
      setLoadingModels(false)
    }
  }

  const handleDownload = (e: React.FormEvent) => {
    e.preventDefault()
    const modelId = useCustom ? customModel : localSelectedModel
    if (!modelId.trim()) return

    onSelectModel({
      model_id: modelId,
      output_dir: "/app/models/base",
      quantization,
    })
  }

  const handleSelectExistingModel = (modelId: string) => {
    onModelSelected(modelId)
  }

  const isModelAvailable = (modelId: string) => {
    return availableModels.some((model) => model.model_id === modelId)
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="rounded-md bg-primary/10 p-2">
              <Cpu className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle className="text-lg">Model Selection</CardTitle>
              <CardDescription>Choose or download a base model for fine-tuning</CardDescription>
            </div>
          </div>
          <Button variant="outline" onClick={loadAvailableModels} disabled={loadingModels}>
            <RefreshCw className={`mr-2 h-4 w-4 ${loadingModels ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {selectedModel && (
          <div className="rounded-md border border-primary/20 bg-primary/5 p-4">
            <div className="flex items-center space-x-2">
              <Check className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium text-primary">Selected Model:</span>
              <span className="text-sm font-mono">{selectedModel}</span>
            </div>
          </div>
        )}

        <div className="flex space-x-1 rounded-md bg-muted p-1">
          <Button
            variant={viewMode === "available" ? "default" : "ghost"}
            onClick={() => setViewMode("available")}
            className="flex-1"
            size="sm"
          >
            Available Models ({availableModels.length})
          </Button>
          <Button
            variant={viewMode === "popular" ? "default" : "ghost"}
            onClick={() => setViewMode("popular")}
            className="flex-1"
            size="sm"
          >
            Download New Model
          </Button>
        </div>

        {viewMode === "available" && (
          <div className="space-y-4">
            {loadingModels ? (
              <div className="flex items-center justify-center py-12">
                <div className="flex items-center space-x-3">
                  <RefreshCw className="h-5 w-5 animate-spin text-primary" />
                  <span className="text-sm text-muted-foreground">Loading available models...</span>
                </div>
              </div>
            ) : availableModels.length > 0 ? (
              <div className="space-y-3">
                {availableModels.map((model) => (
                  <div
                    key={model.model_id}
                    className={`rounded-lg border p-4 transition-colors cursor-pointer ${
                      selectedModel === model.model_id
                        ? "border-primary bg-accent"
                        : "hover:border-accent-foreground/20 hover:bg-accent/50"
                    }`}
                    onClick={() => handleSelectExistingModel(model.model_id)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-3 mb-2">
                          <h4 className="font-medium truncate">{model.model_id}</h4>
                          <Badge variant="secondary">Available</Badge>
                        </div>
                        <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                          <span>Size: {Math.round(model.size_mb / 1024)} GB</span>
                          {model.architecture && <span>Architecture: {model.architecture}</span>}
                          <span className={`${model.has_tokenizer ? "text-primary" : "text-destructive"}`}>
                            Tokenizer: {model.has_tokenizer ? "Available" : "Missing"}
                          </span>
                        </div>
                      </div>
                      {selectedModel === model.model_id && (
                        <div className="flex-shrink-0">
                          <div className="rounded-full bg-primary p-1">
                            <Check className="h-4 w-4 text-primary-foreground" />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="rounded-full bg-muted p-3 mb-4">
                  <AlertCircle className="h-6 w-6 text-muted-foreground" />
                </div>
                <h4 className="text-sm font-medium mb-1">No models available locally</h4>
                <p className="text-sm text-muted-foreground">Download a model to get started</p>
              </div>
            )}
          </div>
        )}

        {viewMode === "popular" && (
          <form onSubmit={handleDownload} className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-3">
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="radio" checked={!useCustom} onChange={() => setUseCustom(false)} className="h-4 w-4" />
                  <div>
                    <span className="text-sm font-medium">Popular Models</span>
                    <p className="text-xs text-muted-foreground">Pre-configured optimized models</p>
                  </div>
                </label>
                {!useCustom && (
                  <div className="ml-7 space-y-3">
                    <select
                      value={localSelectedModel}
                      onChange={(e) => setLocalSelectedModel(e.target.value)}
                      className="w-full p-2 border rounded-md bg-background"
                    >
                      {POPULAR_MODELS.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.name} ({model.size})
                          {isModelAvailable(model.id) ? " - Available" : " - Need to download"}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>

              <div className="space-y-3">
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="radio" checked={useCustom} onChange={() => setUseCustom(true)} className="h-4 w-4" />
                  <div>
                    <span className="text-sm font-medium">Custom Model ID</span>
                    <p className="text-xs text-muted-foreground">Enter any Hugging Face model ID</p>
                  </div>
                </label>
                {useCustom && (
                  <div className="ml-7">
                    <input
                      type="text"
                      value={customModel}
                      onChange={(e) => setCustomModel(e.target.value)}
                      placeholder="e.g., microsoft/DialoGPT-medium"
                      className="w-full p-2 border rounded-md bg-background"
                    />
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-3">
              <label className="text-sm font-medium">Quantization</label>
              <div className="flex space-x-6">
                {(["none", "4bit", "8bit"] as const).map((option) => (
                  <label key={option} className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="radio"
                      value={option}
                      checked={quantization === option}
                      onChange={(e) => setQuantization(e.target.value)}
                      className="h-4 w-4"
                    />
                    <span className="text-sm font-medium capitalize">{option}</span>
                  </label>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                Quantization reduces memory usage but may affect model quality
              </p>
            </div>

            <Button
              type="submit"
              disabled={isLoading || (!localSelectedModel && !customModel.trim())}
              className="w-full"
            >
              <Download className="mr-2 h-4 w-4" />
              {isLoading ? "Downloading Model..." : "Download Model"}
            </Button>
          </form>
        )}
      </CardContent>
    </Card>
  )
}
