"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { Send, MessageSquare, Settings, Trash2, RotateCcw, Sliders } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Separator } from "@/components/ui/separator"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  getChatModelPairs,
  getChatSessions,
  createChatSession,
  getChatSessionHistory,
  deleteChatSession,
  generateChatResponse,
} from "../lib/api"

interface ModelPair {
  pair_id: string
  base_model_name: string
  adapter_name: string
  base_model_size_mb: number
  adapter_size_mb: number
  estimated_memory_mb: number
  status: string
  load_time_ms?: number
}

interface ChatMessage {
  role: string
  content: string
  timestamp: string
  model_type?: string
}

interface ChatSession {
  session_id: string
  session_name: string
  pair_id: string
  model_info: {
    base_model: string
    adapter_name: string
  }
  created_at: string
  last_activity: string
  message_count: number
  status: string
}

export const ComparisonChatInterface: React.FC = () => {
  const [modelPairs, setModelPairs] = useState<ModelPair[]>([])
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [currentMessage, setCurrentMessage] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [showNewSessionDialog, setShowNewSessionDialog] = useState(false)
  const [selectedPairId, setSelectedPairId] = useState<string>("")
  const [sessionName, setSessionName] = useState("")

  // Generation parameters state
  const [generationParams, setGenerationParams] = useState({
    max_new_tokens: 256,
    temperature: 0.7,
    do_sample: true,
    top_p: 0.9,
    top_k: 50,
    repetition_penalty: 1.1,
  })
  const [showParametersSheet, setShowParametersSheet] = useState(false)

  // Refs for auto-scrolling
  const baseMessagesRef = useRef<HTMLDivElement>(null)
  const finetunedMessagesRef = useRef<HTMLDivElement>(null)

  // Load model pairs and sessions on component mount
  useEffect(() => {
    loadModelPairs()
    loadSessions()
  }, [])

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (baseMessagesRef.current) {
      baseMessagesRef.current.scrollTop = baseMessagesRef.current.scrollHeight
    }
    if (finetunedMessagesRef.current) {
      finetunedMessagesRef.current.scrollTop = finetunedMessagesRef.current.scrollHeight
    }
  }, [messages])

  const loadModelPairs = async () => {
    try {
      const response = await getChatModelPairs()
      if (response.status === "success") {
        setModelPairs(response.model_pairs)
      }
    } catch (error) {
      console.error("Failed to load model pairs:", error)
    }
  }

  const loadSessions = async () => {
    try {
      const response = await getChatSessions()
      if (response.status === "success") {
        setSessions(response.sessions)
      }
    } catch (error) {
      console.error("Failed to load sessions:", error)
    }
  }

  const createSession = async () => {
    if (!selectedPairId) return

    try {
      const response = await createChatSession({
        pair_id: selectedPairId,
        session_name: sessionName || undefined,
      })

      if (response.status === "success") {
        const newSession = response.session
        setSessions([...sessions, newSession])
        setActiveSessionId(newSession.session_id)
        setMessages([])
        setShowNewSessionDialog(false)
        setSessionName("")
        setSelectedPairId("")
      }
    } catch (error) {
      console.error("Failed to create session:", error)
    }
  }

  const loadSessionHistory = async (sessionId: string) => {
    try {
      const response = await getChatSessionHistory(sessionId)
      if (response.status === "success") {
        setMessages(response.messages || [])
      }
    } catch (error) {
      console.error("Failed to load session history:", error)
    }
  }

  const deleteSession = async (sessionId: string) => {
    try {
      await deleteChatSession(sessionId)
      setSessions(sessions.filter((s) => s.session_id !== sessionId))
      if (activeSessionId === sessionId) {
        setActiveSessionId(null)
        setMessages([])
      }
    } catch (error) {
      console.error("Failed to delete session:", error)
    }
  }

  const sendMessage = async () => {
    if (!currentMessage.trim() || !activeSessionId || isGenerating) return

    const userMessage: ChatMessage = {
      role: "user",
      content: currentMessage,
      timestamp: new Date().toISOString(),
    }

    // Add user message to display
    setMessages((prev) => [...prev, userMessage])
    setCurrentMessage("")
    setIsGenerating(true)

    try {
      const response = await generateChatResponse({
        session_id: activeSessionId,
        messages: [...messages, userMessage],
        generation_params: generationParams,
      })

      if (response.status === "success") {
        const comparisonResponse = response.response

        // Add both responses
        const baseResponse: ChatMessage = {
          role: "assistant",
          content: comparisonResponse.base_response,
          timestamp: comparisonResponse.timestamp,
          model_type: "base",
        }

        const finetunedResponse: ChatMessage = {
          role: "assistant",
          content: comparisonResponse.finetuned_response,
          timestamp: comparisonResponse.timestamp,
          model_type: "finetuned",
        }

        setMessages((prev) => [...prev, baseResponse, finetunedResponse])
      }
    } catch (error) {
      console.error("Failed to generate response:", error)
    } finally {
      setIsGenerating(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const selectSession = (sessionId: string) => {
    setActiveSessionId(sessionId)
    loadSessionHistory(sessionId)
  }

  const activeSession = sessions.find((s) => s.session_id === activeSessionId)
  const baseMessages = messages.filter((m) => m.role === "user" || m.model_type === "base")
  const finetunedMessages = messages.filter((m) => m.role === "user" || m.model_type === "finetuned")

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <div className="w-80 bg-card border-r border-border flex flex-col">
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <MessageSquare className="w-5 h-5 text-primary" />
              Chat Sessions
            </h2>
            <Button onClick={() => setShowNewSessionDialog(true)}>New Session</Button>
          </div>
        </div>

        {/* Session List */}
        <div className="flex-1 overflow-y-auto">
          {sessions.length === 0 ? (
            <div className="p-4 text-center text-muted-foreground">
              <MessageSquare className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No chat sessions yet</p>
              <p className="text-sm">Create one to start comparing models</p>
            </div>
          ) : (
            <div className="p-2 space-y-2">
              {sessions.map((session) => (
                <div
                  key={session.session_id}
                  className={`p-3 rounded-lg cursor-pointer transition-colors ${
                    activeSessionId === session.session_id
                      ? "bg-primary/10 border border-primary/20"
                      : "bg-muted hover:bg-muted/80"
                  }`}
                  onClick={() => selectSession(session.session_id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium truncate">{session.session_name}</h3>
                      <p className="text-sm text-muted-foreground truncate">
                        {session.model_info.base_model} + {session.model_info.adapter_name}
                      </p>
                      <p className="text-xs text-muted-foreground">{session.message_count} messages</p>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteSession(session.session_id)
                      }}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {activeSession ? (
          <>
            {/* Header */}
            <div className="p-4 border-b border-border bg-card">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-xl font-semibold">{activeSession.session_name}</h1>
                  <p className="text-sm text-muted-foreground">
                    Comparing: {activeSession.model_info.base_model} vs {activeSession.model_info.adapter_name}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="sm" onClick={() => loadSessionHistory(activeSessionId!)}>
                    <RotateCcw className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="sm">
                    <Settings className="w-4 h-4" />
                  </Button>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={() => setShowParametersSheet(true)}
                    title="Generation Parameters"
                  >
                    <Sliders className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Split Screen Chat */}
            <div className="flex-1 flex">
              {/* Base Model Side */}
              <div className="flex-1 flex flex-col border-r border-border">
                <div className="p-3 bg-blue-500/10 border-b border-border">
                  <h3 className="font-medium text-blue-400">Base Model: {activeSession.model_info.base_model}</h3>
                </div>
                <div ref={baseMessagesRef} className="flex-1 overflow-y-auto p-4 space-y-4">
                  {baseMessages.map((message, index) => (
                    <div
                      key={`base-${index}`}
                      className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg p-3 ${
                          message.role === "user"
                            ? "bg-primary text-primary-foreground"
                            : "bg-blue-500/10 border border-blue-500/20"
                        }`}
                      >
                        <div className="whitespace-pre-wrap">{message.content}</div>
                        <div className="text-xs opacity-70 mt-1">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Fine-tuned Model Side */}
              <div className="flex-1 flex flex-col">
                <div className="p-3 bg-primary/10 border-b border-border">
                  <h3 className="font-medium text-primary">Fine-tuned: {activeSession.model_info.adapter_name}</h3>
                </div>
                <div ref={finetunedMessagesRef} className="flex-1 overflow-y-auto p-4 space-y-4">
                  {finetunedMessages.map((message, index) => (
                    <div
                      key={`finetuned-${index}`}
                      className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg p-3 ${
                          message.role === "user"
                            ? "bg-primary text-primary-foreground"
                            : "bg-primary/10 border border-primary/20"
                        }`}
                      >
                        <div className="whitespace-pre-wrap">{message.content}</div>
                        <div className="text-xs opacity-70 mt-1">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-border bg-card">
              <div className="flex gap-2">
                <Textarea
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message here..."
                  className="flex-1 resize-none"
                  rows={2}
                  disabled={isGenerating}
                />
                <Button onClick={sendMessage} disabled={!currentMessage.trim() || isGenerating}>
                  <Send className="w-4 h-4" />
                  {isGenerating ? "Generating..." : "Send"}
                </Button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center bg-muted/20">
            <div className="text-center">
              <MessageSquare className="w-16 h-16 mx-auto mb-4 text-muted-foreground opacity-50" />
              <h2 className="text-xl font-semibold mb-2">Select a Chat Session</h2>
              <p className="text-muted-foreground mb-4">
                Choose a session from the sidebar or create a new one to start comparing models
              </p>
              <Button onClick={() => setShowNewSessionDialog(true)}>Create New Session</Button>
            </div>
          </div>
        )}
      </div>

      {/* New Session Dialog */}
      {showNewSessionDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle>Create New Chat Session</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="session-name">Session Name (Optional)</Label>
                <Input
                  id="session-name"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                  placeholder="My Chat Session"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="model-pair">Model Pair</Label>
                <Select value={selectedPairId} onValueChange={setSelectedPairId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a model pair..." />
                  </SelectTrigger>
                  <SelectContent>
                    {modelPairs.map((pair) => (
                      <SelectItem key={pair.pair_id} value={pair.pair_id}>
                        {pair.base_model_name} + {pair.adapter_name} ({(pair.estimated_memory_mb / 1024).toFixed(1)}GB)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex justify-end gap-2 pt-4">
                <Button
                  variant="outline"
                  onClick={() => {
                    setShowNewSessionDialog(false)
                    setSessionName("")
                    setSelectedPairId("")
                  }}
                >
                  Cancel
                </Button>
                <Button onClick={createSession} disabled={!selectedPairId}>
                  Create Session
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Parameters Sheet */}
      <Sheet open={showParametersSheet} onOpenChange={setShowParametersSheet}>
        <SheetContent side="right" className="w-80 sm:w-96">
          <SheetHeader>
            <SheetTitle>Generation Parameters</SheetTitle>
          </SheetHeader>

          <div className="space-y-6 mt-6 px-4 overflow-y-auto">
            {/* Max New Tokens */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label htmlFor="max-tokens">Max New Tokens</Label>
                <Input
                  id="max-tokens"
                  type="number"
                  value={generationParams.max_new_tokens}
                  onChange={(e) =>
                    setGenerationParams((prev) => ({
                      ...prev,
                      max_new_tokens: parseInt(e.target.value) || 256,
                    }))
                  }
                  className="w-20 text-right"
                  min="1"
                  max="4096"
                />
              </div>
              <Slider
                value={[generationParams.max_new_tokens]}
                onValueChange={(value) =>
                  setGenerationParams((prev) => ({
                    ...prev,
                    max_new_tokens: value[0],
                  }))
                }
                min={1}
                max={4096}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Maximum number of tokens to generate in the response.
              </p>
            </div>

            <Separator />

            {/* Temperature */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label htmlFor="temperature">Temperature</Label>
                <Input
                  id="temperature"
                  type="number"
                  value={generationParams.temperature}
                  onChange={(e) =>
                    setGenerationParams((prev) => ({
                      ...prev,
                      temperature: parseFloat(e.target.value) || 0.7,
                    }))
                  }
                  className="w-20 text-right"
                  min="0"
                  max="2"
                  step="0.01"
                />
              </div>
              <Slider
                value={[generationParams.temperature]}
                onValueChange={(value) =>
                  setGenerationParams((prev) => ({
                    ...prev,
                    temperature: value[0],
                  }))
                }
                min={0}
                max={2}
                step={0.01}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Controls randomness. Higher values make output more creative,
                lower values more focused.
              </p>
            </div>

            <Separator />

            {/* Top P */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label htmlFor="top-p">Top P</Label>
                <Input
                  id="top-p"
                  type="number"
                  value={generationParams.top_p}
                  onChange={(e) =>
                    setGenerationParams((prev) => ({
                      ...prev,
                      top_p: parseFloat(e.target.value) || 0.9,
                    }))
                  }
                  className="w-20 text-right"
                  min="0"
                  max="1"
                  step="0.01"
                />
              </div>
              <Slider
                value={[generationParams.top_p]}
                onValueChange={(value) =>
                  setGenerationParams((prev) => ({
                    ...prev,
                    top_p: value[0],
                  }))
                }
                min={0}
                max={1}
                step={0.01}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Nucleus sampling. Considers only tokens with cumulative
                probability up to this value.
              </p>
            </div>

            <Separator />

            {/* Top K */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label htmlFor="top-k">Top K</Label>
                <Input
                  id="top-k"
                  type="number"
                  value={generationParams.top_k}
                  onChange={(e) =>
                    setGenerationParams((prev) => ({
                      ...prev,
                      top_k: parseInt(e.target.value) || 50,
                    }))
                  }
                  className="w-20 text-right"
                  min="1"
                  max="200"
                />
              </div>
              <Slider
                value={[generationParams.top_k]}
                onValueChange={(value) =>
                  setGenerationParams((prev) => ({
                    ...prev,
                    top_k: value[0],
                  }))
                }
                min={1}
                max={200}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Considers only the top K most likely tokens at each step.
              </p>
            </div>

            <Separator />

            {/* Repetition Penalty */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label htmlFor="rep-penalty">Repetition Penalty</Label>
                <Input
                  id="rep-penalty"
                  type="number"
                  value={generationParams.repetition_penalty}
                  onChange={(e) =>
                    setGenerationParams((prev) => ({
                      ...prev,
                      repetition_penalty: parseFloat(e.target.value) || 1.1,
                    }))
                  }
                  className="w-20 text-right"
                  min="0.5"
                  max="2"
                  step="0.01"
                />
              </div>
              <Slider
                value={[generationParams.repetition_penalty]}
                onValueChange={(value) =>
                  setGenerationParams((prev) => ({
                    ...prev,
                    repetition_penalty: value[0],
                  }))
                }
                min={0.5}
                max={2}
                step={0.01}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Penalty for repeating tokens. Higher values reduce repetition.
              </p>
            </div>

            <Separator />

            {/* Do Sample Toggle */}
            <div className="flex items-center justify-between">
              <Label htmlFor="do-sample" className="text-sm">
                Enable Sampling
              </Label>
              <Switch
                id="do-sample"
                checked={generationParams.do_sample}
                onCheckedChange={(checked) =>
                  setGenerationParams((prev) => ({
                    ...prev,
                    do_sample: checked,
                  }))
                }
              />
            </div>
            <p className="text-xs text-muted-foreground">
              When enabled, uses sampling instead of greedy decoding. Disable
              for more deterministic output.
            </p>

            {/* Preset Buttons */}
            <div className="space-y-3">
              <Label className="text-sm font-medium">Quick Presets</Label>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setGenerationParams({
                      max_new_tokens: 256,
                      temperature: 0.1,
                      do_sample: true,
                      top_p: 0.9,
                      top_k: 50,
                      repetition_penalty: 1.1,
                    })
                  }
                >
                  Conservative
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setGenerationParams({
                      max_new_tokens: 512,
                      temperature: 0.7,
                      do_sample: true,
                      top_p: 0.9,
                      top_k: 50,
                      repetition_penalty: 1.1,
                    })
                  }
                >
                  Balanced
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setGenerationParams({
                      max_new_tokens: 512,
                      temperature: 1.2,
                      do_sample: true,
                      top_p: 0.8,
                      top_k: 40,
                      repetition_penalty: 1.15,
                    })
                  }
                >
                  Creative
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setGenerationParams({
                      max_new_tokens: 256,
                      temperature: 0.0,
                      do_sample: false,
                      top_p: 1.0,
                      top_k: 1,
                      repetition_penalty: 1.0,
                    })
                  }
                >
                  Deterministic
                </Button>
              </div>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </div>
  )
}

export default ComparisonChatInterface
