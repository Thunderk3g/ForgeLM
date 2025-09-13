"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  BarChart3,
  FileText,
  MessageSquare,
  RotateCcw,
  Send,
  Settings,
  Sliders,
  Trash2,
} from "lucide-react";
import type React from "react";
import { useEffect, useRef, useState } from "react";

interface ModelPair {
  pair_id: string;
  base_model_name: string;
  adapter_name: string;
  base_model_size_mb: number;
  adapter_size_mb: number;
  estimated_memory_mb: number;
  status: string;
  load_time_ms?: number;
}

interface ChatMessage {
  role: string;
  content: string;
  timestamp: string;
  model_type?: string;
}

interface ChatSession {
  session_id: string;
  session_name: string;
  pair_id: string;
  model_info: {
    base_model: string;
    adapter_name: string;
  };
  created_at: string;
  last_activity: string;
  message_count: number;
  status: string;
}

export const ComparisonChatInterface: React.FC = () => {
  const [modelPairs, setModelPairs] = useState<ModelPair[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [showNewSessionDialog, setShowNewSessionDialog] = useState(false);
  const [selectedPairId, setSelectedPairId] = useState<string>("");
  const [sessionName, setSessionName] = useState("");
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [analyticsData, setAnalyticsData] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [showSystemPromptDialog, setShowSystemPromptDialog] = useState(false);
  const [customSystemPrompt, setCustomSystemPrompt] = useState("");
  const [useAgentOrchestrator, setUseAgentOrchestrator] = useState(false);
  const [availableAgents, setAvailableAgents] = useState<
    Array<Record<string, unknown>>
  >([]);

  // Generation parameters state
  const [generationParams, setGenerationParams] = useState({
    max_new_tokens: 256,
    temperature: 0.7,
    do_sample: true,
    top_p: 0.9,
    top_k: 50,
    repetition_penalty: 1.1,
  });
  const [showParametersSheet, setShowParametersSheet] = useState(false);

  // Refs for auto-scrolling
  const baseMessagesRef = useRef<HTMLDivElement>(null);
  const finetunedMessagesRef = useRef<HTMLDivElement>(null);

  // Load model pairs, sessions, and agents on component mount
  useEffect(() => {
    loadModelPairs();
    loadSessions();
    loadAvailableAgents();
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (baseMessagesRef.current) {
      baseMessagesRef.current.scrollTop = baseMessagesRef.current.scrollHeight;
    }
    if (finetunedMessagesRef.current) {
      finetunedMessagesRef.current.scrollTop =
        finetunedMessagesRef.current.scrollHeight;
    }
  }, [messages]);

  const loadModelPairs = async () => {
    try {
      const response = await fetch("http://localhost:8000/chat/model-pairs");
      const data = await response.json();
      if (data.status === "success") {
        // Add estimated memory calculation if not present
        const processedPairs = data.model_pairs.map((pair: ModelPair) => ({
          ...pair,
          estimated_memory_mb:
            pair.estimated_memory_mb ||
            pair.base_model_size_mb + pair.adapter_size_mb,
        }));
        setModelPairs(processedPairs);
        console.log("Loaded model pairs:", processedPairs);
      }
    } catch (error) {
      console.error("Failed to load model pairs:", error);
    }
  };

  const loadSessions = async () => {
    try {
      const response = await fetch("http://localhost:8000/chat/sessions");
      const data = await response.json();
      if (data.status === "success") {
        setSessions(data.sessions);
      }
    } catch (error) {
      console.error("Failed to load sessions:", error);
    }
  };

  const loadAvailableAgents = async () => {
    try {
      const response = await fetch("http://localhost:8000/agents/list");
      const data = await response.json();
      if (data.status === "success") {
        setAvailableAgents(data.agents);
      }
    } catch (error) {
      console.error("Failed to load agents:", error);
      // Set empty array as fallback
      setAvailableAgents([]);
    }
  };

  const createSession = async () => {
    if (!selectedPairId) return;

    try {
      const response = await fetch("http://localhost:8000/chat/sessions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          pair_id: selectedPairId,
          session_name: sessionName || undefined,
          custom_system_prompt: customSystemPrompt || undefined,
          use_agent_orchestrator: useAgentOrchestrator,
        }),
      });

      const data = await response.json();
      if (data.status === "success") {
        const newSession = data.session;
        setSessions([...sessions, newSession]);
        setActiveSessionId(newSession.session_id);
        setMessages([]);
        setShowNewSessionDialog(false);
        setSessionName("");
        setSelectedPairId("");
      }
    } catch (error) {
      console.error("Failed to create session:", error);
    }
  };

  const loadSessionHistory = async (sessionId: string) => {
    try {
      const response = await fetch(
        `http://localhost:8000/chat/sessions/${sessionId}`
      );
      const data = await response.json();
      if (data.status === "success") {
        setMessages(data.messages || []);
      }
    } catch (error) {
      console.error("Failed to load session history:", error);
    }
  };

  const deleteSession = async (sessionId: string) => {
    try {
      const response = await fetch(
        `http://localhost:8000/chat/sessions/${sessionId}`,
        {
          method: "DELETE",
        }
      );

      if (response.ok) {
        setSessions(sessions.filter((s) => s.session_id !== sessionId));
        if (activeSessionId === sessionId) {
          setActiveSessionId(null);
          setMessages([]);
        }
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
    }
  };

  const sendMessage = async () => {
    if (!currentMessage.trim() || !activeSessionId || isGenerating) return;

    const userMessage: ChatMessage = {
      role: "user",
      content: currentMessage,
      timestamp: new Date().toISOString(),
    };

    // Add user message to display
    setMessages((prev) => [...prev, userMessage]);
    setCurrentMessage("");
    setIsGenerating(true);

    try {
      const response = await fetch("http://localhost:8000/chat/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: activeSessionId,
          messages: [...messages, userMessage],
          generation_params: generationParams,
          custom_system_prompt: customSystemPrompt || undefined,
          use_agent_orchestrator: useAgentOrchestrator,
        }),
      });

      const data = await response.json();
      if (data.status === "success") {
        const comparisonResponse = data.response;

        // Add both responses
        const baseResponse: ChatMessage = {
          role: "assistant",
          content: comparisonResponse.base_response,
          timestamp: comparisonResponse.timestamp,
          model_type: "base",
        };

        const finetunedResponse: ChatMessage = {
          role: "assistant",
          content: comparisonResponse.finetuned_response,
          timestamp: comparisonResponse.timestamp,
          model_type: "finetuned",
        };

        setMessages((prev) => [...prev, baseResponse, finetunedResponse]);
      }
    } catch (error) {
      console.error("Failed to generate response:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const loadAnalytics = async () => {
    try {
      const response = await fetch(
        "http://localhost:8000/chat/research/analytics"
      );
      const data = await response.json();
      if (data.status === "success") {
        setAnalyticsData(data.analytics);
      }
    } catch (error) {
      console.error("Failed to load analytics:", error);
    }
  };

  const exportResearchData = async () => {
    try {
      const response = await fetch(
        "http://localhost:8000/chat/research/export?format=csv"
      );
      const data = await response.json();
      if (data.status === "success") {
        alert(`Research data exported to: ${data.export_file}`);
      }
    } catch (error) {
      console.error("Failed to export research data:", error);
    }
  };

  const selectSession = (sessionId: string) => {
    setActiveSessionId(sessionId);
    loadSessionHistory(sessionId);
  };

  const activeSession = sessions.find((s) => s.session_id === activeSessionId);
  const baseMessages = messages.filter(
    (m) => m.role === "user" || m.model_type === "base"
  );
  const finetunedMessages = messages.filter(
    (m) => m.role === "user" || m.model_type === "finetuned"
  );

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <div className="w-80 bg-card border-r border-border flex flex-col">
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
              <MessageSquare className="w-5 h-5" />
              Chat Sessions
            </h2>
            <button
              onClick={() => setShowNewSessionDialog(true)}
              className="px-3 py-2 bg-foreground text-background rounded-md hover:bg-foreground/80 text-sm"
            >
              New Session
            </button>
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
                      ? "bg-muted border border-border"
                      : "bg-muted hover:bg-muted/80"
                  }`}
                  onClick={() => selectSession(session.session_id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-foreground truncate">
                        {session.session_name}
                      </h3>
                      <p className="text-sm text-muted-foreground truncate">
                        {session.model_info.base_model} +{" "}
                        {session.model_info.adapter_name}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {session.message_count} messages
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteSession(session.session_id);
                      }}
                      className="p-1 hover:bg-muted rounded"
                    >
                      <Trash2 className="w-4 h-4 text-muted-foreground hover:text-foreground" />
                    </button>
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
                  <h1 className="text-xl font-semibold text-foreground">
                    {activeSession.session_name}
                  </h1>
                  <p className="text-sm text-muted-foreground">
                    Comparing: {activeSession.model_info.base_model} vs{" "}
                    {activeSession.model_info.adapter_name}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => loadSessionHistory(activeSessionId!)}
                    className="p-2 hover:bg-muted rounded-md"
                    title="Refresh"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => {
                      setShowAnalytics(!showAnalytics);
                      if (!showAnalytics) loadAnalytics();
                    }}
                    className="p-2 hover:bg-muted rounded-md"
                    title="Research Analytics"
                  >
                    <BarChart3 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={exportResearchData}
                    className="p-2 hover:bg-muted rounded-md"
                    title="Export Research Data"
                  >
                    <FileText className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setShowSystemPromptDialog(true)}
                    className="p-2 hover:bg-muted rounded-md"
                    title="System Prompt Settings"
                  >
                    <Settings className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setShowParametersSheet(true)}
                    className="p-2 hover:bg-muted rounded-md"
                    title="Generation Parameters"
                  >
                    <Sliders className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Analytics Panel */}
            {showAnalytics && analyticsData && (
              <div className="p-4 bg-muted/50 border-b border-border">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="font-medium text-foreground">
                    Research Analytics
                  </h3>
                  <button
                    onClick={() => setShowAnalytics(false)}
                    className="text-muted-foreground hover:text-foreground"
                  >
                    ×
                  </button>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="bg-background rounded p-2">
                    <div className="text-muted-foreground">
                      Total Comparisons
                    </div>
                    <div className="font-medium">
                      {(analyticsData.total_comparisons as number) ?? 0}
                    </div>
                  </div>
                  <div className="bg-background rounded p-2">
                    <div className="text-muted-foreground">Avg Similarity</div>
                    <div className="font-medium">
                      {(
                        ((analyticsData.average_similarity as number) ?? 0) *
                        100
                      ).toFixed(1)}
                      %
                    </div>
                  </div>
                  <div className="bg-background rounded p-2">
                    <div className="text-muted-foreground">Length Ratio</div>
                    <div className="font-medium">
                      {(
                        (analyticsData.average_length_ratio as number) ?? 0
                      ).toFixed(2)}
                      x
                    </div>
                  </div>
                  <div className="bg-background rounded p-2">
                    <div className="text-muted-foreground">Model Pairs</div>
                    <div className="font-medium">
                      {(analyticsData.unique_model_pairs as number) ?? 0}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Split Screen Chat */}
            <div className="flex-1 flex">
              {/* Base Model Side */}
              <div className="flex-1 flex flex-col border-r border-border">
                <div className="p-3 bg-muted border-b border-border">
                  <h3 className="font-medium text-foreground">
                    Base Model: {activeSession.model_info.base_model}
                  </h3>
                </div>
                <div
                  ref={baseMessagesRef}
                  className="flex-1 overflow-y-auto p-4 space-y-4"
                >
                  {baseMessages.map((message, index) => (
                    <div
                      key={`base-${index}`}
                      className={`flex ${
                        message.role === "user"
                          ? "justify-end"
                          : "justify-start"
                      }`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg p-3 ${
                          message.role === "user"
                            ? "bg-foreground text-background"
                            : "bg-muted text-foreground border border-border"
                        }`}
                      >
                        <div className="whitespace-pre-wrap">
                          {message.content}
                        </div>
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
                <div className="p-3 bg-muted border-b border-border">
                  <h3 className="font-medium text-foreground">
                    Fine-tuned: {activeSession.model_info.adapter_name}
                  </h3>
                </div>
                <div
                  ref={finetunedMessagesRef}
                  className="flex-1 overflow-y-auto p-4 space-y-4"
                >
                  {finetunedMessages.map((message, index) => (
                    <div
                      key={`finetuned-${index}`}
                      className={`flex ${
                        message.role === "user"
                          ? "justify-end"
                          : "justify-start"
                      }`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg p-3 ${
                          message.role === "user"
                            ? "bg-foreground text-background"
                            : "bg-muted text-foreground border border-border"
                        }`}
                      >
                        <div className="whitespace-pre-wrap">
                          {message.content}
                        </div>
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
                <textarea
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message here..."
                  className="flex-1 resize-none p-3 bg-background border border-input rounded-lg text-foreground focus:ring-2 focus:ring-ring focus:border-transparent"
                  rows={2}
                  disabled={isGenerating}
                />
                <button
                  onClick={sendMessage}
                  disabled={!currentMessage.trim() || isGenerating}
                  className="px-4 py-2 bg-foreground text-background rounded-lg hover:bg-foreground/80 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <Send className="w-4 h-4" />
                  {isGenerating ? "Generating..." : "Send"}
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center bg-muted/20">
            <div className="text-center">
              <MessageSquare className="w-16 h-16 mx-auto mb-4 text-muted-foreground opacity-50" />
              <h2 className="text-xl font-semibold text-foreground mb-2">
                Select a Chat Session
              </h2>
              <p className="text-muted-foreground mb-4">
                Choose a session from the sidebar or create a new one to start
                comparing models
              </p>
              <button
                onClick={() => setShowNewSessionDialog(true)}
                className="px-4 py-2 bg-foreground text-background rounded-lg hover:bg-foreground/80"
              >
                Create New Session
              </button>
            </div>
          </div>
        )}
      </div>

      {/* New Session Dialog */}
      {showNewSessionDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card rounded-lg p-6 w-full max-w-md border border-border">
            <h3 className="text-lg font-semibold text-foreground mb-4">
              Create New Chat Session
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Session Name (Optional)
                </label>
                <input
                  type="text"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                  placeholder="My Chat Session"
                  className="w-full p-2 bg-background border border-input rounded-md text-foreground"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Model Pair
                </label>
                <select
                  value={selectedPairId}
                  onChange={(e) => setSelectedPairId(e.target.value)}
                  className="w-full p-2 bg-background border border-input rounded-md text-foreground"
                >
                  <option value="">Select a model pair...</option>
                  {modelPairs.map((pair) => {
                    const memoryGB = pair.estimated_memory_mb
                      ? (pair.estimated_memory_mb / 1024).toFixed(1)
                      : "?.?";
                    const statusText = pair.status === "mock" ? " (Demo)" : "";

                    return (
                      <option key={pair.pair_id} value={pair.pair_id}>
                        📊 {pair.base_model_name} ↔️ {pair.adapter_name} (
                        {memoryGB}GB){statusText}
                      </option>
                    );
                  })}
                </select>

                {selectedPairId && (
                  <div className="mt-2 p-3 bg-muted rounded-md">
                    {(() => {
                      const selectedPair = modelPairs.find(
                        (p) => p.pair_id === selectedPairId
                      );
                      return selectedPair ? (
                        <div className="text-sm">
                          <div className="font-medium mb-2">
                            Selected Model Pair:
                          </div>
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <div className="text-foreground font-medium">
                                Base Model
                              </div>
                              <div className="text-sm text-muted-foreground">
                                {selectedPair.base_model_name}
                              </div>
                              <div className="text-xs">
                                Size:{" "}
                                {selectedPair.base_model_size_mb.toFixed(0)}MB
                              </div>
                            </div>
                            <div>
                              <div className="text-foreground font-medium">
                                Fine-tuned Model
                              </div>
                              <div className="text-sm text-muted-foreground">
                                {selectedPair.adapter_name}
                              </div>
                              <div className="text-xs">
                                Size: {selectedPair.adapter_size_mb.toFixed(0)}
                                MB
                              </div>
                            </div>
                          </div>
                          {selectedPair.status === "mock" && (
                            <div className="mt-2 text-xs text-muted-foreground bg-muted p-2 rounded">
                              🎭 Demo Mode: This will use mock responses to
                              demonstrate the comparison interface
                            </div>
                          )}
                        </div>
                      ) : null;
                    })()}
                  </div>
                )}
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <button
                onClick={() => {
                  setShowNewSessionDialog(false);
                  setSessionName("");
                  setSelectedPairId("");
                }}
                className="px-4 py-2 text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
              <button
                onClick={createSession}
                disabled={!selectedPairId}
                className="px-4 py-2 bg-foreground text-background rounded-md hover:bg-foreground/80 disabled:opacity-50"
              >
                Create Session
              </button>
            </div>
          </div>
        </div>
      )}
      {/* System Prompt Settings Dialog */}
      {showSystemPromptDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card rounded-lg p-6 w-full max-w-2xl border border-border max-h-[80vh] overflow-y-auto">
            <h3 className="text-lg font-semibold text-foreground mb-4">
              System Prompt Settings
            </h3>

            <div className="space-y-6">
              {/* Agent Orchestrator Toggle */}
              <div>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={useAgentOrchestrator}
                    onChange={(e) => setUseAgentOrchestrator(e.target.checked)}
                    className="rounded border-input"
                  />
                  <span className="text-sm font-medium text-foreground">
                    Use Agent Orchestrator (Auto-select specialist agents)
                  </span>
                </label>
                <p className="text-xs text-muted-foreground mt-1">
                  Automatically selects the best AI agent based on your query
                  type
                </p>
              </div>

              {/* Available Agents Preview */}
              {useAgentOrchestrator && availableAgents.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Available Specialist Agents:
                  </label>
                  <div className="grid grid-cols-1 gap-2 max-h-32 overflow-y-auto">
                    {availableAgents.map((agent) => (
                      <div
                        key={agent.agent_id as string}
                        className="p-2 bg-muted rounded text-xs"
                      >
                        <div className="font-medium">
                          {agent.name as string}
                        </div>
                        <div className="text-muted-foreground">
                          {(agent.capabilities as string[])?.join(", ") ||
                            "General assistance"}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Custom System Prompt */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Custom System Prompt {!useAgentOrchestrator && "(Optional)"}
                </label>
                <textarea
                  value={customSystemPrompt}
                  onChange={(e) => setCustomSystemPrompt(e.target.value)}
                  placeholder={
                    useAgentOrchestrator
                      ? "Leave empty to use agent-based prompts, or provide a custom prompt to override..."
                      : "Enter a custom system prompt to guide the fine-tuned model's behavior..."
                  }
                  className="w-full p-3 bg-background border border-input rounded-md text-foreground resize-none"
                  rows={8}
                />
                <div className="text-xs text-muted-foreground mt-1">
                  {customSystemPrompt.length}/2000 characters
                </div>
              </div>

              {/* Example Prompts */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Quick Examples:
                </label>
                <div className="space-y-2">
                  {[
                    {
                      name: "Helpful Assistant",
                      prompt:
                        "You are a helpful, knowledgeable assistant that provides detailed and practical responses. Always be specific and helpful.",
                    },
                    {
                      name: "Code Reviewer",
                      prompt:
                        "You are an expert code reviewer with deep knowledge of software development best practices. Review code for quality, security, performance, and maintainability. Provide specific, actionable feedback.",
                    },
                    {
                      name: "Creative Writer",
                      prompt:
                        "You are a creative companion that helps with brainstorming, creative writing, and artistic projects. Think outside the box and inspire innovative ideas.",
                    },
                    {
                      name: "Business Analyst",
                      prompt:
                        "You are a business analyst with expertise in data-driven decision making. Analyze business problems and provide strategic insights backed by data and market knowledge.",
                    },
                  ].map((example) => (
                    <button
                      key={example.name}
                      onClick={() => setCustomSystemPrompt(example.prompt)}
                      className="text-left p-2 bg-muted hover:bg-muted/80 rounded text-xs w-full"
                    >
                      <div className="font-medium text-foreground">
                        {example.name}
                      </div>
                      <div className="text-muted-foreground truncate">
                        {example.prompt}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <button
                onClick={() => {
                  setShowSystemPromptDialog(false);
                }}
                className="px-4 py-2 text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowSystemPromptDialog(false);
                  // Settings are automatically applied through state
                }}
                className="px-4 py-2 bg-foreground text-background rounded-md hover:bg-foreground/80"
              >
                Apply Settings
              </button>
            </div>
          </div>
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
  );
};

export default ComparisonChatInterface;
