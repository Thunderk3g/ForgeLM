import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  FileText,
  MessageSquare,
  RotateCcw,
  Send,
  Settings,
  Trash2,
  Sliders,
} from "lucide-react";
import React, { useEffect, useRef, useState } from "react";
import { ConfigurationProvider, useConfiguration } from "../contexts/ConfigurationContext";
import { ConfigurationPanel } from "./ConfigurationPanel";
import type { 
  DualModelConfig,
  ModelPair as ModelPairType,
  ChatMessage as ChatMessageType,
  ChatSession as ChatSessionType,
  EnhancedComparisonRequest,
  ComparisonResponse as ComparisonResponseType
} from "../lib/types";
import { 
  generateEnhancedComparison
} from "../lib/api";

// Use types from lib/types.ts
type ModelPair = ModelPairType & { estimated_memory_mb?: number; }
type ChatMessage = ChatMessageType & { model_type?: string; }
type ChatSession = ChatSessionType
type ComparisonResponse = ComparisonResponseType

// Inner component that uses the configuration context
const ComparisonChatInterfaceInner: React.FC = () => {
  const {
    state: configState,
    loadSessionConfig,
    updateConfig,
    applyConfig,
    saveCurrentAsPreset
  } = useConfiguration();
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
  const [analyticsData, setAnalyticsData] = useState<any>(null);
  const [showSystemPromptDialog, setShowSystemPromptDialog] = useState(false);
  const [customSystemPrompt, setCustomSystemPrompt] = useState("");
  const [useAgentOrchestrator, setUseAgentOrchestrator] = useState(false);
  const [availableAgents, setAvailableAgents] = useState<any[]>([]);
  const [newSessionConfig, setNewSessionConfig] = useState<DualModelConfig | null>(null);
  const [showRightSidebar, setShowRightSidebar] = useState(false);

  // Debug logging for right sidebar state
  console.log('ComparisonChatInterface render - showRightSidebar:', showRightSidebar);

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
      const requestBody: any = {
        pair_id: selectedPairId,
        session_name: sessionName || undefined,
        custom_system_prompt: customSystemPrompt || undefined,
        use_agent_orchestrator: useAgentOrchestrator,
      };

      // Include configuration if available
      if (newSessionConfig) {
        requestBody.initial_config = {
          ...newSessionConfig,
          session_id: undefined, // Let the backend set this
        };
      }

      const response = await fetch("http://localhost:8000/chat/sessions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
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
        setNewSessionConfig(null); // Clear the configuration after creating session
        
        // If configuration was provided, update the current config state
        if (newSessionConfig) {
          configState.updateConfig({
            ...newSessionConfig,
            session_id: newSession.session_id,
          });
        }
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
      // Use enhanced comparison with configuration
      const requestData: EnhancedComparisonRequest = {
        session_id: activeSessionId,
        messages: [...messages, userMessage],
        config: configState.currentConfig || undefined,
        use_agent_orchestrator: useAgentOrchestrator,
        return_debug_info: false
      };

      const data = await generateEnhancedComparison(requestData);
      if ((data as any).status === "success") {
        const comparisonResponse = (data as any).response;

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
      
      // Fallback to original API if enhanced comparison fails
      try {
        const response = await fetch("http://localhost:8000/chat/generate", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            session_id: activeSessionId,
            messages: [...messages, userMessage],
            generation_params: {
              max_new_tokens: 1024,
              temperature: 0.7,
              do_sample: true,
              top_p: 0.9,
            },
            custom_system_prompt: customSystemPrompt || undefined,
            use_agent_orchestrator: useAgentOrchestrator,
          }),
        });

        const data = await response.json();
        if (data.status === "success") {
          const comparisonResponse: ComparisonResponse = data.response;

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
      } catch (fallbackError) {
        console.error("Fallback API also failed:", fallbackError);
      }
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

  const selectSession = async (sessionId: string) => {
    setActiveSessionId(sessionId);
    await loadSessionHistory(sessionId);
    
    // Load session configuration
    try {
      await loadSessionConfig(sessionId);
    } catch (error) {
      console.error("Failed to load session configuration:", error);
    }
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
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <MessageSquare className="w-5 h-5 text-primary" />
                Chat Sessions
              </h2>
              {configState.currentConfig && (
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" title="Configuration Active" />
              )}
            </div>
            <button
              onClick={() => setShowNewSessionDialog(true)}
              className="px-3 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 text-sm"
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
                      ? "bg-purple-500/10 border border-purple-500/20"
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
                      className="p-1 hover:bg-red-500/10 rounded"
                    >
                      <Trash2 className="w-4 h-4 text-muted-foreground hover:text-red-500" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex">
        {/* Chat Section */}
        <div className={`flex-1 flex flex-col transition-all duration-300 ${
          showRightSidebar ? 'mr-0' : 'mr-0'
        }`}>
          {activeSession ? (
          <>
            {/* Header */}
            <div className="p-4 border-b border-border bg-card">
              {/* DEBUG: Test Right Sidebar Button */}
              <div className="mb-2">
                <button
                  onClick={() => {
                    console.log('DEBUG TEST: Toggling right sidebar!');
                    setShowRightSidebar(!showRightSidebar);
                  }}
                  className="px-4 py-2 bg-red-500 text-white rounded font-bold"
                  style={{ zIndex: 999 }}
                >
                  DEBUG: Toggle Right Sidebar ({showRightSidebar ? 'OPEN' : 'CLOSED'})
                </button>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <h1 className="text-xl font-semibold text-foreground">
                      {activeSession.session_name}
                    </h1>
                    {configState.currentConfig && (
                      <span className="inline-flex items-center px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">
                        Custom Config
                      </span>
                    )}
                  </div>
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
                    onClick={() => {
                      console.log('Settings button clicked! Current state:', showRightSidebar);
                      setShowRightSidebar(!showRightSidebar);
                    }}
                    className={`p-2 hover:bg-muted rounded-md transition-colors border-2 ${
                      showRightSidebar ? 'bg-primary text-primary-foreground border-primary' : 'hover:bg-muted border-border hover:border-primary'
                    }`}
                    title="Toggle Model Configuration Sidebar - DEBUG MODE"
                    style={{ position: 'relative', zIndex: 10, cursor: 'pointer' }}
                  >
                    <Sliders className={`w-4 h-4 ${showRightSidebar ? 'text-white' : ''}`} />
                  </button>
                  <button
                    onClick={() => setShowSystemPromptDialog(true)}
                    className="p-2 hover:bg-muted rounded-md"
                    title="System Prompt Settings"
                  >
                    <Settings className="w-4 h-4" />
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
                      {analyticsData.total_comparisons || 0}
                    </div>
                  </div>
                  <div className="bg-background rounded p-2">
                    <div className="text-muted-foreground">Avg Similarity</div>
                    <div className="font-medium">
                      {((analyticsData.average_similarity || 0) * 100).toFixed(
                        1
                      )}
                      %
                    </div>
                  </div>
                  <div className="bg-background rounded p-2">
                    <div className="text-muted-foreground">Length Ratio</div>
                    <div className="font-medium">
                      {(analyticsData.average_length_ratio || 0).toFixed(2)}x
                    </div>
                  </div>
                  <div className="bg-background rounded p-2">
                    <div className="text-muted-foreground">Model Pairs</div>
                    <div className="font-medium">
                      {analyticsData.unique_model_pairs || 0}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Split Screen Chat */}
            <div className="flex-1 flex">
              {/* Base Model Side */}
              <div className="flex-1 flex flex-col border-r border-border">
                <div className="p-3 bg-blue-500/10 border-b border-border">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-blue-400">
                      Base Model: {activeSession.model_info.base_model}
                    </h3>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setShowRightSidebar(!showRightSidebar)}
                        className={`p-1 rounded-md transition-colors ${
                          showRightSidebar ? 'bg-blue-500/30 text-blue-300' : 'hover:bg-blue-500/20 text-blue-400'
                        }`}
                        title="Toggle Configuration Sidebar"
                      >
                        <Sliders className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  
                  {/* Base Model Configuration Controls */}
                  {configState.currentConfig && (
                    <div className="flex flex-wrap gap-2 text-xs">
                      <div className="px-2 py-1 bg-blue-500/20 rounded-md">
                        Temp: {configState.currentConfig.base_model.generation_config.temperature}
                      </div>
                      <div className="px-2 py-1 bg-blue-500/20 rounded-md">
                        Max: {configState.currentConfig.base_model.generation_config.max_tokens}
                      </div>
                      <div className="px-2 py-1 bg-blue-500/20 rounded-md">
                        Top-p: {configState.currentConfig.base_model.generation_config.top_p}
                      </div>
                      
                      {/* Agent Orchestrator Integration */}
                      {useAgentOrchestrator && (
                        <div className="px-2 py-1 bg-yellow-500/20 text-yellow-700 rounded-md">
                          🤖 Agent Mode
                        </div>
                      )}
                    </div>
                  )}
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
                            ? "bg-purple-600 text-white"
                            : "bg-blue-500/10 text-foreground border border-blue-500/20"
                        }`}
                      >
                        <div className="whitespace-pre-wrap">
                          {message.content}
                        </div>
                        <div className="text-xs opacity-70 mt-1">
                          {message.timestamp ? new Date(message.timestamp).toLocaleTimeString() : ''}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Fine-tuned Model Side */}
              <div className="flex-1 flex flex-col">
                <div className="p-3 bg-green-500/10 border-b border-border">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-green-400">
                      Fine-tuned: {activeSession.model_info.adapter_name}
                    </h3>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setShowRightSidebar(!showRightSidebar)}
                        className={`p-1 rounded-md transition-colors ${
                          showRightSidebar ? 'bg-green-500/30 text-green-300' : 'hover:bg-green-500/20 text-green-400'
                        }`}
                        title="Toggle Configuration Sidebar"
                      >
                        <Sliders className="w-4 h-4" />
                      </button>
                      
                      {/* Agent Orchestrator Toggle */}
                      <button
                        onClick={() => setUseAgentOrchestrator(!useAgentOrchestrator)}
                        className={`p-1 rounded-md transition-colors ${
                          useAgentOrchestrator 
                            ? 'bg-yellow-500/20 text-yellow-600' 
                            : 'hover:bg-green-500/20 text-green-400'
                        }`}
                        title={useAgentOrchestrator ? "Disable Agent Orchestrator" : "Enable Agent Orchestrator"}
                      >
                        <Settings className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  
                  {/* Fine-tuned Model Configuration Controls */}
                  {configState.currentConfig && (
                    <div className="flex flex-wrap gap-2 text-xs">
                      <div className="px-2 py-1 bg-green-500/20 rounded-md">
                        Temp: {configState.currentConfig.finetuned_model.generation_config.temperature}
                      </div>
                      <div className="px-2 py-1 bg-green-500/20 rounded-md">
                        Max: {configState.currentConfig.finetuned_model.generation_config.max_tokens}
                      </div>
                      <div className="px-2 py-1 bg-green-500/20 rounded-md">
                        Top-p: {configState.currentConfig.finetuned_model.generation_config.top_p}
                      </div>
                      
                      {/* Agent Status */}
                      {useAgentOrchestrator && availableAgents.length > 0 && (
                        <div className="px-2 py-1 bg-yellow-500/20 text-yellow-700 rounded-md">
                          🤖 {availableAgents.length} Agents Available
                        </div>
                      )}
                      
                      {/* Custom System Prompt Indicator */}
                      {customSystemPrompt && (
                        <div className="px-2 py-1 bg-purple-500/20 text-purple-700 rounded-md">
                          📝 Custom Prompt
                        </div>
                      )}
                    </div>
                  )}
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
                            ? "bg-purple-600 text-white"
                            : "bg-green-500/10 text-foreground border border-green-500/20"
                        }`}
                      >
                        <div className="whitespace-pre-wrap">
                          {message.content}
                        </div>
                        <div className="text-xs opacity-70 mt-1">
                          {message.timestamp ? new Date(message.timestamp).toLocaleTimeString() : ''}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-border bg-card">
              {/* Integrated Configuration & Agent Controls */}
              <div className="mb-3 space-y-2">
                {/* Configuration Status & Agent Status */}
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-3">
                    {configState.currentConfig && (
                      <div className="flex items-center gap-2">
                        <Sliders className="w-3 h-3 text-blue-500" />
                        <span className="text-muted-foreground">Config:</span>
                        <span className="font-medium">
                          T:{configState.currentConfig.base_model.generation_config.temperature} | 
                          Max:{configState.currentConfig.base_model.generation_config.max_tokens}
                        </span>
                      </div>
                    )}
                    
                    {/* Agent Orchestrator Status */}
                    {useAgentOrchestrator && (
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
                        <span className="text-yellow-600 font-medium">
                          Agent Mode ({availableAgents.length} available)
                        </span>
                      </div>
                    )}
                    
                    {customSystemPrompt && (
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                        <span className="text-purple-600 font-medium">Custom Prompt</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setShowRightSidebar(!showRightSidebar)}
                      className="text-primary hover:text-primary/80 text-xs"
                    >
                      Configure
                    </button>
                    <button
                      onClick={() => setShowSystemPromptDialog(true)}
                      className="text-primary hover:text-primary/80 text-xs"
                    >
                      Prompts & Agents
                    </button>
                  </div>
                </div>
                
                {/* Quick Configuration Presets */}
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">Quick:</span>
                  
                  {/* Preset Buttons */}
                  <button 
                    onClick={() => {
                      const creativeConfig = {
                        session_id: activeSessionId!,
                        base_model: {
                          system_prompt: customSystemPrompt,
                          context_length: 2048,
                          generation_config: {
                            temperature: 0.9, top_p: 0.95, top_k: 40, max_tokens: 512,
                            presence_penalty: 0.1, frequency_penalty: 0.1, repetition_penalty: 1.1,
                            do_sample: true, early_stopping: false, seed: null, pad_token_id: null, eos_token_id: null
                          },
                          model_id: modelPairs.find(p => p.pair_id === activeSession?.pair_id)?.base_model_name || null,
                          enabled: true
                        },
                        finetuned_model: {
                          system_prompt: customSystemPrompt,
                          context_length: 2048,
                          generation_config: {
                            temperature: 0.9, top_p: 0.95, top_k: 40, max_tokens: 512,
                            presence_penalty: 0.1, frequency_penalty: 0.1, repetition_penalty: 1.1,
                            do_sample: true, early_stopping: false, seed: null, pad_token_id: null, eos_token_id: null
                          },
                          model_id: modelPairs.find(p => p.pair_id === activeSession?.pair_id)?.adapter_name || null,
                          enabled: true
                        },
                        sync_parameters: false,
                        config_name: "Creative Preset",
                        created_at: new Date().toISOString(),
                        updated_at: new Date().toISOString()
                      };
                      updateConfig(creativeConfig);
                    }}
                    className="px-2 py-1 bg-purple-100 hover:bg-purple-200 text-purple-800 rounded-md transition-colors"
                  >
                    🎨 Creative
                  </button>
                  
                  <button 
                    onClick={() => {
                      const factualConfig = {
                        session_id: activeSessionId!,
                        base_model: {
                          system_prompt: customSystemPrompt,
                          context_length: 2048,
                          generation_config: {
                            temperature: 0.3, top_p: 0.85, top_k: 50, max_tokens: 256,
                            presence_penalty: 0.0, frequency_penalty: 0.0, repetition_penalty: 1.0,
                            do_sample: true, early_stopping: false, seed: null, pad_token_id: null, eos_token_id: null
                          },
                          model_id: modelPairs.find(p => p.pair_id === activeSession?.pair_id)?.base_model_name || null,
                          enabled: true
                        },
                        finetuned_model: {
                          system_prompt: customSystemPrompt,
                          context_length: 2048,
                          generation_config: {
                            temperature: 0.3, top_p: 0.85, top_k: 50, max_tokens: 256,
                            presence_penalty: 0.0, frequency_penalty: 0.0, repetition_penalty: 1.0,
                            do_sample: true, early_stopping: false, seed: null, pad_token_id: null, eos_token_id: null
                          },
                          model_id: modelPairs.find(p => p.pair_id === activeSession?.pair_id)?.adapter_name || null,
                          enabled: true
                        },
                        sync_parameters: false,
                        config_name: "Factual Preset",
                        created_at: new Date().toISOString(),
                        updated_at: new Date().toISOString()
                      };
                      updateConfig(factualConfig);
                    }}
                    className="px-2 py-1 bg-blue-100 hover:bg-blue-200 text-blue-800 rounded-md transition-colors"
                  >
                    📊 Factual
                  </button>
                  
                  {/* Agent Orchestrator Quick Toggle */}
                  <button
                    onClick={() => setUseAgentOrchestrator(!useAgentOrchestrator)}
                    className={`px-2 py-1 rounded-md transition-colors ${
                      useAgentOrchestrator 
                        ? 'bg-yellow-100 hover:bg-yellow-200 text-yellow-800' 
                        : 'bg-gray-100 hover:bg-gray-200 text-gray-800'
                    }`}
                  >
                    🤖 {useAgentOrchestrator ? 'Agent ON' : 'Agent OFF'}
                  </button>
                  
                  <button
                    onClick={() => setShowRightSidebar(!showRightSidebar)}
                    className="px-2 py-1 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-md transition-colors"
                  >
                    ⚙️ Full Config
                  </button>
                </div>
              </div>
              
              <div className="flex gap-2">
                <textarea
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={useAgentOrchestrator ? 
                    "Ask your question - agents will be selected automatically..." : 
                    "Type your message here..."
                  }
                  className="flex-1 resize-none p-3 bg-background border border-input rounded-lg text-foreground focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  rows={2}
                  disabled={isGenerating}
                />
                <button
                  onClick={sendMessage}
                  disabled={!currentMessage.trim() || isGenerating}
                  className={`px-4 py-2 text-white rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 ${
                    useAgentOrchestrator 
                      ? 'bg-yellow-600 hover:bg-yellow-700' 
                      : 'bg-purple-600 hover:bg-purple-700'
                  }`}
                >
                  {useAgentOrchestrator && <Settings className="w-4 h-4" />}
                  <Send className="w-4 h-4" />
                  {isGenerating ? "Generating..." : (useAgentOrchestrator ? "Ask Agents" : "Send")}
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
                comparing models with advanced configuration options
              </p>
              
              {/* Feature Highlights */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 max-w-3xl">
                <div className="p-4 bg-card border border-border rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Sliders className="w-5 h-5 text-purple-500" />
                    <h3 className="font-medium">Configuration Tab</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Access the Config tab in the sidebar to fine-tune parameters
                  </p>
                </div>
                <div className="p-4 bg-card border border-border rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Settings className="w-5 h-5 text-blue-500" />
                    <h3 className="font-medium">Quick Presets</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Apply Creative, Factual, or Balanced configurations instantly
                  </p>
                </div>
                <div className="p-4 bg-card border border-border rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <BarChart3 className="w-5 h-5 text-green-500" />
                    <h3 className="font-medium">Real-time Comparison</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    See how different configurations affect model responses
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowNewSessionDialog(true)}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
              >
                Create New Session
              </button>
            </div>
          </div>
        )}
        </div>

        {/* Right Sidebar - Configuration Panel */}
        <div className={`transition-all duration-300 border-l border-border bg-card ${
          showRightSidebar ? 'w-96' : 'w-0'
        } overflow-hidden flex-shrink-0`}>
          {showRightSidebar && (
            <div className="w-96 h-full flex flex-col">
              {/* Right Sidebar Header */}
              <div className="p-4 border-b border-border">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sliders className="w-5 h-5 text-primary" />
                    <h3 className="font-semibold text-foreground">Model Configuration</h3>
                  </div>
                  <button
                    onClick={() => setShowRightSidebar(false)}
                    className="p-1 hover:bg-muted rounded-md transition-colors"
                    title="Close Configuration"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
                <p className="text-sm text-muted-foreground mt-1">
                  {activeSession ? `Adjust parameters for ${activeSession.session_name}` : 'Select a session to configure parameters'}
                </p>
              </div>
              
              {/* Configuration Panel */}
              <div className="flex-1 overflow-hidden">
                {activeSession ? (
                  <ConfigurationPanel
                    sessionId={activeSessionId!}
                    isOpen={true}
                    onToggle={() => setShowRightSidebar(false)}
                    modelPair={modelPairs.find(p => p.pair_id === activeSession?.pair_id) || null}
                    currentConfig={configState.currentConfig}
                    onConfigChange={updateConfig}
                    onApplyConfig={async () => {
                      await applyConfig(activeSessionId!);
                    }}
                    onSavePreset={async (name, description) => {
                      await saveCurrentAsPreset(activeSessionId!, name, description, "Custom");
                    }}
                    isApplyingConfig={configState.isApplyingConfig}
                    className="h-full border-0 rounded-none shadow-none"
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-center p-8">
                    <Sliders className="w-16 h-16 text-muted-foreground/30 mb-4" />
                    <h3 className="font-medium text-foreground mb-2">No Session Selected</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      Select a chat session from the left sidebar to configure model parameters
                    </p>
                    <button
                      onClick={() => setShowRightSidebar(false)}
                      className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
                    >
                      Close Panel
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Collapsed State - Show expand button */}
          {!showRightSidebar && (
            <div className="w-0 h-full relative">
              <button
                onClick={() => setShowRightSidebar(true)}
                className="absolute -left-10 top-4 p-2 bg-primary text-primary-foreground rounded-l-md hover:bg-primary/90 transition-colors z-10"
                title="Open Configuration Sidebar"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Enhanced New Session Dialog with Configuration Panel */}
      {console.log('showNewSessionDialog:', showNewSessionDialog)}
      {showNewSessionDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-card rounded-lg w-full max-w-6xl h-[80vh] border border-border flex overflow-hidden">
            
            {/* Left Side - Configuration Panel */}
            <div className="w-1/2 border-r border-border">
              <div className="p-6 border-b border-border">
                <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-purple-500" />
                  Model Configuration
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Configure parameters for your new chat session
                </p>
              </div>
              
              <div className="h-full overflow-y-auto">
                {selectedPairId ? (
                  <div className="p-4">
                    {console.log('Rendering ConfigurationPanel with selectedPairId:', selectedPairId)}
                    <ConfigurationPanel
                      sessionId="new-session-temp"
                      isOpen={true}
                      onToggle={() => {}}
                      modelPair={modelPairs.find(p => p.pair_id === selectedPairId) || null}
                      currentConfig={newSessionConfig}
                      onConfigChange={(config) => setNewSessionConfig(config)}
                      onApplyConfig={async () => {}}
                      onSavePreset={async (name: string, description: string) => {
                        console.log('Save preset:', name, description);
                      }}
                      isApplyingConfig={false}
                      className="border-0 shadow-none bg-transparent"
                    />
                  </div>
                ) : (
                  <div className="p-8 text-center text-muted-foreground">
                    <Sliders className="w-16 h-16 mx-auto mb-4 opacity-30" />
                    <h4 className="font-medium mb-2">Select a Model Pair First</h4>
                    <p className="text-sm">
                      Choose your models on the right to configure parameters
                    </p>
                    
                    {/* Quick Preset Preview */}
                    <div className="mt-6 space-y-3">
                      <h5 className="font-medium text-foreground">Available Presets:</h5>
                      <div className="grid gap-2">
                        <div className="p-3 bg-purple-50 rounded-lg border border-purple-200">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-purple-600">🎨</span>
                            <span className="font-medium text-purple-800">Creative</span>
                          </div>
                          <p className="text-xs text-purple-600">High temperature, imaginative responses</p>
                        </div>
                        <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-blue-600">📊</span>
                            <span className="font-medium text-blue-800">Factual</span>
                          </div>
                          <p className="text-xs text-blue-600">Low temperature, precise responses</p>
                        </div>
                        <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-green-600">⚖️</span>
                            <span className="font-medium text-green-800">Balanced</span>
                          </div>
                          <p className="text-xs text-green-600">Moderate settings, versatile use</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Side - Session Settings */}
            <div className="w-1/2 flex flex-col">
              <div className="p-6 border-b border-border">
                <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                  <MessageSquare className="w-5 h-5 text-purple-500" />
                  Session Details
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Basic session settings and model selection
                </p>
              </div>
              
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {/* Session Name */}
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Session Name (Optional)
                  </label>
                  <input
                    type="text"
                    value={sessionName}
                    onChange={(e) => setSessionName(e.target.value)}
                    placeholder="My Chat Session"
                    className="w-full p-3 bg-background border border-input rounded-lg text-foreground focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>

                {/* Model Pair Selection */}
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Model Pair
                  </label>
                  <select
                    value={selectedPairId}
                    onChange={(e) => {
                      setSelectedPairId(e.target.value);
                      // Initialize configuration when model pair is selected
                      if (e.target.value) {
                        const selectedPair = modelPairs.find(p => p.pair_id === e.target.value);
                        if (selectedPair && !newSessionConfig) {
                          const defaultConfig = {
                            session_id: "new-session-temp",
                            base_model: {
                              system_prompt: customSystemPrompt || "",
                              context_length: 2048,
                              generation_config: {
                                temperature: 0.7, top_p: 0.9, top_k: 50, max_tokens: 384,
                                presence_penalty: 0.0, frequency_penalty: 0.0, repetition_penalty: 1.0,
                                do_sample: true, early_stopping: false, seed: null, pad_token_id: null, eos_token_id: null
                              },
                              model_id: selectedPair.base_model_name,
                              enabled: true
                            },
                            finetuned_model: {
                              system_prompt: customSystemPrompt || "",
                              context_length: 2048,
                              generation_config: {
                                temperature: 0.7, top_p: 0.9, top_k: 50, max_tokens: 384,
                                presence_penalty: 0.0, frequency_penalty: 0.0, repetition_penalty: 1.0,
                                do_sample: true, early_stopping: false, seed: null, pad_token_id: null, eos_token_id: null
                              },
                              model_id: selectedPair.adapter_name,
                              enabled: true
                            },
                            sync_parameters: false,
                            config_name: "Default Configuration",
                            created_at: new Date().toISOString(),
                            updated_at: new Date().toISOString()
                          };
                          setNewSessionConfig(defaultConfig);
                        }
                      } else {
                        setNewSessionConfig(null);
                      }
                    }}
                    className="w-full p-3 bg-background border border-input rounded-lg text-foreground focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  >
                    <option value="">Select a model pair...</option>
                    {modelPairs.map((pair) => {
                      const memoryGB = pair.estimated_memory_mb
                        ? (pair.estimated_memory_mb / 1024).toFixed(1)
                        : "?.?";
                      const statusText = pair.status === "mock" ? " (Demo)" : "";

                      return (
                        <option key={pair.pair_id} value={pair.pair_id}>
                          📊 {pair.base_model_name} ↔️ {pair.adapter_name} ({memoryGB}GB){statusText}
                        </option>
                      );
                    })}
                  </select>
                </div>

                {/* Model Pair Preview */}
                {selectedPairId && (
                  <div className="p-4 bg-muted/50 rounded-lg border border-border">
                    {(() => {
                      const selectedPair = modelPairs.find(p => p.pair_id === selectedPairId);
                      return selectedPair ? (
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <h4 className="font-medium text-foreground">Selected Model Pair</h4>
                            {selectedPair.status === "mock" && (
                              <span className="px-2 py-1 bg-orange-100 text-orange-700 text-xs rounded-full">
                                🎭 Demo Mode
                              </span>
                            )}
                          </div>
                          
                          <div className="grid grid-cols-2 gap-4">
                            <div className="p-3 bg-blue-50 rounded-lg">
                              <div className="flex items-center gap-2 mb-2">
                                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                                <span className="font-medium text-blue-700">Base Model</span>
                              </div>
                              <p className="text-sm text-blue-600 mb-1">{selectedPair.base_model_name}</p>
                              <p className="text-xs text-blue-500">
                                Size: {selectedPair.base_model_size_mb.toFixed(0)}MB
                              </p>
                            </div>
                            
                            <div className="p-3 bg-green-50 rounded-lg">
                              <div className="flex items-center gap-2 mb-2">
                                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                                <span className="font-medium text-green-700">Fine-tuned</span>
                              </div>
                              <p className="text-sm text-green-600 mb-1">{selectedPair.adapter_name}</p>
                              <p className="text-xs text-green-500">
                                Size: {selectedPair.adapter_size_mb.toFixed(0)}MB
                              </p>
                            </div>
                          </div>
                          
                          {selectedPair.status === "mock" && (
                            <div className="p-3 bg-orange-50 rounded-lg border border-orange-200">
                              <p className="text-xs text-orange-700">
                                🎭 <strong>Demo Mode:</strong> This will use intelligent mock responses to demonstrate the comparison interface
                              </p>
                            </div>
                          )}
                        </div>
                      ) : null;
                    })()}
                  </div>
                )}

                {/* Agent Orchestrator Settings */}
                <div className="p-4 bg-muted/30 rounded-lg border border-border">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Settings className="w-4 h-4 text-muted-foreground" />
                      <span className="font-medium text-foreground">Agent Orchestrator</span>
                    </div>
                    <button
                      onClick={() => setUseAgentOrchestrator(!useAgentOrchestrator)}
                      className={`px-3 py-1 rounded-full text-xs transition-colors ${
                        useAgentOrchestrator 
                          ? 'bg-yellow-100 text-yellow-700 border border-yellow-200' 
                          : 'bg-gray-100 text-gray-600 border border-gray-200'
                      }`}
                    >
                      {useAgentOrchestrator ? '🤖 ON' : '🤖 OFF'}
                    </button>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">
                    {useAgentOrchestrator 
                      ? 'Specialist agents will be automatically selected based on your queries'
                      : 'Use standard model responses without agent selection'
                    }
                  </p>
                  {useAgentOrchestrator && availableAgents.length > 0 && (
                    <div className="text-xs text-yellow-700 bg-yellow-50 p-2 rounded border border-yellow-200">
                      {availableAgents.length} specialist agents available for automatic selection
                    </div>
                  )}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="p-6 border-t border-border">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    {newSessionConfig && (
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span>Configuration Ready</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex gap-2">
                    <button
                      onClick={() => {
                        setShowNewSessionDialog(false);
                        setSessionName("");
                        setSelectedPairId("");
                        setNewSessionConfig(null);
                      }}
                      className="px-4 py-2 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={createSession}
                      disabled={!selectedPairId}
                      className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
                    >
                      <MessageSquare className="w-4 h-4" />
                      Create Session
                    </button>
                  </div>
                </div>
              </div>
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
                  Automatically selects the best AI agent based on your query type
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
                      <div key={agent.agent_id} className="p-2 bg-muted rounded text-xs">
                        <div className="font-medium">{agent.name}</div>
                        <div className="text-muted-foreground">
                          {agent.capabilities?.join(", ") || "General assistance"}
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
                      prompt: "You are a helpful, knowledgeable assistant that provides detailed and practical responses. Always be specific and helpful."
                    },
                    {
                      name: "Code Reviewer",
                      prompt: "You are an expert code reviewer with deep knowledge of software development best practices. Review code for quality, security, performance, and maintainability. Provide specific, actionable feedback."
                    },
                    {
                      name: "Creative Writer",
                      prompt: "You are a creative companion that helps with brainstorming, creative writing, and artistic projects. Think outside the box and inspire innovative ideas."
                    },
                    {
                      name: "Business Analyst", 
                      prompt: "You are a business analyst with expertise in data-driven decision making. Analyze business problems and provide strategic insights backed by data and market knowledge."
                    }
                  ].map((example) => (
                    <button
                      key={example.name}
                      onClick={() => setCustomSystemPrompt(example.prompt)}
                      className="text-left p-2 bg-muted hover:bg-muted/80 rounded text-xs w-full"
                    >
                      <div className="font-medium text-purple-400">{example.name}</div>
                      <div className="text-muted-foreground truncate">{example.prompt}</div>
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
                className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
              >
                Apply Settings
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Main component with configuration provider
export const ComparisonChatInterface: React.FC = () => {
  return (
    <ConfigurationProvider>
      <ComparisonChatInterfaceInner />
    </ConfigurationProvider>
  );
};
