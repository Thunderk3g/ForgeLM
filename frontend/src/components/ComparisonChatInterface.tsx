import React, { useState, useEffect, useRef } from 'react';
import { Send, MessageSquare, Settings, Trash2, Download, RotateCcw } from 'lucide-react';

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

interface ComparisonResponse {
  base_response: string;
  finetuned_response: string;
  generation_time_ms: {
    base: number;
    finetuned: number;
    total: number;
  };
  model_info: {
    base_model: string;
    adapter_name: string;
    pair_id: string;
  };
  cache_metrics: {
    cache_hits: number;
    cache_misses: number;
  };
  session_id: string;
  timestamp: string;
}

export const ComparisonChatInterface: React.FC = () => {
  const [modelPairs, setModelPairs] = useState<ModelPair[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showNewSessionDialog, setShowNewSessionDialog] = useState(false);
  const [selectedPairId, setSelectedPairId] = useState<string>('');
  const [sessionName, setSessionName] = useState('');

  // Refs for auto-scrolling
  const baseMessagesRef = useRef<HTMLDivElement>(null);
  const finetunedMessagesRef = useRef<HTMLDivElement>(null);

  // Load model pairs and sessions on component mount
  useEffect(() => {
    loadModelPairs();
    loadSessions();
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (baseMessagesRef.current) {
      baseMessagesRef.current.scrollTop = baseMessagesRef.current.scrollHeight;
    }
    if (finetunedMessagesRef.current) {
      finetunedMessagesRef.current.scrollTop = finetunedMessagesRef.current.scrollHeight;
    }
  }, [messages]);

  const loadModelPairs = async () => {
    try {
      const response = await fetch('http://localhost:8000/chat/model-pairs');
      const data = await response.json();
      if (data.status === 'success') {
        setModelPairs(data.model_pairs);
      }
    } catch (error) {
      console.error('Failed to load model pairs:', error);
    }
  };

  const loadSessions = async () => {
    try {
      const response = await fetch('http://localhost:8000/chat/sessions');
      const data = await response.json();
      if (data.status === 'success') {
        setSessions(data.sessions);
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const createSession = async () => {
    if (!selectedPairId) return;

    try {
      const response = await fetch('http://localhost:8000/chat/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pair_id: selectedPairId,
          session_name: sessionName || undefined,
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        const newSession = data.session;
        setSessions([...sessions, newSession]);
        setActiveSessionId(newSession.session_id);
        setMessages([]);
        setShowNewSessionDialog(false);
        setSessionName('');
        setSelectedPairId('');
      }
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  const loadSessionHistory = async (sessionId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/chat/sessions/${sessionId}`);
      const data = await response.json();
      if (data.status === 'success') {
        setMessages(data.messages || []);
      }
    } catch (error) {
      console.error('Failed to load session history:', error);
    }
  };

  const deleteSession = async (sessionId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/chat/sessions/${sessionId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setSessions(sessions.filter(s => s.session_id !== sessionId));
        if (activeSessionId === sessionId) {
          setActiveSessionId(null);
          setMessages([]);
        }
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  };

  const sendMessage = async () => {
    if (!currentMessage.trim() || !activeSessionId || isGenerating) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString(),
    };

    // Add user message to display
    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsGenerating(true);

    try {
      const response = await fetch('http://localhost:8000/chat/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: activeSessionId,
          messages: [...messages, userMessage],
          generation_params: {
            max_new_tokens: 256,
            temperature: 0.7,
            do_sample: true,
            top_p: 0.9,
          },
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        const comparisonResponse: ComparisonResponse = data.response;

        // Add both responses
        const baseResponse: ChatMessage = {
          role: 'assistant',
          content: comparisonResponse.base_response,
          timestamp: comparisonResponse.timestamp,
          model_type: 'base',
        };

        const finetunedResponse: ChatMessage = {
          role: 'assistant',
          content: comparisonResponse.finetuned_response,
          timestamp: comparisonResponse.timestamp,
          model_type: 'finetuned',
        };

        setMessages(prev => [...prev, baseResponse, finetunedResponse]);
      }
    } catch (error) {
      console.error('Failed to generate response:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const selectSession = (sessionId: string) => {
    setActiveSessionId(sessionId);
    loadSessionHistory(sessionId);
  };

  const activeSession = sessions.find(s => s.session_id === activeSessionId);
  const baseMessages = messages.filter(m => m.role === 'user' || m.model_type === 'base');
  const finetunedMessages = messages.filter(m => m.role === 'user' || m.model_type === 'finetuned');

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <div className="w-80 bg-card border-r border-border flex flex-col">
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
              <MessageSquare className="w-5 h-5 text-purple-500" />
              Chat Sessions
            </h2>
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
                      ? 'bg-purple-500/10 border border-purple-500/20'
                      : 'bg-muted hover:bg-muted/80'
                  }`}
                  onClick={() => selectSession(session.session_id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-foreground truncate">
                        {session.session_name}
                      </h3>
                      <p className="text-sm text-muted-foreground truncate">
                        {session.model_info.base_model} + {session.model_info.adapter_name}
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
                    Comparing: {activeSession.model_info.base_model} vs {activeSession.model_info.adapter_name}
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
                    className="p-2 hover:bg-muted rounded-md"
                    title="Settings"
                  >
                    <Settings className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Split Screen Chat */}
            <div className="flex-1 flex">
              {/* Base Model Side */}
              <div className="flex-1 flex flex-col border-r border-border">
                <div className="p-3 bg-blue-500/10 border-b border-border">
                  <h3 className="font-medium text-blue-400">
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
                      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg p-3 ${
                          message.role === 'user'
                            ? 'bg-purple-600 text-white'
                            : 'bg-blue-500/10 text-foreground border border-blue-500/20'
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
                <div className="p-3 bg-green-500/10 border-b border-border">
                  <h3 className="font-medium text-green-400">
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
                      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg p-3 ${
                          message.role === 'user'
                            ? 'bg-purple-600 text-white'
                            : 'bg-green-500/10 text-foreground border border-green-500/20'
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
                <textarea
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message here..."
                  className="flex-1 resize-none p-3 bg-background border border-input rounded-lg text-foreground focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  rows={2}
                  disabled={isGenerating}
                />
                <button
                  onClick={sendMessage}
                  disabled={!currentMessage.trim() || isGenerating}
                  className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <Send className="w-4 h-4" />
                  {isGenerating ? 'Generating...' : 'Send'}
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
                Choose a session from the sidebar or create a new one to start comparing models
              </p>
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
                  {modelPairs.map((pair) => (
                    <option key={pair.pair_id} value={pair.pair_id}>
                      {pair.base_model_name} + {pair.adapter_name} 
                      ({(pair.estimated_memory_mb / 1024).toFixed(1)}GB)
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <button
                onClick={() => {
                  setShowNewSessionDialog(false);
                  setSessionName('');
                  setSelectedPairId('');
                }}
                className="px-4 py-2 text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
              <button
                onClick={createSession}
                disabled={!selectedPairId}
                className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50"
              >
                Create Session
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};