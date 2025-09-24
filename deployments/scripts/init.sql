-- LLMForge Database Initialization Script
-- This script sets up the initial database schema for the microservices

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create sessions table for chat comparison service
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_name VARCHAR(255),
    pair_id VARCHAR(255) NOT NULL,
    base_model_name VARCHAR(255) NOT NULL,
    adapter_name VARCHAR(255) NOT NULL,
    custom_system_prompt TEXT,
    use_agent_orchestrator BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    model_type VARCHAR(20) CHECK (model_type IN ('base', 'finetuned')),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create model pairs table
CREATE TABLE IF NOT EXISTS model_pairs (
    pair_id VARCHAR(255) PRIMARY KEY,
    base_model_name VARCHAR(255) NOT NULL,
    base_model_path VARCHAR(500) NOT NULL,
    adapter_name VARCHAR(255) NOT NULL,
    adapter_path VARCHAR(500) NOT NULL,
    base_model_size_mb DECIMAL(10,2),
    adapter_size_mb DECIMAL(10,2),
    status VARCHAR(50) DEFAULT 'available',
    load_time_ms INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create training jobs table
CREATE TABLE IF NOT EXISTS training_jobs (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) UNIQUE NOT NULL,
    base_model_dir VARCHAR(500) NOT NULL,
    dataset_dir VARCHAR(500) NOT NULL,
    output_dir VARCHAR(500) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    config JSONB,
    progress JSONB,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create datasets table for data generation service
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(255) NOT NULL,
    dataset_type VARCHAR(100) NOT NULL,
    template_name VARCHAR(255),
    generation_config JSONB,
    file_path VARCHAR(500),
    size_bytes BIGINT,
    record_count INTEGER,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create model configurations table
CREATE TABLE IF NOT EXISTS model_configurations (
    config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    config_name VARCHAR(255),
    base_model_config JSONB NOT NULL,
    finetuned_model_config JSONB NOT NULL,
    sync_parameters BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create service registry table for orchestrator
CREATE TABLE IF NOT EXISTS service_registry (
    service_id VARCHAR(255) PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    host VARCHAR(255) NOT NULL,
    port INTEGER NOT NULL,
    health_endpoint VARCHAR(255) DEFAULT '/health',
    status VARCHAR(50) DEFAULT 'unknown',
    capabilities TEXT[],
    metadata JSONB,
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create research logs table for analytics
CREATE TABLE IF NOT EXISTS research_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    log_type VARCHAR(100) NOT NULL,
    user_message TEXT,
    base_response TEXT,
    finetuned_response TEXT,
    model_info JSONB,
    generation_params JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created_at ON chat_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_training_jobs_created_at ON training_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status);
CREATE INDEX IF NOT EXISTS idx_service_registry_service_name ON service_registry(service_name);
CREATE INDEX IF NOT EXISTS idx_research_logs_session_id ON research_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_research_logs_created_at ON research_logs(created_at);

-- Insert default model pairs (examples)
INSERT INTO model_pairs (pair_id, base_model_name, base_model_path, adapter_name, adapter_path, base_model_size_mb, adapter_size_mb, status)
VALUES 
    ('llama2_7b_alpaca', 'Llama-2-7b-chat-hf', '/app/models/base/Llama-2-7b-chat-hf', 'alpaca-adapter', '/app/models/finetuned', 13000.0, 50.0, 'mock'),
    ('mistral_7b_custom', 'Mistral-7B-Instruct-v0.1', '/app/models/base/Mistral-7B-Instruct-v0.1', 'custom-adapter', '/app/models/finetuned', 14000.0, 65.0, 'mock')
ON CONFLICT (pair_id) DO NOTHING;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables
CREATE TRIGGER update_chat_sessions_updated_at BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_pairs_updated_at BEFORE UPDATE ON model_pairs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_jobs_updated_at BEFORE UPDATE ON training_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_configurations_updated_at BEFORE UPDATE ON model_configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();