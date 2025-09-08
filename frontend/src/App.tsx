import { useState, useEffect } from 'react'
import { ThemeProvider } from './components/theme-provider'

// Import the new page components
import HomePage from './components/HomePage'
import WorkflowPage from './components/workflow-page'
import DocsPage from './components/DocsPage'
import LLMForgeApp from './components/llmforge-app'
import ComparisonChatInterface from './components/comparison-chat-interface'
import { VisualizationDashboard } from './components/VisualizationDashboard'

// API
import { healthCheck } from './lib/api'

type PageType = "home" | "training" | "workflow" | "docs" | "chat" | "analytics"

function App() {
  const [currentPage, setCurrentPage] = useState<PageType>("home")
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking')

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck()
        setApiStatus('online')
      } catch (error) {
        setApiStatus('offline')
      }
    }
    checkHealth()
  }, [])

  const handleNavigation = (page: PageType) => {
    setCurrentPage(page)
  }

  const renderCurrentPage = () => {
    switch (currentPage) {
      case "home":
        return <HomePage onNavigate={handleNavigation} apiStatus={apiStatus} />
      case "training":
        return <LLMForgeApp onNavigate={handleNavigation} />
      case "workflow":
        return <WorkflowPage onNavigate={handleNavigation} />
      case "docs":
        return <DocsPage onNavigate={handleNavigation} />
      case "chat":
        return <ComparisonChatInterface />
      case "analytics":
        return (
          <main className="container mx-auto max-w-7xl p-4 md:p-6 lg:p-8">
            <VisualizationDashboard 
              currentTaskId={null}
              selectedDatasetPath={null}
              onNavigateToTraining={() => setCurrentPage('training')}
            />
          </main>
        )
      default:
        return <HomePage onNavigate={handleNavigation} apiStatus={apiStatus} />
    }
  }

  return (
    <ThemeProvider defaultTheme="dark" storageKey="llmforge-ui-theme">
      {renderCurrentPage()}
    </ThemeProvider>
  )
}

export default App