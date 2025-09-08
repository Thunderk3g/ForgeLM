"use client"

import type React from "react"
import { CheckCircle, Clock, AlertCircle, Play } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"

interface StatusCardProps {
  title: string
  status: "pending" | "running" | "completed" | "error"
  message?: string
  details?: Record<string, any>
}

const statusConfig = {
  pending: {
    icon: Clock,
    color: "text-muted-foreground",
    bgColor: "bg-muted/50",
    dotColor: "bg-muted-foreground",
  },
  running: {
    icon: Play,
    color: "text-blue-500",
    bgColor: "bg-blue-50/10",
    dotColor: "bg-blue-500",
  },
  completed: {
    icon: CheckCircle,
    color: "text-primary",
    bgColor: "bg-primary/10",
    dotColor: "bg-primary",
  },
  error: {
    icon: AlertCircle,
    color: "text-destructive",
    bgColor: "bg-destructive/10",
    dotColor: "bg-destructive",
  },
}

export const StatusCard: React.FC<StatusCardProps> = ({ title, status, message, details }) => {
  const config = statusConfig[status] || statusConfig.pending
  const Icon = config?.icon || Clock

  return (
    <Card className="transition-colors">
      <CardContent className="p-4">
        <div className="flex items-start space-x-3">
          <div className={`rounded-md p-2 ${config.bgColor}`}>
            <Icon className={`h-4 w-4 ${config.color}`} />
          </div>
          <div className="flex-1 space-y-2">
            <div className="flex items-center space-x-2">
              <h4 className="text-sm font-semibold">{title}</h4>
              <div className={`h-2 w-2 rounded-full ${config.dotColor}`} />
            </div>
            {message && <p className="text-sm text-muted-foreground">{message}</p>}
            {details && Object.keys(details).length > 0 && (
              <div className="space-y-1">
                {Object.entries(details).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground capitalize">{key.replace(/_/g, " ")}</span>
                    <span className="font-mono font-medium">{String(value)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
