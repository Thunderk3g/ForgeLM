"use client"

import type React from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface MetricData {
  epoch: number
  step: number
  loss?: number
  eval_loss?: number
  learning_rate?: number
}

interface MetricsChartProps {
  data: MetricData[]
  title?: string
}

export const MetricsChart: React.FC<MetricsChartProps> = ({ data, title = "Training Metrics" }) => {
  if (data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            No metrics data available yet...
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" label={{ value: "Training Steps", position: "insideBottom", offset: -5 }} />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="lr" orientation="right" />
              <Tooltip
                formatter={(value, name) => [
                  typeof value === "number"
                    ? name === "Learning Rate"
                      ? value.toExponential(2)
                      : value.toFixed(4)
                    : value,
                  name,
                ]}
                labelFormatter={(step) => `Step: ${step}`}
              />
              <Legend />
              {data.some((d) => d.loss !== undefined) && (
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="hsl(var(--destructive))"
                  strokeWidth={2}
                  name="Training Loss"
                  dot={false}
                  yAxisId="left"
                />
              )}
              {data.some((d) => d.eval_loss !== undefined) && (
                <Line
                  type="monotone"
                  dataKey="eval_loss"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  name="Validation Loss"
                  dot={false}
                  yAxisId="left"
                />
              )}
              {data.some((d) => d.learning_rate !== undefined) && (
                <Line
                  type="monotone"
                  dataKey="learning_rate"
                  stroke="hsl(var(--secondary))"
                  strokeWidth={2}
                  name="Learning Rate"
                  dot={false}
                  yAxisId="lr"
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
