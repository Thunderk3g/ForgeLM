"use client"

import type React from "react"
import { useState } from "react"
import { Database, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Checkbox } from "@/components/ui/checkbox"
import type { PrepareDataRequest } from "../lib/types"

interface DataPreparationProps {
  onPrepareData: (request: PrepareDataRequest) => void
  isLoading: boolean
}

export const DataPreparation: React.FC<DataPreparationProps> = ({ onPrepareData, isLoading }) => {
  const [inputFile, setInputFile] = useState("/app/data/input/dataset.csv")
  const [task, setTask] = useState<"unsupervised" | "supervised" | "instruction">("instruction")
  const [augment, setAugment] = useState(false)
  const [balance, setBalance] = useState(false)
  const [testSize, setTestSize] = useState(0.2)
  const [maxLength, setMaxLength] = useState(512)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onPrepareData({
      input_file: inputFile,
      task,
      augment,
      balance,
      test_size: testSize,
      max_length: maxLength,
    })
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center space-x-3">
          <div className="rounded-md bg-secondary/10 p-2">
            <Database className="h-5 w-5 text-secondary" />
          </div>
          <div>
            <CardTitle className="text-lg">Prepare Training Data</CardTitle>
            <CardDescription>Configure data preprocessing and validation settings</CardDescription>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="input-file">Input File Path</Label>
            <Input
              id="input-file"
              type="text"
              value={inputFile}
              onChange={(e) => setInputFile(e.target.value)}
              placeholder="/app/data/input/dataset.csv"
            />
          </div>

          <div className="space-y-3">
            <Label>Task Type</Label>
            <RadioGroup value={task} onValueChange={(value) => setTask(value as typeof task)}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="instruction" id="instruction" />
                <Label htmlFor="instruction" className="cursor-pointer">
                  <div>
                    <p className="font-medium">Instruction Learning</p>
                    <p className="text-sm text-muted-foreground">Train on instruction-response pairs</p>
                  </div>
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="supervised" id="supervised" />
                <Label htmlFor="supervised" className="cursor-pointer">
                  <div>
                    <p className="font-medium">Supervised Learning</p>
                    <p className="text-sm text-muted-foreground">Standard supervised fine-tuning</p>
                  </div>
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="unsupervised" id="unsupervised" />
                <Label htmlFor="unsupervised" className="cursor-pointer">
                  <div>
                    <p className="font-medium">Unsupervised Learning</p>
                    <p className="text-sm text-muted-foreground">Train on unlabeled text data</p>
                  </div>
                </Label>
              </div>
            </RadioGroup>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="test-size">Test Size</Label>
              <Input
                id="test-size"
                type="number"
                value={testSize}
                onChange={(e) => setTestSize(Number.parseFloat(e.target.value))}
                min="0.1"
                max="0.5"
                step="0.05"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="max-length">Max Length</Label>
              <Input
                id="max-length"
                type="number"
                value={maxLength}
                onChange={(e) => setMaxLength(Number.parseInt(e.target.value))}
                min="128"
                max="4096"
                step="128"
              />
            </div>
          </div>

          <div className="flex space-x-6">
            <div className="flex items-center space-x-2">
              <Checkbox id="augment" checked={augment} onCheckedChange={setAugment} />
              <Label htmlFor="augment" className="cursor-pointer">
                Data Augmentation
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox id="balance" checked={balance} onCheckedChange={setBalance} />
              <Label htmlFor="balance" className="cursor-pointer">
                Balance Classes
              </Label>
            </div>
          </div>

          <Button type="submit" disabled={isLoading} className="w-full">
            <Upload className="mr-2 h-4 w-4" />
            {isLoading ? "Preparing Data..." : "Prepare Data"}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
