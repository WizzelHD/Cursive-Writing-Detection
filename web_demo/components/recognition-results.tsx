"use client"

import { Brain, Type, Loader2 } from "lucide-react"

interface RecognitionResultsProps {
  crnnResult: string | null
  resnetResult: string | null
  crnnConfidence: number | null
  resnetConfidence: number | null
  isLoading: boolean
}

function ConfidenceBar({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100)
  return (
    <div className="mt-2 flex items-center gap-3">
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full rounded-full bg-primary transition-all duration-700 ease-out"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="min-w-[3rem] text-right font-mono text-xs text-muted-foreground">
        {percentage}%
      </span>
    </div>
  )
}

export function RecognitionResults({
  crnnResult,
  resnetResult,
  crnnConfidence,
  resnetConfidence,
  isLoading,
}: RecognitionResultsProps) {
  if (isLoading) {
    return (
      <div className="flex flex-col items-center gap-4 rounded-xl border border-border bg-card p-8">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <div className="text-center">
          <p className="text-sm font-medium text-foreground">Analyse läuft...</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Bilder werden mit CRNN und ResNet verarbeitet
          </p>
        </div>
      </div>
    )
  }

  if (!crnnResult && !resnetResult) return null

  return (
    <div className="flex flex-col gap-4">
      {/* CRNN Result */}
      {crnnResult !== null && (
        <div className="overflow-hidden rounded-xl border border-border bg-card">
          <div className="flex items-center gap-2 border-b border-border bg-secondary/50 px-4 py-3">
            <Type className="h-4 w-4 text-primary" />
            <span className="text-xs font-semibold uppercase tracking-wider text-primary">
              CRNN — Texterkennung
            </span>
          </div>
          <div className="p-4">
            <p className="font-mono text-lg text-foreground">
              {crnnResult || "Kein Text erkannt"}
            </p>
            {crnnConfidence !== null && (
              <ConfidenceBar confidence={crnnConfidence} />
            )}
          </div>
        </div>
      )}

      {/* ResNet Result */}
      {resnetResult !== null && (
        <div className="overflow-hidden rounded-xl border border-border bg-card">
          <div className="flex items-center gap-2 border-b border-border bg-secondary/50 px-4 py-3">
            <Brain className="h-4 w-4 text-primary" />
            <span className="text-xs font-semibold uppercase tracking-wider text-primary">
              ResNet — Klassifikation
            </span>
          </div>
          <div className="p-4">
            <p className="font-mono text-lg text-foreground">
              {resnetResult || "Nicht erkannt"}
            </p>
            {resnetConfidence !== null && (
              <ConfidenceBar confidence={resnetConfidence} />
            )}
          </div>
        </div>
      )}
    </div>
  )
}
