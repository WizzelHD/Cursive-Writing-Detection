"use client"

import { X, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ImagePreviewProps {
  imageData: string
  onClear: () => void
  onRetry: () => void
  hasResults: boolean
}

export function ImagePreview({ imageData, onClear, onRetry, hasResults }: ImagePreviewProps) {
  return (
    <div className="relative overflow-hidden rounded-xl border border-border bg-card">
      <div className="relative">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageData}
          alt="Aufgenommenes oder hochgeladenes Bild"
          className="w-full object-contain"
          style={{ maxHeight: "40vh" }}
        />
        <div className="absolute right-2 top-2 flex gap-2">
          {hasResults && (
            <Button
              variant="secondary"
              size="icon"
              onClick={onRetry}
              className="h-8 w-8 rounded-full bg-card/80 backdrop-blur-sm hover:bg-card"
              aria-label="Erneut analysieren"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          )}
          <Button
            variant="secondary"
            size="icon"
            onClick={onClear}
            className="h-8 w-8 rounded-full bg-card/80 backdrop-blur-sm hover:bg-card"
            aria-label="Bild entfernen"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
