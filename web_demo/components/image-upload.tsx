"use client"

import { useRef, useCallback, useState } from "react"
import { Upload, ImageIcon } from "lucide-react"

interface ImageUploadProps {
  onUpload: (imageData: string) => void
}

export function ImageUpload({ onUpload }: ImageUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return

      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result as string
        if (result) {
          onUpload(result)
        }
      }
      reader.readAsDataURL(file)
    },
    [onUpload]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={() => fileInputRef.current?.click()}
      className={`group cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-all ${
        isDragging
          ? "border-primary bg-primary/10"
          : "border-border hover:border-primary/50 hover:bg-secondary/50"
      }`}
      role="button"
      tabIndex={0}
      aria-label="Bild hochladen"
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          fileInputRef.current?.click()
        }
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileInput}
        className="hidden"
        aria-hidden="true"
      />
      <div className="flex flex-col items-center gap-3">
        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-secondary transition-colors group-hover:bg-primary/20">
          {isDragging ? (
            <ImageIcon className="h-7 w-7 text-primary" />
          ) : (
            <Upload className="h-7 w-7 text-muted-foreground transition-colors group-hover:text-primary" />
          )}
        </div>
        <div>
          <p className="text-sm font-medium text-foreground">
            Bild hierher ziehen oder klicken
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            JPG, PNG, WebP
          </p>
        </div>
      </div>
    </div>
  )
}
