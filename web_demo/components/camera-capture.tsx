"use client"

import { useRef, useState, useCallback, useEffect } from "react"
import { Camera, SwitchCamera, X } from "lucide-react"
import { Button } from "@/components/ui/button"

interface CameraCaptureProps {
  onCapture: (imageData: string) => void
  onClose: () => void
}

export function CameraCapture({ onCapture, onClose }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [facingMode, setFacingMode] = useState<"user" | "environment">("environment")
  const [error, setError] = useState<string | null>(null)

  const startCamera = useCallback(async (facing: "user" | "environment") => {
    try {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: facing,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      })
      setStream(mediaStream)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
      setError(null)
    } catch {
      setError("Kamerazugriff verweigert. Bitte erlaube den Kamerazugriff in deinen Browsereinstellungen.")
    }
  }, [stream])

  useEffect(() => {
    startCamera(facingMode)
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const switchCamera = useCallback(() => {
    const newFacing = facingMode === "user" ? "environment" : "user"
    setFacingMode(newFacing)
    startCamera(newFacing)
  }, [facingMode, startCamera])

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctx.drawImage(video, 0, 0)
    const imageData = canvas.toDataURL("image/jpeg", 0.9)

    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
    }

    onCapture(imageData)
  }, [stream, onCapture])

  const handleClose = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
    }
    onClose()
  }, [stream, onClose])

  if (error) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/95 p-6">
        <div className="flex max-w-sm flex-col items-center gap-4 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-destructive/20">
            <Camera className="h-8 w-8 text-destructive" />
          </div>
          <p className="text-foreground">{error}</p>
          <Button onClick={handleClose} variant="outline">
            Schliessen
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-background">
      <div className="relative flex-1">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full object-cover"
        />
        <canvas ref={canvasRef} className="hidden" />
      </div>

      <div className="flex items-center justify-center gap-8 bg-card/90 px-6 py-6 backdrop-blur-sm">
        <Button
          variant="ghost"
          size="icon"
          onClick={handleClose}
          className="h-12 w-12 rounded-full text-foreground hover:bg-secondary"
          aria-label="Kamera schliessen"
        >
          <X className="h-6 w-6" />
        </Button>

        <button
          onClick={capturePhoto}
          className="flex h-18 w-18 items-center justify-center rounded-full border-4 border-primary bg-primary/20 transition-all hover:bg-primary/40 active:scale-95"
          aria-label="Foto aufnehmen"
        >
          <div className="h-12 w-12 rounded-full bg-primary" />
        </button>

        <Button
          variant="ghost"
          size="icon"
          onClick={switchCamera}
          className="h-12 w-12 rounded-full text-foreground hover:bg-secondary"
          aria-label="Kamera wechseln"
        >
          <SwitchCamera className="h-6 w-6" />
        </Button>
      </div>
    </div>
  )
}
