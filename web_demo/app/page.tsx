"use client"

import { useState, useCallback, useEffect } from "react"
import { Camera, Upload, Zap, Cpu, Scan, Server, ServerOff, Bug } from "lucide-react"
import { Button } from "@/components/ui/button"
import { CameraCapture } from "@/components/camera-capture"
import { ImageUpload } from "@/components/image-upload"
import { ImagePreview } from "@/components/image-preview"
import { RecognitionResults } from "@/components/recognition-results"

type BackendStatus = "checking" | "online" | "offline"

export default function Home() {
  const [showCamera, setShowCamera] = useState(false)
  const [imageData, setImageData] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [crnnResult, setCrnnResult] = useState<string | null>(null)
  const [resnetResult, setResnetResult] = useState<string | null>(null)
  const [crnnConfidence, setCrnnConfidence] = useState<number | null>(null)
  const [resnetConfidence, setResnetConfidence] = useState<number | null>(null)
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking")
  const [backendInfo, setBackendInfo] = useState<{
    resnet_loaded: boolean
    crnn_loaded: boolean
    device: string
  } | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Check backend health on mount and periodically
  const checkBackend = useCallback(async () => {
    try {
      const res = await fetch("/api/health")
      if (res.ok) {
        const data = await res.json()
        setBackendStatus("online")
        setBackendInfo(data)
      } else {
        setBackendStatus("offline")
        setBackendInfo(null)
      }
    } catch {
      setBackendStatus("offline")
      setBackendInfo(null)
    }
  }, [])

  useEffect(() => {
    checkBackend()
    const interval = setInterval(checkBackend, 10000)
    return () => clearInterval(interval)
  }, [checkBackend])

  const analyzeImage = useCallback(
    async (image: string, isBase64: boolean) => {
      setIsAnalyzing(true)
      setCrnnResult(null)
      setResnetResult(null)
      setCrnnConfidence(null)
      setResnetConfidence(null)
      setErrorMsg(null)

      try {
        let res: Response

        if (isBase64) {
          // Camera capture sends base64
          res = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image }),
          })
        } else {
          // File upload sends the data URL, convert to blob
          const blob = await fetch(image).then((r) => r.blob())
          const formData = new FormData()
          formData.append("file", blob, "image.jpg")
          res = await fetch("/api/analyze", {
            method: "POST",
            body: formData,
          })
        }

        const data = await res.json()

        if (!res.ok) {
          setErrorMsg(data.error || "Analyse fehlgeschlagen")
          return
        }

        if (data.crnn) {
          setCrnnResult(data.crnn.text)
          setCrnnConfidence(data.crnn.confidence)
        }
        if (data.resnet) {
          setResnetResult(data.resnet.label)
          setResnetConfidence(data.resnet.confidence)
        }
      } catch (error) {
        console.error("Analysis failed:", error)
        setErrorMsg("Verbindung zum Backend fehlgeschlagen. Laeuft der Python-Server?")
      } finally {
        setIsAnalyzing(false)
      }
    },
    []
  )

  const handleCapture = useCallback(
    (data: string) => {
      setImageData(data)
      setShowCamera(false)
      analyzeImage(data, true)
    },
    [analyzeImage]
  )

  const handleUpload = useCallback(
    (data: string) => {
      setImageData(data)
      analyzeImage(data, false)
    },
    [analyzeImage]
  )

  const handleClear = useCallback(() => {
    setImageData(null)
    setCrnnResult(null)
    setResnetResult(null)
    setCrnnConfidence(null)
    setResnetConfidence(null)
    setErrorMsg(null)
  }, [])

  const handleRetry = useCallback(() => {
    if (imageData) {
      analyzeImage(imageData, imageData.startsWith("data:"))
    }
  }, [imageData, analyzeImage])

  return (
    <>
      {showCamera && (
        <CameraCapture
          onCapture={handleCapture}
          onClose={() => setShowCamera(false)}
        />
      )}

      <main className="min-h-screen bg-background">
        {/* Header */}
        <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
          <div className="mx-auto flex max-w-2xl items-center justify-between px-4 py-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15">
                <Scan className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h1 className="text-base font-semibold text-foreground">
                  Vision AI
                </h1>
                <p className="text-xs text-muted-foreground">
                  ResNet + CRNN Bilderkennung
                </p>
              </div>
            </div>
            <button
              onClick={checkBackend}
              className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-medium transition-colors ${
                backendStatus === "online"
                  ? "bg-primary/10 text-primary"
                  : backendStatus === "checking"
                  ? "bg-muted text-muted-foreground"
                  : "bg-destructive/10 text-destructive"
              }`}
              title="Klicken zum Aktualisieren"
            >
              {backendStatus === "online" ? (
                <Server className="h-3 w-3" />
              ) : (
                <ServerOff className="h-3 w-3" />
              )}
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  backendStatus === "online"
                    ? "bg-primary"
                    : backendStatus === "checking"
                    ? "bg-muted-foreground animate-pulse"
                    : "bg-destructive"
                }`}
              />
              {backendStatus === "online"
                ? "Backend Online"
                : backendStatus === "checking"
                ? "Pruefen..."
                : "Backend Offline"}
            </button>
          </div>
        </header>

        <div className="mx-auto max-w-2xl px-4 py-6">
          {/* Backend Status Banner */}
          {backendStatus === "offline" && (
            <div className="mb-6 rounded-xl border border-destructive/30 bg-destructive/5 p-4">
              <div className="flex items-start gap-3">
                <ServerOff className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
                <div>
                  <p className="text-sm font-medium text-foreground">
                    Python Backend nicht erreichbar
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
                    Starte den Server mit:{" "}
                    <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[11px] text-foreground">
                      python backend/server.py
                    </code>
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
                    Lege deine Modelle ab in:{" "}
                    <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[11px] text-foreground">
                      backend/models/resnet.pth
                    </code>
                    {" und "}
                    <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[11px] text-foreground">
                      backend/models/crnn.pth
                    </code>
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Model Status Cards */}
          <div className="mb-6 grid grid-cols-2 gap-3">
            <div className="flex items-center gap-3 rounded-xl border border-border bg-card p-3">
              <div
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
                  backendInfo?.crnn_loaded ? "bg-primary/15" : "bg-muted"
                }`}
              >
                <Cpu
                  className={`h-4 w-4 ${
                    backendInfo?.crnn_loaded
                      ? "text-primary"
                      : "text-muted-foreground"
                  }`}
                />
              </div>
              <div className="min-w-0">
                <p className="text-xs font-semibold text-foreground">CRNN</p>
                <p className="truncate text-[10px] text-muted-foreground">
                  {backendInfo?.crnn_loaded ? "Geladen" : "Nicht geladen"}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3 rounded-xl border border-border bg-card p-3">
              <div
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
                  backendInfo?.resnet_loaded ? "bg-primary/15" : "bg-muted"
                }`}
              >
                <Zap
                  className={`h-4 w-4 ${
                    backendInfo?.resnet_loaded
                      ? "text-primary"
                      : "text-muted-foreground"
                  }`}
                />
              </div>
              <div className="min-w-0">
                <p className="text-xs font-semibold text-foreground">ResNet</p>
                <p className="truncate text-[10px] text-muted-foreground">
                  {backendInfo?.resnet_loaded ? "Geladen" : "Nicht geladen"}
                </p>
              </div>
            </div>
          </div>

          {/* Device Info */}
          {backendInfo && (
            <div className="mb-6 flex items-center justify-center">
              <span className="rounded-full bg-secondary px-3 py-1 font-mono text-[10px] text-muted-foreground">
                Device: {backendInfo.device}
              </span>
            </div>
          )}

          {/* Image Section */}
          {imageData ? (
            <div className="mb-6">
              <ImagePreview
                imageData={imageData}
                onClear={handleClear}
                onRetry={handleRetry}
                hasResults={crnnResult !== null || resnetResult !== null}
              />
            </div>
          ) : (
            <div className="mb-6 flex flex-col gap-4">
              {/* Camera Button */}
              <Button
                onClick={() => setShowCamera(true)}
                disabled={backendStatus !== "online"}
                className="flex h-14 w-full items-center justify-center gap-3 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                <Camera className="h-5 w-5" />
                <span className="font-medium">
                  {backendStatus === "online"
                    ? "Foto aufnehmen"
                    : "Backend starten um zu beginnen"}
                </span>
              </Button>

              {/* Divider */}
              <div className="flex items-center gap-4">
                <div className="h-px flex-1 bg-border" />
                <span className="text-xs text-muted-foreground">oder</span>
                <div className="h-px flex-1 bg-border" />
              </div>

              {/* Upload Area */}
              <div
                className={
                  backendStatus === "online"
                    ? ""
                    : "pointer-events-none opacity-50"
                }
              >
                <ImageUpload onUpload={handleUpload} />
              </div>
            </div>
          )}

          {/* Error Message */}
          {errorMsg && (
            <div className="mb-4 rounded-xl border border-destructive/30 bg-destructive/5 p-4">
              <p className="text-sm text-destructive">{errorMsg}</p>
            </div>
          )}

          {/* Results */}
          <RecognitionResults
            crnnResult={crnnResult}
            resnetResult={resnetResult}
            crnnConfidence={crnnConfidence}
            resnetConfidence={resnetConfidence}
            isLoading={isAnalyzing}
          />

          {/* New Image Button */}
          {imageData && !isAnalyzing && (crnnResult || resnetResult) && (
            <div className="mt-4">
              <Button
                onClick={handleClear}
                variant="outline"
                className="w-full gap-2 rounded-xl border-border"
              >
                <Upload className="h-4 w-4" />
                Neues Bild analysieren
              </Button>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="mt-8 border-t border-border py-6">
          <div className="mx-auto max-w-2xl px-4">
            <p className="text-center text-xs text-muted-foreground">
              Modelle: CRNN (Texterkennung) & ResNet (Klassifikation)
              <br />
              <span className="font-mono text-[10px]">
                Next.js Frontend + Python/PyTorch Backend
              </span>
            </p>
          </div>
        </footer>
      </main>
    </>
  )
}
