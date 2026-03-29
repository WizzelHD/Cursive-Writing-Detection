import { NextRequest, NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const contentType = request.headers.get("content-type") || ""

    // Check if Python backend is running
    let backendAvailable = false
    try {
      const health = await fetch(`${BACKEND_URL}/health`, {
        signal: AbortSignal.timeout(2000),
      })
      backendAvailable = health.ok
    } catch {
      backendAvailable = false
    }

    if (!backendAvailable) {
      return NextResponse.json(
        { error: "Python Backend nicht erreichbar." },
        { status: 503 }
      )
    }

    // Handle base64 (camera) — convert to file upload for the backend
    if (contentType.includes("application/json")) {
      const body = await request.json()
      let imageData: string = body.image || ""
      if (imageData.includes(",")) {
        imageData = imageData.split(",")[1]
      }
      const buffer = Buffer.from(imageData, "base64")
      const blob = new Blob([buffer], { type: "image/jpeg" })
      const formData = new FormData()
      formData.append("file", blob, "image.jpg")

      const response = await fetch(`${BACKEND_URL}/debug/preprocess`, {
        method: "POST",
        body: formData,
      })
      const data = await response.json()
      return NextResponse.json(data, { status: response.status })
    }

    // Forward multipart form data
    const formData = await request.formData()
    const file = formData.get("file") as File | null

    if (!file) {
      return NextResponse.json({ error: "Kein Bild gesendet" }, { status: 400 })
    }

    const backendForm = new FormData()
    backendForm.append("file", file)

    const response = await fetch(`${BACKEND_URL}/debug/preprocess`, {
      method: "POST",
      body: backendForm,
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error("Debug preprocess API error:", error)
    return NextResponse.json(
      { error: "Interner Serverfehler" },
      { status: 500 }
    )
  }
}
