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
        {
          error: "Python Backend nicht erreichbar. Starte es mit: python backend/server.py",
          hint: "Stelle sicher, dass der Python-Server auf Port 8000 laeuft.",
        },
        { status: 503 }
      )
    }

    // Handle base64 (camera capture) vs multipart (file upload)
    if (contentType.includes("application/json")) {
      const body = await request.json()
      const response = await fetch(`${BACKEND_URL}/analyze/base64`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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

    const response = await fetch(`${BACKEND_URL}/analyze`, {
      method: "POST",
      body: backendForm,
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error("API route error:", error)
    return NextResponse.json(
      { error: "Interner Serverfehler" },
      { status: 500 }
    )
  }
}
