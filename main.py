import os
import asyncio
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .qkd_ibm import QKDRunner, QKDConfig


app = FastAPI(title="QKD Backend", version="1.0.0")

# CORS: allow local dev and static hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    n_trials: int = 50
    with_eve: bool = False
    shots_per_job: int = 1
    instance: Optional[str] = None
    token: Optional[str] = None


runner: Optional[QKDRunner] = None


@app.on_event("startup")
async def on_startup() -> None:
    global runner
    # IBM auth via env; the service will look it up. Keep lazy init in runner
    runner = QKDRunner()


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/qkd/start")
async def start_qkd(req: StartRequest):
    assert runner is not None
    # Kick off a run; results will be streamed on websocket
    await runner.start(config=QKDConfig(
        n_trials=req.n_trials,
        shots_per_job=req.shots_per_job,
        with_eve=req.with_eve,
        instance=req.instance,
        token=req.token,
    ))
    return {"status": "started"}


@app.websocket("/qkd/stream")
async def qkd_stream(ws: WebSocket):
    assert runner is not None
    await ws.accept()
    try:
        # Immediately notify client that the stream is open
        await ws.send_json({"type": "status", "message": "Stream connected"})

        # Heartbeat task to keep some intermediaries from closing idle WS
        async def _heartbeat():
            try:
                while True:
                    await asyncio.sleep(15)
                    await ws.send_json({"type": "heartbeat", "ts": asyncio.get_event_loop().time()})
            except Exception:
                pass

        hb_task = asyncio.create_task(_heartbeat())
        async for event in runner.events():
            await ws.send_json(event)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
    finally:
        try:
            hb_task.cancel()  # type: ignore[name-defined]
        except Exception:
            pass
        await ws.close()


# Local dev: uvicorn amaravati.backend.main:app --reload --port 8000

