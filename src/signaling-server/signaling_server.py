"""
signaling_server.py
-------------------
Minimal HTTP signaling server.
Edge node POSTs its SDP offer → server stores it.
Cloud node GETs the offer, POSTs its SDP answer → server stores it.
Edge node polls GET /answer until the answer arrives.

Endpoints:
  POST /offer          { "sdp": "...", "type": "offer" }
  GET  /offer          → { "sdp": "...", "type": "offer" }
  POST /answer         { "sdp": "...", "type": "answer" }
  GET  /answer         → { "sdp": "...", "type": "answer" } or 204
  DELETE /session      reset both sides
"""

import asyncio
import json
from aiohttp import web

_offer:  dict | None = None
_answer: dict | None = None


async def post_offer(request: web.Request) -> web.Response:
    global _offer, _answer
    _offer  = await request.json()
    _answer = None                          # reset answer for new session
    print(f"[signal] offer received (type={_offer.get('type')})")
    return web.Response(status=200)


async def get_offer(request: web.Request) -> web.Response:
    if _offer is None:
        return web.Response(status=204)
    return web.json_response(_offer)


async def post_answer(request: web.Request) -> web.Response:
    global _answer
    _answer = await request.json()
    print(f"[signal] answer received (type={_answer.get('type')})")
    return web.Response(status=200)


async def get_answer(request: web.Request) -> web.Response:
    if _answer is None:
        return web.Response(status=204)
    return web.json_response(_answer)


async def delete_session(request: web.Request) -> web.Response:
    global _offer, _answer
    _offer = _answer = None
    return web.Response(status=200)


app = web.Application()
app.router.add_post("/offer",   post_offer)
app.router.add_get ("/offer",   get_offer)
app.router.add_post("/answer",  post_answer)
app.router.add_get ("/answer",  get_answer)
app.router.add_delete("/session", delete_session)

if __name__ == "__main__":
    print("[signal] listening on :8080")
    web.run_app(app, host="0.0.0.0", port=8080)