"""ContextWeave — Personal Long-Term Memory & Context Engine."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from contextweave.api.rate_limit import limiter
from contextweave.api.routes import router
from contextweave.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        # *.supabase.co: the browser talks to Supabase Auth directly on sign-in
        "connect-src 'self' https://*.supabase.co; "
        "object-src 'none'; "
        "base-uri 'self'; "
        # huggingface.co must be allowed to frame us or the Space page
        # (which embeds the app in an iframe) shows "refused to connect"
        "frame-ancestors 'self' https://huggingface.co"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # microphone=(self): the Listen button (Web Speech API) needs
    # same-origin mic access — an empty allowlist blocks our own feature
    "Permissions-Policy": "camera=(), microphone=(self), geolocation=()",
}

# Swagger/ReDoc load their bundles from a CDN with inline bootstrap scripts,
# so the strict CSP would blank them out — send the other headers only.
CSP_EXEMPT_PATHS = {"/docs", "/redoc"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.groq_api_key:
        logger.info("Groq API key configured — LLM reasoning and entity extraction enabled")
    else:
        logger.warning(
            "CW_GROQ_API_KEY is not set — running degraded: regex-only entity "
            "extraction and citation-list answers. Get a free key at console.groq.com"
        )
    logger.info(
        "Config: embedding=%s reasoning=%s sqlite=%s chroma=%s",
        settings.embedding_model,
        settings.reasoning_model,
        settings.sqlite_db_path,
        settings.chroma_persist_dir,
    )

    from contextweave.notify import mailer
    from contextweave.notify.scheduler import scheduler_loop

    digest_task = None
    if mailer.email_configured():
        digest_task = asyncio.create_task(scheduler_loop())
        logger.info("Pushed daily digest enabled — hourly delivery sweep scheduled")
    else:
        logger.info(
            "No email transport configured — pushed digests disabled (pull via /api/digest)"
        )

    yield

    if digest_task:
        digest_task.cancel()


app = FastAPI(
    title="ContextWeave",
    description="A personal long-term memory and context reasoning engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Public demo API without cookies or auth — wildcard origins are fine,
# but credentials must stay off with a wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        if name == "Content-Security-Policy" and request.url.path in CSP_EXEMPT_PATHS:
            continue
        response.headers[name] = value
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.port))
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=port,
        reload=True,
    )
