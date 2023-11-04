import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from onepoint_document_chat.config import cfg

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory=cfg.webserver_files), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "onepoint_document_chat.server.http_server:app",
        host=cfg.webserver_server,
        port=cfg.webserver_port,
        reload=True,
    )
