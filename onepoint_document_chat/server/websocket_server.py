import socketio
from pathlib import Path

from asyncer import asyncify
from aiohttp import web

from onepoint_document_chat.config import cfg
from onepoint_document_chat.log_init import logger
from onepoint_document_chat.server.session import add_message, delete_session
from onepoint_document_chat.service.qa_service import answer_question, ResponseText


sio = socketio.AsyncServer(cors_allowed_origins=cfg.websocket_cors_allowed_origins)
app = web.Application()
sio.attach(app)


routes = web.RouteTableDef()


@routes.get("/")
async def get_handler(request):
    raise web.HTTPFound("/index.html")


@sio.event
def connect(sid, environ):
    logger.info("connect %s ", sid)


@sio.event
async def question(sid, data):
    logger.info("question %s: %s", sid, data)
    session = add_message(sid, data)
    res: ResponseText = await asyncify(answer_question)(
        data, session.messages_history_str()
    )
    logger.info("response: %s", res.response)
    await sio.emit("response", res.json(), room=sid)


@sio.event
def disconnect(sid):
    logger.info("disconnect %s", sid)
    delete_session(sid)


@routes.post("/upload")
async def upload_file(request):
    """
    Used to upload a single file. Expects the request to have a parameteer "file"
    """
    data = await request.post()
    file = data.get("file")
    if file is None:
        return web.Response(text="""{"status": "error", "description": "Parameter 'file' missing"}""", content_type="application/json")
    file_name = file.filename
    target_file: Path = cfg.webserver_upload_folder/file_name
    content = file.file.read()
    target_file.write_bytes(content)
    return web.Response(text='{"status": "ok"}', content_type="application/json")


if __name__ == "__main__":
    app.add_routes(routes)
    app.router.add_static("/files/", path=cfg.data_folder.as_posix(), name="files")
    app.router.add_static("/", path=cfg.ui_folder.as_posix(), name="ui")
    web.run_app(app, host=cfg.websocket_server, port=cfg.websocket_port)
