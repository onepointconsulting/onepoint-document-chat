import socketio

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
    name = request.query.get("name", "Anonymous")
    text = "Hello, " + name
    return web.Response(text=text)


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


if __name__ == "__main__":
    app.add_routes(routes)
    app.router.add_static('/files/', path=cfg.data_folder.as_posix(), name='files')
    web.run_app(app, host=cfg.websocket_server, port=cfg.websocket_port)
