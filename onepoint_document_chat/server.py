import socketio

from onepoint_document_chat.config import cfg

sio = socketio.Server(cors_allowed_origins=cfg.websocket_cors_allowed_origins)

app = socketio.WSGIApp(sio)