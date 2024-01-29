import asyncio
import uuid
import lsprotocol.types as types



class LspClient:
    async def start(self, cmd, args):
        self.server = await asyncio.create_subprocess_exec(
            cmd,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def send_request(method, params=None, callback=None, msg_id=None):
        if msg_id is None:
            msg_id = str(uuid.uuid4())

        request_type = self.get_message_type(method) or JsonRPCRequestMessage
        logger.debug('Sending request with id "%s": %s %s', msg_id, method, params)

        request = request_type(
            id=msg_id,
            method=method,
            params=params,
            jsonrpc=JsonRPCProtocol.VERSION,
        )