import asyncio
import inspect
import json
import logging
from asyncio import Queue, CancelledError
from sanic import Sanic, Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from sanic.response import text
from typing import Text, Dict, Any, Optional, Callable, Awaitable, NoReturn
import requests
import pdb
import rasa.utils.endpoints
import http3
from rasa import utils
from rasa.core.channels.channel import (
    InputChannel,
    CollectingOutputChannel,
    UserMessage,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WabizRestInput(InputChannel):
    """A custom http input channel.

    This implementation is the basis for a custom implementation of a chat
    frontend. You can customize this to send messages to Rasa and
    retrieve responses from the assistant."""

    @classmethod
    def name(cls) -> Text:
        return "wabiz"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        # pytype: disable=attribute-error
        return cls(
            credentials.get("account_id"),
            credentials.get("auth_token"),
            credentials.get("api_host"),
        )
        # pytype: enable=attribute-error

    def __init__(
            self,
            account_id: Optional[Text],
            auth_token: Optional[Text],
            api_host: Optional[Text]
    ) -> None:
        logger.info('Inin Wabiz Channel')
        self.account_id = account_id
        self.auth_token = auth_token
        self.api_host = api_host

    @staticmethod
    async def on_message_wrapper(
            on_new_message: Callable[[UserMessage], Awaitable[Any]],
            text: Text,
            queue: Queue,
            sender_id: Text,
            input_channel: Text,
            metadata: Optional[Dict[Text, Any]],
    ) -> None:
        collector = WabizQueueOutputChannel(queue)

        message = UserMessage(
            text, collector, sender_id, input_channel=input_channel, metadata=metadata
        )
        await on_new_message(message)

        await queue.put("DONE")  # pytype: disable=bad-return-type

    async def _extract_sender(self, req: Request) -> Optional[Text]:
        return req.json.get("sender", None)

    # noinspection PyMethodMayBeStatic
    def _extract_message(self, req: Request) -> Optional[Text]:
        return req.json.get("message", None)

    def _extract_input_channel(self, req: Request) -> Text:
        return req.json.get("input_channel") or self.name()

    def set_user_phone_number(self, wa_number: Text) -> Optional[Text]:
        print("////////set_user_phone_number//////")
        data = {"event": "slot", "name": "user-phone-number", "value": wa_number, "timestamp": 0}
        wa_id = '0'
        url = self.api_host + "/conversations/" + wa_id + "/tracker/events?include_events=NONE"

        result = requests.post(url, data)

    def stream_response(
            self,
            on_new_message: Callable[[UserMessage], Awaitable[None]],
            text: Text,
            sender_id: Text,
            input_channel: Text,
            metadata: Optional[Dict[Text, Any]],
    ) -> Callable[[Any], Awaitable[None]]:
        async def stream(resp: Any) -> None:
            logging.info("========= stream response da D ====")
            logging.info("text=" + text)
            q = Queue()
            task = asyncio.ensure_future(
                self.on_message_wrapper(
                    on_new_message, text, q, sender_id, input_channel, metadata
                )
            )
            result = None  # declare variable up front to avoid pytype error
            while True:
                result = await q.get()
                if result == "DONE":
                    break
                else:
                    await resp.write(json.dumps(result))
            await task

        return stream  # pytype: disable=bad-return-type

    def blueprint(
            self, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Blueprint:

        custom_webhook = Blueprint(
            "custom_webhook_{}".format(type(self).__name__),
            inspect.getmodule(self).__name__,
        )

        # noinspection PyUnusedLocal
        @custom_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            logger.info('Wabiz get status')
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> HTTPResponse:
            """
            request.json:   {
                                'contacts': [{'wa_id': int}],
                                'messages': [{'text':
                                                        {'body': str}
                                             }],
                                'status': int,
                                'message_id': str
                            }
            """

            logger.info(json.dumps(request.json, indent=4))

            if request.json.get("contacts", None) is not None:
                wa_id = request.json.get("contacts")[0]["wa_id"]
                logger.info(f"> Reading wa_id = {wa_id}")
                if wa_id is not None:
                    data = {"event": "slot",
                            "name": "user-phone-number",
                            "value": wa_id,
                            "timestamp": 0}

                    url = self.api_host + "/conversations/" + wa_id + "/tracker/events?include_events=NONE"

                    client = http3.AsyncClient()
                    r = await client.post(url, data=json.dumps(data))
                    logger.info(" result: ")
                    logger.info(r)

            else:
                logger.error(f"NO WA_ID FOUND")
                return response.json('{ "status": :"no wa_id intercepted"}')

            # Message
            if request.json.get("messages", None) is not None:
                message_text = request.json.get("messages")[0]["text"]["body"]
                logger.info("> Reading text=" + message_text)
            else:
                logger.error("NO TEXT FOUND")
                message_text = ''

            # Status
            if request.json.get("status", None) is not None:
                status = request.json.get("status")
                logger.info("> Reading status: " + status)
            else:
                logger.info("NO STATUS FOUND")

            # Message ID
            if request.json.get("message_id", None) is not None:
                message_id = request.json.get("message_id")
                logger.info("Reading message_id: " + message_id)
            else:
                logger.info("NO MESSAGE_ID FOUND")

            sender_id = wa_id
            text = message_text
            should_use_stream = rasa.utils.endpoints.bool_arg(
                request, "stream", default=False
            )
            input_channel = self._extract_input_channel(request)
            metadata = self.get_metadata(request)
            # logging.info( "*****|||||||||  metadata: " )
            # if metadata is None:
            #    metadata = {"userphone": sender_id}
            # logging.info(metadata)
            if should_use_stream:
                return response.stream(
                    self.stream_response(
                        on_new_message, text, sender_id, input_channel, metadata
                    ),
                    content_type="text/event-stream",
                )
            else:
                logger.info("========NO STREAM but collecting ========")
                collector = CollectingOutputChannel()
                # noinspection PyBroadException
                try:
                    await on_new_message(
                        UserMessage(
                            text,
                            collector,
                            sender_id,
                            input_channel=input_channel,
                            metadata=metadata,
                        )
                    )
                except CancelledError:
                    logger.error(
                        f"Message handling timed out for " f"user message '{text}'."
                    )
                except Exception:
                    logger.exception(
                        f"An exception occured while handling "
                        f"user message '{text}'."
                    )
                logger.info("======== return collection====")
                logger.info(collector.messages)
                wabizMess = ""
                counter = 0
                for val in collector.messages:
                    counter = counter + 1
                    toadd = val["text"]
                    logger.info("================== toadd tocall wabiz=" + toadd)
                    logger.info("==== counter=" + str(counter))
                    wabizMess = wabizMess + " \n " + toadd
                if wabizMess != "":
                    wabizRequest = {"id": sender_id, "message": wabizMess, "token": "yTThH0FJ16LNEmEvnWsu"}
                    logger.info("======wabizRequest:")
                    logger.info(wabizRequest)
                    logger.info("======")
                    result = requests.post('https://waserver.makeyoudigit.com/message', data=wabizRequest)
                    logger.info("sent response to wabiz: " + result.text)
                return response.json(collector.messages)

        logger.info(f'webhook_name: {custom_webhook.name}')
        logger.info(f'webhook_host: {custom_webhook.host}')
        logger.info(f'webhook_url: {custom_webhook.url_prefix}')
        logger.info(f'webhook_routes: {custom_webhook.routes}')

        return custom_webhook


class WabizQueueOutputChannel(CollectingOutputChannel):
    """Output channel that collects send messages in a list

    (doesn't send them anywhere, just collects them)."""

    @classmethod
    def name(cls) -> Text:
        return "wabizqueue"

    # noinspection PyMissingConstructor
    def __init__(self, message_queue: Optional[Queue] = None) -> None:
        super().__init__()
        self.messages = Queue() if not message_queue else message_queue

    def latest_output(self) -> NoReturn:
        raise NotImplementedError("A queue doesn't allow to peek at messages.")

    async def _persist_message(self, message) -> None:
        await self.messages.put(message)  # pytype: disable=bad-return-type
