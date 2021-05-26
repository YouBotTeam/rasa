import asyncio
import inspect
import json
import logging
import time
from asyncio import Queue, CancelledError

from ruamel import yaml
from sanic import Sanic, Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from sanic.response import text
from typing import Text, Dict, Any, Optional, Callable, Awaitable, NoReturn
import requests
# import requests_async as requests
import pdb
import rasa.utils.endpoints
import http3
from rasa.core.channels.channel import (
    InputChannel,
    CollectingOutputChannel,
    UserMessage,
)
MAIN_INVOCATION_INTENT = "actions.intent.MAIN"
END_CONVERSATION_UTTER_1 = 'utter_congedo'
END_CONVERSATION_UTTER_2 = 'utter_goodbye'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleAssistantRestInput(InputChannel):
    """A custom http input channel.

    This implementation is the basis for a custom implementation of a chat
    frontend. You can customize this to send messages to Rasa and
    retrieve responses from the assistant."""

    @classmethod
    def name(cls) -> Text:
        return "google_assistant"

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
        logger.info('Init google_assistant Channel')
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
        collector = GoogleAssistantQueueOutputChannel(queue)

        message = UserMessage(
            text, collector, sender_id, input_channel=input_channel, metadata=metadata
        )
        await on_new_message(message)

        await queue.put("DONE")

    def _extract_input_channel(self, req: Request) -> Text:
        return req.json.get("input_channel") or self.name()

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
            logger.info('google_assistant get status')
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> HTTPResponse:
            logger.info(f'>> google_assistant:')
            logger.debug(json.dumps(request.json, indent=4))

            intent = request.json.get('intent', None)
            session = request.json.get('session', None)

            assert intent is not None, 'Error: intent is None'
            assert session is not None, 'Error: session is None'

            session_id = session['id']
            input_text = intent['query']
            intent_name = intent['name']

            logger.info(">> google_assistant Webhook Post Call ")
            logger.info(f'    > Message: {input_text}')
            logger.info(f'    > Conversation_id: {session_id}')

            data = {"event": "slot",
                    "name": "session_id",
                    "value": session_id,
                    "timestamp": 0}

            # Saving session into tracker
            tracker_post_url = self.api_host + f"/conversations/{session_id}/tracker/events?include_events=NONE&token={self.auth_token}"
            client = http3.AsyncClient()
            logger.info(f'Post Call: {tracker_post_url}')
            r = await client.post(tracker_post_url, data=json.dumps(data))
            logger.info(f'    > Response: {r}')
            try:
                logger.info(r.headers.get('Content-Type'))
                event_response = r.json()
                logger.info(f'Event Response: {event_response}')
                last_intent = event_response['latest_message']['intent'].get('name')
                logger.info(f'Last Intent: {last_intent}')
            except Exception as error:
                logger.error(f'Json response error: {error}')
                last_intent = None

            should_use_stream = rasa.utils.endpoints.bool_arg(
                request, "stream", default=False
            )
            input_channel = self._extract_input_channel(request)
            logger.info(f'Input_channel: {input_channel}')
            metadata = self.get_metadata(request)

            main_invocation = intent_name == MAIN_INVOCATION_INTENT
            if main_invocation:
                input_text = '/welcome'

            if should_use_stream:
                return response.stream(
                    self.stream_response(
                        on_new_message, input_text, session_id, input_channel, metadata
                    ),
                    content_type="text/event-stream",
                )
            else:
                # logging.info("========NO STREAM but collecting ========")
                collector = CollectingOutputChannel()
                # noinspection PyBroadException
                try:
                    await on_new_message(
                        UserMessage(
                            input_text,
                            collector,
                            session_id,
                            input_channel=input_channel,
                            metadata=metadata,
                        )
                    )
                except CancelledError:
                    logger.error(
                        f"Message handling timed out for " f"user message '{input_text}'."
                    )
                except Exception:
                    logger.exception(
                        f"An exception occured while handling "
                        f"user message '{input_text}'."
                    )
                logger.info(f"Rasa Response: {collector.messages}")

                google_assistant_message = ""

                for val in collector.messages:
                    message = val.get("text")
                    google_assistant_message = google_assistant_message + " \n " + message if message is not None else google_assistant_message

                # Getting tracker story
                tracker_get_url = self.api_host + f'/conversations/{session_id}/story?token={self.auth_token}'
                client = http3.AsyncClient()
                logger.info(f'Post Get: {tracker_post_url}')
                r = await client.get(tracker_get_url)
                logger.info(f'    > Response: {r}')
                try:
                    logger.info(r.headers.get('Content-Type'))
                    logger.info(f'Text: {r.text}')
                    story = yaml.safe_load(r.text)
                    stories = story['stories']
                    logger.info(f'Story: {stories}')
                    utter_list = [step['action'] for step in stories[0]['steps'] if 'action' in step.keys()]
                    logger.info(f'Utter_list: {utter_list}')

                except Exception as error:
                    logger.error(f'Reading tracker story error: {error}')
                    utter_list = []

                # main_invocation = intent_name == MAIN_INVOCATION_INTENT
                end_conversation = END_CONVERSATION_UTTER_1 in utter_list or END_CONVERSATION_UTTER_2 in utter_list

                # if main_invocation:
                #     response_data = {"session": {
                #         "id": session_id,
                #         "params": {}
                #     },
                #         "prompt": {
                #             "override": 'false',
                #             "firstSimple": {
                #                 "speech": 'Ciao sono la demo di YouAI, cosa posso fare per te?',
                #                 "text": ""
                #             }
                #         }
                #     }
                # elif not main_invocation and end_conversation:
                if end_conversation:
                    response_data = {"session": {
                        "id": session_id,
                        "params": {}
                    },
                        "prompt": {
                            "override": 'false',
                            "firstSimple": {
                                "speech": google_assistant_message,
                                "text": ""
                            }
                        },
                        "scene": {
                            "name": "Main",
                            "slots": {},
                            "next": {
                                "name": "actions.scene.END_CONVERSATION"
                            }
                        }
                    }

                else:
                    response_data = {"session": {
                                        "id": session_id,
                                        "params": {}
                                      },
                                     "prompt": {
                                        "override": 'false',
                                        "firstSimple": {
                                          "speech": google_assistant_message,
                                          "text": ""
                                        }
                                      }
                                    }

                logger.debug(f'Rasa to Google assistance: {response_data}')
                return response.json(response_data)

        logger.info(f'webhook_name: {custom_webhook.name}')
        logger.info(f'webhook_host: {custom_webhook.host}')
        logger.info(f'webhook_url: {custom_webhook.url_prefix}')
        logger.info(f'webhook_routes: {custom_webhook.routes}')

        return custom_webhook


class GoogleAssistantQueueOutputChannel(CollectingOutputChannel):
    """Output channel that collects send messages in a list

    (doesn't send them anywhere, just collects them)."""

    @classmethod
    def name(cls) -> Text:
        return "googleassistantqueue"

    # noinspection PyMissingConstructor
    def __init__(self, message_queue: Optional[Queue] = None) -> None:
        super().__init__()
        self.messages = Queue() if not message_queue else message_queue

    def latest_output(self) -> NoReturn:
        raise NotImplementedError("A queue doesn't allow to peek at messages.")

    async def _persist_message(self, message) -> None:
        await self.messages.put(message)  # pytype: disable=bad-return-type
