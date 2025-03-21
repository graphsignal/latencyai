import json
import logging
import openai
from openai import AsyncOpenAI
from openai import OpenAIError


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#openai.logging.enable()
#openai.logging.set_level(openai.logging.DEBUG)

class OpenAIChat:
    def __init__(self, developer_message, model_name="o3-mini"):
        self.client = AsyncOpenAI()
        self.model_name = model_name

        self.messages = []
        self.messages.append({"role": "system", "content": developer_message})

    def developer(self, message: str):
        self.add_message('developer', message)
 
    def assistant(self, message: str):
        self.add_message('assistant', message)

    def user(self, message: str):
        self.add_message('user', message)

    def add_message(self, role: str, message: str):
        self.messages.append({"role": role, "content": message})
        logger.debug(f"Chat message: {role}: {message}")

    async def generate_completion(self) -> str:
        try:
            #logger.debug(f"Messages: {self.messages}")
            logger.debug("Generating completion...")

            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages
            )

            completion_content = completion.choices[0].message.content.strip()
            #logger.debug(f"Completion: {completion_content}")
            return completion_content
        except OpenAIError as e:
            raise e

    async def generate_code(self) -> str:
        try:
            self.user("Pass generated Python code to 'code' argument of 'execute_code' function.")
            #logger.debug(f"Messages: {self.messages}")
            logger.debug("Generating code...")

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "description": "Executes Python code",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "Python code to execute"
                                }
                            },
                            "required": ["code"]
                        }
                    }
                }
            ]

            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                tools=tools,
                tool_choice="auto"
            )
            #logger.debug(f"Completion: {completion}")
            args_json = completion.choices[0].message.tool_calls[0].function.arguments
            args = json.loads(args_json)
            code = args['code'].strip()
            #logger.debug(f"Code: {code}")
            return code
        except OpenAIError as e:
            raise e

    def get_messages(self):
        return self.messages

    def reset(self):
        self.messages = []

    async def close(self):
        await self.client.close()