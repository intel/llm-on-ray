class ChatModel:
    human_id = "<human>"
    bot_id = "<bot>"
    unknown_id = "<unknown>"
    MEANINGLESS_WORDS = ['<pad>', '</s>', '<|endoftext|>', '<br>']
    stop_words = ["<human>"]

    def __init__(self, human_id, bot_id, stop_words, prompt_prefix="", generation_config=None) -> None:
        self.human_id = human_id
        self.bot_id = bot_id
        self.stop_words = stop_words
        self.MEANINGLESS_WORDS.extend(self.stop_words)
        self.prompt_prefix = prompt_prefix
        self._generation_config = generation_config

    def prepare_prompt(self, messages: list):
        """Prepare prompt from history messages."""
        prompt = ''
        for msg in messages:
            role, content = msg.role, msg.content
            if role == "user":
                prompt += f"{self.human_id}: {content}\n"
            elif role == "assistant":
                prompt += f"{self.bot_id}: {content}\n"
            else:
                prompt += f"{self.unknown_id}: {content}\n"
        prompt += f"{self.bot_id}:"
        return prompt

    def convert_output(self, output: str):
        """Convert the model output to final answer."""
        bot_turn = output.split(self.human_id)[0]
        bot_turn = bot_turn.split(self.bot_id)[0]
        for word in self.MEANINGLESS_WORDS:
            bot_turn = bot_turn.replace(word, "")
        text = bot_turn.strip()
        # remove partial human_id or bot id
        if '\n' in text and (self.human_id.startswith(text[text.rfind('\n')+1:]) or
                             self.bot_id.startswith(text[text.rfind('\n')+1])):
            text = text[:text.rfind('\n')]
        return text

    def get_prompt(self ,messages):
        """Generate response based on messages."""
        prompt = self.prepare_prompt(messages)
        return prompt
        

class ChatModelGptJ(ChatModel):
    def __init__(self, human_id, bot_id, stop_words, prompt_prefix="", generation_config=None):
        super().__init__(human_id, bot_id, stop_words, prompt_prefix, generation_config)

    def prepare_prompt(self, messages: list):
        """Prepare prompt from history messages."""
        prompt = self.prompt_prefix
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "user":
                prompt += f"{self.human_id}:\n{content}\n"
            elif role == "assistant":
                prompt += f"{self.bot_id}:\n{content}\n"
            else:
                prompt += f"### Unknown:\n{content}\n"
        prompt += f"{self.bot_id}:\n"
        return prompt

if __name__ == "__main__":
    process_tool = ChatModelGptJ("### Instruction", "### Response", stop_words=["##", "### Instruction"])
