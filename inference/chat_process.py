class ChatModel:
    human_id = "<human>"
    bot_id = "<bot>"
    unknown_id = "<unknown>"
    MEANINGLESS_WORDS = ['<pad>', '</s>', '<|endoftext|>', '<br>']
    stop_words = ["<human>"]

    def __init__(self, intro, human_id, bot_id, stop_words) -> None:
        self.intro = intro
        self.human_id = human_id
        self.bot_id = bot_id
        self.stop_words = stop_words
        self.MEANINGLESS_WORDS.extend(self.stop_words)

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
        human_id = self.human_id.strip()
        bot_id = self.bot_id.strip()
        if human_id != "":
            output = output.split(human_id)[0]
        if bot_id != "":
            output = output.split(bot_id)[0]
        for word in self.MEANINGLESS_WORDS:
            output = output.replace(word, "")
        text = output
        # remove partial human_id or bot id
        if '\n' in text and (human_id.startswith(text[text.rfind('\n')+1:]) or
                             bot_id.startswith(text[text.rfind('\n')+1])):
            text = text[:text.rfind('\n')]
        return text

    def get_prompt(self ,messages):
        """Generate response based on messages."""
        prompt = self.prepare_prompt(messages)
        return prompt
    
class ChatModelGptJ(ChatModel):
    def __init__(self, intro, human_id, bot_id, stop_words):
        super().__init__(intro, human_id, bot_id, stop_words)

    def prepare_prompt(self, messages: list):
        """Prepare prompt from history messages."""
        prompt = self.intro
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "user":
                if self.human_id != "":
                    prompt += f"{self.human_id}:\n{content}\n"
                else:
                    prompt += f"{content}\n"
            elif role == "assistant":
                if self.bot_id != "":
                    prompt += f"{self.bot_id}:\n{content}\n"
                else:
                    prompt += f"{content}\n"
            else:
                prompt += f"### Unknown:\n{content}\n"
        if self.bot_id != "":
            prompt += f"{self.bot_id}:\n"
        return prompt

class ChatModelLLama(ChatModel):
    def __init__(self, intro, human_id, bot_id, stop_words):
        super().__init__(intro, human_id, bot_id, stop_words)

    def prepare_prompt(self, messages: list):
        """Prepare prompt from history messages."""
        prompt = self.intro
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "user":
                if self.human_id != "":
                    prompt += self.human_id.format(content)
                else:
                    prompt += f"{content}\n"
            elif role == "assistant":
                prompt += f"{content}\n"
            else:
                prompt += f"### Unknown:\n{content}\n"
        if self.bot_id != "":
            prompt += f"{self.bot_id}:\n"
        return prompt

if __name__ == "__main__":
    process_tool = ChatModelGptJ("### Instruction", "### Response", stop_words=["##", "### Instruction"])
