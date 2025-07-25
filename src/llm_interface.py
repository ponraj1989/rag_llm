from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LLMInterface:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.llm = None
        self._initialize_model()

    def _initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            load_in_8bit=True
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

    def get_llm(self):
        return self.llm