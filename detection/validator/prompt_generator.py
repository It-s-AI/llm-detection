import sys

import numpy as np
from transformers import AutoTokenizer, Pipeline

from detection.validator.text_completion import OllamaModel
from prompting.agent import HumanAgent
from prompting.conversation import create_task
import bittensor as bt


class MyLLMPipeline:
    def __init__(self, model_name='zephyr:7b-beta'):
        self.model = OllamaModel(model_name,
                                 num_predict=512)
        self.tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')

    def __call__(self, prompt, **kwargs):
        res = self.model(prompt)
        return [{'generated_text': res}]


class PromptGenerator:
    def __init__(self,
                 model_id='HuggingFaceH4/zephyr-7b-beta',
                 tasks=['summarization', 'qa', 'debugging', 'math', 'date_qa'],
                 task_p=[0.25, 0.25, 0.0, 0.25, 0.25],
                 device='cuda'):

        self.tasks = tasks
        self.model_id = model_id
        self.task_p = task_p

        self.llm_pipeline = MyLLMPipeline()
        # self.llm_pipeline = load_pipeline(
        #     model_id=model_id,
        #     torch_dtype=torch.bfloat16,
        #     device=device,
        #     mock=False,
        # )

    def get_challenge(self, task_name=None):
        while True:
            bt.logging.info(
                f"📋 Selecting task... from {self.tasks} with distribution {self.task_p}"
            )

            if task_name is None:
                task_name = np.random.choice(
                    self.tasks, p=self.task_p
                )

            bt.logging.info(f"📋 Creating {task_name} task... ")

            try:
                task = create_task(llm_pipeline=self.llm_pipeline, task_name=task_name)
                break
            except Exception as e:
                bt.logging.error(
                    f"Failed to create {task_name} task. {sys.exc_info()}. Skipping to next task."
                )
                continue

        bt.logging.info(f"🤖 Creating agent for {task_name} task... ")
        agent = HumanAgent(
            task=task, llm_pipeline=self.llm_pipeline, begin_conversation=True
        )

        res = {'prompt': agent.challenge,
               'persona_profile': agent.persona.profile,
               'persona_mood': agent.persona.mood,
               'person_ton': agent.persona.tone, }

        for k, v in task.__state_dict__().items():
            res[k] = v

        return res


if __name__ == '__main__':
    generator = PromptGenerator()
    challenge = generator.get_challenge(None)
    print(challenge)
