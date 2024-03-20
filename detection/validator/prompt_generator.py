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
            bt.logging.debug(
                f"📋 Selecting task... from {self.tasks} with distribution {self.task_p}"
            )

            if task_name is None:
                cur_task = np.random.choice(
                    self.tasks, p=self.task_p
                )
            else:
                cur_task = task_name

            bt.logging.debug(f"📋 Creating {cur_task} task... ")

            try:
                task = create_task(llm_pipeline=self.llm_pipeline, task_name=cur_task)
                break
            except Exception as e:
                bt.logging.error(
                    f"Failed to create {cur_task} task. {sys.exc_info()}. Skipping to next task."
                )
                continue

        bt.logging.debug(f"🤖 Creating agent for {cur_task} task... ")
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
