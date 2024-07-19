import openai
import re
import httpx
import os
from dotenv import load_dotenv
_ = load_dotenv()
from openai import OpenAI

def test_openai_api_call():
    client = OpenAI()
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello world"}]
    )
    print(chat_completion.choices[0].message.content)

class Agent:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.messages: list[str] = []
        if self.system_prompt:
            self.messages.append({"role":"user", "content":system_prompt})
        self.client = OpenAI()

    def execute(self, messages: list[str]) -> str:
        chat_completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return chat_completion.choices[0].message.content

    def __call__(self, message: str) -> str:
        self.messages.append({"role":"user", "content":message})
        assistant_message = self.execute(self.messages)
        self.messages.append({"role":"assistant", "content":assistant_message})
        return assistant_message

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()

def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}

def example_turn():
    abot = Agent(prompt)
    result = abot("How much does a toy poodle weigh?")
    print(result)
    result = average_dog_weight("Toy Poodle")
    print(result)
    next_prompt = f"Observation: {result}"
    answer = abot(next_prompt)
    print(answer)

def example_turn_2():
    abot = Agent(prompt)
    question = """I have 2 dogs, a border collie and a scottish terrier. \
    What is their combined weight"""
    result = abot(question)
    print(result)
    result = average_dog_weight("Border Collie")
    print(result)
    next_prompt = f"Observation: {result}"
    result = abot(next_prompt)
    print(result)
    result = average_dog_weight("Scottish Terrier")
    print(result)
    next_prompt = f"Observation: {result}"
    result = abot(next_prompt)
    print(result)
    result = calculate("37 + 20")
    print(result)
    next_prompt = f"Observation: {result}"
    result = abot(next_prompt)
    print(result)

def loop():
    action_re = re.compile('^Action: (\w+): (.*)$')
    def query(question, max_turns=5):
        i = 0
        bot = Agent(prompt)
        next_prompt = question 
        while i < max_turns:
            i += 1
            result = bot(next_prompt)
            print(result)

            actions = [
                action_re.match(a)
                for a in result.split('\n')
                if action_re.match(a)
            ]

            if actions:
                action, action_input = actions[0].groups()
                if action not in known_actions:
                    raise Exception(f"Unknown action: {action}, {action_input}")
                print(f" -- running {action} {action_input}")
                observation = known_actions[action](action_input)
                print(f"Observation: {observation}")
                next_prompt = f"Observation: {observation}"
            else:
                return 
    question = """I have 2 dogs, a border collie and a scottish terrier. \
    What is their combined weight"""
    query(question)

if __name__ == '__main__':
    # test_openai_api_call()
    # example_turn()
    # example_turn_2()
    loop()
