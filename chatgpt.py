import openai
import time

client = openai.OpenAI()

input_context = f"Given a context, please generate related questions as comprehensively as possible with bullet points and answers.\n"
input_context += '''This is an example:
Context: A small coastal town has a beach known for its colorful sea glass. The town hosts an annual festival celebrating this unique feature with art and conservation efforts.

Question: What attracts tourists to the small coastal town annually? Answer: The unique sea glass beach.
Question: What is celebrated at the town's annual festival? Answer: The natural phenomenon of sea glass.
Question: What type of activities are featured at the festival? Answer: Glass art exhibitions, environmental workshops, and local music performances.
Question: What is the purpose of the workshops at the festival? Answer: To promote environmental awareness among visitors.
Question: How does the festival impact the local economy? Answer: It boosts local businesses by attracting tourists.
'''

def gpt_generate(context, gpt_model):
    input = f'Now, please generate related questions based on the following context: \n{context}'
    success = False
    while not success:
        try:
            completion = client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": input_context},
                    {"role": "user", "content": input}
                ],
                temperature=1
            )
            success = True
        except:
            print('api failed')
            time.sleep(10)

    return completion.choices[0].message.content.replace('\n', '')
