from openai import OpenAI
# MODEL_NAME = "deepseek-reasoner"  
MODEL_NAME = "deepseek-chat"  
TEMP = 0.8
MAX_MD_CHARS = 900_000               # 防止超 token，太长就截断
SLEEP_SEC_ON_RATE_LIMIT = 20
client = OpenAI(
    api_key="sk-d992b4434046444191435e9d3fbe82c7",
    base_url="https://api.deepseek.com"
)


def llm_generate(user_contents,assistant_contents=None):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    # messages.append({"role": "user", "content": user_contents})
    if assistant_contents is None:
        assistant_contents = []
    elif isinstance(assistant_contents, str):
        assistant_contents = [assistant_contents]
    if isinstance(user_contents, str):
        user_contents = [user_contents]
    num_assistant_mes = len(assistant_contents)
    for idx in range(num_assistant_mes):
        messages.append({"role": "user", "content": user_contents[idx]})
        messages.append({"role": "assistant", "content": assistant_contents[idx]})
    messages.append({"role": "user", "content": user_contents[-1]})
    
    
    
    
    
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        #prompt=prompt,
        messages=messages,
        #max_tokens=max_tokens,#不知道删掉行不行
        temperature=TEMP
    )
    result = ''
    for choice in response.choices: 
        result += choice.message.content 
    return result


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = llm_generate(prompt)
    print(response)