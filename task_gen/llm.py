from openai import OpenAI
MODEL_NAME = "deepseek-chat"  # "deepseek-reasoner"
TEMPERATURE = 0.8

client = OpenAI(
    api_key="sk-d992b4434046444191435e9d3fbe82c7",
    base_url="https://api.deepseek.com"
)

def llm_generate(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE
    )

    result = ""
    for choice in response.choices: 
        result += choice.message.content 
    return result

if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = llm_generate(prompt)
    print(response)