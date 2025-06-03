def llm_generate(prompt):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": model,
        "messages": messages,
        #"max_tokens": 32000,#不知道删掉行不行
        # "temperature": temperature
    }

    response = requests.post("http://localhost:12589/generate", json=data)

    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")

    return response.json()