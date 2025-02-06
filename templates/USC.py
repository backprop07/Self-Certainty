def usc_prompt(question: str, outputs: list):
    prompt = f"I have generated the following responses to the question: {question}\n"
    for i in range(len(outputs)):
        prompt += f"Response {i}: {outputs[i]}\n\n"
    prompt += "Evaluate these responses.\n"
    prompt += "Select the most consistent response based on majority consensus.\n"
    prompt += 'Format you answer as "The most consistent response is Response X" (wihtout quotes).\n' 
    return prompt