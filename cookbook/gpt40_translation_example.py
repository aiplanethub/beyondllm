from beyondllm.llms import GPT4oOpenAIModel
llm = GPT4oOpenAIModel(api_key="YOUR_OPENAI_API_KEY")
text_to_translate = "どこへ行くの？外は暑すぎるよ"
target_language = "English"
prompt = f"Translate the following text to {target_language}: '{text_to_translate}'"
translated_text = llm.predict(prompt)
print(translated_text)
