import ollama
response = ollama.chat(
    model='llava',
    messages=[{
        'role': 'user',
        # 'content': 'Describe this image in English.',
        "content": 'Is there any person in this image? Return "Yes" or "No".',
        'images': ['./photo2.jpg']
    }]
)
print(response['message']['content'])
