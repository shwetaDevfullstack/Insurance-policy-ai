from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def generate_answer(context, question):

    prompt = f"""
                You are an insurance policy assistant.

                Answer the question ONLY using the provided context.

                If the answer is not present in the context, say:
                "I could not find this information in the policy."

                Context:
                {context}

                Question:
                {question}

                Answer:
            """

    response = generator(
        prompt,
        max_new_tokens=150
    )

    return response[0]["generated_text"]