"""Generation: turn retrieved passages into a cited answer."""
import os
from openai import OpenAI
from retrieve import Retriever

MODEL = "gpt-4o-mini"
DEFAULT_K = 8   # Recall@8 is 87.5% vs 81.2% at k=4; more context, more tokens

REFUSAL = "The retrieved sections do not answer this question."

SYSTEM = f"""You answer questions about Indian labour law.

- Use ONLY the provided sections. Do not use outside knowledge.
- If they do not contain the answer, say exactly: "{REFUSAL}"
- Cite the sections you used as [1], [2].
- Be concise. Quote specific numbers, time periods and thresholds where given.
"""


class Answerer:
    """The model never sees the corpus - only the passages in one message.

    The refusal instruction is the most important line in the prompt. Without it
    the model always produces something, and a confident wrong answer about
    employment law is the dangerous failure mode.
    """

    def __init__(self, retriever=None, device="cpu"):
        self.retriever = retriever or Retriever(device=device)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def answer(self, question, k=DEFAULT_K):
        passages = self.retriever.search(question, k=k)
        context = "\n\n".join(
            f"[{i + 1}] ({p['code']}, page {p['page']})\n{p['text']}"
            for i, p in enumerate(passages)
        )
        response = self.client.chat.completions.create(
            model=MODEL,
            temperature=0,   # reproducible, which matters when measuring
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Sections:\n\n{context}\n\nQuestion: {question}"},
            ],
        )
        return {"answer": response.choices[0].message.content, "sources": passages}


if __name__ == "__main__":
    a = Answerer()
    result = a.answer("How much notice must be given before retrenching a worker?")
    print(result["answer"])
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s['code']} p.{s['page']}")
