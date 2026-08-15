import json

from config.settings import settings
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel, Field


class SOPResponse(BaseModel):
    answer: str = Field(description="Grounded answer based only on SOP context")
    citations: list[str] = Field(description="Relevant SOP sections/pages")
    confidence: float = Field(description="Confidence score from 0 to 1")


async def generate_sop_context(question: str, data: list[dict] | list[str]) -> SOPResponse:

    llm = Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=settings.ollama_request_timeout,
    )

    prompt = PromptTemplate("""
    You are answering questions about warehouse Standard Operating Procedures
    (SOPs) for pharmaceutical and medical supply management. Your answer will
    be read by a warehouse worker who needs to follow it exactly, so
    correctness and completeness matter more than brevity.

    Question:
    {question}

    Context:
    {data}

    How to use the context:
    - The context may contain multiple chunks, each labelled with its source
      section and page number (e.g. "[Chunk 2 - Basic Payments, p35]"). Some
      chunks may be irrelevant to the question - these are realistic
      retrieval noise. Identify which chunk(s) actually answer the question
      and ignore the rest; do not blend unrelated chunks into the answer.
    - Base the answer ONLY on the supplied context. Do not use outside
      knowledge of mSupply, warehousing, or pharmaceutical logistics, even if
      you believe you know the answer - the context is the sole source of
      truth for this task.
    - If the question has multiple parts (e.g. it asks for a sequence of
      steps, every role involved, a comparison between two procedures, or a
      list of conditions), address every part. An answer that only covers
      part of a multi-part question is incomplete.
    - Preserve exact procedural detail from the context: specific numbers,
      thresholds, time windows, temperatures, percentages, button/field
      names, and role titles. Do not round, generalise, or paraphrase these
      away.
    - If the context does not contain enough information to answer the
      question (in full or in part), say so explicitly in the answer instead
      of guessing or inventing a plausible-sounding procedure. State clearly
      which part of the question the context does not cover. It is always
      better to admit a gap than to fabricate a policy, number, or step.

    Citations:
    - List the page number(s) shown in the labels of the chunk(s) you
      actually relied on, e.g. "35" or "29-30". Do not cite a chunk you
      ignored as irrelevant. Do not invent a page number that isn't present
      in the context.

    Confidence:
    - 0.9-1.0: the context fully and directly supports every part of the
      answer.
    - 0.5-0.8: the context supports most of the answer, but some minor detail
      is inferred, ambiguous, or only partially covered.
    - 0.0-0.4: the context does not meaningfully answer the question, or the
      answer is mostly a statement that the information isn't available.

    Return ONLY valid JSON in exactly this format:

    {{
        "answer": "your answer here",
        "citations": ["page number"],
        "confidence": 0.0
    }}

    Rules:
    - Do not return markdown.
    - Do not use ```json fences.
    - Do not return any text before or after the JSON.
    - confidence must be a number between 0 and 1.
    """)

    llm_generated_data = await llm.acomplete(
        prompt.format(question=question, data=json.dumps(data))
    )

    raw = llm_generated_data.text.strip()

    parsed = json.loads(raw)

    return SOPResponse.model_validate(parsed)

