# Measure Retrieval and generation seperately
import asyncio
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from config import settings
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall, ContextEntityRecall, NoiseSensitivity

GROUND_TRUTH = [

    {
        "question": "What happens to an item if it fails the R&I Committee's inspection?",
        "reference": (
            "The item is rejected and immediately moved to the quarantine area, "
            "with each carton clearly marked 'Not For Use - Quarantined'. The "
            "Warehouse Manager is notified to remove the rejected product from "
            "the Goods Receipt in mSupply, and the Contract Management Chief is "
            "notified to inform the supplier of the problem. Minor issues are "
            "resolved and the item released; significant issues require the "
            "supplier to replace, destroy, or refund the stock at their own cost."
        ),
        "reference_page": "24",
    },
    {
        "question": "Who finalises a Supplier Invoice, and what does that step actually do?",
        "reference": (
            "The Warehouse Director finalises the Supplier Invoice, after the "
            "Warehouse Manager has already confirmed it. Finalising blocks any "
            "further editing and completes the goods receipt process."
        ),
        "reference_page": "29-33",
    },
    {
        "question": "Before a supplier can actually be paid, what does the Contract Management Chief need to do first?",
        "reference": (
            "The Contract Management Chief double-checks the payment amount on "
            "the Supplier Invoice against other local records, then creates a "
            "payment request attaching the Supplier Invoice, Purchase Order, "
            "contract, and other relevant documents, and sends these to the NMS "
            "President for endorsement before they can be forwarded to a "
            "Finance Officer."
        ),
        "reference_page": "35",
    },
    {
        "question": "How does mSupply decide which batch of stock to give out first when fulfilling a customer order?",
        "reference": (
            "mSupply automatically allocates stock according to FEFO (First "
            "Expiry, First Out) when the Customer Invoice is created."
        ),
        "reference_page": "39",
    },
    {
        "question": "What's the difference between using 'Split Stock' and 'Consolidate Stock' in mSupply?",
        "reference": (
            "Split Stock moves part of a single batch to a different location, "
            "used when the same batch needs to be divided across multiple "
            "storage locations. Consolidate Stock does the opposite - it brings "
            "stock of the same batch that is currently spread across multiple "
            "locations together into one single location."
        ),
        "reference_page": "63-64",
    },
    {
        "question": "If I'm removing damaged stock in mSupply, when should I use a Negative Inventory Adjustment versus a Stocktake?",
        "reference": (
            "Use a Negative Inventory Adjustment if you want to enter the "
            "quantity of stock being removed. Use a Stocktake if you'd rather "
            "enter the total quantity of stock remaining after the unusable "
            "goods have already been removed or destroyed."
        ),
        "reference_page": "66-68",
    },
    {
        "question": "What temperature range do cold chain items need to be kept at, and what's the maximum the general warehouse can reach?",
        "reference": (
            "Cold chain items must be kept in a fridge or cold room maintained "
            "at 2-8°C. The general warehouse temperature should stay below 25°C "
            "whenever possible and must never exceed 30°C."
        ),
        "reference_page": "72-73",
    },
    {
        "question": "What needs to be cleared up in mSupply before a full stocktake can start?",
        "reference": (
            "Every pending order must be processed, every outstanding Goods "
            "Receipt must be processed, and every order awaiting dispatch must "
            "be processed and excluded from the stocktake."
        ),
        "reference_page": "79",
    },
    {
        "question": "Who carries out the monthly warehouse audit, and how is its accuracy percentage worked out?",
        "reference": (
            "The monthly warehouse audit is carried out by 2 non-warehouse "
            "staff from the National Pharmacy Division, who check a randomly "
            "selected 20 items. Accuracy is calculated as (number of correct "
            "items / total number of items checked) x 100%."
        ),
        "reference_page": "82",
    },
    {
        "question": "How often does stock have to be checked against the DDA Register, and who does that check?",
        "reference": (
            "Controlled substances must be audited against the DDA Register at "
            "least once every 3 months, carried out by two staff members: the "
            "Warehouse Manager and the DDA Coordinator (selected on a "
            "rotational basis)."
        ),
        "reference_page": "83",
    },
]

class RetrivalMetrics:

    def __init__(self):
        self.__llm_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        self.__llm = llm_factory("gpt-4.1-mini", client=self.__llm_client)
        self.__retriever = settings.index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=1
        )
        self.results = {}
        self.__semaphore = asyncio.Semaphore(4)

    async def context_precision(self, question: str, reference: str, retrieved_contexts: list[str]):
        context_precision = ContextPrecision(llm=self.__llm)

        async with self.__semaphore:
            result = await context_precision.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )

        return result.value

    async def context_recall(self, question: str, reference: str, retrieved_contexts: list[str]):
        context_recall = ContextRecall(llm=self.__llm)

        async with self.__semaphore:
            result = await context_recall.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )

        return result.value

    async def evaluate( self,
        question: str,
        reference: str
    ):
        retrieved_nodes = await self.__retriever.aretrieve(question)
        retrieved_contexts = [
            node.node.get_content()
            for node in retrieved_nodes
        ]

        precision, recall = await asyncio.gather(
            self.context_precision(question, reference, retrieved_contexts),
            self.context_recall(question, reference, retrieved_contexts)
        )

        self.results[question] = {
            "context_precision": precision,
            "context_recall": recall
        }

    async def get_results(self):
        return self.results


async def main():
    metrics = RetrivalMetrics()
    tasks = [
        metrics.evaluate(
            question=data["question"],
            reference=data["reference"],
        )
        for data in GROUND_TRUTH
    ]


    await tqdm_asyncio.gather(
        *tasks,
        desc="Evaluating questions"
    )

    result = await metrics.get_results()

    for key, value in result.items():
        print(f"Question: {key}")
        print(f"Context_precision: {value['context_precision']}")
        print(f"Recall: {value['context_recall']}")




if __name__ == "__main__":

    asyncio.run(main())