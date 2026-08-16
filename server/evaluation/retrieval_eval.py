import asyncio

from config import settings
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
)
from tqdm.asyncio import tqdm_asyncio

GROUND_TRUTH = [
    {
        "question": "A carton of vaccines arrives and the temperature monitor inside reads too low. What does the SOP require before that carton can be accepted?",
        "reference": (
            "If a temperature monitor in any carton is too low (below 2°C), the "
            "entire carton must be subjected to a Shake Test. If the secondary "
            "test also fails, that carton must be rejected and a thorough "
            "inspection of all remaining cartons must be carried out. For "
            "vaccines, a Vaccines Arrival Report (VAR) must also be completed "
            "and emailed to the supplier within 36 hours of receipt."
        ),
        "reference_page": "20",
    },
    {
        "question": "Who needs to be told straight away if the quarterly controlled-drugs count doesn't match the register?",
        "reference": (
            "Any discrepancy found during the periodic stocktake of controlled "
            "substances against the DDA Register (done at least once every 3 "
            "months by the Warehouse Manager and the rotational DDA "
            "Coordinator) must be immediately reported to the NMS President "
            "and the National Director of Pharmacy to initiate an "
            "investigation."
        ),
        "reference_page": "83",
    },
    {
        "question": "During the annual full count, a team finds expired items sitting in a normal picking location. Are they allowed to move or bin them on the spot?",
        "reference": (
            "No. During a stocktake no stock is to be moved: expired items or "
            "stock in the wrong location should still be counted, and only "
            "moved or destroyed at the end of the stocktake. Any item found on "
            "the shelf but not on the stocktake list must also be recorded."
        ),
        "reference_page": "79",
    },
    {
        "question": "A charity drops off a pallet of medicines we never ordered, and the charity isn't in our system. How does that stock get into mSupply?",
        "reference": (
            "First add the donor as a new Supplier: navigate to the Suppliers "
            "tab, click New Supplier, and enter at minimum the Code, Charge To "
            "and Name. Then record the stock with a manual Supplier Invoice: "
            "Suppliers tab, New Supplier Invoice, enter the supplier, click "
            "New Line for each received item, enter Number of Packs, Pack "
            "Size, Batch, Expiry and Location, then check Finalise and click "
            "OK."
        ),
        "reference_page": "69-70",
    },
    {
        "question": "A batch won't fit on a single shelf, so some packs have to live somewhere else. Which mSupply transaction records that, and what exactly do you enter?",
        "reference": (
            "Split Stock. Open the item via the Item tab, Item List, search "
            "and double-click the item, then open the Stock tab. Click the "
            "line to split, click Split, enter the Quantity to Split (the "
            "amount being moved) and the New Shelf Location, then click OK. "
            "The batch's total quantity is unchanged - it is simply stored "
            "across two locations."
        ),
        "reference_page": "62-64",
    },
    {
        "question": "How many people can key results into the mSupply stocktake screen at the same time, and how should data entry be organised during the annual count?",
        "reference": (
            "The mSupply Stocktake screen can only be opened by one person at "
            "a time. Completed stocktake sheets are taken periodically to the "
            "Warehouse Manager, who enters the actual quantities consistently "
            "throughout the week rather than at the very end; the Warehouse "
            "Manager may ask Team Leaders to assist so that data entry never "
            "becomes a bottleneck."
        ),
        "reference_page": "80-81",
    },
    {
        "question": "How long must the paperwork proving the fridge thermometers were serviced and calibrated be kept on file?",
        "reference": (
            "Certificates of Servicing and Calibration must be retained with "
            "the Master Calibration Log for a minimum of 5 years. All "
            "thermometers are calibrated initially when purchased and annually "
            "thereafter, and every temperature gauge must be accurate to "
            "within 0.5°C."
        ),
        "reference_page": "74-75",
    },
    {
        "question": "There's a percentage chart pinned up in the warehouse each month. What is it, who produces it, and how is the number worked out?",
        "reference": (
            "It is the warehouse audit accuracy figure. In the middle of each "
            "month, 2 non-warehouse staff from the National Pharmacy Division "
            "check 20 randomly selected items; an item counts as correct only "
            "if its location, batch, expiry and quantity all match mSupply. "
            "The percentage is calculated as (number of correct items / total "
            "number of items checked, i.e. 20) x 100%, then graphed each month "
            "and displayed in the warehouse for all staff to see."
        ),
        "reference_page": "82",
    },
    {
        "question": "Right before a goods receipt is finalised in the system, someone physically walks the warehouse. Who is it and what are they checking for?",
        "reference": (
            "The Warehouse Manager walks through the warehouse to ensure the "
            "proposed locations have sufficient space for the incoming "
            "products. Only when the Warehouse Manager is satisfied everything "
            "is correct do they finalise the Goods Receipt in mSupply, which "
            "generates the Supplier Invoice."
        ),
        "reference_page": "29",
    },
    {
        "question": "A supplier flat-out refuses to take back or replace stock that failed inspection. What happens next?",
        "reference": (
            "The Warehouse Manager removes the product from the Purchase Order "
            "in mSupply, and Procurement and Finance commence dispute "
            "processes according to the specific contract conditions. The "
            "stock is then sourced elsewhere by the Contract Management Chief. "
            "The supplier must refund any money already paid on the contract "
            "(as a credit to NMS) and may forfeit a percentage of their "
            "performance security, depending on the contract terms."
        ),
        "reference_page": "24",
    },
    {
        "question": "Some health facilities still send handwritten orders. How do those get into the system, and what happens if an approval request just sits there unactioned?",
        "reference": (
            "If a facility is not using mSupply, the relevant Team Leader must "
            "manually enter the paper order as a Requisition in mSupply. Where "
            "authorisation is required before orders are sent, a request that "
            "is not attended to within a predetermined period (e.g. 4 days) is "
            "automatically authorised."
        ),
        "reference_page": "39",
    },
    {
        "question": "Which items are deliberately kept on low, easy-to-reach shelving, and which are allowed to go up high?",
        "reference": (
            "Items with short expiry dates should be stored in easily "
            "accessible, low-shelf locations, while bulk items with long "
            "expiry dates may be stored in higher, more inaccessible areas. If "
            "'next to expire' stock is found in a high, inaccessible location, "
            "it should be moved to a low, accessible one."
        ),
        "reference_page": "59",
    },
]


class RetrivalMetrics:
    def __init__(self):
        self.__llm_client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value()
        )
        self.__llm = llm_factory(
            "gpt-4.1-mini", client=self.__llm_client, max_tokens=4096
        )
        self.__retriever = settings.index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.HYBRID, similarity_top_k=3
        )
        self.results = {}
        self.__semaphore = asyncio.Semaphore(2)

    async def context_precision(
        self, question: str, reference: str, retrieved_contexts: list[str]
    ):
        context_precision = ContextPrecision(llm=self.__llm)

        async with self.__semaphore:
            result = await context_precision.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )

        return result.value

    async def context_recall(
        self, question: str, reference: str, retrieved_contexts: list[str]
    ):
        context_recall = ContextRecall(llm=self.__llm)

        async with self.__semaphore:
            result = await context_recall.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )

        return result.value

    async def evaluate(self, question: str, reference: str):
        retrieved_nodes = await self.__retriever.aretrieve(question)
        retrieved_contexts = [node.node.get_content() for node in retrieved_nodes]

        precision, recall = await asyncio.gather(
            self.context_precision(question, reference, retrieved_contexts),
            self.context_recall(question, reference, retrieved_contexts),
        )

        self.results[question] = {
            "context_precision": precision,
            "context_recall": recall,
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

    await tqdm_asyncio.gather(*tasks, desc="Evaluating questions")

    result = await metrics.get_results()

    for key, value in result.items():
        print(f"Question: {key}")
        print(f"Context_precision: {value['context_precision']}")
        print(f"Recall: {value['context_recall']}")


if __name__ == "__main__":
    asyncio.run(main())
