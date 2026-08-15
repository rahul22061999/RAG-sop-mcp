import asyncio

from config import settings
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness
from tools.rag_generator import generate_sop_context
from tqdm.asyncio import tqdm_asyncio

GROUND_TRUTH = [
    {
        "question": "A carton of vaccines arrives with its temperature monitor reading too high. Walk me through exactly what must happen, including any deadlines.",
        "context": (
            "[Chunk 1 - Goods Arrival, p20] Cold chain: Inform the Cold Chain "
            "Coordinator, who must immediately: check the temperature monitor "
            "inside each box, and the quantity and condition of stock against "
            "the invoice, and move the stock to the cool room. For vaccines: "
            "complete a Vaccines Arrival Report (VAR) and email it to the "
            "supplier within 36 hours of receipt. If a temperature monitor in "
            "any carton is too low (below 2°C), the entire carton must be "
            "subject to a Shake Test. If the temperature monitor in any carton "
            "is too high, each individual Vaccine Vial Monitor (VVM) must be "
            "checked. If the secondary test also fails (too high or too low), "
            "that carton must be rejected and a thorough inspection of all "
            "remaining cartons must be carried out.\n\n"
            "[Chunk 2 - Supplier Payments, p37] mSupply Procedure: Supplier "
            "Payments. Navigate to the Suppliers tab. Click the New Payment "
            "button. Enter the Name of the supplier you are paying. View the "
            "outstanding invoice payments in the table and the Total "
            "Outstanding below. Choose the Payment Currency from the dropdown "
            "list, if applicable. Enter the Payment amount and press tab. "
            "Click Distribute to automatically distribute the payment to each "
            "outstanding invoice. Click OK to finalise the payment."
        ),
        "reference_page": "20",
    },
    {
        "question": "Trace the full paper trail from the moment a Goods Receipt is finalised to the moment the supplier's payment is recorded. Name every role involved and what each one does.",
        "context": (
            "[Chunk 1 - Goods Receipt, p29-30] When the Warehouse Manager is "
            "satisfied everything is correct, they finalise the Goods Receipt "
            "in mSupply - this will generate a Supplier Invoice. Warehouse "
            "Manager confirms the Supplier Invoice in mSupply - this brings "
            "the items into stock, allowing them to be issued later. Warehouse "
            "Manager prints the confirmed Supplier Invoice and sends it to the "
            "Warehouse Director who finalises it in mSupply. This prevents "
            "further editing and completes the goods receipt process. "
            "Warehouse Director prints and sends finalised Supplier Invoice to "
            "Contract Management Chief for payment per SOP: Basic Payments.\n\n"
            "[Chunk 2 - Basic Payments, p35] Contract Management Chief double "
            "checks payment amount on Supplier Invoice with information "
            "recorded in other local systems, creates a payment request and "
            "attaches the Supplier Invoice, Purchase Order, contract and other "
            "relevant local documents. These documents are sent to the NMS "
            "President for endorsement (usually a signed cover letter), then "
            "returned to the Contract Management Chief who forwards them to a "
            "Finance Officer. Finance Officer arranges payment, including "
            "obtaining relevant approvals from the Finance Director and "
            "others. When payment is made, Finance Officer informs Contract "
            "Management Chief (including a copy of receipts). Contract "
            "Management Chief records the payment in mSupply."
        ),
        "reference_page": "29-30, 35",
    },
    {
        "question": "I have 30 packs of one batch on the Blue Shelf and I need 5 of them on the Red Shelf. Give me the exact clicks in mSupply, and tell me what the stock screen shows afterwards.",
        "context": (
            "[Chunk 1 - Moving Stock, p62-64] Navigate to the Item tab. Click "
            "the Item List button. Search for the item, then click Find. "
            "Double click the item to be moved. Click the Stock tab to see all "
            "the available stock of that item, separated by batch and "
            "location. Split Stock: If you want to move some stock of the same "
            "batch to a different location, you need to Split the stock. "
            "Click the line to split. Click Split. Enter the Quantity to Split "
            "(i.e. the quantity to move). Enter the New Shelf Location of the "
            "moved stock. In this example, 5 of 30 packs are being moved to "
            "the Red Shelf. Click OK. We can see 5 packs are now stored in the "
            "Red Shelf, while the remaining 25 packs of the same batch are "
            "still in the Blue Shelf. The total quantity is still 30.\n\n"
            "[Chunk 2 - Order Processing, p39] Facilities place their orders "
            "in mSupply according to published schedules: mSupply Desktop = "
            "Internal Order, mSupply Mobile = Supplier Requisition, not using "
            "mSupply = paper order. Internal Orders appear as Requisitions in "
            "mSupply at NMS. Team Leaders monitor incoming Requisitions for "
            "their allocated facilities or programs."
        ),
        "reference_page": "62-64",
    },
    {
        "question": "An entire batch arrived water-damaged and has to be written off. Which mSupply transaction applies, what does each field mean, and what's the alternative if I'd rather count what's left?",
        "context": (
            "[Chunk 1 - Removing or Destroying Stock, p66-68] If you want to "
            "remove stock from the warehouse for destruction or quarantine, "
            "you need to perform either a Negative Inventory Adjustment "
            "(enter the quantity removed) OR Stocktake (enter the quantity "
            "remaining after stock is removed). Negative Inventory Adjustment: "
            "Navigate to the Item tab. Click the Inventory Adjustment - "
            "Reduce Stock button. Select a Category from the dropdown list to "
            "explain why stock is being reduced. Click New Line, double click "
            "the relevant Line, making sure the Batch and Location are "
            "correct, and enter the Quantity of that item line being removed. "
            "Check the Finalise box and click OK. Stocktake: Navigate to the "
            "Item tab, click Stocktakes, click New Stocktake, search for the "
            "item. Review the Snapshot Quantity (what mSupply thinks you "
            "have) and enter the new correct quantity in the Enter Quantity "
            "column. Click Create Inventory Adjustments and Confirm.\n\n"
            "[Chunk 2 - Storage Conditions, p72] To facilitate room "
            "temperature regulation: the main roller doors should only be "
            "opened when receiving or dispatching stock. Keep all external "
            "doors and windows closed unless in use. The air conditioning "
            "system must be regularly maintained, with filters cleaned "
            "monthly."
        ),
        "reference_page": "66-68",
    },
    {
        "question": "A batch failed inspection but the supplier says the problem is minor and easily sorted. Contrast what happens in that case versus when the defect is serious, including who pays and the time limit involved.",
        "context": (
            "[Chunk 1 - Goods Inspection, p24] R&I Committee immediately "
            "notifies the Contract Management Chief, who notifies the company "
            "of the problem (including a copy of the R&I inspection report). "
            "If minor, the issue can be resolved and the company simply "
            "notified. The item is released into the system. If significant "
            "and the item needs to be replaced, the Contract Management Chief "
            "liaises with the company. The item may either be destroyed or "
            "returned; all costs are to be borne by the supplier. Replacement "
            "stock is ordered at no additional charge. If the original "
            "supplier cannot satisfactorily replace the stock, they must "
            "refund any money already paid on the contract, giving a credit "
            "to NMS. Companies are given 30 days to remove the rejected stock "
            "and replace it. If removed before 30 days, no further "
            "warehousing fees should be charged.\n\n"
            "[Chunk 2 - Warehouse Storage, p59] Items should be stored on "
            "shelves. If absolutely necessary, only waterproof, non-perishable "
            "items may be stored on the floor. Items with short expiry dates "
            "should be stored in easily accessible, low-shelf locations. "
            "Fragile items should be moved by hand where possible."
        ),
        "reference_page": "24",
    },
    {
        "question": "Describe a Team Leader's daily spot-check stocktake routine from choosing the item to filing the sheet - including the check they must do before starting and what to do about a big discrepancy.",
        "context": (
            "[Chunk 1 - Spot-Check Stocktake, p81-82] Each day, each Team "
            "Leader conducts a stocktake on one item. The Warehouse Manager "
            "coordinates and allocates the items to be stocktaked. Before "
            "commencing, the Team Leader checks if there are any outstanding "
            "Customer Invoices on that item. If there are none, the stocktake "
            "can proceed; if one is nearly complete they may help complete "
            "and confirm it; if there are several outstanding invoices they "
            "may choose another item. Team Leader generates and prints a "
            "stocktake for that item in mSupply, and gives it to Warehouse "
            "Staff, who perform the stocktake. If Warehouse Staff notice a "
            "large discrepancy (e.g. missing stock or batch), they should "
            "investigate and try to find it - asking all staff members or "
            "doing a visual search; details are communicated to the Team "
            "Leader for updating in mSupply. When complete, Warehouse Staff "
            "return the sheet to the Team Leader who enters the data in "
            "mSupply, then gives the sheet to the Warehouse Manager for "
            "filing.\n\n"
            "[Chunk 2 - Cold Chain, p73] Store cold chain items in "
            "fit-for-purpose cool rooms. Refrigerators that open on the top "
            "are more efficient than vertical ones, because hot air rises "
            "while cold air falls. Always have enough frozen icepacks to "
            "transport cold chain items in cold boxes."
        ),
        "reference_page": "81-82",
    },
    {
        "question": "The warehouse is closing for the annual full stocktake. List everything that must be finished beforehand and every rule in force while counting is underway.",
        "context": (
            "[Chunk 1 - Stocktake Requirements, p79] Before the stocktake: "
            "every pending order must be processed; every outstanding Goods "
            "Receipt must be processed; every order awaiting dispatch must be "
            "processed and excluded from the stocktake. During the stocktake: "
            "no new Customer Invoices can be created; any items processed but "
            "not dispatched before the stocktake must be clearly set aside "
            "and not counted - in addition, those Customer Invoices should be "
            "confirmed; no stock is to be moved - expired items or stock in "
            "the wrong location should be counted, and then moved or "
            "destroyed at the end of the stocktake; any item not on the list "
            "but on the shelf must be recorded.\n\n"
            "[Chunk 2 - Adding Unordered Stock, p69] mSupply Procedure: "
            "Adding Unordered Stock (e.g. Donations or Samples). Navigate to "
            "the Suppliers tab. Click the New Supplier Invoice button. Enter "
            "the Supplier. Click New Line to start entering received items. "
            "Enter Number of Packs and Pack Size received."
        ),
        "reference_page": "79",
    },
    {
        "question": "Explain how the monthly audit's 20-item sample is put together, what counts as a 'correct' item, and why this method can still miss missing stock.",
        "context": (
            "[Chunk 1 - Warehouse Audit, p82] In the middle of each month, a "
            "warehouse audit should be undertaken to determine the accuracy "
            "of the data in mSupply. This is carried out by 2 non-warehouse "
            "staff from the National Pharmacy Division. A stocktake is "
            "generated with 20 randomly selected items (medicines and "
            "consumables) in mSupply and printed. For items with multiple "
            "locations, one location may be selected for the purpose of the "
            "audit, so that a total of 20 items/locations are checked. If the "
            "item location, batch, expiry and quantity are correct, this item "
            "is marked as correct. % Accuracy = (Number of correct items / "
            "Total number of items checked (20)) x 100%. Spot-check "
            "stocktakes may be difficult, as they do not necessarily capture "
            "'missing stock' that has not been recorded properly and is in an "
            "unknown location. This stock is captured in a full stocktake, as "
            "eventually someone will come to count that location and 'find' "
            "the missing stock.\n\n"
            "[Chunk 2 - Goods Receipt editing, p31] To edit an item's batch, "
            "expiry date or location, double click the line to open the item "
            "details. Click OK & Next to move to the next item, or click OK "
            "to return to the main Goods Receiving screen."
        ),
        "reference_page": "82",
    },
    {
        "question": "A donated pallet arrives from an organisation that has never supplied us before. Cover both halves of the process: getting the donor into mSupply and getting the stock on the books.",
        "context": (
            "[Chunk 1 - Adding Unordered Stock, p69-70] Navigate to the "
            "Suppliers tab. Click the New Supplier Invoice button. Enter the "
            "Supplier. Click New Line to start entering received items. Enter "
            "the received Item details, Number of Packs and Pack Size "
            "received (the Total Quantity will be automatically calculated). "
            "Carefully enter the Batch, Expiry, Location and other details. "
            "Click OK & Next to keep entering more received items; when "
            "finished, check the Finalise box and click OK. If you are "
            "receiving stock from someone who isn't a regular supplier (e.g. "
            "an organisation donating stock), you will need to add them as a "
            "new Supplier in mSupply first: Navigate to the Suppliers tab, "
            "click the New Supplier button, and enter the details. The "
            "minimum details to enter are the Code, Charge To (automatically "
            "completes when you enter the code) and Name. When you are "
            "finished, click OK. Now you can receive stock from this "
            "Supplier.\n\n"
            "[Chunk 2 - Full Stocktake, p80] Warehouse Manager prepares and "
            "prints stocktake sheets at close of business on the working day "
            "before commencement. To ensure stock is not counted twice, each "
            "location should only be printed once."
        ),
        "reference_page": "69-70",
    },
    {
        "question": "What financial penalty does the SOP set for suppliers who deliver goods later than the agreed date?",
        "context": (
            "[Chunk 1 - Basic Payments, p35-36] Contract Management Chief "
            "double checks payment amount on Supplier Invoice with "
            "information recorded in other local systems. Contract Management "
            "Chief creates a payment request according to local procedures, "
            "and attaches the Supplier Invoice, Purchase Order, contract and "
            "other relevant local documents. These documents are sent to the "
            "NMS President for endorsement. Finance Officer arranges payment "
            "according to local procedures. When payment is made, Finance "
            "Officer informs Contract Management Chief. Contract Management "
            "Chief records the payment in mSupply. Performance Indicators: % "
            "Annual budget executed; Total value of invoices received; Total "
            "value of invoices paid.\n\n"
            "[Chunk 2 - Supplier Payments, p37] Navigate to the Suppliers "
            "tab. Click the New Payment button. Enter the Name of the "
            "supplier you are paying. View the outstanding invoice payments "
            "in the table and the Total Outstanding below. Enter the Payment "
            "amount and press tab. Click Distribute. Click OK to finalise "
            "the payment."
        ),
        "reference_page": "35-37 (trap: not specified in context)",
    },
]


class RAGGenerationEvaluation:

    def __init__(self):
        self.__llm_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        self.__llm = llm_factory("gpt-4.1-mini", client=self.__llm_client, max_tokens=4096)
        self.__embeddings = embedding_factory(
            provider="openai", model="text-embedding-3-small", client=self.__llm_client
        )
        self.results = {}
        self.__semaphore = asyncio.Semaphore(8)
        self.__ollama_semaphore = asyncio.Semaphore(1)


    async def context_faithfulness(self, question: str, llm_response: str, retrieved_contexts: list[str]) -> float:
        context_faithfulness = Faithfulness(llm=self.__llm)

        async with self.__semaphore:
           result = await context_faithfulness.ascore(
               user_input=question,
               response=llm_response,
               retrieved_contexts=retrieved_contexts
           )

           return result.value

    async def context_relevance(self, question: str, llm_response: str) -> float:
        context_relevance = AnswerRelevancy(llm=self.__llm, embeddings=self.__embeddings)

        async with self.__semaphore:
           result = await context_relevance.ascore(
               user_input=question,
               response=llm_response,
           )

           return result.value


    async def evaluate(
            self,
            question: str,
            context: str
        ):

        async with self.__ollama_semaphore:
            generated = await generate_sop_context(question, [context])

        llm_response = generated.answer

        faithfulness, relevance = await asyncio.gather(
            self.context_faithfulness(question, llm_response, [context]),
            self.context_relevance(question, llm_response)
        )

        self.results[question] = {
            "answer": llm_response,
            "context_faithfulness": faithfulness,
            "context_relevance": relevance,
        }

    async def get_results(self):
        return  self.results



async def main():
    metrics = RAGGenerationEvaluation()

    tasks = [
        metrics.evaluate(
            question=data["question"],
            context=data["context"],
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
        print(f"Answer: {value['answer']}")
        print(f"Faithfulness: {value['context_faithfulness']}")
        print(f"Relevance: {value['context_relevance']}")
        print()




if __name__ == "__main__":

    asyncio.run(main())