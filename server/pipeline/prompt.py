PAGE_TO_MARKDOWN_PROMPT = """
Convert this PDF page into faithful Markdown for a warehouse SOP.

Preserve the original reading order, headings, numbered steps, nested lists,
and exact wording. Recreate tables with every readable column and row in the
correct position. Include relevant text inside screenshots, forms, and diagrams,
especially UI labels, highlighted controls, names, numbers, and dates.

Do not invent missing text, values, or table cells. Write [unreadable] wherever
the image is too unclear to transcribe. Do not repeat text that appears both
in the procedure and a screenshot.

Output only Markdown.
""".strip()
