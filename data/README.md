# Dataset

The experiment uses the **Customer Support Tickets** dataset by Tobi Bueck.

Dataset identifier / DOI: `10.57967/hf/6184`

The source dataset contains support-ticket text and metadata, including subject, body, answer, ticket type, queue, priority, language, version, and tags.

## File used

`aa_dataset-tickets-multi-lang-5-2-50-version.csv`

The audited experiment used **28,587 records** and **16 columns**.

- Languages: English and German
- Ticket types: Incident, Request, Problem, Change
- Queue classes: 10
- Priority classes used in the experiment: 3 (Low, Medium, High)
- Exact duplicate rows: 0
- Missing subject values: 3,838 (13.43%)
- Missing body values: 0

## Download / attribution

Obtain the dataset from its Tobi Bueck / Hugging Face dataset source using the identifier above, subject to the dataset's stated license and terms.

The raw CSV is **not stored in this GitHub repository**. After obtaining it, place it at the repository root so the experiment scripts can find:

```text
aa_dataset-tickets-multi-lang-5-2-50-version.csv
```

This keeps the repository focused on the implementation and avoids committing the full dataset.
