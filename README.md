# Generating new drug brand names with neural networks

## Steps

1. Downloaded a [database of drugs](https://www.fda.gov/downloads/Drugs/InformationOnDrugs/UCM527389.zip) from the [FDA's website](https://www.fda.gov/drugs/informationondrugs/ucm079750.htm)
2. Massaged data to:
    1. Remove generic names (if DrugName !== ActiveIngredient)
    2. Remove duplicate names