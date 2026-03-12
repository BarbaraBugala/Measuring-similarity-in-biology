# Measuring similarity in biology

This project aims to answer two research questions:
- Do latent representations of machine learning models capture known biological patterns?
- Does the capture of known biological patterns improve model performance? Is it correlated?


## Protein Families Used

### Insulin
Insulin is a small peptide hormone that regulates blood glucose levels by binding to the insulin receptor.  
It is **highly conserved across vertebrates** due to strong structural constraints, including conserved cysteine residues that form disulfide bonds.  

Sequences used in this dataset have lengths **50–200 amino acids**.

### Hemoglobin
Hemoglobin is an **oxygen-transport protein** found in red blood cells. It consists of subunits containing heme groups that bind oxygen molecules.

Although the overall structure is conserved, sequence variation occurs across species, resulting in **moderate evolutionary diversity**.  

Sequences used here have lengths **130–160 amino acids**.

### Protein Kinases
Protein kinases are enzymes that regulate cellular processes by **phosphorylating target proteins**.  
They share a conserved catalytic kinase domain but belong to a **large and diverse protein family** involved in many signaling pathways.

Because of this diversity, kinase sequences exhibit **substantial evolutionary variation**.

## Project Structure

```text
.
├── data/                       # Raw and processed datasets
├── experiments/                # Scripts for various metric calculations
├── models/                     # Model-related code and embeddings
├── plots/                      # Visualization scripts and saved figures
├── results/                    # Output data and experimental results
