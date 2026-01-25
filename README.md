# COM7023 Mathematics for Data Science

## Academic Portfolio

This repository contains the complete mathematical analysis portfolio for the COM7023 Mathematics for Data Science module. The portfolio demonstrates advanced understanding of mathematical concepts and their practical application in data science contexts.

---

## Student Information

| Field | Details |
|-------|---------|
| **Module Title** | Maths for Data Science |
| **Module Code** | COM7023 |
| **Assignment Title** | Maths for Data Science |
| **Student Number** | 24154844 |
| **Student Name** | Farid Negahbnai |
| **Tutor Name** | Ali Vaisifard |
| **University** | Arden University |

---

## Project Overview

This portfolio applies mathematical principles to two distinct datasets, demonstrating versatility in data science applications. Each mathematical concept is implemented twice: once using the course marking matrix dataset and once using real-world demographic data on multiple births.

### Datasets Used

**Dataset 1: Course Marking Matrix**

The marking matrix provides assessment criteria for the COM7023 module across different grade boundaries. This structured dataset allows demonstration of text processing, categorical analysis, and matrix operations.

Reference: Arden University (2024) *COM7023 Mathematics for Data Science Marking Matrix*. Arden University.

**Dataset 2: Human Multiple Births Database**

This dataset contains demographic information about multiple births (twins, triplets) across multiple countries and years. The French subset is used for probability distribution analysis and statistical modelling.

Reference: Human Multiple Births Database (2024) *FRA_InputData_25.11.2024.xlsx*. Available at: https://www.twinbirths.org/en/data-metadata/ (Accessed: 25 November 2024).

---

## Mathematical Topics Covered

The portfolio addresses the following mathematical concepts as outlined in the module specification:

1. **Data Loading and Exploration**: Understanding data structures and initial analysis
2. **Descriptive Statistics**: Measures of central tendency and dispersion
3. **Normalisation and Standardisation**: Data transformation techniques
4. **Probability Distributions**: Discrete and continuous probability models
5. **Linear Algebra**: Matrix operations and vector spaces
6. **Calculus and Optimisation**: Derivatives, integrals, and optimisation problems
7. **Correlation and Regression**: Statistical relationships and predictive modelling
8. **Hypothesis Testing**: Statistical inference and decision making

---

## Installation and Setup

### Prerequisites

Ensure Python 3.8 or higher is installed on your system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Farid89-data/COM7023_Mathematics_for_Data_Science.git
cd COM7023_Mathematics_for_Data_Science

COM7023_Mathematics_for_Data_Science/
│
├── README.md                                    # Main documentation
├── requirements.txt                             # Python dependencies
├── LICENSE                                      # MIT License
│
├── datasets/
│   ├── COM7023_Mathematics_for_Data_Science_Marking_Matrix.csv
│   └── FRA_InputData_25.11.2024.xlsx
│
├── 01_data_loading_and_exploration/
│   ├── data_loading_marking_matrix.py
│   ├── data_loading_triplet_births.py
│   └── README.md
│
├── 02_descriptive_statistics/
│   ├── descriptive_statistics_marking_matrix.py
│   ├── descriptive_statistics_triplet_births.py
│   └── README.md
│
├── 03_normalisation_and_standardisation/
│   ├── normalisation_marking_matrix.py
│   ├── normalisation_triplet_births.py
│   └── README.md
│
├── 04_probability_distributions/
│   ├── probability_distributions_marking_matrix.py
│   ├── poisson_distribution_triplet_births.py
│   └── README.md
│
├── 05_linear_algebra/
│   ├── linear_algebra_marking_matrix.py
│   ├── linear_algebra_triplet_births.py
│   └── README.md
│
├── 06_calculus_and_optimisation/
│   ├── calculus_marking_matrix.py
│   ├── calculus_triplet_births.py
│   └── README.md
│
├── 07_correlation_and_regression/
│   ├── correlation_regression_marking_matrix.py
│   ├── correlation_regression_triplet_births.py
│   └── README.md
│
├── 08_hypothesis_testing/
│   ├── hypothesis_testing_marking_matrix.py
│   ├── hypothesis_testing_triplet_births.py
│   └── README.md
│
└── outputs/
    ├──results/
    └── figures/
```
### Step 2: Create Virtual Environment

```bash
python -m venv venv
x
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

```
### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```
### Step 4: Add Datasets
Place the following files in the `datasets/` directory:

`COM7023_Mathematics_for_Data_Science_Marking_Matrix.csv`
`FRA_InputData_25.11.2024.xlsx` (download from twinbirths.org)

### Step 5: Run Scripts
Navigate to any topic folder and run the Python scripts:

```bash
cd 01_data_loading_and_exploration
python data_loading_marking_matrix.py
python data_loading_triplet_births.py
```
#### Dependencies
This project uses only the following libraries as specified in the module requirements:

**pandas:** Data manipulation and analysis

**numpy:** Numerical computing 

**matplotlib:** Data visualisation

**seaborn:** Statistical visualisation

