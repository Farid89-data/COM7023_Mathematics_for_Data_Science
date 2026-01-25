# COM7023 Mathematics for Data Science

<div align="center">

Python
Pandas
NumPy
Matplotlib
License

A comprehensive mathematical analysis portfolio for data science applications

Overview •
Installation •
Usage •
Topics •
Datasets •
License

</div>

## 📋 Overview
This repository contains the complete mathematical analysis portfolio for the **COM7023 Mathematics for Data Science** module at Arden University. The portfolio demonstrates advanced understanding of mathematical concepts and their practical application in real-world data science contexts.

Each mathematical concept is implemented **twice** using different datasets, demonstrating versatility and deep understanding of the underlying principles.

## 🎓 Student Information

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

## 🚀 Installation
* Prerequisites
* Python 3.8 or higher
* pip (Python package installer)
* Git

**Quick Start**
```bash
# Clone the repository
git clone https://github.com/Farid89-data/COM7023_Mathematics_for_Data_Science.git

# Navigate to project directory
cd COM7023_Mathematics_for_Data_Science

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
**Verify Installation**
```bash
python --version    # Should show Python 3.8+
pip list           # Should show pandas, numpy, matplotlib, seaborn
```
## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|--------|
|pandas|	≥1.5.0	|Data manipulation and analysis|
|numpy|	≥1.21.0|	Numerical computing and array operations|
|matplotlib|	≥3.5.0	|Static data visualisation|
|seaborn|	≥0.11.0	|Statistical data visualisation|

Install all dependencies with:
```bash
pip install -r requirements.txt
```
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
├── 📄 README.md                          # Project documentation
├── 📄 requirements.txt                   # Python dependencies
├── 📄 LICENSE                            # MIT License
│
├── 📂 datasets/
│   ├── COM7023_Mathematics_for_Data_Science_Marking_Matrix.csv
│   └── FRA_InputData_25.11.2024.xlsx
│
├── 📂 01_data_loading_and_exploration/
│   ├── data_loading_marking_matrix.py
│   ├── data_loading_triplet_births.py
│   └── README.md
│
├── 📂 02_descriptive_statistics/
│   ├── descriptive_statistics_marking_matrix.py
│   ├── descriptive_statistics_triplet_births.py
│   └── README.md
│
├── 📂 03_normalisation_and_standardisation/
│   ├── normalisation_marking_matrix.py
│   ├── normalisation_triplet_births.py
│   └── README.md
│
├── 📂 04_probability_distributions/
│   ├── probability_distributions_marking_matrix.py
│   ├── poisson_distribution_triplet_births.py
│   └── README.md
│
├── 📂 05_linear_algebra/
│   ├── linear_algebra_marking_matrix.py
│   ├── linear_algebra_triplet_births.py
│   └── README.md
│
├── 📂 06_calculus_and_optimisation/
│   ├── calculus_marking_matrix.py
│   ├── calculus_triplet_births.py
│   └── README.md
│
├── 📂 07_correlation_and_regression/
│   ├── correlation_regression_marking_matrix.py
│   ├── correlation_regression_triplet_births.py
│   └── README.md
│
├── 📂 08_hypothesis_testing/
│   ├── hypothesis_testing_marking_matrix.py
│   ├── hypothesis_testing_triplet_births.py
│   └── README.md
│
└── 📂 outputs/
    ├── 📂 results/                       # Numerical outputs and tables
    └── 📂 figures/                       # Generated visualisations
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
## 💻 Usage
**Running Individual Scripts**
Navigate to any topic folder and execute the Python scripts:
```bash
# Example: Run data loading scripts
cd 01_data_loading_and_exploration
python data_loading_marking_matrix.py
python data_loading_triplet_births.py
```
**Running All Scripts**
```bash
# Run all scripts for a specific topic
cd 02_descriptive_statistics
python descriptive_statistics_marking_matrix.py
python descriptive_statistics_triplet_births.py
```
Expected Output
Each script generates:

* **Console output:** Statistical summaries and analysis results
* **Figures:** Saved to `outputs/figures/` directory
* **Results:** Saved to `outputs/results/` directory


📊 Mathematical Topics
#	Topic	Description	Key Concepts
01	Data Loading & Exploration	Initial data analysis and structure understanding	pandas.read_csv(), DataFrame.info(), DataFrame.describe()
02	Descriptive Statistics	Measures of central tendency and dispersion	Mean, median, mode, variance, standard deviation
03	Normalisation & Standardisation	Data transformation techniques	Min-max scaling, Z-score normalisation
04	Probability Distributions	Discrete and continuous probability models	Binomial, Poisson, Normal distributions
05	Linear Algebra	Matrix operations and vector spaces	Matrix multiplication, determinants, eigenvalues
06	Calculus & Optimisation	Derivatives and optimisation problems	Gradient descent, critical points, integrals
07	Correlation & Regression	Statistical relationships and prediction	Pearson correlation, linear regression, R²
08	Hypothesis Testing	Statistical inference and decision making	t-tests, p-values, confidence intervals
## 📈 Datasets
**Dataset 1: Course Marking Matrix**
Assessment criteria for the COM7023 module across different grade boundaries. Used for text processing, categorical analysis, and matrix operations.

|Property	|Value|
|-----------|------|
|Format|	CSV|
|Source|	Arden University|
|Purpose|	Categorical analysis, matrix operations|

**Reference:**

> Arden University (2024). COM7023 Mathematics for Data Science Marking Matrix. Arden University.

**Dataset 2: Human Multiple Births Database (HMBD)**
Demographic information about multiple births (twins, triplets) across multiple countries and years. The French subset is used for probability distribution analysis and statistical modelling.

|Property|	Value|
|--------|-------|
|Format|	XLSX|
|Source|	Human Multiple Births Database|
|Country|	France|
|Purpose|	Statistical modelling, probability distributions|

**Available Variables in HMBD:**

|Variable|	Description|
|--------|-------------|
|Country|	Country name|
|Year	|Year of reference|
|Twin_deliveries|	Number of twin deliveries|
|Triplet_deliveries|	Number of triplet deliveries|
|Multiple_deliveries|	Total number of multiple deliveries|
|Total_deliveries|	Total number of deliveries|
|Twinning_rate|	Twin deliveries per 1,000 total deliveries|
|Multiple_rate|	Multiple deliveries per 1,000 total deliveries|

**Reference:**
> Torres, C., Caporali, A., Pison, G. (2023). The Human Multiple Births Database (HMBD): An international database on twin and other multiple births. Demographic Research, 48(4): 89-106. DOI: 10.4054/DemRes.2023.48.4

**Data Source:**
> Human Multiple Births Database (2024). FRA_InputData_25.11.2024.xlsx. Available at: https://www.twinbirths.org/en/data-metadata/ (Accessed: 25 November 2024).

