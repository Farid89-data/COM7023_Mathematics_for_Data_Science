# COM7023 Mathematics for Data Science - AI Agent Instructions

## Project Overview
This is an academic portfolio demonstrating mathematical concepts in data science. Each topic (01-08) implements analysis on two datasets: marking matrix (text-based) and triplet births (numerical demographic data).

## Key Patterns & Conventions

### Script Structure
All Python scripts follow this standardized format:
- Detailed docstring header with student info, module details, mathematical concepts, dataset reference
- Section headers using `===` for major sections, `---` for subsections
- Imports: `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`
- Configuration: `pd.set_option('display.max_columns', None)`, `plt.style.use('seaborn-v0_8-whitegrid')`
- 7-step workflow: Load → Structure → Preview → Quality → Summary → Visualize → Findings
- Save plots to `../outputs/figures/` with descriptive names
- Console output with formatted print statements

### File Paths & Data Flow
- Datasets: `../datasets/` (relative from topic folders)
- Outputs: `../outputs/figures/` (plots saved as PNG)
- Marking matrix scripts: Focus on text analysis (lengths, word counts)
- Triplet births scripts: Focus on numerical statistics and distributions

### Dependencies
Install from `requirements.txt`: pandas≥1.5.0, numpy≥1.23.0, matplotlib≥3.6.0, seaborn≥0.12.0, openpyxl≥3.0.0

### Running Scripts
```bash
cd [topic_folder]
python [script_name].py
```
Scripts are self-contained with error handling for missing files.

### Code Style Examples
- Use `try/except` for file loading with informative error messages
- Text cleaning: `' '.join(text.split())` for normalization
- Visualization: `plt.savefig('../outputs/figures/[filename].png', dpi=150, bbox_inches='tight')`
- Statistical summaries: Include both numerical results and interpretations

### Key Reference Files
- `01_data_loading_and_exploration/data_loading_marking_matrix.py`: Exemplifies complete script structure
- `README.md`: Project overview and setup instructions
- `requirements.txt`: Exact dependency versions</content>
<parameter name="filePath">/home/codespace/COM7023_Mathematics_for_Data_Science/.github/copilot-instructions.md