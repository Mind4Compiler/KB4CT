# KB4CT: Knowledge Base for Compiler Tuning

## Project Overview

KB4CT is a knowledge-based compiler tuning system that optimizes LLVM pass sequences through a combination of offline empirical prototype discovery and an online knowledge-guided personalized evolutionary algorithm. The system operates in two main stages: Offline Knowledge Base Construction and Online Knowledge-Guided Personalized Optimization.

## Dataset Preparation

### Extract Dataset from Supplementary Material

Extract the LLVM IR datasets from the supplementary material.
Place the extracted datasets in the project root directory, ensuring the structure is as follows:

```
KB4CT/
├── dataset/
│   ├── train/          
│   │   ├── dataset1/
│   │   ├── dataset2/
│   │   └── ...
│   └── test/           
│       ├── dataset1/
│       ├── dataset2/
│       └── ...
│── DLL
├── LLVMEnv/
├── llvm_tools/
├── output/
└── KB4CT.py
```


## Running Instructions

### Basic Execution

```bash
python KB4CT.py
```

### Configuration

You can configure the parameters in the if __name__ == '__main__': section of KB4CT.py.

#### Offline GA Parameters 

```python
"offline_ga_params": {
    "seq_len": 100,           # Sequence length
    "population_size": 100,   # Population size
    "generations": 30,        # Number of generations
    "elite_size": 20,         # Number of elite individuals
    "crossover_rate": 0.8,    # Crossover probability
    "mutation_rate": 0.8      # Mutation probability
}
```

#### Online GA Parameters

```python
"online_ga_params": {
    "population_size": 50,     # Population size
    "generations": 5,          # Number of generations
    "elite_size": 10,          # Number of elite individuals
    "crossover_rate": 0.8,     # Crossover probability
    "mutation_rate": 0.99      # Mutation probability
}
```

## Output Results

After execution, the results will be saved in the `output/` directory:

- `pass_embeddings_visualization.png`: Visualization of pass embeddings
- `pass_clusters_visualization.png`: Visualization of pass clusters
- `ablation_study_*.png`: Ablation study result figures
- `ablation_study_report.txt`: Ablation study report
- `ablation_detailed_results.json`: Detailed result data

## Ablation Study Modes

The system supports the following ablation study modes:

1. **full**: Full knowledge-guided method
2. **no_knowledge_crossover**: No knowledge-guided crossover
3. **no_knowledge_mutation**: No knowledge-guided mutation
4. **random_init**: Random population initialization
5. **no_knowledge**: Standard GA without any knowledge guidance
