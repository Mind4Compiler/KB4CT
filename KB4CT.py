import os
import json
import random
from pathlib import Path
from enum import Enum
from tqdm import tqdm
import concurrent.futures
from collections import defaultdict
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
import networkx as nx

# --- Assumed Imports ---
from LLVMEnv.actionspace.llvm10_0_0.actions import Actions_LLVM_10_0_0
from LLVMEnv.common import get_instrcount
from LLVMEnv.obsUtility.Autophase import get_autophase_obs
from LLVMEnv.actionspace.IRinstcount_pairs import synergy_pairs_dict
# --- End Imports ---

# ==============================================================================
# Ablation Experiment Tracker
# ==============================================================================
class AblationTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sample_count = 0
        self.fitness_history = []
        self.generation_history = []
        self.best_fitness_history = []
    
    def log_evaluation(self, fitness_scores):
        """Logs fitness evaluation for each sample"""
        self.sample_count += len(fitness_scores)
        best_fitness = max(fitness_scores) if fitness_scores else 0.0
        self.fitness_history.extend(fitness_scores)
        self.best_fitness_history.append(best_fitness)
    
    def log_generation(self, generation, best_fitness):
        """Logs best fitness for each generation"""
        self.generation_history.append((generation, self.sample_count, best_fitness))

# ==============================================================================
# Phase One: Offline "Ignorant" Empirical Prototype Finder
# ==============================================================================
class EmpiricalPrototypeFinder:
    def __init__(self, all_passes_list, llvm_tools_path, num_workers=None):
        self.all_passes_list = all_passes_list
        self.llvm_tools_path = llvm_tools_path
        self.num_workers = num_workers or os.cpu_count()

    def _initialize_population(self, pop_size, seq_len):
        """Initializes the population"""
        population = []
        for _ in range(pop_size):
            individual = random.sample(self.all_passes_list, min(seq_len, len(self.all_passes_list)))
            population.append(individual)
        return population

    def _calculate_fitness_on_cluster(self, population, programs_in_cluster):
        """Calculates fitness on a cluster"""
        total_fitness_scores = np.zeros(len(population))
        valid_programs = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_prog = {executor.submit(self._evaluate_population_on_one_program, prog_path, population): prog_path for prog_path in programs_in_cluster}
            for future in concurrent.futures.as_completed(future_to_prog):
                scores = future.result()
                if scores is not None:
                    total_fitness_scores += np.array(scores)
                    valid_programs += 1
        return (total_fitness_scores / valid_programs).tolist() if valid_programs > 0 else [0.0] * len(population)

    def _evaluate_population_on_one_program(self, prog_path, population):
        """Evaluates individual fitness on a single program"""
        try:
            ll_code = prog_path.read_text()
            size_oz = get_instrcount(ll_code, ["-Oz"], llvm_tools_path=self.llvm_tools_path)
            if size_oz == 0: return None
            scores = [(size_oz - get_instrcount(ll_code, ind, llvm_tools_path=self.llvm_tools_path)) / size_oz for ind in population]
            return scores
        except Exception: return None

    def _selection(self, population, fitness_scores, elite_size):
        """Selection"""
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elites = [population[i] for i in elite_indices]
        selected = list(elites)
        for _ in range(len(population) - elite_size):
            tournament = random.sample(list(enumerate(fitness_scores)), k=3)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(population[winner[0]])
        return selected

    def _crossover(self, parent1, parent2, crossover_rate):
        """Crossover"""
        if random.random() > crossover_rate: return list(parent1), list(parent2)
        size = min(len(parent1), len(parent2))
        if size < 2: return list(parent1), list(parent2)
        cx_point = random.randint(1, size - 1)
        c1, c2 = parent1[:cx_point] + parent2[cx_point:], parent2[:cx_point] + parent1[cx_point:]
        return list(dict.fromkeys(c1)), list(dict.fromkeys(c2)) # Remove duplicates

    def _mutation(self, individual, mutation_rate):
        """Mutation"""
        if random.random() > mutation_rate: return individual
        mutated_individual = list(individual)
        if not mutated_individual: return []
        idx = random.randint(0, len(mutated_individual) - 1)
        mutated_individual[idx] = random.choice(self.all_passes_list)
        return list(dict.fromkeys(mutated_individual)) # Remove duplicates

    def find_prototype(self, programs_in_cluster, **ga_params):
        """Finds the prototype"""
        pop_size, gens, seq_len = ga_params.get('population_size'), ga_params.get('generations'), ga_params.get('seq_len')
        elite_size, cross_rate, mut_rate = ga_params.get('elite_size'), ga_params.get('crossover_rate'), ga_params.get('mutation_rate')
        population = self._initialize_population(pop_size, seq_len)
        best_overall = (None, -float('inf'))
        for gen in range(gens):
            fitness = self._calculate_fitness_on_cluster(population, programs_in_cluster)
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_overall[1]:
                best_overall = (list(population[best_idx]), fitness[best_idx])
            selected = self._selection(population, fitness, elite_size)
            next_pop = list(selected[:elite_size])
            while len(next_pop) < pop_size:
                p1, p2 = random.sample(selected, 2)
                c1, c2 = self._crossover(p1, p2, cross_rate)
                next_pop.append(self._mutation(c1, mut_rate))
                if len(next_pop) < pop_size: next_pop.append(self._mutation(c2, mut_rate))
            population = next_pop
        return best_overall[0]

# ==============================================================================
# Phase Two: Online Knowledge-Guided Personalization Evolver (Supports Ablation Studies)
# ==============================================================================
import itertools

class KnowledgeGuidedEvolver:
    def __init__(self, prototype_library, pass_embeddings, pass_clusters, pass_graph, llvm_tools_path, num_workers=None):
        self.prototype_library = prototype_library
        self.pass_embeddings = pass_embeddings
        self.pass_clusters = pass_clusters
        self.pass_graph = pass_graph
        self.llvm_tools_path = llvm_tools_path
        self.num_workers = num_workers or os.cpu_count()
        self.pass_to_cluster_map = {p: cid for cid, passes in pass_clusters.items() for p in passes}
        self.tracker = AblationTracker()

    def _initialize_population(self, ll_code, top_k, use_prototypes=True):
        """Initializes the population, supporting ablation experiments"""
        if not use_prototypes:
            # Ablation: Random Initialization
            population = []
            for _ in range(top_k):
                seq_len = random.randint(3, 10)
                individual = random.sample(list(self.pass_embeddings.keys()), min(seq_len, len(self.pass_embeddings)))
                population.append(individual)
            return population
        
        # Original: Prototype-based Initialization
        size_oz = get_instrcount(ll_code, ["-Oz"], llvm_tools_path=self.llvm_tools_path)
        evaluated_prototypes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_proto = {executor.submit(get_instrcount, ll_code, proto, llvm_tools_path=self.llvm_tools_path): proto for proto in self.prototype_library.values()}
            for future in concurrent.futures.as_completed(future_to_proto):
                proto = future_to_proto[future]
                try:
                    size_seq = future.result()
                    perf = (size_oz - size_seq) / size_oz if size_oz > 0 else -float('inf')
                    evaluated_prototypes.append((proto, perf))
                except Exception:
                    evaluated_prototypes.append((proto, -float('inf')))

        evaluated_prototypes.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in evaluated_prototypes[:top_k]]

    def _calculate_fitness(self, population, ll_code, size_oz):
        fitness_scores = [0.0] * len(population)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_idx = {executor.submit(get_instrcount, ll_code, ind, llvm_tools_path=self.llvm_tools_path): i for i, ind in enumerate(population)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    size_seq = future.result()
                    fitness_scores[idx] = (size_oz - size_seq) / size_oz if size_oz > 0 else -float('inf')
                except Exception as e:
                    fitness_scores[idx] = -float('inf')
        
        # Log evaluation
        self.tracker.log_evaluation(fitness_scores)
        return fitness_scores

    def _selection(self, population, fitness_scores, elite_size):
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elites = [population[i] for i in elite_indices]
        selected = list(elites)
        for _ in range(len(population) - elite_size):
            tournament = random.sample(list(enumerate(fitness_scores)), k=3)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(population[winner[0]])
        return selected

    def _to_blocks(self, seq):
        if not seq: return []
        return [list(g) for k, g in itertools.groupby(seq, key=lambda p: self.pass_to_cluster_map.get(p))]

    def _crossover(self, p1, p2, prog_cluster_id, cross_rate, use_knowledge=True):
        """Crossover operation supporting ablation experiments"""
        if random.random() > cross_rate: 
            return list(p1), list(p2)
        
        if not use_knowledge:
            # Ablation: Simple Single-point Crossover
            min_len = min(len(p1), len(p2))
            if min_len < 2:
                return list(p1), list(p2)
            
            cx_point = random.randint(1, min_len - 1)
            c1 = p1[:cx_point] + p2[cx_point:]
            c2 = p2[:cx_point] + p1[cx_point:]
            return list(dict.fromkeys(c1)), list(dict.fromkeys(c2))
        
        # Original: Knowledge-Guided Crossover
        p1_blocks, p2_blocks = self._to_blocks(p1), self._to_blocks(p2)
        if not p1_blocks or not p2_blocks: return list(p1), list(p2)

        min_len = min(len(p1_blocks), len(p2_blocks))
        child_blocks = []

        for i in range(min_len):
            b1, b2 = p1_blocks[i], p2_blocks[i]
            
            s1, s2 = 0.0, 0.0
            for p in b1:
                embedding = self.pass_embeddings.get(p)
                if embedding and 0 <= prog_cluster_id < len(embedding):
                    s1 += embedding[prog_cluster_id]
            for p in b2:
                embedding = self.pass_embeddings.get(p)
                if embedding and 0 <= prog_cluster_id < len(embedding):
                    s2 += embedding[prog_cluster_id]
            
            norm_s1 = max(s1 + 100, 1) # Add a constant to avoid division by zero or negative values
            norm_s2 = max(s2 + 100, 1) # And normalize
            
            if random.random() < norm_s1 / (norm_s1 + norm_s2):
                child_blocks.append(b1)
            else:
                child_blocks.append(b2)
        
        if len(p1_blocks) > min_len and random.random() < 0.7:
            child_blocks.extend(p1_blocks[min_len:])
        elif len(p2_blocks) > min_len and random.random() < 0.7:
            child_blocks.extend(p2_blocks[min_len:])
        
        child = [p for block in child_blocks for p in block]
        # Return only one child, the other can be p2, but for simplicity returning p1,p2 structure for compatibility.
        # In a real GA, you'd generate 2 children, this is simplified for the specific knowledge-guided crossover.
        return child, list(p2)

    def _mutation(self, individual, ll_code, size_oz, mutation_rate, prog_cluster_id, use_knowledge=True, use_pass_graph=True):
        """Mutation operation supporting ablation experiments"""
        if random.random() > mutation_rate:
            return individual
        
        if not use_knowledge:
            # Ablation: Simple Random Mutation
            mutated_individual = list(individual)
            if not mutated_individual:
                return []
            
            # Randomly select a position for mutation
            idx = random.randint(0, len(mutated_individual) - 1)
            mutated_individual[idx] = random.choice(list(self.pass_embeddings.keys()))
            return list(dict.fromkeys(mutated_individual))
        
        # Original: Knowledge-Guided Mutation
        return self._marginal_contribution_mutation(individual, ll_code, size_oz, mutation_rate, prog_cluster_id, use_pass_graph)

    def _marginal_contribution_mutation(self, individual, ll_code, size_oz, mutation_rate, prog_cluster_id, use_pass_graph=True):
        """Knowledge-guided Mutation Operation"""
        blocks = self._to_blocks(individual)
        if len(blocks) <= 1:
            return individual

        contributions = []
        for block in blocks:
            if not block:
                contributions.append(0.0)
                continue
            block_scores = [self.pass_embeddings.get(p, [])[prog_cluster_id] if self.pass_embeddings.get(p) and 0 <= prog_cluster_id < len(self.pass_embeddings.get(p)) else 0.0 for p in block]
            avg_contribution = sum(block_scores) / len(block_scores) if block_scores else 0.0
            contributions.append(avg_contribution)

        if not contributions: return individual
            
        worst_block_idx = np.argmin(contributions)
        
        p_anchor = blocks[worst_block_idx - 1][-1] if worst_block_idx > 0 else None
        
        candidate_pool = set()
        
        # Determine whether to use Pass relationship graph based on ablation settings
        if use_pass_graph and p_anchor and self.pass_graph.has_node(p_anchor):
            for _, tgt, data in self.pass_graph.out_edges(p_anchor, data=True):
                if data.get('type') == 'synergy':
                    candidate_pool.add(tgt)
        
        # Supplement candidates from similar functional clusters regardless of Pass graph usage
        if len(candidate_pool) < 5:
            if blocks[worst_block_idx]:
                worst_block_cid_key = self.pass_to_cluster_map.get(blocks[worst_block_idx][0])
                if worst_block_cid_key is not None:
                    candidate_pool.update(self.pass_clusters.get(str(worst_block_cid_key), []))
        
        # If candidate pool is still too small, add random candidates
        if len(candidate_pool) < 3:
            all_passes = set(self.pass_embeddings.keys())
            candidate_pool.update(random.sample(list(all_passes), min(10, len(all_passes))))
                
        if not candidate_pool:
            return individual

        num_candidates = 32
        target_len = 3 if len(blocks[worst_block_idx]) == 1 else len(blocks[worst_block_idx])
        candidate_blocks = []
        for _ in range(num_candidates):
            new_block = random.sample(list(candidate_pool), min(len(candidate_pool), target_len))
            candidate_blocks.append(new_block)

        best_replacement_block, best_replacement_fitness = None, -float('inf')

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_block = {}
            for new_block in candidate_blocks:
                temp_blocks = list(blocks)
                temp_blocks[worst_block_idx] = new_block
                temp_seq = [p for b in temp_blocks for p in b]
                
                future = executor.submit(get_instrcount, ll_code, temp_seq, llvm_tools_path=self.llvm_tools_path)
                future_to_block[future] = new_block

            for future in concurrent.futures.as_completed(future_to_block):
                block = future_to_block[future]
                try:
                    size_seq = future.result()
                    fitness = (size_oz - size_seq) / size_oz if size_oz > 0 else -float('inf')
                    
                    if fitness > best_replacement_fitness:
                        best_replacement_fitness = fitness
                        best_replacement_block = block
                except Exception as e:
                    pass
        
        original_fitness = self._calculate_fitness([individual], ll_code, size_oz)[0]
        if best_replacement_block is not None and best_replacement_fitness > original_fitness:
            blocks[worst_block_idx] = best_replacement_block
            return [p for b in blocks for p in b]
            
        return individual

    def evolve(self, ll_code, prog_cluster_id, ablation_mode='full', **ga_params):
        """
        Performs evolution, supporting ablation experiments
        
        Possible values for ablation_mode:
        - 'full': Full knowledge-guided method
        - 'no_knowledge_crossover': No knowledge-guided crossover
        - 'no_knowledge_mutation': No knowledge-guided mutation  
        - 'random_init': Random population initialization
        - 'no_program_cluster': No program cluster information
        - 'no_knowledge': No knowledge at all (standard GA)
        """
        self.tracker.reset()
        
        pop_size = ga_params.get('population_size', 20)
        generations = ga_params.get('generations', 10)
        elite_size = ga_params.get('elite_size', 2)
        cross_rate = ga_params.get('crossover_rate', 0.8)
        mut_rate = ga_params.get('mutation_rate', 0.3)

        # Adjust parameters based on ablation mode
        use_prototypes = ablation_mode not in ['random_init', 'no_knowledge']
        use_knowledge_crossover = ablation_mode not in ['no_knowledge_crossover', 'no_knowledge']
        use_knowledge_mutation = ablation_mode not in ['no_knowledge_mutation', 'no_knowledge']
        use_program_cluster = ablation_mode not in ['no_program_cluster', 'no_knowledge']
        use_pass_graph = ablation_mode not in ['no_pass_graph', 'no_knowledge'] # Assuming 'no_pass_graph' can be a mode

        # If program clustering is not used, use a random cluster ID
        effective_cluster_id = prog_cluster_id if use_program_cluster else random.randint(0, len(next(iter(self.pass_embeddings.values()))) - 1)

        population = self._initialize_population(ll_code, pop_size, use_prototypes)
        if not population: return None, self.tracker

        size_oz = get_instrcount(ll_code, ["-Oz"], llvm_tools_path=self.llvm_tools_path)
        initial_fitness = self._calculate_fitness(population, ll_code, size_oz)
        best_overall = (list(population[0]), initial_fitness[0])
        
        pbar = tqdm(range(generations), desc=f"GA-{ablation_mode}", leave=False)
        for gen in pbar:
            fitness = self._calculate_fitness(population, ll_code, size_oz) if gen > 0 else initial_fitness
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_overall[1]:
                best_overall = (list(population[best_idx]), fitness[best_idx])
            
            # Log generation information
            self.tracker.log_generation(gen, best_overall[1])
            pbar.set_description(f"GA-{ablation_mode} (Best: {best_overall[1] * 100:.2f}%)")
            
            selected = self._selection(population, fitness, elite_size)
            next_pop = list(selected[:elite_size])
            while len(next_pop) < pop_size:
                p1, p2 = random.sample(selected, 2)
                c1, _ = self._crossover(p1, p2, effective_cluster_id, cross_rate, use_knowledge_crossover)
                c1 = self._mutation(c1, ll_code, size_oz, mut_rate, effective_cluster_id, use_knowledge_mutation, use_pass_graph)
                next_pop.append(c1)
            population = next_pop
        pbar.close()
        print(f"Best ({ablation_mode}): {best_overall[1]*100:.2f}%")
        
        # Always returns two values: sequence and tracker
        return best_overall[0], self.tracker

class PassFingerprintGenerator:
    def __init__(self, training_dir: str, test_dir:str, llvm_tools_path: str, output_dir: str, 
                 num_clusters: int = 10, num_workers: int = None):
        self.training_dir = Path(training_dir); self.test_dir = Path(test_dir)
        self.llvm_tools_path = llvm_tools_path; self.output_dir = Path(output_dir)
        self.num_clusters = num_clusters; self.num_workers = num_workers or os.cpu_count()
        self.all_passes_list = [a.value for a in Actions_LLVM_10_0_0]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.empirical_prototype_cache_file = self.output_dir / "empirical_prototype_sequences.json"
        self.training_dir = Path(training_dir); self.test_dir = Path(test_dir)
        self.llvm_tools_path = llvm_tools_path; self.output_dir = Path(output_dir)
        self.num_clusters = num_clusters; self.num_workers = num_workers or os.cpu_count()
        self.all_passes_list = [a.value for a in Actions_LLVM_10_0_0]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.program_clusters_cache_file = self.output_dir / "program_clusters.pkl"
        self.pass_embeddings_cache_file = self.output_dir / "pass_embeddings.json"
        self.prog_kmeans_model_cache_file = self.output_dir / "prog_kmeans_model.pkl"
        self.prog_scaler_cache_file = self.output_dir / "prog_scaler.pkl"
        self.prog_normalizer_cache_file = self.output_dir / "prog_normalizer.pkl"
        self.pass_clusters_cache_file = self.output_dir / "pass_clusters.json"
        self.pass_graph_cache_file = self.output_dir / "pass_relationship_graph.graphml"
        self.program_paths = []; self.program_clusters = defaultdict(list)
        
        # Ablation experiment related
        self.ablation_results = {}  # Stores ablation experiment results
        self.representative_benchmarks = []  # Stores representative benchmarks

    def _find_ll_files(self, directory: Path):
        print(f"[*] Recursively searching for .ll files in {directory}..."); files = list(directory.rglob("*.ll"))
        if not files: print(f"[!] Warning: No .ll files found in {directory}.")
        else: print(f"[+] Found {len(files)} .ll files.")
        return files

    def step1_cluster_programs(self):
        if self.program_clusters_cache_file.exists():
            print(f"[+] Found cached program cluster file, loading from {self.program_clusters_cache_file}...")
            with open(self.program_clusters_cache_file, 'rb') as f: self.program_clusters = pickle.load(f)
            print("[+] Program clustering results loaded."); return
        self.program_paths = self._find_ll_files(self.training_dir)
        if not self.program_paths: raise FileNotFoundError(f"No files found in training directory {self.training_dir}.")
        print("\n--- Step 1: Program Clustering (using Cosine Similarity-based KMeans) ---")
        feature_list, path_list = [], []
        for path in tqdm(self.program_paths, desc="Extracting features"):
            try:
                features = get_autophase_obs(path.read_text())
                if features and len(features) == 56: feature_list.append(features); path_list.append(path)
                else: print(f"[!] Warning: Could not get valid Autophase features from {path}. Skipping.")
            except Exception as e: print(f"[!] Error: Feature extraction failed when processing {path}: {e}")
        if not feature_list: raise ValueError("Failed to successfully extract features for any program.")
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(np.array(feature_list))
        normalizer = Normalizer(norm='l2'); X_normalized = normalizer.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10); labels = kmeans.fit_predict(X_normalized)
        for i, path in enumerate(path_list): self.program_clusters[labels[i]].append(str(path))
        print("[*] Caching program clustering model and preprocessors...")
        with open(self.prog_kmeans_model_cache_file, 'wb') as f: pickle.dump(kmeans, f)
        with open(self.prog_scaler_cache_file, 'wb') as f: pickle.dump(scaler, f)
        with open(self.prog_normalizer_cache_file, 'wb') as f: pickle.dump(normalizer, f)
        print("[+] Model and preprocessors cached.")
        print(f"[*] Caching clustering results to {self.program_clusters_cache_file}...")
        with open(self.program_clusters_cache_file, 'wb') as f: pickle.dump(self.program_clusters, f)
        print("[+] Program clustering completed and cached.")
    
    def _calculate_fingerprint_for_pass(self, action: Enum) -> tuple:
        pass_flag = action.value; feature_vector = []
        for cluster_id in range(self.num_clusters):
            programs_in_cluster = [Path(p) for p in self.program_clusters.get(cluster_id, [])]
            if not programs_in_cluster: feature_vector.append(0.0); continue
            cluster_reductions = []
            for prog_path in programs_in_cluster:
                try:
                    ll_code = prog_path.read_text()
                    initial_size = get_instrcount(ll_code, llvm_tools_path=self.llvm_tools_path)
                    optimized_size = get_instrcount(ll_code, pass_flag, llvm_tools_path=self.llvm_tools_path)
                    if initial_size > 0: cluster_reductions.append((initial_size - optimized_size) / initial_size * 100)
                except Exception: pass
            feature_vector.append(np.mean(cluster_reductions) if cluster_reductions else 0.0)
        return pass_flag, feature_vector
    
    def step2_generate_pass_embeddings(self) -> dict:
        if self.pass_embeddings_cache_file.exists():
            print(f"\n--- Step 2: Loading Pass Embeddings from Cache ---")
            with open(self.pass_embeddings_cache_file, 'r') as f: return json.load(f)
        if not self.program_clusters: print("[!] Warning: Programs not yet clustered."); return {}
        print(f"\n--- Step 2: Generating Pass Embeddings (using {self.num_workers} threads) ---")
        pass_embeddings = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._calculate_fingerprint_for_pass, action) for action in Actions_LLVM_10_0_0]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(list(Actions_LLVM_10_0_0)), desc="Calculating Pass fingerprints"):
                try: pass_flag, f_vec = future.result(); pass_embeddings[pass_flag] = f_vec
                except Exception as e: print(f"[!] Critical Error: A fingerprint calculation task failed: {e}")
        sorted_embeddings = { a.value: pass_embeddings[a.value] for a in Actions_LLVM_10_0_0 if a.value in pass_embeddings }
        with open(self.pass_embeddings_cache_file, 'w') as f: json.dump(sorted_embeddings, f, indent=2)
        print("\n[+] Behavioral fingerprints for all Passes generated and cached!")
        return sorted_embeddings

    def step3_visualize_pass_embeddings(self, pass_embeddings: dict, method: str):
        if not pass_embeddings: print("[!] Warning: Pass Embeddings are empty."); return
        print(f"\n--- Step 3: Visualizing Pass Embeddings (using {method.upper()}) ---")
        labels, data = list(pass_embeddings.keys()), np.array(list(pass_embeddings.values()))
        if method == 'tsne':
            perp = min(30, len(labels) - 1);
            if perp < 1: print("[!] Warning: Too few Passes."); return
            reducer, name = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000), "t-SNE"
        elif method == 'pca': reducer, name = PCA(n_components=2, random_state=42), "PCA"
        else: raise ValueError(f"Unknown dimensionality reduction method: {method}")
        embeddings_2d = reducer.fit_transform(data); plt.figure(figsize=(16, 12)); sns.set_theme(style="whitegrid")
        sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], s=120, alpha=0.7)
        plt.title(f'2D Visualization of Pass Embeddings ({name})', fontsize=16); plt.xlabel(f'{name} D1'); plt.ylabel(f'{name} D2')
        out_path = self.output_dir / f"pass_embeddings_visualization_{method}.png"; plt.savefig(out_path, dpi=100, bbox_inches='tight')
        print(f"[+] Visualization image saved: {out_path}"); plt.close()

    def step4a_find_optimal_pass_clusters(self, pass_embeddings: dict) -> int:
        print("\n--- Step 4a: Finding Optimal k using Elbow Method ---")
        data = np.array(list(pass_embeddings.values())); normalizer = Normalizer(norm='l2'); data_norm = normalizer.fit_transform(data)
        cl_range, inertias = range(2, 51), []
        for k in tqdm(cl_range, desc="Testing k values"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10); kmeans.fit(data_norm); inertias.append(kmeans.inertia_)
        kneedle = KneeLocator(cl_range, inertias, curve='convex', direction='decreasing'); optimal_k = kneedle.elbow or 15
        if kneedle.elbow is None: print("[!] Warning: No elbow found, using default k=15.")
        else: print(f"[+] Optimal number of clusters k = {optimal_k}")
        plt.figure(figsize=(10, 6)); kneedle.plot_knee(); plt.xlabel("k"); plt.ylabel("Inertia")
        plt.title("Elbow Method"); plt.xticks(list(cl_range)[::2] + [optimal_k]); plt.grid(True)
        out_path = self.output_dir / "pass_cluster_elbow_method.png"; plt.savefig(out_path, dpi=100); plt.close()
        print(f"[+] Elbow method image saved: {out_path}")
        return optimal_k

    def step4b_cluster_passes(self, pass_embeddings: dict, num_pass_clusters: int) -> dict:
        print(f"\n--- Step 4b: Clustering LLVM Passes (k={num_pass_clusters}) ---")
        pass_names, data = list(pass_embeddings.keys()), np.array(list(pass_embeddings.values()))
        normalizer = Normalizer(norm='l2'); data_norm = normalizer.fit_transform(data)
        kmeans = KMeans(n_clusters=num_pass_clusters, random_state=42, n_init=10); labels = kmeans.fit_predict(data_norm)
        pass_clusters = defaultdict(list)
        for i, name in enumerate(pass_names): pass_clusters[labels[i]].append(name)
        for cid in pass_clusters: pass_clusters[cid].sort()
        out_str, sorted_clusters = "", sorted(pass_clusters.items())
        for cid, passes in sorted_clusters: out_str += f"\n--- Cluster {cid} ---\n" + '\n'.join([f"  - {p}" for p in passes]) + "\n"
        print("\n[+] LLVM Pass Clustering Results:" + out_str)
        with open(self.output_dir / "pass_clusters.txt", 'w') as f: f.write(out_str)
        with open(self.pass_clusters_cache_file, 'w') as f: json.dump({str(k): v for k, v in sorted_clusters}, f, indent=2)
        print(f"[+] Pass clustering results saved to .txt and .json files.")
        return dict(pass_clusters)

    def step5_build_graph(self, pass_embeddings: dict, pass_clusters: dict):
        print("\n--- Step 5: Building Pass Relationship Graph ---")
        if not pass_clusters: print("[!] Warning: Pass clustering results are empty."); return
        G, names = nx.MultiDiGraph(), list(pass_embeddings.keys())
        for cid, passes in pass_clusters.items():
            for name in passes:
                if name in names: G.add_node(name, cluster_id=int(cid))
        data = np.array(list(pass_embeddings.values())); cos_mat = cosine_similarity(data)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if cos_mat[i, j] > 0.5: G.add_edge(names[i], names[j], key='similarity', type='similarity', weight=float(cos_mat[i, j]))
        syn_sum = defaultdict(int)
        for (src, _), count in synergy_pairs_dict.items(): syn_sum[src] += count
        for (src, tgt), count in synergy_pairs_dict.items():
            if G.has_node(src) and G.has_node(tgt) and syn_sum[src] > 0:
                G.add_edge(src, tgt, key='synergy', type='synergy', weight=float(count / syn_sum[src]), raw_count=count)
        nx.write_graphml(G, self.pass_graph_cache_file)
        print(f"\n[+] Graph construction complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges. Saved to {self.pass_graph_cache_file}")

    def step6_find_empirical_prototypes(self, **offline_ga_params):
        print("\n--- (Offline) Step 6: Finding Empirical Prototype Sequences ---")
        if self.empirical_prototype_cache_file.exists():
            print(f"[+] Found cached prototype sequences, loading from {self.empirical_prototype_cache_file}.")
            return

        print(f"[*] Offline GA configuration: {offline_ga_params}")
        finder = EmpiricalPrototypeFinder(self.all_passes_list, self.llvm_tools_path, self.num_workers)
        
        prototype_sequences = {}
        for cluster_id, prog_paths in self.program_clusters.items():
            print(f"\n--- Running GA for program cluster {cluster_id} (containing {len(prog_paths)} programs) ---")
            programs_in_cluster = [Path(p) for p in prog_paths]
            best_seq = finder.find_prototype(programs_in_cluster, **offline_ga_params)
            prototype_sequences[str(cluster_id)] = best_seq
        
        with open(self.empirical_prototype_cache_file, 'w') as f:
            json.dump(prototype_sequences, f, indent=2)
        print("\n[+] All empirical prototype sequences generated and cached.")

    def step7_knowledge_guided_personalization(self, **online_ga_params):
        print("\n--- (Online) Step 7: Knowledge-Guided Personalization Evolution ---")
        print(f"[*] Online GA configuration: {online_ga_params}")
        try:
            with open(self.prog_kmeans_model_cache_file, 'rb') as f: kmeans = pickle.load(f)
            with open(self.prog_scaler_cache_file, 'rb') as f: scaler = pickle.load(f)
            with open(self.prog_normalizer_cache_file, 'rb') as f: normalizer = pickle.load(f)
            with open(self.empirical_prototype_cache_file, 'r') as f: prototype_library = json.load(f)
            with open(self.pass_embeddings_cache_file, 'r') as f: pass_embeddings = json.load(f)
            with open(self.pass_clusters_cache_file, 'r') as f: pass_clusters = {int(k): v for k, v in json.load(f).items()}
            pass_graph = nx.read_graphml(self.pass_graph_cache_file)
        except FileNotFoundError as e: print(f"[!] Error: Missing cache file: {e}."); return

        evolver = KnowledgeGuidedEvolver(prototype_library, pass_embeddings, pass_clusters, pass_graph, self.llvm_tools_path, self.num_workers)
        
        datasets = [d for d in self.test_dir.iterdir() if d.is_dir()]
        if not datasets: print(f"[!] No datasets found under {self.test_dir}."); return
        
        all_results, total_perf, total_files = [], 0.0, 0
        for ds_dir in datasets:
            print(f"\n--- Processing dataset: {ds_dir.name} ---")
            test_files = list(ds_dir.rglob("*.ll"))
            if not test_files: continue
            ds_perf, ds_files_count = 0.0, 0
            for ll_file in test_files:
                print(f"[*] Running knowledge-guided GA for {ll_file.name}...")
                try:
                    ll_code = ll_file.read_text(); features = np.array(get_autophase_obs(ll_code))
                    if features.size != 56: continue
                    prog_cluster_id = int(kmeans.predict(normalizer.transform(scaler.transform(features.reshape(1,-1))))[0])
                    
                    best_seq, tracker = evolver.evolve(ll_code, prog_cluster_id, **online_ga_params)
                    if not best_seq: continue
                    
                    size_oz = get_instrcount(ll_code, ["-Oz"], llvm_tools_path=self.llvm_tools_path)
                    if size_oz == 0: continue
                    size_seq = get_instrcount(ll_code, best_seq, llvm_tools_path=self.llvm_tools_path)
                    perf = (size_oz - size_seq) / size_oz * 100
                    ds_perf += perf; ds_files_count += 1
                except Exception as e: print(f"\n[!] Error processing {ll_file.name}: {e}")
                
            if ds_files_count > 0:
                avg_perf = ds_perf / ds_files_count
                all_results.append({"dataset": ds_dir.name, "file_count": ds_files_count, "avg_perf": avg_perf})
                total_perf += ds_perf; total_files += ds_files_count
        
        print("\n" + "="*20 + " Average Performance Evaluation Results by Dataset (Knowledge-Guided GA) " + "="*20)
        output_str = "Average Performance Improvement (vs -Oz)\n" + "-"*50 + "\n"
        output_str += f"{'Dataset':<30} {'File Count':>8} {'Performance Improvement (%)':>15}\n" + "-"*50 + "\n"
        for res in all_results:
            line = f"{res['dataset']:<30} {res['file_count']:>8} {res['avg_perf']:>15.2f}\n"
            print(line, end=""); output_str += line
        if total_files > 0:
            total_avg = total_perf / total_files
            summary = "-"*50 + f"\n{'Overall Average':<30} {total_files:>8} {total_avg:>15.2f}\n" + "="*50 + "\n"
            print(summary, end=""); output_str += summary
        with open(self.output_dir / "recommendations_knowledge_guided_ga.txt", 'w') as f: f.write(output_str)
        print(f"\n[+] Evaluation report saved to: {self.output_dir / 'recommendations_knowledge_guided_ga.txt'}")

    def _find_representative_benchmarks(self, test_files, max_benchmarks=6):
        """
        Finds representative benchmark programs
        Selection based on program size, complexity, and other features
        """
        print(f"[*] Selecting {max_benchmarks} representative benchmarks from {len(test_files)} files...")
        
        candidates = []
        for ll_file in test_files:
            try:
                ll_code = ll_file.read_text()
                # Calculate program features
                size_oz = get_instrcount(ll_code, ["-Oz"], llvm_tools_path=self.llvm_tools_path)
                if size_oz == 0:
                    continue
                    
                features = get_autophase_obs(ll_code)
                if not features or len(features) != 56:
                    continue
                
                # Calculate complexity metric
                complexity_score = np.std(features) + np.mean(features)
                
                candidates.append({
                    'file': ll_file,
                    'size': size_oz,
                    'complexity': complexity_score,
                    'features': features
                })
            except Exception:
                continue
        
        if not candidates:
            return []
        
        # Sort by size and complexity, select representative ones
        candidates.sort(key=lambda x: (x['size'], x['complexity']))
        
        # Select programs of varying scales and complexities
        selected = []
        step = max(1, len(candidates) // max_benchmarks)
        for i in range(0, min(len(candidates), max_benchmarks * step), step):
            selected.append(candidates[i])
        
        # If not enough, supplement with remaining
        while len(selected) < max_benchmarks and len(selected) < len(candidates):
            remaining = [c for c in candidates if c not in selected]
            if remaining:
                selected.append(remaining[0])
            else:
                break
        
        print(f"[+] Selected {len(selected)} representative benchmarks")
        return [item['file'] for item in selected[:max_benchmarks]]

    def _visualize_ablation_results(self, ablation_results, output_prefix):
        """Visualizes ablation experiment results"""
        
        # 1. Generates line plots for representative benchmarks
        if self.representative_benchmarks:
            plt.figure(figsize=(20, 12))
            
            n_benchmarks = len(self.representative_benchmarks)
            rows = 2
            cols = 3
            
            for i, benchmark_file in enumerate(self.representative_benchmarks):
                if i >= 6:  # Show up to 6
                    break
                    
                plt.subplot(rows, cols, i + 1)
                
                for mode, data in ablation_results.items():
                    benchmark_name = benchmark_file.name
                    if benchmark_name in data['benchmark_results']:
                        tracker = data['benchmark_results'][benchmark_name]['tracker']
                        if tracker and tracker.generation_history:
                            sample_counts = [item[1] for item in tracker.generation_history]
                            fitness_values = [item[2] * 100 for item in tracker.generation_history]  # Convert to percentage
                            
                            plt.plot(sample_counts, fitness_values, 
                                   marker='o', label=mode, linewidth=2, markersize=4)
                
                plt.title(f'{benchmark_name}', fontsize=12, fontweight='bold')
                plt.xlabel('Number of Samples')
                plt.ylabel('Performance over -Oz (%)')
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{output_prefix}_convergence_curves.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[+] Convergence curves saved: {output_prefix}_convergence_curves.png")
        
        # 2. Generates overall performance comparison chart
        plt.figure(figsize=(12, 8))
        
        modes = list(ablation_results.keys())
        avg_performances = []
        std_performances = []
        
        for mode in modes:
            perfs = []
            for dataset_name, dataset_data in ablation_results[mode]['dataset_results'].items():
                if dataset_data['file_count'] > 0:
                    perfs.append(dataset_data['avg_perf'])
            
            if perfs:
                avg_performances.append(np.mean(perfs))
                std_performances.append(np.std(perfs))
            else:
                avg_performances.append(0.0)
                std_performances.append(0.0)
        
        x_pos = np.arange(len(modes))
        bars = plt.bar(x_pos, avg_performances, yerr=std_performances, 
                      capsize=5, alpha=0.8, edgecolor='black')
        
        # Add numerical labels to each bar
        for i, (bar, avg_perf) in enumerate(zip(bars, avg_performances)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_performances[i] + 0.1,
                    f'{avg_perf:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Ablation Mode')
        plt.ylabel('Average Performance over -Oz (%)')
        plt.title('Ablation Study Results Comparison')
        plt.xticks(x_pos, modes, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"{output_prefix}_performance_comparison.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[+] Performance comparison chart saved: {output_prefix}_performance_comparison.png")

    def _generate_ablation_report(self, ablation_results, output_filename):
        """Generates ablation experiment report"""
        
        # Prepare table data
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(" " * 25 + "ABLATION STUDY RESULTS")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Output results table by dataset
        report_lines.append("PERFORMANCE BY DATASET:")
        report_lines.append("-" * 80)
        
        # Header
        header = f"{'Dataset':<25}"
        for mode in ablation_results.keys():
            header += f"{mode:<15}"
        report_lines.append(header)
        report_lines.append("-" * 80)
        
        # Get all datasets
        all_datasets = set()
        for mode_data in ablation_results.values():
            all_datasets.update(mode_data['dataset_results'].keys())
        
        # Output results for each dataset
        for dataset in sorted(all_datasets):
            line = f"{dataset:<25}"
            for mode in ablation_results.keys():
                dataset_data = ablation_results[mode]['dataset_results'].get(dataset, {'avg_perf': 0.0})
                line += f"{dataset_data['avg_perf']:<15.2f}"
            report_lines.append(line)
        
        report_lines.append("-" * 80)
        
        # Overall Statistics
        report_lines.append("")
        report_lines.append("OVERALL STATISTICS:")
        report_lines.append("-" * 50)
        
        for mode, mode_data in ablation_results.items():
            all_perfs = []
            total_files = 0
            for dataset_data in mode_data['dataset_results'].values():
                if dataset_data['file_count'] > 0:
                    all_perfs.append(dataset_data['avg_perf'])
                    total_files += dataset_data['file_count']
            
            if all_perfs:
                avg_perf = np.mean(all_perfs)
                std_perf = np.std(all_perfs)
                report_lines.append(f"{mode}:")
                report_lines.append(f"  - Average Performance: {avg_perf:.2f}% (±{std_perf:.2f}%)")
                report_lines.append(f"  - Total Files Processed: {total_files}")
                report_lines.append("")
        
        # Save report
        report_content = "\n".join(report_lines)
        with open(self.output_dir / output_filename, 'w') as f:
            f.write(report_content)
        
        print("\n" + report_content)
        print(f"\n[+] Ablation experiment report saved: {output_filename}")

    def step8_comprehensive_ablation_study(self, **online_ga_params):
        """
        Comprehensive Ablation Study: Single run to get all ablation data
        """
        print("\n" + "="*60)
        print(" " * 15 + "COMPREHENSIVE ABLATION STUDY")
        print("="*60)
        
        # Define ablation experiment modes
        ablation_modes = {
            'full': 'Full knowledge-guided method',
            'no_knowledge_crossover': 'No knowledge-guided crossover',
            'no_knowledge_mutation': 'No knowledge-guided mutation',
            'random_init': 'Random population initialization',
            'no_knowledge': 'No knowledge at all (standard GA)'
        }
        
        print(f"[*] Ablation experiment configuration: {online_ga_params}")
        print(f"[*] Experiment modes: {list(ablation_modes.keys())}")
        
        try:
            # Load necessary models and data
            with open(self.prog_kmeans_model_cache_file, 'rb') as f: kmeans = pickle.load(f)
            with open(self.prog_scaler_cache_file, 'rb') as f: scaler = pickle.load(f)
            with open(self.prog_normalizer_cache_file, 'rb') as f: normalizer = pickle.load(f)
            with open(self.empirical_prototype_cache_file, 'r') as f: prototype_library = json.load(f)
            with open(self.pass_embeddings_cache_file, 'r') as f: pass_embeddings = json.load(f)
            with open(self.pass_clusters_cache_file, 'r') as f: pass_clusters = {int(k): v for k, v in json.load(f).items()}
            pass_graph = nx.read_graphml(self.pass_graph_cache_file)
        except FileNotFoundError as e: 
            print(f"[!] Error: Missing cache file: {e}."); 
            return

        # Get test datasets
        datasets = [d for d in self.test_dir.iterdir() if d.is_dir()]
        if not datasets: 
            print(f"[!] No datasets found under {self.test_dir}."); 
            return
        
        # Collect all test files and find representative benchmarks
        all_test_files = []
        for ds_dir in datasets:
            test_files = list(ds_dir.rglob("*.ll"))
            all_test_files.extend(test_files)
        
        self.representative_benchmarks = self._find_representative_benchmarks(all_test_files, 6)
        
        # Initialize results storage
        ablation_results = {}
        
        # Run experiments for each ablation mode
        for mode_name, mode_description in ablation_modes.items():
            print(f"\n--- Running Ablation Experiment: {mode_name} ({mode_description}) ---")
            
            evolver = KnowledgeGuidedEvolver(prototype_library, pass_embeddings, pass_clusters, 
                                           pass_graph, self.llvm_tools_path, self.num_workers)
            
            mode_results = {
                'dataset_results': {},
                'benchmark_results': {}
            }
            
            # Process each dataset
            for ds_dir in datasets:
                print(f"\n[*] Processing dataset: {ds_dir.name} (Mode: {mode_name})")
                test_files = list(ds_dir.rglob("*.ll"))
                if not test_files: 
                    continue
                
                ds_perf, ds_files_count = 0.0, 0
                
                for ll_file in test_files:
                    try:
                        ll_code = ll_file.read_text()
                        features = np.array(get_autophase_obs(ll_code))
                        if features.size != 56: 
                            continue
                        
                        prog_cluster_id = int(kmeans.predict(normalizer.transform(scaler.transform(features.reshape(1,-1))))[0])
                        
                        # Run GA
                        best_seq, tracker = evolver.evolve(ll_code, prog_cluster_id, 
                                                         ablation_mode=mode_name, **online_ga_params)
                        if not best_seq: 
                            continue
                        
                        # Calculate performance
                        size_oz = get_instrcount(ll_code, ["-Oz"], llvm_tools_path=self.llvm_tools_path)
                        if size_oz == 0: 
                            continue
                        size_seq = get_instrcount(ll_code, best_seq, llvm_tools_path=self.llvm_tools_path)
                        perf = (size_oz - size_seq) / size_oz * 100
                        
                        ds_perf += perf
                        ds_files_count += 1
                        
                        # If a representative benchmark, record detailed information
                        if ll_file in self.representative_benchmarks:
                            mode_results['benchmark_results'][ll_file.name] = {
                                'performance': perf,
                                'tracker': tracker,
                                'best_sequence': best_seq
                            }
                        
                    except Exception as e: 
                        print(f"[!] Error processing {ll_file.name}: {e}")
                
                # Record dataset results
                if ds_files_count > 0:
                    avg_perf = ds_perf / ds_files_count
                    mode_results['dataset_results'][ds_dir.name] = {
                        'file_count': ds_files_count,
                        'avg_perf': avg_perf,
                        'total_perf': ds_perf
                    }
            
            ablation_results[mode_name] = mode_results
        
        # Generate visualizations and reports
        self._visualize_ablation_results(ablation_results, "ablation_study")
        self._generate_ablation_report(ablation_results, "ablation_study_report.txt")
        
        # Save detailed results
        detailed_results = {}
        for mode, results in ablation_results.items():
            detailed_results[mode] = {
                'dataset_results': results['dataset_results'],
                'benchmark_count': len(results['benchmark_results'])
            }
        
        with open(self.output_dir / "ablation_detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n[✔] Comprehensive ablation study completed! {len(ablation_modes)} configurations tested")
        print(f"[+] Results saved to {self.output_dir}")
        
        self.ablation_results = ablation_results
        return ablation_results

    def run_pipeline(self, **kwargs):
        self.step1_cluster_programs()
        self.step2_generate_pass_embeddings()
        pass_embeddings = self.step2_generate_pass_embeddings()
        self.step3_visualize_pass_embeddings(pass_embeddings, kwargs.get('reduction_method', 'tsne'))
        optimal_k = self.step4a_find_optimal_pass_clusters(pass_embeddings)
        pass_clusters = self.step4b_cluster_passes(pass_embeddings, optimal_k)
        if pass_clusters: self.step5_build_graph(pass_embeddings, pass_clusters)
        if kwargs.get('run_step6_find_prototypes'):
            self.step6_find_empirical_prototypes(**kwargs.get('offline_ga_params', {}))
        if kwargs.get('run_step7_personalize'):
            self.step7_knowledge_guided_personalization(**kwargs.get('online_ga_params', {}))
        if kwargs.get('run_step8_comprehensive_ablation_study'):
            self.step8_comprehensive_ablation_study(**kwargs.get('online_ga_params', {}))
        print("\n[✔] Pipeline execution complete!")


if __name__ == '__main__':
    BASE_DIR = Path("./")
    TRAINING_SET_PATH = BASE_DIR / "dataset/train/"
    TEST_SET_PATH = BASE_DIR / "dataset/test/"
    LLVM_TOOLS_PATH = BASE_DIR / "llvm_tools/"
    OUTPUT_DIR = BASE_DIR / "output/"
    for p in [TRAINING_SET_PATH, TEST_SET_PATH, LLVM_TOOLS_PATH]: p.mkdir(parents=True, exist_ok=True)

    pipeline_params = {
        "num_clusters": 100, "num_workers": 16, "reduction_method": 'tsne',
        "run_step6_find_prototypes": False, # Set to True to run offline prototype finding
        "run_step7_personalize": True,      # Set to True to run knowledge-guided personalization
        "run_step8_comprehensive_ablation_study": True, # Set to True to run comprehensive ablation study
        # Offline GA parameters (pure brute-force search, requires more time and population)
        "offline_ga_params": {
            "seq_len": 100, "population_size": 100, "generations": 30,
            "elite_size": 20, "crossover_rate": 0.8, "mutation_rate": 0.8
        },
        # Online GA parameters (knowledge-guided, can be lighter)
        "online_ga_params": {
            "population_size": 50, "generations": 5,
            "elite_size": 10, "crossover_rate": 0.8, "mutation_rate": 0.99
        }
    }
    generator = PassFingerprintGenerator(
        training_dir=str(TRAINING_SET_PATH), test_dir=str(TEST_SET_PATH),
        llvm_tools_path=str(LLVM_TOOLS_PATH), output_dir=str(OUTPUT_DIR),
        num_clusters=pipeline_params["num_clusters"], num_workers=pipeline_params["num_workers"]
    )
    generator.run_pipeline(**pipeline_params)