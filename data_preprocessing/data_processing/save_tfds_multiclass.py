import scipy.sparse as sp
import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from rdkit import Chem # Trebaće za potencijalno učitavanje Mola, ali za ovu funkciju samo putanje
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

# Categorizes an AC50 value into one of five classes.
def categorize_ac50(ac50_value):
    if ac50_value < 0.09:
        # Strong
        return 0
    elif 0.09 <= ac50_value < 0.18:
        # Moderate
        return 1
    elif 0.18 <= ac50_value < 20:
        # Weak
        return 2
    elif 20 <= ac50_value < 800:
        # Very Weak
        return 3
    elif ac50_value >= 800: 
        # Inactive
        return 4
    

# Loads all data from specified CSV and MOL directories into two maps.
def load_all_data_into_maps(
    csv_files_directory: str,
    mol_files_parent_directory: str
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    csv_dataframes = {}
    mol_paths_by_csv_name = {}

    print(f"Scanning CSV directory: {csv_files_directory}")
    # --- Loading all CSV files ---
    for item_name in os.listdir(csv_files_directory):
        item_path = os.path.join(csv_files_directory, item_name)
        
        if os.path.isfile(item_path) and item_name.lower().endswith('.csv'):
            csv_name_without_ext = os.path.splitext(item_name)[0]
            try:
                df = pd.read_csv(item_path)
                csv_dataframes[csv_name_without_ext] = df
                print(f"  Loaded CSV: {item_name} with {len(df)} rows.")
            except Exception as e:
                print(f"  Error loading CSV '{item_name}': {e}. Skipping.")
    
    print(f"\nScanning MOL directory: {mol_files_parent_directory}")
    # --- Mapping MOL file paths ---
    for subfolder_name in os.listdir(mol_files_parent_directory):
        subfolder_path = os.path.join(mol_files_parent_directory, subfolder_name)

        if os.path.isdir(subfolder_path):
            current_mol_map = {}
            
            for mol_file_name in os.listdir(subfolder_path):
                if mol_file_name.lower().endswith('.mol'):
                    mol_name_without_ext = os.path.splitext(mol_file_name)[0]
                    full_mol_path = os.path.join(subfolder_path, mol_file_name)
                    
                    # ### APPLYING STANDARDIZATION TO MOL FILE NAME ###
                    standardized_mol_name = standardize_mol_name(mol_name_without_ext)
                    current_mol_map[standardized_mol_name] = full_mol_path
            
            if current_mol_map:
                mol_paths_by_csv_name[subfolder_name] = current_mol_map
                print(f"  Found {len(current_mol_map)} MOL files in subfolder '{subfolder_name}'.")
            else:
                print(f"  No MOL files found in subfolder '{subfolder_name}'.")

    print("\nData loading complete.")
    print(f"Total CSV files loaded: {len(csv_dataframes)}")
    print(f"Total MOL subfolders mapped: {len(mol_paths_by_csv_name)}")

    return csv_dataframes, mol_paths_by_csv_name

# Determines the assay type ('tox21' or 'binders') based on the CSV file name.
def determine_assay_type(csv_name: str) -> str:
    csv_name_lower = csv_name.lower()
    if 'tox21' in csv_name_lower:
        return 'tox21'
    else: 
        return 'binders'
    
# Standardizes a molecular name by converting it to lowercase
# and removing non-alphanumeric characters.
def standardize_mol_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name
    
# Main function for processing and combining molecular data.
def process_and_combine_molecular_data(
    csv_dataframes: dict[str, pd.DataFrame],
    mol_paths_by_csv_name: dict[str, dict[str, str]],
    column_mapping: dict,
    common_label_name: str,
    common_global_features_list: list[str]
) -> dict[str, dict]:
    all_molecules_processed_data = {}
    total_processed_molecules = 0
    total_skipped_molecules = 0

    print("\nStarting the process of combining molecular data...")

    for csv_name, df in csv_dataframes.items():
        print(f"\nProcessing data from CSV: '{csv_name}'")
        
        assay_type = determine_assay_type(csv_name)
        print(f"Assay type: {assay_type}")
        if assay_type not in column_mapping:
            print(f"Warning: Assay type '{assay_type}' (from '{csv_name}') not recognized or defined in the column map. Skipping this CSV.")
            total_skipped_molecules += len(df)
            continue

        current_assay_col_names = column_mapping[assay_type]
        
        mol_id_col_source = current_assay_col_names['mol_id_col']
        ac50_col_source = current_assay_col_names[common_label_name]

        df[mol_id_col_source] = df[mol_id_col_source].astype(str)
        df[mol_id_col_source] = df[mol_id_col_source].apply(standardize_mol_name)


        required_cols = [mol_id_col_source, ac50_col_source] + [current_assay_col_names.get(feat, feat) for feat in common_global_features_list]
        print('Required columns:', required_cols)
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Error: CSV '{csv_name}' (type {assay_type}) is missing required columns: {missing_cols}. Skipping.")
            total_skipped_molecules += len(df)
            continue
        
        df_indexed = df.set_index(mol_id_col_source).copy()

        current_mol_paths_map = mol_paths_by_csv_name.get(csv_name, {})
        print('Number of MOL files in CSV:', len(current_mol_paths_map))
        
        if not current_mol_paths_map:
            print(f"Warning: No MOL files mapped for subfolder '{csv_name}'. Skipping this CSV.")
            total_skipped_molecules += len(df)
            continue
        print(len(df_indexed))
        
        for mol_name, row_data in df_indexed.iterrows():
            if mol_name in current_mol_paths_map:
                mol_file_path = current_mol_paths_map[mol_name]
                
                if not os.path.exists(mol_file_path):
                    total_skipped_molecules += 1
                    continue

                label_ac_value = row_data.get(ac50_col_source)
                label_class_value = categorize_ac50(label_ac_value)
                if pd.isna(label_ac_value) or label_ac_value == ' ':
                    print(f" Warning: Label '{ac50_col_source}' is NaN for molecule '{mol_name}'. Skipping.\n Setting to 800")
                    label_class_value = categorize_ac50(800)
                    total_skipped_molecules += 1
                    
                try:
                    hitc_value = row_data.get('hitc')
                except Exception as e:
                    hitc_value = row_data.get('HITC')
                
                if hitc_value not in [0, 1]:
                    if hitc_value == 'Active':
                        hitc_value = 1.0
                    else: hitc_value = 0.0
                
                label_class_value = int(label_class_value)

                global_features_values = {}
                for common_feat_name in COMMON_GLOBAL_FEATURES_LIST:
                    source_col = current_assay_col_names.get(common_feat_name, common_feat_name)
                    feature_val = row_data.get(source_col)
                    
                    if source_col == 'preferredName':
                        global_features_values[common_feat_name] = feature_val
                        continue
                    if pd.isna(feature_val):
                        global_features_values[common_feat_name] = 'NaN'
                    else:
                        try:
                            global_features_values[common_feat_name] = float(feature_val)
                        except (ValueError, TypeError):
                            global_features_values[common_feat_name] = 'NaN'
                
                if mol_name in all_molecules_processed_data:
                    mol_name = mol_name + str(total_skipped_molecules)
                    total_skipped_molecules += 1 
                
                esr = mol_file_path.split('/')[2].split("-")
                
                if 'ESR' in esr[-2]:
                    esr = f'{esr[-1]}-{esr[-2]}'
                else:
                    esr = esr[-1]
                all_molecules_processed_data[mol_name] = {
                    'mol_file_path': mol_file_path,
                    'preferredName': mol_name,
                    'esr': esr,
                    'hitc': hitc_value,
                    'label': label_class_value,
                    'global_features': global_features_values,
                    'source_assay_name': csv_name
                }
                total_processed_molecules += 1
            else:
                print(f"Warning: MOL file for '{mol_name}' not found in the map for '{csv_name}'. Skipping.")
                total_skipped_molecules += 1
    
    print(f"\nProcessing complete. Total molecules processed: {total_processed_molecules}, Skipped: {total_skipped_molecules}.")
    all_labels = [data['label'] for data in all_molecules_processed_data.values()]

    label_counts = Counter(all_labels)

    print("\nDistribution of molecules by class:")
    for label, count in sorted(label_counts.items()):
        print(f"  Class {label}: {count} molecules")

    print(f"\nTotal number of molecules: {len(all_labels)}")
    return all_molecules_processed_data


# --- Global definitions (from previous code, these are the UNIFIED names we want in the output) ---
COMMON_LABEL_NAME = 'ac50Class' # Name of the label column in the final dataset
COMMON_GLOBAL_FEATURES_LIST = [ # Names of columns for global features in the final dataset
     'monoisotopicMass', 'ac50', 
]

ACTUAL_COLUMN_NAMES_PER_ASSAY_TYPE = {
    'binders': {
        'mol_id_col': 'preferredName',  # Actual column name for molecule ID in 'binders' CSVs
        COMMON_LABEL_NAME: 'ac50',      # Actual column name for 'hitc' in 'binders' CSVs
        'averageMass':'averageMass',
        'hitc': 'hitc',
        'monoisotopicMass': 'monoisotopicMass',
        'ac50':'ac50'
    },
    'tox21': {
        'mol_id_col': 'preferredName',  
        COMMON_LABEL_NAME: 'AC50',      
        'hitc': 'HITC',
        'monoisotopicMass': 'MONOISOTOPIC MASS',
        'ac50': 'AC50',
    }
}


# --- HELPER FUNCTIONS FOR GRAPH FEATURE EXTRACTION ---
# Must be defined before calling process_mols_for_gnn_dataset
# Extracts a list of features for a given RDKit atom object.
def get_atom_features(atom):
    features = [
        atom.GetAtomicNum(),  # Atomic number
        atom.GetDegree(),     # Number of neighbors
        atom.GetTotalNumHs(), # Total number of hydrogens
        atom.GetImplicitValence(), # Implicit valence
        atom.GetFormalCharge(), # Formal charge
        atom.GetIsAromatic(), # Is aromatic
        atom.IsInRingSize(3), atom.IsInRingSize(4), atom.IsInRingSize(5), # In a ring of size 3,4,5
        atom.IsInRingSize(6), atom.IsInRingSize(7), atom.IsInRingSize(8), # In a ring of size 6,7,8
    ]
    return np.array(features, dtype=np.float32)

# Extracts a list of features for a given RDKit bond object.
def get_bond_features(bond):
    features = [
        bond.GetBondTypeAsDouble(), # Bond type (1.0 for single, 2.0 for double, etc.)
        bond.GetIsConjugated(),      # Is conjugated
        bond.IsInRing(),             # Is in a ring
    ]
    return np.array(features, dtype=np.float32)


# Converts molecular data into a graph representation for use in Graph Neural Networks.
def process_mols_for_gnn_dataset(all_molecules_processed_data: dict) -> list[dict]:
    dataset_ready_molecules = []
    skipped_count = 0

    print("\n--- Starting graph representation generation for TF Dataset ---")
    total_mols = len(all_molecules_processed_data)

    for i, (mol_name, data) in enumerate(all_molecules_processed_data.items()):
        if i % 100 == 0:
            print(f"  Processing molecule {i+1}/{total_mols}: {mol_name}")

        mol_file_path = data['mol_file_path']

        # --- Loading MOL file and generating graph features ---
        try:
            mol = Chem.MolFromMolFile(mol_file_path)
            if mol is None:
                print(f"    Warning: RDKit failed to load MOL file '{mol_file_path}' for molecule '{mol_name}'. Skipping.")
                skipped_count += 1
                continue

            # Node Features (Atom features)
            node_features = []
            for atom in mol.GetAtoms():
                node_features.append(get_atom_features(atom))
            
            if not node_features:
                print(f"    Warning: Molecule '{mol_name}' has no atoms. Skipping.")
                skipped_count += 1
                continue
            
            node_features = np.array(node_features, dtype=np.float32)

            # Edge Features (Bond features) and Adjacency Matrix
            edge_features = []
            adjacency_indices = [] 
            adjacency_values = [] 

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                bond_feat = get_bond_features(bond)

                edge_features.append(bond_feat)
                adjacency_indices.append([i, j])
                adjacency_values.append(1.0)

                edge_features.append(bond_feat)
                adjacency_indices.append([j, i])
                adjacency_values.append(1.0)
            
            if not adjacency_indices:
                dummy_bond_feat_dim = 3
                
                adjacency_indices = np.empty((0, 2), dtype=np.int64)
                adjacency_values = np.empty((0,), dtype=np.float32)
                edge_features = np.empty((0, dummy_bond_feat_dim), dtype=np.float32)
                num_edges = 0
            else:
                adjacency_indices = np.array(adjacency_indices, dtype=np.int64)
                adjacency_values = np.array(adjacency_values, dtype=np.float32)
                edge_features = np.array(edge_features, dtype=np.float32)
                num_edges = adjacency_indices.shape[0]

            # --- Packaging the data ---
            processed_data = {
                'preferredName': data['preferredName'],
                'esr': data.get('esr', 'unknown'),
                'hitc': data.get('hitc', -1),
                'label': data['label'],
                'global_features': data['global_features'],
                'node_features': node_features,
                'edge_features': edge_features,
                'adjacency_indices': adjacency_indices,
                'adjacency_values': adjacency_values,
                'num_nodes': node_features.shape[0],
                'num_edges': num_edges,
                'source_assay_name': data['source_assay_name']
            }
            dataset_ready_molecules.append(processed_data)

        except Exception as e:
            print(f"    Error generating graph features for '{mol_name}' from '{mol_file_path}': {e}. Skipping molecule.")
            skipped_count += 1
            continue
            
    print(f"\n--- Graph representation generation complete. Total processed: {len(dataset_ready_molecules)}, Skipped: {skipped_count}. ---")
    return dataset_ready_molecules
    
# --- Helper to extract base graph features from a MOL file (returns NumPy arrays) ---
# This is essentially the core logic extracted from your original process_mols_for_gnn_dataset
# and _generate_graph_features_for_mol that deals with RDKit.
def _extract_base_graph_features_from_mol_file(mol_file_path: str, mol_name: str):
    mol = Chem.MolFromMolFile(mol_file_path)
    if mol is None:
        raise ValueError(f"RDKit failed to load MOL file: {mol_file_path}")

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    node_features = np.array(node_features, dtype=np.float32)

    if node_features.shape[0] == 0:
         raise ValueError(f"Molecule '{mol_name}' has no atoms.")

    adjacency_indices = []
    adjacency_values = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency_indices.append([i, j])
        adjacency_values.append(1.0)
        adjacency_indices.append([j, i]) # For undirected graph
        adjacency_values.append(1.0)

    if not adjacency_indices:
        # Fallback for dummy_bond_feat_dim
        dummy_bond_feat_dim = 3 # Assuming get_bond_features returns 3 features based on prior context
        
        adjacency_indices = np.empty((0, 2), dtype=np.int64)
        adjacency_values = np.empty((0,), dtype=np.float32)
        num_edges = 0
    else:
        adjacency_indices = np.array(adjacency_indices, dtype=np.int64)
        adjacency_values = np.array(adjacency_values, dtype=np.float32)
        num_edges = adjacency_indices.shape[0]

    return {
        'node_features': node_features,
        'adjacency_indices': adjacency_indices,
        'adjacency_values': adjacency_values,
        'num_nodes': node_features.shape[0],
        'num_edges': num_edges,
    }


# --- Function to apply augmentation to the entire dataset ---
def augment_molecular_data(
    all_molecules_processed_data: dict,
    augment_label_1: bool = True,
    num_augmentations_per_sample: int = 2
) -> dict[str, dict]:
    """
    Applies node permutation augmentation to molecular graph data.
    Returns a new dictionary containing original and augmented molecules.
    """
    augmented_data = {}
    total_original_mols = len(all_molecules_processed_data)
    processed_count = 0

    print(f"\n--- Starting data augmentation (node permutation) for {total_original_mols} molecules ---")
    target_count_per_class = 2000
    original_counts = {
        2: 3561,
        3: 13981,
    }

    # calculate probability for every dominant class
    prob_to_keep_class_2 = target_count_per_class / original_counts[2]
    prob_to_keep_class_3 = target_count_per_class / original_counts[3]

    # determine posibility for every class
    prob_to_keep_per_class = {
        2.0: prob_to_keep_class_2,
        3.0: prob_to_keep_class_3
    }
    for mol_name, data in all_molecules_processed_data.items():
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"  Augmenting molecule {processed_count}/{total_original_mols}: {mol_name}")

        try:
            # Extract base graph features from MOL file for the current molecule
            base_graph_features = _extract_base_graph_features_from_mol_file(data['mol_file_path'], mol_name)
            
            # Combine all features for the original molecule
            
            original_entry = {
                **data, # Copy all existing metadata (label, global features, etc.)
                **base_graph_features, # Add the extracted graph features (NumPy arrays)
            }
            if (data['label'] == 2.0 or data['label'] == 3.0) and base_graph_features['num_nodes'] > 0:
                if random.random() < prob_to_keep_per_class[data['label']]:
                    augmented_data[mol_name] = original_entry
                continue #Skip augmentation for this class molecules
            #for those that will be augmented, add entry
            augmented_data[mol_name] = original_entry

            # Check if augmentation should be applied
            should_augment = False
            if data['label'] == 0.0:
                should_augment = True
                iter_count = 2
            if data['label'] == 1.0:
                should_augment = True
                iter_count = 10
            if data['label'] == 4.0:
                should_augment = True
                iter_count = 2

            #oversampling with isomorphism
            if should_augment and base_graph_features['num_nodes'] > 0:
                for aug_idx in range(iter_count):
                    permutation = np.random.permutation(base_graph_features['num_nodes'])
                    
                    # Permute node features
                    x_permuted = base_graph_features['node_features'][permutation, :]

                    # Permute adjacency indices
                    a_indices_permuted = np.copy(base_graph_features['adjacency_indices'])
                    
                    if a_indices_permuted.size > 0:
                        old_to_new_idx_map = {old_idx: new_idx for old_idx, new_idx in enumerate(permutation)}
                        a_indices_permuted[:, 0] = np.vectorize(old_to_new_idx_map.get)(a_indices_permuted[:, 0])
                        a_indices_permuted[:, 1] = np.vectorize(old_to_new_idx_map.get)(a_indices_permuted[:, 1])
                    
                    # Create a new entry for the augmented molecule
                    augmented_mol_name = f"{mol_name}_aug_perm{aug_idx+1}"
                    augmented_entry = {
                        **data, # Copy original metadata
                        'preferredName': augmented_mol_name, # Update preferredName
                        'node_features': x_permuted,
                        'adjacency_indices': a_indices_permuted,
                        'adjacency_values': base_graph_features['adjacency_values'],
                        'num_nodes': base_graph_features['num_nodes'],
                        'num_edges': base_graph_features['num_edges'],
                    }
                    augmented_data[augmented_mol_name] = augmented_entry

        except Exception as e:
            print(f"    Warning: Skipping molecule '{mol_name}' due to error during feature extraction or augmentation: {e}")
            continue

    print(f"\n--- Data augmentation finished. Total molecules (original + augmented): {len(augmented_data)} ---")
    return augmented_data

# Applies oversampling to a molecular dataset by augmenting minority classes
# to match the size of the dominant class.
def augment_molecular_data_oversampling(
    all_molecules_processed_data: dict,
) -> dict[str, dict]:
    """
    Applies node permutation augmentation to molecular graph data.
    Oversamples all minority classes to the size of the dominant class.
    
    Returns a new dictionary containing original and augmented molecules.
    """
    augmented_data = {}
    
    # 1. Calculate class distribution
    pre_balance_labels = [data['label'] for data in all_molecules_processed_data.values()]
    pre_balance_counts = Counter(pre_balance_labels)
    
    print("\n--- Label distribution before balancing ---")
    for label, count in sorted(pre_balance_counts.items()):
        print(f"  Class {label}: {count} molecules")
        
    # 2. Determine the dominant class and target count
    dominant_label = max(pre_balance_counts, key=pre_balance_counts.get)
    target_count = pre_balance_counts[dominant_label]
    
    print(f"\n--- The target count for all classes is {target_count}, based on the dominant class {dominant_label} ---")

    augment_factors = {}
    for label, count in pre_balance_counts.items():
        if label != dominant_label:
            # Calculate the number of augmentations for minority classes
            num_to_add = target_count - count
            iter_count = num_to_add // count + 1 if count > 0 else 1
            augment_factors[label] = iter_count
            print(f"  Augmenting class {label}: {num_to_add} more instances needed. We will permute each instance {iter_count} times.")
        else:
            print(f"  Dominant class {dominant_label} is not augmented, keeping all {target_count} instances.")

    
    # 3. Iterate through molecules and apply oversampling
    total_original_mols = len(all_molecules_processed_data)
    processed_count = 0
    print(f"\n--- Starting to balance the dataset with {total_original_mols} molecules ---")

    for mol_name, data in all_molecules_processed_data.items():
        processed_count += 1
        if processed_count % 500 == 0:
            print(f"  Processed {processed_count}/{total_original_mols} molecules.")
            
        try:
            # Extract graph features from the MOL file
            base_graph_features = _extract_base_graph_features_from_mol_file(data['mol_file_path'], mol_name)
            
            # If the molecule has no atoms, skip it
            if base_graph_features['num_nodes'] == 0:
                print(f"    Warning: Skipping molecule '{mol_name}' because it has no atoms.")
                continue

            # Create the original instance
            original_entry = {
                **data, 
                **base_graph_features, 
            }
            
            # First, add the original instance
            augmented_data[mol_name] = original_entry

            # Oversample minority classes
            if data['label'] in augment_factors:
                iter_count = augment_factors[data['label']]
                
                for aug_idx in range(iter_count):
                    permutation = np.random.permutation(base_graph_features['num_nodes'])
                    
                    x_permuted = base_graph_features['node_features'][permutation, :]
                    
                    a_indices_permuted = np.copy(base_graph_features['adjacency_indices'])
                    if a_indices_permuted.size > 0:
                        old_to_new_idx_map = {old_idx: new_idx for old_idx, new_idx in enumerate(permutation)}
                        a_indices_permuted[:, 0] = np.vectorize(old_to_new_idx_map.get)(a_indices_permuted[:, 0])
                        a_indices_permuted[:, 1] = np.vectorize(old_to_new_idx_map.get)(a_indices_permuted[:, 1])
                    
                    augmented_mol_name = f"{mol_name}_aug_perm{aug_idx+1}"
                    augmented_entry = {
                        **data,
                        'preferredName': augmented_mol_name,
                        'node_features': x_permuted,
                        'adjacency_indices': a_indices_permuted,
                        'adjacency_values': base_graph_features['adjacency_values'],
                        'num_nodes': base_graph_features['num_nodes'],
                        'num_edges': base_graph_features['num_edges'],
                    }
                    augmented_data[augmented_mol_name] = augmented_entry

        except Exception as e:
            print(f"    Warning: Skipping molecule '{mol_name}' due to error: {e}")
            continue

    print(f"\n--- Balancing complete. Total molecules (original + augmented): {len(augmented_data)} ---")
    
    final_counts = Counter([data['label'] for data in augmented_data.values()])
    print("--- Distribution after balancing ---")
    for label, count in sorted(final_counts.items()):
        print(f"  Class {label}: {count} molecules")

    return augmented_data

# --- _generate_graph_features_for_mol to accept pre-extracted features ---
# This function will now focus on converting NumPy arrays (which may be augmented) to TF Tensors.
def _generate_graph_features_for_mol(mol_name: str, mol_data: dict, common_global_features_list: list[str]):
    # mol_data now already contains 'node_features', 'adjacency_indices', etc. as NumPy arrays
    # It also contains 'mol_file_path' and other metadata.
    
    try:
        # Extract features that are already in mol_data (NumPy arrays)
        node_features = mol_data['node_features']
        adjacency_indices = mol_data['adjacency_indices']
        adjacency_values = mol_data['adjacency_values']
        num_nodes = mol_data['num_nodes']
        num_edges = mol_data['num_edges']

        # Global features need to be converted to a TensorFlow tensor
        processed_global_features = []
        for k in common_global_features_list:
            v = mol_data['global_features'].get(k)
            if v == 'NaN': # assuming 'NaN' is used for missing global features
                processed_global_features.append(np.nan)
            else:
                processed_global_features.append(float(v))
        global_features_tensor = tf.constant(processed_global_features, dtype=tf.float32)

        return {
            'preferredName': mol_data['preferredName'],
            'esr': mol_data.get('esr', 'unknown'),
            'label': mol_data['label'], # Label is already a float
            'hitc': mol_data['hitc'],
            'global_features': global_features_tensor,
            'node_features': node_features, # These are already NumPy arrays
            'edge_features': np.empty((0, 3), dtype=np.float32), # No edge features used here, default empty
            'adjacency_indices': adjacency_indices,
            'adjacency_values': adjacency_values,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'source_assay_name': mol_data['source_assay_name']
        }
    except Exception as e:
        print(f"Error generating TF features for '{mol_name}': {e}. Skipping.")
        return None
    
# Creates a TensorFlow `tf.data.Dataset` for Graph Neural Networks (GNNs) by using a generator to
# process molecular data. It iterates through a provided list of molecule IDs, fetches the corresponding
# pre-processed data, and converts it into a structured format containing graph features (nodes, edges,
# adjacency) and the molecule's label, which is then yielded for the dataset.
def create_tf_dataset_from_ids(
    mol_ids: list[str],
    all_molecules_processed_data: dict,
    common_global_features_list: list[str],
    output_signature: tuple
) -> tf.data.Dataset:
    def generator_fn():
        for mol_id in mol_ids:
            mol_data = all_molecules_processed_data.get(mol_id)
            if mol_data is None:
                print(f"Warning: Data for mol_id '{mol_id}' not found. Skipping.")
                continue

            processed_mol = _generate_graph_features_for_mol(mol_id, mol_data, common_global_features_list)
            if processed_mol is None:
                continue

            yield {
                'node_features': processed_mol['node_features'],
                'edge_features': processed_mol['edge_features'],
                'adjacency_indices': processed_mol['adjacency_indices'],
                'adjacency_values': processed_mol['adjacency_values'],
                'global_features': processed_mol['global_features'],
                'num_nodes': processed_mol['num_nodes'],
                'num_edges': processed_mol['num_edges'],
                'preferredName': processed_mol['preferredName'],
                'esr': processed_mol['esr'],
                'hitc':processed_mol['hitc'],
                'source_assay_name': processed_mol['source_assay_name']
            }, tf.constant(processed_mol['label'], dtype=tf.float32)

    return tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=output_signature
    )
def split_data_and_create_tf_datasets_by_id(
    all_molecules_processed_data: dict,
    test_split_ratio: float = 0.2,
    val_split_ratio: float = 0.2,
    random_seed: int = 42,
    common_global_features_list: list[str] = None
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    if not all_molecules_processed_data:
        print("Upozorenje: Prazan ulazni rečnik podataka. Vraćam prazne datasete.")
        return tf.data.Dataset.from_tensor_slices({}), tf.data.Dataset.from_tensor_slices({}), tf.data.Dataset.from_tensor_slices({})

    mol_ids = list(all_molecules_processed_data.keys())
    labels = np.array([all_molecules_processed_data[mol_id]['label'] for mol_id in mol_ids])

    train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
        mol_ids, labels,
        test_size=test_split_ratio,
        stratify=labels,
        random_state=random_seed
    )
    print(f"\nUkupno podataka: {len(mol_ids)}")
    print(f"Podela na Test ({test_split_ratio*100}%): {len(test_ids)} uzoraka.")
    print(f"Preostalo za Train+Val ({(1-test_split_ratio)*100}%): {len(train_val_ids)} uzoraka.")

    train_ids, val_ids, train_labels, val_labels = train_test_split(
        train_val_ids, train_val_labels,
        test_size=val_split_ratio,
        stratify=train_val_labels,
        random_state=random_seed
    )
    print(f"Podela Train+Val: Train ({len(train_ids)} uzoraka), Val ({len(val_ids)} uzoraka).")
    print(f"Finalni odnosi: Train: {len(train_ids)/len(mol_ids):.2f}, "
          f"Val: {len(val_ids)/len(mol_ids):.2f}, "
          f"Test: {len(test_ids)/len(mol_ids):.2f}")

    # determine output_signature
    try:
        dummy_atom_dim = get_atom_features(Chem.Atom(0)).shape[0]
    except Exception:
        print("Upozorenje: Nije moguće dobiti dimenziju atomskih feature-a iz Chem.Atom(0). Pretpostavljam 12.")
        dummy_atom_dim = 12

    try:
        dummy_mol_for_bond_dim = Chem.MolFromSmiles('CC')
        if dummy_mol_for_bond_dim and dummy_mol_for_bond_dim.GetNumBonds() > 0:
            dummy_bond_dim = get_bond_features(dummy_mol_for_bond_dim.GetBonds()[0]).shape[0]
        else:
            raise ValueError("Could not get bond feature dimension from dummy molecule.")
    except Exception:
        print("Upozorenje: Nije moguće dobiti dimenziju veza feature-a iz Chem.Bond() ili dummy molekula. Pretpostavljam 3.")
        dummy_bond_dim = 3

    if common_global_features_list is None or not common_global_features_list:
        if mol_ids:
            first_mol_id = mol_ids[0]
            first_mol_data = all_molecules_processed_data[first_mol_id]
            global_features_dim = len(first_mol_data['global_features'])
            print(f"Determinisana dimenzija globalnih karakteristika: {global_features_dim}")
        else:
            global_features_dim = 0
            print("Upozorenje: Nema podataka za određivanje dimenzije globalnih karakteristika. Postavljeno na 0.")
    else:
        global_features_dim = len(common_global_features_list)

    output_signature = (
        {
            'node_features': tf.TensorSpec(shape=(None, dummy_atom_dim), dtype=tf.float32),
            'edge_features': tf.TensorSpec(shape=(None, dummy_bond_dim), dtype=tf.float32),
            'adjacency_indices': tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
            'adjacency_values': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'global_features': tf.TensorSpec(shape=(global_features_dim,), dtype=tf.float32),
            'num_nodes': tf.TensorSpec(shape=(), dtype=tf.int32),
            'num_edges': tf.TensorSpec(shape=(), dtype=tf.int32),
            'preferredName': tf.TensorSpec(shape=(), dtype=tf.string),
            'esr': tf.TensorSpec(shape=(), dtype=tf.string),
            'hitc': tf.TensorSpec(shape=(), dtype=tf.float32),
            'source_assay_name': tf.TensorSpec(shape=(), dtype=tf.string)
        },
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )

    train_tf_dataset = create_tf_dataset_from_ids(
        train_ids, all_molecules_processed_data, common_global_features_list, output_signature
    )
    val_tf_dataset = create_tf_dataset_from_ids(
        val_ids, all_molecules_processed_data, common_global_features_list, output_signature
    )
    test_tf_dataset = create_tf_dataset_from_ids(
        test_ids, all_molecules_processed_data, common_global_features_list, output_signature
    )

    return train_tf_dataset, val_tf_dataset, test_tf_dataset


# --- storing datasets ---
def save_tf_datasets(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    output_directory: str = 'processed_tf_datasets'
):
    """
   Saves the training, validation, and test TensorFlow Datasets to the specified directory.

    Args:
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.
        test_ds (tf.data.Dataset): The test dataset.
        output_directory (str): The path to the directory where the datasets will be saved.
    """
    os.makedirs(output_directory, exist_ok=True)

    train_path = os.path.join(output_directory, 'train_dataset')
    val_path = os.path.join(output_directory, 'val_dataset')
    test_path = os.path.join(output_directory, 'test_dataset')

    print(f"\n--- Počinjem čuvanje TF Datasetova u {output_directory} ---")
    
    try:
        train_ds.save(train_path)
        print(f"Trening dataset sačuvan na: {train_path}")
    except Exception as e:
        print(f"Greška pri čuvanju trening dataseta: {e}")

    try:
        val_ds.save(val_path)
        print(f"Validacioni dataset sačuvan na: {val_path}")
    except Exception as e:
        print(f"Greška pri čuvanju validacionog dataseta: {e}")

    try:
        test_ds.save(test_path)
        print(f"Test dataset sačuvan na: {test_path}")
    except Exception as e:
        print(f"Greška pri čuvanju test dataseta: {e}")

    print("\n--- Čuvanje TF Datasetova završeno ---")



if __name__ == "__main__":
    test_base_dir = 'data'
    csv_dir = os.path.join(test_base_dir, 'binders_and_aa')
    mol_dir = os.path.join(test_base_dir, 'mols_from_original_csv')
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(mol_dir, exist_ok=True)

    csv_dataframes_map, mol_paths_map = load_all_data_into_maps(csv_dir, mol_dir)

    
    # 1. Process and combine molecular data (gets metadata and mol file paths)
    all_molecules_metadata = process_and_combine_molecular_data(
        csv_dataframes_map,
        mol_paths_map,
        ACTUAL_COLUMN_NAMES_PER_ASSAY_TYPE,
        COMMON_LABEL_NAME,
        COMMON_GLOBAL_FEATURES_LIST
    )

    
    # 2. PERFORM AUGMENTATION HERE
    # This function will load MOL files, extract graph features, and apply permutations.
    # It returns a new dictionary with original AND augmented molecules.
    """ stari pristup
    augmented_all_molecules_data = augment_molecular_data(
        all_molecules_metadata,
        augment_label_1=True,
        num_augmentations_per_sample=2 # Adjust as needed
    )""" 
    #novi pristup, cuvamo originale vecinske klase, a radimo oversampling ostalih sa isomorphismom
    augmented_all_molecules_data = augment_molecular_data_oversampling(
        all_molecules_metadata
    )

    
    print("\n--- Content of combined and augmented molecular data (first 5) ---")
    # For display, convert TensorFlow tensors back to numpy if needed
    for mol_name, data in list(augmented_all_molecules_data.items())[:5]:
        print(f"Molecule: {mol_name}")
        # 'mol_file_path' might not exist for augmented samples, check if you need it here
        if 'mol_file_path' in data:
            print(f"  Original MOL file path: {data['mol_file_path']}")
        print(f"Preferred name: {data['preferredName']}")
        print(f"  Label ('{COMMON_LABEL_NAME}'): {data['label']}")
        # Global features are now a TF Tensor, convert to numpy for printing if needed
        # print(f"  Global features ({len(data['global_features'])}): {data['global_features'].numpy()}") 
        print(f"  Node features shape: {data['node_features'].shape}")
        print(f"  Adjacency indices shape: {data['adjacency_indices'].shape}")
        print("-" * 20)
    
    if len(augmented_all_molecules_data) > 5:
        print(f"... and {len(augmented_all_molecules_data) - 5} more molecules.")
    print(f"Total processed and augmented molecules: {len(augmented_all_molecules_data)}")


    # 3. Split the AUGMENTED data and create TF Datasets
    train_ds, val_ds, test_ds = split_data_and_create_tf_datasets_by_id(
        augmented_all_molecules_data, # Pass the new, augmented dictionary here
        test_split_ratio=0.2, # Adjust ratios based on the increased dataset size
        val_split_ratio=0.25,
        common_global_features_list=COMMON_GLOBAL_FEATURES_LIST
    )

    # 4. Save the created datasets
    output_dir = '../data/tfrecords_multi_b_oversampled'
    os.makedirs(output_dir, exist_ok=True)

    save_tf_datasets(train_ds, val_ds, test_ds, output_directory=output_dir)
    #"""
