import os
import sys
import torch
import numpy as np
import pandas as pd

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
EXCEL_PATH = os.path.join(DATA_DIR, 'gore_data.xlsx')
OUTPUT_PATH = os.path.join(DATA_DIR, 'Gore_processed_data.pt')


def parse_species_sequences(excel_path):
    """
    Extracts 16S sequences and creates the canonical 0-indexed species mapping.
    Species1 -> 0, Species2 -> 1, ..., Species48 -> 47.
    """
    df_seq = pd.read_excel(excel_path, sheet_name='Species 16s sequence')
    
    species_list = []
    seq_dict = {}
    
    # Handle header row (which contains Species1 and its sequence as column names)
    col0 = str(df_seq.columns[0]).strip()
    col1 = str(df_seq.columns[1]).strip()
    species_list.append(col0)
    seq_dict[col0] = col1
    
    # Remaining species (Species2 to Species48)
    for idx, row in df_seq.iterrows():
        sp = str(row.iloc[0]).strip()
        seq = str(row.iloc[1]).strip()
        species_list.append(sp)
        seq_dict[sp] = seq
        
    species_to_idx = {sp: i for i, sp in enumerate(species_list)}
    assert len(species_list) == 48, f"Expected 48 species, found {len(species_list)}"
    return species_list, species_to_idx, seq_dict


def parse_abundance_sheet(excel_path, sheet_name, species_to_idx):
    """
    Parses one of the species abundance sheets ('LN species abundances',
    'MN species abundances', 'HN Species abundances').
    
    Returns:
        trajs_raw: dict of (S, comm_name) -> (timepoints, 48) raw read counts
        reps_d10_raw: dict of (S, comm_name) -> (3, 48) Day 10 replicate read counts
        days_per_comm: dict of (S, comm_name) -> list of day strings
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    current_S = 3
    current_comm = None
    trajs_raw = {}
    reps_d10_raw = {}
    days_per_comm = {}
    
    # Tracking current community species and reads
    comm_species = []
    comm_ts_rows = []
    comm_rep_rows = []
    
    def save_current_comm():
        if current_comm is not None and len(comm_species) > 0:
            key = (current_S, current_comm)
            num_sp = len(comm_species)
            
            ts_matrix = np.array(comm_ts_rows, dtype=np.float32) # (num_sp, time_len)
            rep_matrix = np.array(comm_rep_rows, dtype=np.float32) # (num_sp, 3)
            
            num_timepoints = ts_matrix.shape[1]
            full_ts = np.zeros((num_timepoints, 48), dtype=np.float32)
            full_reps = np.zeros((3, 48), dtype=np.float32)
            
            for s_idx, sp_name in enumerate(comm_species):
                if sp_name in species_to_idx:
                    g_idx = species_to_idx[sp_name]
                    full_ts[:, g_idx] = ts_matrix[s_idx, :]
                    full_reps[:, g_idx] = rep_matrix[s_idx, :]
                    
            trajs_raw[key] = full_ts
            reps_d10_raw[key] = full_reps
            
            if num_timepoints == 11:
                days_per_comm[key] = [f'Day{d}' for d in range(11)]
            else:
                days_per_comm[key] = ['Day7', 'Day8', 'Day9', 'Day10']
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        row_str = ' '.join([str(v) for v in row.values])
        
        # Check S indicator
        if 'S=' in row_str:
            for val in row.values:
                if isinstance(val, str) and 'S=' in val:
                    s_val = val.split('=')[1].strip()
                    if s_val.isdigit():
                        save_current_comm()
                        current_S = int(s_val)
                        current_comm = None
                        comm_species = []
                        comm_ts_rows = []
                        comm_rep_rows = []
                        
        first_val = str(row.iloc[0]).strip()
        if first_val.startswith('Community'):
            save_current_comm()
            current_comm = first_val
            comm_species = []
            comm_ts_rows = []
            comm_rep_rows = []
            continue
            
        # Detect species column dynamically (col 11 for 11-day layout, col 4 for 4-day layout)
        sp_name = None
        for col_cand in [11, 4]:
            val = str(row.iloc[col_cand]).strip()
            if val.startswith('Species'):
                sp_name = val
                break
                
        if sp_name is not None and current_comm is not None:
            comm_species.append(sp_name)
            
            if 'MN' in sheet_name and current_S <= 12:
                time_cols = [0, 1, 2, 3] # Day7, Day8, Day9, Day10
                rep_cols = [5, 6, 7] # Day10 Rep1, Rep2, Rep3
            else:
                time_cols = list(range(11)) # Day0..Day10
                rep_cols = [12, 13, 14] # Day10 Rep1, Rep2, Rep3
                
            ts_vals = [pd.to_numeric(row.iloc[c], errors='coerce') for c in time_cols]
            rep_vals = [pd.to_numeric(row.iloc[c], errors='coerce') for c in rep_cols]
            
            ts_vals = [0.0 if np.isnan(v) else v for v in ts_vals]
            rep_vals = [0.0 if np.isnan(v) else v for v in rep_vals]
            
            comm_ts_rows.append(ts_vals)
            comm_rep_rows.append(rep_vals)
            
    # Save the final community (Community1 of S=48)
    save_current_comm()
    
    assert len(trajs_raw) == 33, f"Expected 33 communities in {sheet_name}, extracted {len(trajs_raw)}"
    return trajs_raw, reps_d10_raw, days_per_comm


def parse_biomass_sheet(excel_path, sheet_name):
    """
    Parses optical density (OD600) biomass sheets ('LN Biomass (OD)',
    'MN Biomass (OD)', 'HN Biomass (OD)').
    
    Each sheet contains 33 communities across S in {3, 6, 12, 24, 48}.
    Each community has 3 rows of replicates across 11 days (Day 0 to Day 10).
    
    Returns:
        biomass_dict: dict of (S, comm_name) -> np.ndarray of shape (3, 11)
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # Locate all Replicate1 rows
    rep1_indices = df[df.iloc[:, 11] == 'Replicate1'].index.tolist()
    rep1_indices = rep1_indices[:33]
    
    expected_s_order = [3]*8 + [6]*8 + [12]*8 + [24]*8 + [48]*1
    expected_comm_order = [f'Community{i}' for i in range(1, 9)] * 4 + ['Community1']
    
    biomass_dict = {}
    for block_idx, r in enumerate(rep1_indices):
        s_val = expected_s_order[block_idx]
        c_name = expected_comm_order[block_idx]
        key = (s_val, c_name)
        
        rep_rows = []
        for rep_offset in range(3):
            row_vals = [pd.to_numeric(df.iloc[r + rep_offset, c], errors='coerce') for c in range(11)]
            row_vals = [0.0 if np.isnan(v) else v for v in row_vals]
            rep_rows.append(row_vals)
            
        biomass_dict[key] = np.array(rep_rows, dtype=np.float32)
        
    assert len(biomass_dict) == 33, f"Expected 33 communities in {sheet_name}, extracted {len(biomass_dict)}"
    return biomass_dict


def normalize_to_relative_abundance(raw_reads, presence_mask, pseudo_count=1e-6):
    """
    Converts raw sequencing read counts to relative abundances.
    Adds a small pseudo-count only to member species of the community's pool,
    preserving exact zeros for absent species.
    """
    total_reads = raw_reads.sum(axis=-1, keepdims=True)
    safe_total = np.where(total_reads > 0, total_reads, 1.0)
    rel_raw = raw_reads / safe_total
    
    mask_expanded = presence_mask.reshape((1, -1)) if rel_raw.ndim == 2 else presence_mask.reshape((1, 1, -1))
    rel_eps = rel_raw + pseudo_count * mask_expanded
    total_eps = rel_eps.sum(axis=-1, keepdims=True)
    total_eps = np.where(total_eps > 0, total_eps, 1.0)
    rel_normalized = rel_eps / total_eps
    
    return rel_raw, rel_normalized


def calculate_stability_metrics(late_abundances):
    """
    Computes species abundance Coefficient of Variation (CV) over Days 7 to 10.
    A community is classified as 'fluctuating' if mean CV >= 0.25 (Gore et al. Science 2022).
    """
    mean_ab = late_abundances.mean(axis=0)
    std_ab = late_abundances.std(axis=0)
    
    surv_mask = mean_ab >= 1e-3
    if np.any(surv_mask):
        cv_per_species = std_ab[surv_mask] / mean_ab[surv_mask]
        mean_cv = float(np.mean(cv_per_species))
    else:
        mean_cv = 0.0
        
    is_fluctuating = bool(mean_cv >= 0.25)
    return mean_cv, is_fluctuating


def main():
    print(f"Loading and processing {EXCEL_PATH}...")
    
    # 1. Parse species sequences and mapping
    species_list, species_to_idx, seq_dict = parse_species_sequences(EXCEL_PATH)
    print(f"Loaded {len(species_list)} species ({species_list[0]} to {species_list[-1]}).")
    
    # 2. Parse species abundances across conditions
    conditions = ['LN', 'MN', 'HN']
    sheet_map = {
        'LN': 'LN species abundances',
        'MN': 'MN species abundances',
        'HN': 'HN Species abundances'
    }
    
    raw_reads_by_cond = {}
    reps_d10_by_cond = {}
    days_by_cond = {}
    
    for cond in conditions:
        sheet = sheet_map[cond]
        trajs, reps, days = parse_abundance_sheet(EXCEL_PATH, sheet, species_to_idx)
        raw_reads_by_cond[cond] = trajs
        reps_d10_by_cond[cond] = reps
        days_by_cond[cond] = days
        print(f"Sheet '{sheet}': {len(trajs)} communities parsed.")
        
    # 3. Parse biomass (OD600)
    biomass_sheet_map = {
        'LN': 'LN Biomass (OD)',
        'MN': 'MN Biomass (OD)',
        'HN': 'HN Biomass (OD)'
    }
    biomass_by_cond = {}
    for cond in conditions:
        sheet = biomass_sheet_map[cond]
        biomass_dict = parse_biomass_sheet(EXCEL_PATH, sheet)
        biomass_by_cond[cond] = biomass_dict
        print(f"Sheet '{sheet}': {len(biomass_dict)} biomass trajectories parsed.")
        
    # 4. Generate Presence Masks & Community Metadata
    standard_keys = []
    for s in [3, 6, 12, 24]:
        for c in range(1, 9):
            standard_keys.append((s, f'Community{c}'))
    standard_keys.append((48, 'Community1'))
    
    presence_masks = {}
    metadata_rows = []
    
    for comm_id, key in enumerate(standard_keys):
        s_val, c_name = key
        all_reads_ln = raw_reads_by_cond['LN'][key]
        sp_indices = np.where(all_reads_ln.sum(axis=0) > 0)[0].tolist()
        
        assert len(sp_indices) == s_val, f"Community {key} expected {s_val} species, found {len(sp_indices)}"
        
        mask = np.zeros(48, dtype=bool)
        mask[sp_indices] = True
        presence_masks[key] = torch.tensor(mask, dtype=torch.bool)
        
        member_species_names = [species_list[i] for i in sp_indices]
        
        metadata_rows.append({
            'community_id': comm_id,
            'S': s_val,
            'community_name': c_name,
            'key': key,
            'species_indices': sp_indices,
            'species_names': member_species_names
        })
        
    metadata_df = pd.DataFrame(metadata_rows)
    print(f"\nMetadata generated for {len(metadata_df)} communities.")
    
    # 5. Normalization & Stability Calculation
    # Apply dispersal floor (1e-6) to member species so no member species is ever at exact 0.0
    rel_abundances = {}
    rel_abundances_raw = {}
    stability_metrics = {}
    
    for cond in conditions:
        stability_metrics[cond] = {}
        for key in standard_keys:
            raw_ts = raw_reads_by_cond[cond][key]
            mask = presence_masks[key].numpy()
            
            rel_raw, rel_floored = normalize_to_relative_abundance(raw_ts, mask, pseudo_count=1e-6)
            rel_abundances[(cond, key[0], key[1])] = torch.tensor(rel_floored, dtype=torch.float32)
            rel_abundances_raw[(cond, key[0], key[1])] = torch.tensor(rel_raw, dtype=torch.float32)
            
            # Days 7..10 for stability (using raw frequencies)
            late_ab = rel_raw[-4:, :]
            mean_cv, is_fluc = calculate_stability_metrics(late_ab)
            stability_metrics[cond][key] = {
                'mean_cv': mean_cv,
                'is_fluctuating': is_fluc
            }
            
    # Add stability to metadata_df
    for cond in conditions:
        metadata_df[f'{cond}_CV'] = [stability_metrics[cond][k]['mean_cv'] for k in standard_keys]
        metadata_df[f'{cond}_fluctuating'] = [stability_metrics[cond][k]['is_fluctuating'] for k in standard_keys]
        
    print("\nEmpirical Stability Summary (Fraction Fluctuating by S):")
    for s in [3, 6, 12, 24, 48]:
        sub = metadata_df[metadata_df['S'] == s]
        ln_fluc = sub['LN_fluctuating'].sum()
        mn_fluc = sub['MN_fluctuating'].sum()
        hn_fluc = sub['HN_fluctuating'].sum()
        total = len(sub)
        print(f"  S={s:2d} ({total} comms): LN={ln_fluc}/{total} | MN={mn_fluc}/{total} | HN={hn_fluc}/{total}")
        
    # Convert all raw reads, reps, biomass to PyTorch tensors
    torch_raw_reads = {
        (cond, k[0], k[1]): torch.tensor(raw_reads_by_cond[cond][k], dtype=torch.float32)
        for cond in conditions for k in standard_keys
    }
    torch_day10_reps = {
        (cond, k[0], k[1]): torch.tensor(reps_d10_by_cond[cond][k], dtype=torch.float32)
        for cond in conditions for k in standard_keys
    }
    torch_biomass = {
        (cond, k[0], k[1]): torch.tensor(biomass_by_cond[cond][k], dtype=torch.float32)
        for cond in conditions for k in standard_keys
    }
    torch_presence_masks = {
        k: presence_masks[k] for k in standard_keys
    }
    
    # 6. Save unified dataset
    processed_dataset = {
        'relative_abundance': rel_abundances,         # Floored with 1e-6 on member species (no 0 -> bloom issues)
        'raw_relative_abundance': rel_abundances_raw, # Exact raw fractions (unfloored)
        'raw_reads': torch_raw_reads,                 # Raw sequencing read counts
        'day10_replicates': torch_day10_reps,         # (cond, S, comm) -> (3, 48)
        'biomass_od': torch_biomass,                  # (cond, S, comm) -> (3, 11)
        'presence_masks': torch_presence_masks,       # (S, comm) -> (48,) bool
        'metadata': metadata_df,                      # DataFrame of all 33 communities
        'species': species_list,                      # List of 48 species names
        'species_to_idx': species_to_idx,             # Dict mapping species -> 0..47
        'sequences': seq_dict,                        # 16S sequences
        'dilution_factor': 30.0,                      # 30-fold daily dilution
        'dispersal_rate': 1e-6,                       # 10^-6 daily dispersal rate
        'timepoints': days_by_cond                    # Day names for each trajectory
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(processed_dataset, OUTPUT_PATH)
    print(f"\n[SUCCESS] Successfully saved standardized dataset to:\n  {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")


if __name__ == '__main__':
    main()
