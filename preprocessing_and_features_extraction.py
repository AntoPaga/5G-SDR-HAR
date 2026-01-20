import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def extract_csi_matrix(df, num_sub):
    csi_list_complex = []
    total_iterations = len(df)

    for index, row in tqdm(df.iterrows(), total=total_iterations, desc="Estrazione CSI"):
        csi_curr = row["CSI"].replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")
        csi_list_float = [float(vv) for vv in csi_curr]

        for csi_index in range(0, len(csi_list_float), 2):
            x_real = csi_list_float[csi_index]
            x_imag = csi_list_float[csi_index + 1]
            csi_list_complex.append(complex(x_real, x_imag))

    csi_list_complex = np.array(csi_list_complex)
    n = csi_list_complex.size
    N = num_sub
    M = n // N
    csi_matrix = csi_list_complex.reshape((M, N))
    return csi_matrix



def compute_features_stats(matrix, subcarrier_index):
    phases = np.unwrap(np.angle(matrix))
    relative_phase_shift = np.diff(phases[:, subcarrier_index])

    df_stats = pd.DataFrame(relative_phase_shift, columns=['phase'])
    stats = df_stats.describe()
    stats = stats.drop(index="count", errors='ignore')
    return stats['phase'].to_dict()



def process_file(file_name, target_sym_size, num_sub, subcarrier_index, step=2000):
    data = []
    features = []
    indices = []

    with open(file_name, 'r') as file:
        total_lines = sum(1 for _ in file)

    for start in range(2000, 100000, step):
        end = start + step
        
        with open(file_name, 'r') as file:
            for cc, line in enumerate(file):
                if cc < start:
                    continue
                if cc >= end:
                    break

                try:
                    elements = line.strip().split(';')
                    data.append(elements)
                except Exception as e:
                    print(f"Errore durante l'elaborazione della riga: {e}")
        
        df = pd.DataFrame(data, columns=[' ', 'Timestamp', 'DMRS_id', 'Layer(MIMO)', 'Sym_size', 'CSI'])
        filtered_df = df[df['Sym_size'] == target_sym_size]
        if filtered_df.empty:
            data = []
            continue

        csi_matrix = extract_csi_matrix(filtered_df, num_sub)
        stats_dict = compute_features_stats(csi_matrix, subcarrier_index)

        features.append(stats_dict)
        indices.append(start)
        data = []  

    return features, indices



def plot_features(features, indices, label):
    features = np.array(features)
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Feature Extraction for {label} - Subcarrier 64', fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(indices, features[:, 0], marker='o')
    plt.title('Phase Mean')
    plt.xlabel('File Segment Index')
    plt.ylabel('Value')

    plt.subplot(1, 2, 2)
    plt.plot(indices, features[:, 1], marker='o')
    plt.title('Phase Variance')
    plt.xlabel('File Segment Index')
    plt.ylabel('Value')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



# ------------------ MAIN ------------------

ROOT_DIR = "C:\\Users\\giova\\Downloads"
OUTPUT_DIR = "dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input files and labels
file_names = [
    {"name": os.path.join(ROOT_DIR, "IDLE_9_35_80cm.csv"), "label": "IDLE"},
    {"name": os.path.join(ROOT_DIR, "WALK_10_12_80cm.csv"), "label": "WALK"},
    {"name": os.path.join(ROOT_DIR, "SIT_11_15_80cm.csv"), "label": "SIT"},
]
target_sym_size = '276'
num_sub = 276
subcarrier_index = 200  

combined_df = pd.DataFrame()

for file in file_names:
    file_name = file["name"]
    label = file["label"]

    features, indices = process_file(file_name, target_sym_size, num_sub, subcarrier_index)
    
    features_df = pd.DataFrame(features)
    features_df['index'] = indices
    features_df['label'] = label
    
    combined_df = pd.concat([combined_df, features_df], ignore_index=True)

    output_csv_file = os.path.join(OUTPUT_DIR, f'{label}_features.csv')
    features_df.to_csv(output_csv_file, index=False)
    print(f"DataFrame salvato in: {output_csv_file}")

combined_output_csv_file = os.path.join(OUTPUT_DIR, 'combined_features.csv')
combined_df.to_csv(combined_output_csv_file, index=False)
print(f"DataFrame combinato salvato in: {combined_output_csv_file}")

shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
shuffled_output_csv_file = os.path.join(OUTPUT_DIR, 'dataset.csv')
shuffled_df.to_csv(shuffled_output_csv_file, index=False)
print(f"DataFrame combinato dopo lo shuffle salvato in: {shuffled_output_csv_file}")

for label in combined_df['label'].unique():
    label_df = combined_df[combined_df['label'] == label]
    plot_features(
        label_df[['mean', 'std']].values,  
        label_df['index'].values,
        label
    )
