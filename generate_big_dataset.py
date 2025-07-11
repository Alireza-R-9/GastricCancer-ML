import pandas as pd
import numpy as np
import os

REAL_DATA_PATH = "data/gastric.csv"
OUTPUT_PATH = "data/gastric_big.csv"


def main():
    print("🔍 Reading original dataset...")
    df_real = pd.read_csv(REAL_DATA_PATH)
    n_real = df_real.shape[0]
    feature_cols = df_real.columns[:-1]  # All except 'Diagnosis'
    target_col = df_real.columns[-1]  # 'Diagnosis'

    print(f"✅ Real data shape: {df_real.shape}")

    # Count real distribution
    label_counts = df_real[target_col].value_counts()
    ratio_cancer = label_counts.get('Cancer', 0) / n_real
    ratio_healthy = label_counts.get('Healthy', 0) / n_real

    n_target = 2_000_000
    n_synthetic = n_target - n_real
    n_cancer = int(n_synthetic * ratio_cancer)
    n_healthy = n_synthetic - n_cancer

    print(f"📊 Synthetic: Cancer={n_cancer}, Healthy={n_healthy}")

    # Calculate stats for each feature
    df_stats = df_real[feature_cols].describe().T

    def generate_samples(n, label):
        data = {}
        for col in feature_cols:
            mean = df_stats.loc[col, "mean"]
            std = df_stats.loc[col, "std"]
            data[col] = np.random.normal(loc=mean, scale=std, size=n).round(2)
        data[target_col] = [label] * n
        return pd.DataFrame(data)

    print("⚙️ Generating synthetic samples...")
    df_synth_cancer = generate_samples(n_cancer, "Cancer")
    df_synth_healthy = generate_samples(n_healthy, "Healthy")
    df_synth = pd.concat([df_synth_cancer, df_synth_healthy], ignore_index=True)

    print("🔗 Combining with real data...")
    df_final = pd.concat([df_real, df_synth], ignore_index=True).sample(frac=1, random_state=42)

    os.makedirs("data", exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Final dataset saved: {OUTPUT_PATH}")
    print(f"📦 Final shape: {df_final.shape}")


if __name__ == "__main__":
    main()
