from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

INPUT_CSV = Path("data/splits/duplicate_report.csv")
OUTPUT_CSV = Path("data/splits/final_split_5fold.csv")

TEST_SIZE = 0.15
N_FOLDS = 5
RANDOM_STATE = 42


def main():
    df = pd.read_csv(INPUT_CSV)

    df["kept"] = df["kept"].astype(str).str.lower().eq("true")
    df = df[df["kept"]].copy().reset_index(drop=True)

    trainval_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["class"],
        random_state=RANDOM_STATE,
    )

    trainval_df = trainval_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    trainval_df["subset"] = "trainval"
    test_df["subset"] = "test"

    trainval_df["fold"] = -1
    test_df["fold"] = -1

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold_idx, (_, val_idx) in enumerate(
        skf.split(trainval_df["path"], trainval_df["class"])
    ):
        trainval_df.loc[val_idx, "fold"] = fold_idx

    final_df = pd.concat([trainval_df, test_df], ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Total imágenes limpias: {len(df)}")
    print(f"Trainval: {len(trainval_df)}")
    print(f"Test fijo: {len(test_df)}")
    print(f"Archivo guardado en: {OUTPUT_CSV}")
    print()
    print("Distribución test:")
    print(test_df["class"].value_counts().sort_index())
    print()
    print("Distribución trainval por fold:")
    print(pd.crosstab(trainval_df["fold"], trainval_df["class"]))


if __name__ == "__main__":
    main()