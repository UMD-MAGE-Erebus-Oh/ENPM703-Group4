"""

Dataset splits:
Train: 2022-23, 2023-24
Val: 2024-25
Test: 2025-26

splitting by season prevents dataset poisoning

"""

from data_utils import get_full_flu_dataset


def main():

    datasets = get_full_flu_dataset()

    for k in datasets:
        print(f"{k}: {datasets[k].shape}")
    


if __name__ == "__main__":
    main()