import os

import pandas as pd


def calculate_vqa_accuracy(csv_path):
    """Calculate overall VQA accuracy from a result CSV."""
    df = pd.read_csv(csv_path)

    answers = df["Answer"].values
    model_responses = df["Model_Response"].values

    correct_count = 0
    total_count = len(answers)

    for answer, model_response in zip(answers, model_responses):
        model_answer = str(model_response).strip().split(".")[0].strip()
        if model_answer == answer:
            correct_count += 1

    accuracy = correct_count / total_count if total_count else 0.0
    return accuracy, correct_count, total_count


def calculate_accuracy_by_category(csv_path, category_col="Broad Category"):
    """Calculate VQA accuracy grouped by category."""
    df = pd.read_csv(csv_path)
    categories = df[category_col].unique()
    category_stats = {}

    for category in categories:
        category_df = df[df[category_col] == category]
        answers = category_df["Answer"].values
        model_responses = category_df["Model_Response"].values

        correct_count = 0
        total_count = len(answers)

        for answer, model_response in zip(answers, model_responses):
            model_answer = str(model_response).strip().split(".")[0].strip()
            if model_answer == answer:
                correct_count += 1

        accuracy = correct_count / total_count if total_count else 0.0
        category_stats[category] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "wrong_count": total_count - correct_count,
        }

    return category_stats


def main():
    csv_path = "eval_results/SlideBench-VQA-TCGA_results.csv"

    if not os.path.exists(csv_path):
        print(f"Error: file does not exist: {csv_path}")
        return

    accuracy, correct_count, total_count = calculate_vqa_accuracy(csv_path)

    print("=" * 50)
    print("VQA accuracy results - overall")
    print("=" * 50)
    print(f"Total samples: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Wrong: {total_count - correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 50)

    print("\n")
    print("=" * 50)
    print("VQA accuracy results - by Broad Category")
    print("=" * 50)

    category_stats = calculate_accuracy_by_category(csv_path)

    print(f"{'Category':<20} {'Total':<10} {'Correct':<10} {'Wrong':<10} {'Accuracy':<10}")
    print("-" * 60)

    for category, stats in category_stats.items():
        print(
            f"{category:<20} {stats['total_count']:<10} {stats['correct_count']:<10} "
            f"{stats['wrong_count']:<10} {stats['accuracy']:.4f} ({stats['accuracy'] * 100:.2f}%)"
        )

    print("=" * 50)


if __name__ == "__main__":
    main()
