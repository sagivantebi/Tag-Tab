import warnings
import datetime
import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import matplotlib
import os
import pandas as pd

logging.getLogger().setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level='ERROR')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_metrics(scores, labels):
    labels = np.array(labels, dtype=int)  # Ensure labels are in binary format
    scores = np.array(scores)  # Ensure scores are in the correct format

    # Calculate ROC curve and AUROC
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)

    # Check if AUROC is below 0.5 and invert scores if necessary
    if auroc < 0.5:
        scores = -scores  # Invert scores
        fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
        auroc = auc(fpr_list, tpr_list)

    # Calculate FPR at TPR >= 0.95
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]

    # Calculate TPR at FPR <= 0.05
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]

    # Calculate accuracy
    acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)

    return auroc, fpr95, tpr05, acc


def do_plot(prediction, answers, legend="", output_dir=None):
    fpr, tpr, auc_score, acc = get_metrics(prediction, answers)
    print(f"FPR: {fpr}")
    print(f"TPR: {tpr}")
    print(f"Type of FPR: {type(fpr)}, Type of TPR: {type(tpr)}")
    if len(fpr) == 0 or len(tpr) == 0:
        print(f"No valid FPR/TPR values for {legend}")
        low = 0.0
    else:
        fpr_below_threshold = fpr[fpr < .05]
        if len(fpr_below_threshold) == 0:
            print(f"No FPR values below 0.05 for {legend}")
            low = 0.0
        else:
            low = tpr[np.where(fpr < .05)[0][-1]]
    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\n' % (legend, auc_score, acc, low))
    metric_text = 'auc=%.3f' % auc_score
    plt.plot(fpr, tpr, label=legend + metric_text)
    return legend, auc_score, acc, low


def fig_fpr_tpr(all_output, output_dir):
    print("output_dir", output_dir)
    answers = None
    metric2predictions = defaultdict(list)

    # Collect labels and predictions
    for ex in all_output:
        if answers is None:
            answers = ex["label"]
            if not isinstance(answers, list):
                answers = [answers]  # Ensure labels are a list
            print(f"Collected {len(answers)} labels from one example.")
        for metric, scores in ex["pred"].items():
            if not isinstance(scores, list):
                scores = [scores]  # Ensure scores are a list
            print(f"Collected {len(scores)} scores for metric {metric} from one example.")
            metric2predictions[metric].extend(scores)

    print(f"Total collected labels: {len(answers)}")

    # Check if lengths of answers and each set of predictions match
    valid_metrics = {}
    for metric, predictions in metric2predictions.items():
        print(f"Checking lengths for metric {metric}: {len(predictions)} predictions vs {len(answers)} answers")
        if len(predictions) != len(answers):
            print(f"Length mismatch for metric {metric}: {len(predictions)} predictions vs {len(answers)} answers")
            continue
        valid_metrics[metric] = predictions

    print(f"Valid metrics: {list(valid_metrics.keys())}")

    # Plotting and writing to file
    auc_file_path = f"{output_dir}/auc.txt"
    with open(auc_file_path, "w") as f:
        for metric, predictions in valid_metrics.items():
            print(f"Processing metric {metric} with {len(predictions)} predictions and {len(answers)} answers")
            plt.figure(figsize=(4, 3))
            legend, auc_score, acc, low = do_plot(predictions, answers, legend=metric, output_dir=output_dir)
            f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n' % (legend, auc_score, acc, low))

            plt.semilogx()
            plt.semilogy()
            plt.xlim(1e-5, 1)
            plt.ylim(1e-5, 1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.plot([0, 1], [0, 1], ls='--', color='gray')
            plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
            plt.legend(fontsize=8)
            plt.savefig(f"{output_dir}/{metric}_auc.png")
            plt.close()  # Close the plot to avoid overlap in subsequent iterations
            print(f"Saved plot for metric {metric} to {output_dir}/{metric}_auc.png")

    print(f"AUC results saved to {auc_file_path}")


def save_metrics_to_csv(results, output_dir, model_name, dataset):
    # Convert the results dictionary to a DataFrame
    df = pd.DataFrame(results)

    # Specify the attacks to keep
    attacks_to_keep = [
        "ppl", "ppl_zlib",
        "Min_20.0% Prob", "Max_20.0% Prob", 'recall', 'DC-PDD_Score',
        "MinK++_20.0% Prob"] + ["tag_tab_AT_k=" + str(i) for i in range(1,11)] + ["random_words_mean_prob_k=" + str(i) for i in range(1,11)]

    # Filter the DataFrame to include only the specified attacks
    df_filtered = df[df['method'].isin(attacks_to_keep)]

    # Drop the fpr95 column
    df_filtered = df_filtered.drop(columns=['fpr95'])

    # Specify the CSV file path
    csv_file_name = f'metrics_results_{model_name}_{dataset}.csv'
    csv_file_path = os.path.join(output_dir, csv_file_name)

    # Save the DataFrame to CSV with the specified header
    df_filtered.to_csv(csv_file_path, index=False, header=["method", "auroc", "tpr05", "acc"])

    print(f"Results saved to {csv_file_path}")


def evaluate_and_save_results(scores_dict, data, dataset, model_id):
    labels = [d['label'] for d in data]  # Ensure labels are binary
    results = defaultdict(list)
    all_output = []

    for method, scores in scores_dict.items():
        clean_scores = []
        clean_labels = []
        for i, score in enumerate(scores):
            try:
                numeric_score = float(score)
                if not (np.isnan(numeric_score) or np.isinf(numeric_score)):
                    clean_scores.append(numeric_score)
                    clean_labels.append(labels[i])
            except ValueError:
                continue

        if len(clean_scores) != len(clean_labels):  # Ensure lengths match
            print(f"Length mismatch for method {method}: {len(clean_scores)} scores vs {len(clean_labels)} labels")
            continue

        if not clean_scores:  # Check if clean_scores is empty
            print(f"No valid scores for method: {method}")
            continue

        auroc, fpr95, tpr05, acc = get_metrics(clean_scores, clean_labels)
        results['method'].append(method)
        results['auroc'].append(f"{auroc:.1%}")
        results['fpr95'].append(f"{fpr95:.1%}")
        results['tpr05'].append(f"{tpr05:.1%}")
        results['acc'].append(f"{acc:.1%}")

        all_output.append({
            "label": clean_labels,
            "pred": {method: clean_scores}
        })

    df = pd.DataFrame(results)
    print(df)

    save_root = f"results/{dataset}"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # Save the results to CSV
    save_metrics_to_csv(results, save_root, model_id.split('_')[0], dataset)

    if os.path.isfile(os.path.join(save_root, f"{model_id}.csv")):
        df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False, mode='w', header=False)
    else:
        df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False)

    # fig_fpr_tpr(all_output, save_root)


def evaluate_books(csv_path):
    # Extract dataset and model information from the path
    parts = csv_path.split('/')
    dataset = parts[0]
    model_info = parts[2]

    # Extract model name and date from the model_info
    model_name = model_info.split('_')[0]
    date_info = datetime.datetime.now().strftime("%Y_%m_%d")
    model_id = f"{model_name}_{date_info}"

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Exclude the 'FILE_PATH' and 'label' columns to get all other features dynamically
    attacks = [col for col in df.columns if col not in ['FILE_PATH', 'label']]

    # Extract labels
    labels = df['label'].tolist()

    # Prepare scores dictionary
    scores_dict = {attack: df[attack].tolist() for attack in attacks}

    # Evaluate and save results
    evaluate_and_save_results(scores_dict, [{'label': label} for label in labels], dataset, model_id)


if __name__ == "__main__":
    results_books = \
        [
            "BookMIA/BookMIA_Results_To_be_covered/M=llama-7b_K=5_D=BookMIA_2024_07_24_16_48_25.csv",
            "BookMIA/BookMIA_Results_To_be_covered/M=llama-13b_K=5_D=BookMIA_2024_07_24_18_32_21.csv",
            "BookMIA/BookMIA_Results_To_be_covered/M=pythia-6.9b_K=5_D=BookMIA_2024_07_25_02_32_52.csv",
        ]

    for r in results_books:
        evaluate_books(r)
