import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import matplotlib
import os
import seaborn as sns
import pandas as pd
from datetime import datetime

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

    return fpr_list, tpr_list, auroc, fpr95, tpr05, acc


def do_plot(prediction, answers, legend="", output_dir=None):
    fpr, tpr, auc_score, fpr95, tpr05, acc = get_metrics(prediction, answers)
    # print(f"FPR: {fpr}")
    # print(f"TPR: {tpr}")
    # print(f"Type of FPR: {type(fpr)}, Type of TPR: {type(tpr)}")
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
    # print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\n' % (legend, auc_score, acc, low))
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
            # print(f"Collected {len(answers)} labels from one example.")
        for metric, scores in ex["pred"].items():
            if not isinstance(scores, list):
                scores = [scores]  # Ensure scores are a list
            # print(f"Collected {len(scores)} scores for metric {metric} from one example.")
            metric2predictions[metric].extend(scores)

    # print(f"Total collected labels: {len(answers)}")

    # Check if lengths of answers and each set of predictions match
    valid_metrics = {}
    for metric, predictions in metric2predictions.items():
        # print(f"Checking lengths for metric {metric}: {len(predictions)} predictions vs {len(answers)} answers")
        if len(predictions) != len(answers):
            # print(f"Length mismatch for metric {metric}: {len(predictions)} predictions vs {len(answers)} answers")
            continue
        valid_metrics[metric] = predictions

    print(f"Valid metrics: {list(valid_metrics.keys())}")

    # Plotting and writing to file
    auc_file_path = f"{output_dir}/auc.txt"
    with open(auc_file_path, "w") as f:
        for metric, predictions in valid_metrics.items():
            # print(f"Processing metric {metric} with {len(predictions)} predictions and {len(answers)} answers")
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
            # plt.savefig(f"{output_dir}/{metric}_auc.png")
            plt.close()  # Close the plot to avoid overlap in subsequent iterations
            # print(f"Saved plot for metric {metric} to {output_dir}/{metric}_auc.png")

    print(f"AUC results saved to {auc_file_path}")


def save_metrics_to_csv(results, output_dir, model_name, dataset):
    # Convert the results dictionary to a DataFrame
    df = pd.DataFrame(results)

    # Specify the attacks to keep
    attacks_to_keep = [
        "ppl", "ppl_zlib",
        "Min_20.0% Prob", "Max_20.0% Prob", "MinK++_20.0% Prob", 'recall', 'DC-PDD_Score',
        "tag_tab_FT_k=1", "tag_tab_FT_k=4", "tag_tab_FT_k=10"
    ]

    # Filter the DataFrame to include only the specified attacks
    df_filtered = df[df['method'].isin(attacks_to_keep)]

    # Drop the fpr95 column
    df_filtered = df_filtered.drop(columns=['fpr95'])

    # Specify the CSV file path
    csv_file_name = f'metrics_results_{model_name}_{dataset}.csv'
    csv_file_path = os.path.join(output_dir, csv_file_name)

    # Save the DataFrame to CSV with the specified header
    df_filtered.to_csv(csv_file_path, index=False, header=["method", "AUC", "TPR@FPR=5%", "Accuracy", "pile_name"])

    print(f"Results saved to {csv_file_path}")


def evaluate_top_name_piles(csv_path, number_of_piles_to_check=15):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Extract dataset and model information from the path
    parts = csv_path.split('/')
    dataset = parts[0]
    model_info = parts[2]

    # Extract model name and date from the model_info
    model_name = model_info.split('_')[0]
    date_info = datetime.now().strftime("%Y_%m_%d")
    model_id = f"{model_name}_{date_info}"

    # Extract the Pythia model name
    llm_model = model_info.split('_')[0]

    # Get the top most frequent 'Pile Name' appearances
    top_name_piles = df['Pile Name'].value_counts().head(number_of_piles_to_check).index.tolist()

    # Evaluate each of the top 'Pile Name' groups
    for name_pile in top_name_piles:
        print(f"Analyzing pile: {name_pile}")
        subset_df = df[df['Pile Name'] == name_pile]

        # Prepare scores dictionary
        attacks = [col for col in subset_df.columns if col not in ['FILE_PATH', 'Pile Name', 'label']]
        scores_dict = {attack: subset_df[attack].tolist() for attack in attacks}
        labels = subset_df['label'].tolist()

        # Extract the actual pile name from the dictionary-like string
        pile_name = eval(name_pile).get('pile_set_name', 'Unknown')

        # Create the output directory with the Pythia model name
        output_dir = f"results/{dataset}/{llm_model}/{pile_name}"

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Evaluate and save results
        evaluate_and_save_results(scores_dict, [{'label': label} for label in labels], dataset, model_id, output_dir,
                                  pile_name)


def evaluate_and_save_results(scores_dict, data, dataset, model_id, output_dir, name_pile):
    labels = [d['label'] for d in data]  # Ensure labels are binary
    results = defaultdict(list)
    all_output = []

    for method, scores in scores_dict.items():
        # Remove NaN and infinity values
        clean_scores = [score for score in scores if not (np.isnan(score) or np.isinf(score))]
        clean_labels = [labels[i] for i, score in enumerate(scores) if not (np.isnan(score) or np.isinf(score))]

        if len(clean_scores) != len(clean_labels):  # Ensure lengths match
            print(f"Length mismatch for method {method}: {len(clean_scores)} scores vs {len(clean_labels)} labels")
            continue

        if not clean_scores:  # Check if clean_scores is empty
            print(f"No valid scores for method: {method}")
            continue

        fpr, tpr, auroc, fpr95, tpr05, acc = get_metrics(clean_scores, clean_labels)

        results['method'].append(method)
        results['AUC'].append(f"{auroc:.1%}")
        results['fpr95'].append(f"{fpr95:.1%}")
        results['TPR@FPR=5%'].append(f"{tpr05:.1%}")
        results['Accuracy'].append(f"{acc:.1%}")
        results['pile_name'].append(name_pile)

        all_output.append({
            "label": clean_labels,
            "pred": {method: clean_scores}
        })

    df = pd.DataFrame(results)
    print(df.drop(columns=['pile_name']))  # Print the DataFrame without the 'pile_name' column

    save_root = output_dir
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # Save the results to CSV
    save_metrics_to_csv(results, save_root, model_id.split('_')[0], dataset)

    if os.path.isfile(os.path.join(save_root, f"{model_id}.csv")):
        df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False, mode='w', header=False)
    else:
        df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False)

    fig_fpr_tpr(all_output, save_root)


def merge_csvs(csv1_path, csv2_path, output_path):
    # Read the CSV files
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Add the 'train' column
    df1['train'] = 1
    df2['train'] = 0

    # Merge the dataframes
    merged_df = pd.concat([df1, df2])

    # Rename 'label' to 'Pile Name'
    merged_df.rename(columns={'label': 'Pile Name'}, inplace=True)

    # Rename 'train' to 'label'
    merged_df.rename(columns={'train': 'label'}, inplace=True)

    # Write the merged dataframe to a new CSV file
    merged_df.to_csv(output_path, index=False)


def clean_csv(input_path, output_path):
    # Read the CSV file
    df = pd.read_csv(input_path)

    # Remove rows with empty spaces, NaN values, -inf, or inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df[~df.apply(lambda x: x.str.contains('^\s*$', regex=True).any(), axis=1)]

    # Write the cleaned dataframe to a new CSV file
    df.to_csv(output_path, index=False)


def create_acc_csv(results_dir, output_csv, mode_metric="TPR@FPR=5%"):
    acc_dict = defaultdict(dict)

    # Loop through all subdirectories in the results directory
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.csv') and 'metrics_results' in file:
                pile_name = root.split('/')[-1]
                file_path = os.path.join(root, file)

                # Read the CSV file
                df = pd.read_csv(file_path)

                for _, row in df.iterrows():
                    attack_name = row['method']
                    acc = row[mode_metric]
                    if mode_metric == "TPR@FPR=5%":
                        acc = acc[:-1]
                        acc = float(acc)
                        acc *= 10
                        acc = str(acc)
                        acc = acc[:-1] + "%"

                    acc_dict[attack_name][pile_name] = acc

    # Convert the dictionary to a DataFrame
    acc_df = pd.DataFrame(acc_dict).transpose()

    # Save the DataFrame to a CSV file
    acc_df.to_csv(output_csv, index=True)

    print(f"Accuracy results saved to {output_csv}")


def rename_rows_and_generate_heatmap(input_csv_path, output_image_path, mode_metric):
    # Load the CSV file
    df = pd.read_csv(input_csv_path)

    # Set the index to the first column
    df = df.set_index('Unnamed: 0')

    # Print index before renaming
    print("Index before renaming:", df.index)

    # Rename the specified rows
    rename_dict = {
        'ppl': 'PPL',
        'ppl_zlib': 'Zlib',
        'Min_20.0% Prob': 'MIN-20% PROB',
        'Max_20.0% Prob': 'MAX-20% PROB',
        "MinK++_20.0% Prob": 'MinK++-20% PROB',
        "neighbourhood_loss": "Neighbor",
        'recall': 'ReCall',
        "DC-PDD_Score": "DC-PDD",
        'tag_tab_FT_k=1': 'Ours (Tag&Tab K=1 (FT))',
        "tag_tab_FT_k=4": 'Ours (Tag&Tab K=4 (FT))',
        "tag_tab_FT_k=10": 'Ours (Tag&Tab K=10 (FT))'
    }
    df.rename(index=rename_dict, inplace=True)

    # Print index after renaming
    print("Index after renaming:", df.index)

    # Clean the data
    df_cleaned = df.applymap(lambda x: float(x.strip('%')) / 100)

    # Sort the columns by the values in the 'Ours' row
    if 'Ours' not in df_cleaned.index:
        raise KeyError(
            "'Ours' not found in DataFrame index. Ensure that 'sentence_entropy_log_likelihood_k=2' was renamed correctly.")

    sorted_columns = df_cleaned.loc['Ours'].sort_values(ascending=False).index
    df_cleaned = df_cleaned[sorted_columns]

    # Plot the heatmap
    plt.figure(figsize=(10, 4))
    sns.heatmap(df_cleaned, annot=True, cmap='coolwarm', center=0.5, fmt=".2f")
    plt.xlabel('Pile Names')
    plt.ylabel('Attacks')
    plt.title(mode_metric + ' Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the heatmap to a file
    plt.savefig(output_image_path.replace('.png', '.pdf'))

    # Show the plot
    plt.show()

    print(f"Heatmap saved to {output_image_path}")


def sanitize_filename(filename):
    return "".join([c if c.isalnum() or c in (' ', '.', '_') else '_' for c in filename])


def evaluate_leave_one_out(csv_path, results_dir, number_of_piles_to_check=15, mode_metric="AUC"):
    df = pd.read_csv(csv_path)
    # Get the top most frequent 'Pile Name' appearances
    top_name_piles = df['Pile Name'].value_counts().head(number_of_piles_to_check).index.tolist()

    model_name = "1"
    dataset = "2"

    combined_results = defaultdict(dict)

    for leave_out_pile in top_name_piles:
        train_df = df[df['Pile Name'] != leave_out_pile]
        val_df = df[df['Pile Name'] == leave_out_pile]

        train_scores_dict = {col: train_df[col].tolist() for col in train_df.columns if
                             col not in ['FILE_PATH', 'Pile Name', 'label']}
        train_labels = train_df['label'].tolist()

        val_scores_dict = {col: val_df[col].tolist() for col in val_df.columns if
                           col not in ['FILE_PATH', 'Pile Name', 'label']}
        val_labels = val_df['label'].tolist()

        output_dir = os.path.join(results_dir, f"leave_out_{sanitize_filename(str(leave_out_pile))}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        evaluate_and_save_results(train_scores_dict, [{'label': label} for label in train_labels], dataset, model_name,
                                  output_dir, "train")
        evaluate_and_save_results(val_scores_dict, [{'label': label} for label in val_labels], dataset, model_name,
                                  output_dir, "val")

        create_acc_csv(output_dir, os.path.join(output_dir, "results.csv"), mode_metric=mode_metric)

        # Collect results for combined heatmap
        results_df = pd.read_csv(os.path.join(output_dir, "results.csv"))
        print("Results DataFrame:")
        print(results_df.head())  # Print the first few rows for inspection
        print("Columns in results_df:", results_df.columns)  # Print the columns for debugging

        if 'acc' in results_df.columns:
            results_df.set_index('acc', inplace=True)
        else:
            results_df.set_index(results_df.columns[0], inplace=True)

        accuracy_column_name = results_df.columns[-1]  # Dynamically get the name of the accuracy column
        for metric in results_df.index:
            combined_results[metric][leave_out_pile] = results_df.loc[metric, accuracy_column_name]

    # Generate combined heatmap
    generate_combined_heatmap(combined_results, os.path.join(results_dir, "combined_heatmap.png"))


def generate_combined_heatmap(combined_results, output_image_path):
    combined_df = pd.DataFrame(combined_results)

    # Clean the data
    combined_df_cleaned = combined_df.applymap(lambda x: float(str(x).strip('%')) / 100 if isinstance(x, str) else x)

    # Sort the columns by the values in the 'sentence_entropy_log_likelihood_k=2' row in descending order
    if 'tag_tab_FT_k=1' in combined_df_cleaned.index:
        sorted_columns = combined_df_cleaned.loc['tag_tab_FT_k=1'].sort_values(ascending=False).index
        combined_df_cleaned = combined_df_cleaned[sorted_columns]

    # Plot the heatmap with switched axes
    plt.figure(figsize=(10, 4))
    sns.heatmap(combined_df_cleaned.T, annot=True, cmap='coolwarm', center=0.5, fmt=".2f",
                cbar_kws={'label': 'Accuracy'})
    plt.ylabel('Pile Names')
    plt.xlabel('Metrics')
    plt.title('Combined Accuracy Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the heatmap to a file
    plt.savefig(output_image_path)
    plt.show()

    print(f"Combined heatmap saved to {output_image_path}")


if __name__ == "__main__":
    # # Paths to the CSV files
    csv1_path = 'PILE/PILE_Results_To_be_covered/M=llama-7b_K=5_train_D=PILE_2024_07_22_21_10_43.csv'
    csv2_path = 'PILE/PILE_Results_To_be_covered/M=llama-7b_K=5_validation_D=PILE_2024_07_24_08_48_37.csv'
    temp_str = (csv1_path.split("/"))[-1]
    temp_str = "_".join(temp_str.split("."))
    # Extract dataset and model information from the path
    merged_output_path = 'PILE/PILE_Results_To_be_covered/' + temp_str + "_All.csv"

    parts = merged_output_path.split('/')
    model_info = parts[2]
    # Extract the Pythia model name
    llm_model = model_info.split('_')[0]

    cleaned_output_path = merged_output_path

    # Merge the CSV files
    merge_csvs(csv1_path, csv2_path, merged_output_path)

    # Clean the merged CSV file
    clean_csv(merged_output_path, cleaned_output_path)
    #
    evaluate_top_name_piles(merged_output_path, number_of_piles_to_check=20)

    # modes = "Accuracy" , "TPR@FPR=5%", "AUC"
    mode_metric = "AUC"
    results_directory = "results/PILE/" + llm_model + "/"
    output_acc_csv = results_directory + mode_metric + '_results.csv'
    create_acc_csv(results_directory, output_acc_csv, mode_metric=mode_metric)

    input_csv_path = output_acc_csv
    output_image_path = results_directory + llm_model + "_" + mode_metric + '_heatmap.png'
    rename_rows_and_generate_heatmap(input_csv_path, output_image_path, mode_metric)
    # evaluate_leave_one_out(merged_output_path, results_directory)
