import pandas as pd
import numpy as np
from collections import defaultdict
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#Dataset Paths
ground_truths_dataset = 'output_features_CV004_capture_5_complete_v2.csv'
dataset1_path = r'1_Data_CSV\640_n_init_data_yolov8s_100_epch_CV004_capture5.csv'
dataset2_path  = r'1_Data_CSV\640_n_aug_data_yolov8s_100_epch_CV004_capture5.csv'
dataset3_path  = r'1_Data_CSV\640_n_exp_data_yolov8s_100_epch_CV004_capture5.csv'
dataset4_path  = r'1_Data_CSV\640_n_aug_exp_data_yolov8s_100_epch_CV004_capture5.csv'

#Techniques
dataset1 = 'Initial Dataset No Noise Reduction'
dataset2 = 'Augmented Dataset No Noise Reduction'
dataset3 = 'Expanded Dataset No Noise Reduction'
dataset4 = 'Expanded and Augmented Dataset No Noise Reduction'

#Capture Name
capture_name = 'CV004 Capture 5'

# Define color palette
colors = {
    dataset1: 'blue',  # Pink
    dataset2: 'orange',     # Turquiose
    dataset3: 'red',    # Blue
    dataset4: 'purple' # Yellow
}

# Function to ensure 'background' is included for empty feature lists and set confidence to 1
def add_background_label(features, confidence):
    if not features or features == ['background']:
        return ['background'], 1.0
    return [i.lower() for i in features], confidence

# Function to normalize classes in DataFrames
def normalize_classes(df, features_col='Classes', confidence_col='Confidence Level'):
    if confidence_col not in df.columns:
        df[confidence_col] = 1.0
    df[features_col] = df[features_col].apply(ast.literal_eval)  # Convert string representation of lists to actual lists
    df[features_col], df[confidence_col] = zip(*df.apply(lambda row: add_background_label(row[features_col], row[confidence_col]), axis=1))
    return df

# Function to classify frames and calculate confidence levels and counts
def classify_frames_and_confidence_with_counts(ground_truths, predictions, confidences):
    true_confidences = []
    halftrue_confidences = []
    incorrect_confidences = []

    true_counts = 0
    halftrue_counts = 0
    incorrect_counts = 0
    background_true_counts = 0
    background_incorrect_counts = 0

    for gt, pred, conf in zip(ground_truths, predictions, confidences):
        gt_set = set(gt)
        pred_set = set(pred)

        tp_set = gt_set & pred_set
        fp_set = pred_set - gt_set
        fn_set = gt_set - pred_set

        if tp_set and not fp_set and not fn_set:
            if 'background' in tp_set:
                background_true_counts += 1
            else:
                true_confidences.append(conf)
                true_counts += 1
        elif tp_set and (fp_set or fn_set):
            if 'background' in tp_set:
                background_true_counts += 1
            else:
                halftrue_confidences.append(conf)
                halftrue_counts += 1
        elif not tp_set and (fp_set or fn_set):
            if 'background' in fp_set or 'background' in fn_set:
                background_incorrect_counts += 1
            else:
                incorrect_confidences.append(conf)
                incorrect_counts += 1

    avg_true_conf = np.mean(true_confidences) if true_confidences else 0
    avg_halftrue_conf = np.mean(halftrue_confidences) if halftrue_confidences else 0
    avg_incorrect_conf = np.mean(incorrect_confidences) if incorrect_confidences else 0

    return (avg_true_conf, avg_halftrue_conf, avg_incorrect_conf), (true_counts, halftrue_counts, incorrect_counts, background_true_counts, background_incorrect_counts)

# Function to calculate true positives for each feature
def count_true_positives(ground_truths, predictions):
    true_positives = defaultdict(int)
    
    for gt, pred in zip(ground_truths, predictions):
        gt_set = set(gt)
        pred_set = set(pred)
        
        for feature in gt_set & pred_set:
            true_positives[feature] += 1
            
    return true_positives

# Function to aggregate false positives and false negatives for each feature
def count_false_positives_negatives(ground_truths, predictions):
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    incorrect_fp = defaultdict(int)
    incorrect_fn = defaultdict(int)
    
    for gt, pred in zip(ground_truths, predictions):
        gt_set = set(gt)
        pred_set = set(pred)
        
        fp_set = pred_set - gt_set
        fn_set = gt_set - pred_set
        tp_set = gt_set & pred_set
        
        for feature in fp_set:
            false_positives[feature] += 1
            if not tp_set:
                incorrect_fp[feature] += 1
        
        for feature in fn_set:
            false_negatives[feature] += 1
            if not tp_set:
                incorrect_fn[feature] += 1
            
    return false_positives, false_negatives, incorrect_fp, incorrect_fn

# Function to count the occurrences of each feature in the ground truths
def count_ground_truths(ground_truths):
    feature_counts = defaultdict(int)
    
    for gt in ground_truths:
        for feature in set(gt):
            feature_counts[feature] += 1
    
    return feature_counts

# Function to process confidence data and counts
def process_confidence_and_counts_data(df, ground_truths):
    ground_truth_frames = ground_truths['Features'].tolist()
    predicted_frames = df['Classes'].tolist()
    confidences = df['Confidence Level'].tolist()
    return classify_frames_and_confidence_with_counts(ground_truth_frames, predicted_frames, confidences)

def plot_true_positives_and_ground_truths(true_positives_final, true_positives_additional, true_positives_third, true_positives_fourth, ground_truth_counts, features):
    background_index = features.index("background")
    other_indices = [i for i in range(len(features)) if i != background_index]

    # Extract counts for the two groups
    tp_counts_final_background = true_positives_final["background"]
    tp_counts_additional_background = true_positives_additional["background"]
    tp_counts_third_background = true_positives_third["background"]
    tp_counts_fourth_background = true_positives_fourth["background"]
    ground_truths_counts_background = ground_truth_counts["background"]
    
    tp_counts_final_others = [true_positives_final[features[i]] for i in other_indices]
    tp_counts_additional_others = [true_positives_additional[features[i]] for i in other_indices]
    tp_counts_third_others = [true_positives_third[features[i]] for i in other_indices]
    tp_counts_fourth_others = [true_positives_fourth[features[i]] for i in other_indices]
    ground_truths_counts_others = [ground_truth_counts[features[i]] for i in other_indices]
    
    other_features = [features[i] for i in other_indices]

    fig = make_subplots(rows=1, cols=2)

    # Plot "background" feature
    fig.add_trace(go.Bar(x=["background"], y=[tp_counts_final_background], name=dataset1, marker_color=colors[dataset1]), row=1, col=1)
    fig.add_trace(go.Bar(x=["background"], y=[tp_counts_additional_background], name=dataset2, marker_color=colors[dataset2]), row=1, col=1)
    fig.add_trace(go.Bar(x=["background"], y=[tp_counts_third_background], name=dataset3, marker_color=colors[dataset3]), row=1, col=1)
    fig.add_trace(go.Bar(x=["background"], y=[tp_counts_fourth_background], name=dataset4, marker_color=colors[dataset4]), row=1, col=1)
    fig.add_trace(go.Bar(x=["background"], y=[ground_truths_counts_background], name='Ground Truths', marker_color='green'), row=1, col=1)

    # Plot other features
    fig.add_trace(go.Bar(x=other_features, y=tp_counts_final_others, name=dataset1, marker_color=colors[dataset1], showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=other_features, y=tp_counts_additional_others, name=dataset2, marker_color=colors[dataset2], showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=other_features, y=tp_counts_third_others, name=dataset3, marker_color=colors[dataset3], showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=other_features, y=tp_counts_fourth_others, name=dataset4, marker_color=colors[dataset4], showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=other_features, y=ground_truths_counts_others, name='Ground Truths', marker_color='green', showlegend=False), row=1, col=2)

    fig.update_layout(
        title=f'Number of True Positives and Ground Truths for each Feature ran on {capture_name}',
        xaxis=dict(title='Features'),
        xaxis2=dict(title='Features'),
        yaxis=dict(title='Counts'),
        barmode='group',
        margin=dict(l=50, r=50, t=50, b=50),

    )

    fig.show()

 

# Function to plot false positives for each feature with adjustments for centering and separate scales for "damage"
def plot_false_positives(false_positives_final, false_positives_additional, false_positives_third, false_positives_fourth, features_fp):
    damage_index = features_fp.index("damage")
    other_indices = [i for i in range(len(features_fp)) if i != damage_index]

    # Extract counts for the two groups
    fp_counts_final_damage = false_positives_final["damage"]
    fp_counts_additional_damage = false_positives_additional["damage"]
    fp_counts_third_damage = false_positives_third["damage"]
    fp_counts_fourth_damage = false_positives_fourth["damage"]

    fp_counts_final_others = [false_positives_final[features_fp[i]] for i in other_indices]
    fp_counts_additional_others = [false_positives_additional[features_fp[i]] for i in other_indices]
    fp_counts_third_others = [false_positives_third[features_fp[i]] for i in other_indices]
    fp_counts_fourth_others = [false_positives_fourth[features_fp[i]] for i in other_indices]

    other_features = [features_fp[i] for i in other_indices]

    fig = make_subplots(rows=1, cols=2)

    # Plot "damage" feature
    fig.add_trace(go.Bar(x=["damage"], y=[fp_counts_final_damage], name=dataset1, marker_color=colors[dataset1]), row=1, col=1)
    fig.add_trace(go.Bar(x=["damage"], y=[fp_counts_additional_damage], name=dataset2, marker_color=colors[dataset2]), row=1, col=1)
    fig.add_trace(go.Bar(x=["damage"], y=[fp_counts_third_damage], name=dataset3, marker_color=colors[dataset3]), row=1, col=1)
    fig.add_trace(go.Bar(x=["damage"], y=[fp_counts_fourth_damage], name=dataset4, marker_color=colors[dataset4]), row=1, col=1)

    # Plot other features
    fig.add_trace(go.Bar(x=other_features, y=fp_counts_final_others, name=dataset1, marker_color=colors[dataset1],showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=other_features, y=fp_counts_additional_others, name=dataset2, marker_color=colors[dataset2],showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=other_features, y=fp_counts_third_others, name=dataset3, marker_color=colors[dataset3],showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=other_features, y=fp_counts_fourth_others, name=dataset4, marker_color=colors[dataset4],showlegend=False), row=1, col=2)

    fig.update_layout(
        title=f'False Positives for each Feature ran on {capture_name}',
        xaxis=dict(title='Features'),
        xaxis2=dict(title='Features'),
        yaxis=dict(title='Counts'),
       

        barmode='group',
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.show()

# Function to plot false negatives for each feature with adjustments for centering and separate subplot for "background"
def plot_false_negatives(false_negatives_final, false_negatives_additional, false_negatives_third, false_negatives_fourth, features_fn):
    background_index = features_fn.index("background")
    other_indices = [i for i in range(len(features_fn)) if i != background_index]

    # Extract counts for the two groups
    fn_counts_final_background = false_negatives_final["background"]
    fn_counts_additional_background = false_negatives_additional["background"]
    fn_counts_third_background = false_negatives_third["background"]
    fn_counts_fourth_background = false_negatives_fourth["background"]

    fn_counts_final_others = [false_negatives_final[features_fn[i]] for i in other_indices]
    fn_counts_additional_others = [false_negatives_additional[features_fn[i]] for i in other_indices]
    fn_counts_third_others = [false_negatives_third[features_fn[i]] for i in other_indices]
    fn_counts_fourth_others = [false_negatives_fourth[features_fn[i]] for i in other_indices]

    other_features = [features_fn[i] for i in other_indices]

    fig = make_subplots(rows=1, cols=2)

    # Plot "background" feature
    fig.add_trace(go.Bar(x=["background"], y=[fn_counts_final_background], name=dataset1, marker_color=colors[dataset1]), row=1, col=1)
    fig.add_trace(go.Bar(x=["background"], y=[fn_counts_additional_background], name=dataset2, marker_color=colors[dataset2]), row=1, col=1)
    fig.add_trace(go.Bar(x=["background"], y=[fn_counts_third_background], name=dataset3, marker_color=colors[dataset3]), row=1, col=1)
    fig.add_trace(go.Bar(x=["background"], y=[fn_counts_fourth_background], name=dataset4, marker_color=colors[dataset4]), row=1, col=1)

    # Plot other features
    fig.add_trace(go.Bar(x=other_features, y=fn_counts_final_others, name=dataset1, marker_color=colors[dataset1],showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=other_features, y=fn_counts_additional_others, name=dataset2, marker_color=colors[dataset2],showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=other_features, y=fn_counts_third_others, name=dataset3, marker_color=colors[dataset3],showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=other_features, y=fn_counts_fourth_others, name=dataset4, marker_color=colors[dataset4],showlegend=False), row=1, col=2)

    fig.update_layout(
        title=f'False Negatives for each Feature ran on {capture_name}',
        xaxis=dict(title='Features'),
        xaxis2=dict(title='Features'),
        yaxis=dict(title='Counts'),
       

        barmode='group',
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.show()

def plot_confidence_comparison_mean(confidence_data, techniques):
    labels = ['True Confidence', 'Halftrue Confidence', 'Incorrect Confidence']
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig = go.Figure()

    for technique in techniques:
        true_conf, halftrue_conf, incorrect_conf = confidence_data[technique]
        fig.add_trace(go.Bar(
            x=labels,
            y=[true_conf, halftrue_conf, incorrect_conf],
            name=technique,
            marker_color=colors[technique]
        ))

    fig.update_layout(
        title=f'Mean Confidence of Feature Prediction Comparison ran on {capture_name}',
        xaxis=dict(title='Prediction Type'),
        yaxis=dict(title='Confidence Level'),
        barmode='group',
        margin=dict(l=50, r=50, t=50, b=50),
        
    )

    fig.show()


# Function to plot confidence comparison and counts for true, halftrue, and incorrect predictions using Plotly
def plot_confidence_and_counts_comparison(confidence_data, counts_data, techniques):
    labels = ['True', 'Halftrue', 'Incorrect']
    background_labels = ['Background Incorrect']
    
    true_counts_data = {tech: counts_data[tech][0] for tech in techniques}
    halftrue_counts_data = {tech: counts_data[tech][1] for tech in techniques}
    incorrect_counts_data = {tech: counts_data[tech][2] for tech in techniques}
    background_incorrect_counts_data = {tech: counts_data[tech][4] for tech in techniques}

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Confidence for Feature Predictions", "Incorrect Background Predictions"))

    # Plot confidence data
    for i, technique in enumerate(techniques):
        fig.add_trace(go.Bar(
            x=labels,
            y=[true_counts_data[technique], halftrue_counts_data[technique], incorrect_counts_data[technique]],
            name=f'{technique}',
            marker_color=colors[technique]
        ), row=1, col=1)

    # Plot background counts data
    for i, technique in enumerate(techniques):
        fig.add_trace(go.Bar(
            x=background_labels,
            y=[background_incorrect_counts_data[technique]],
            name=f'{technique} Confidence for Background Predictions',
            marker_color=colors[technique],
            showlegend=False
        ), row=1, col=2)

    fig.update_layout(
        title=f'Confidence and Background Counts Comparison ran on {capture_name}',
        barmode='group',
        height=800
    )

    fig.show()


# Main function to run the analysis
def run_analysis(ground_truths_path, final_df_path, additional_final_df_path, third_final_df_path, fourth_final_df_path):
    # Load datasets
    ground_truths = pd.read_csv(ground_truths_path)
    final_df = pd.read_csv(final_df_path)
    additional_final_df = pd.read_csv(additional_final_df_path)
    third_final_df = pd.read_csv(third_final_df_path)
    fourth_final_df = pd.read_csv(fourth_final_df_path)
    
    # Normalize classes
    ground_truths = normalize_classes(ground_truths, 'Features', 'Confidence Level')
    final_df = normalize_classes(final_df, 'Classes', 'Confidence Level')
    additional_final_df = normalize_classes(additional_final_df, 'Classes', 'Confidence Level')
    third_final_df = normalize_classes(third_final_df, 'Classes', 'Confidence Level')
    fourth_final_df = normalize_classes(fourth_final_df, 'Classes', 'Confidence Level')
    
    # Calculate true positives
    true_positives_final = count_true_positives(ground_truths['Features'], final_df['Classes'])
    true_positives_additional = count_true_positives(ground_truths['Features'], additional_final_df['Classes'])
    true_positives_third = count_true_positives(ground_truths['Features'], third_final_df['Classes'])
    true_positives_fourth = count_true_positives(ground_truths['Features'], fourth_final_df['Classes'])
    ground_truth_counts = count_ground_truths(ground_truths['Features'])
    
    # Plot true positives and ground truths for each feature
    features = list(set(true_positives_final.keys()) | set(true_positives_additional.keys()) | set(true_positives_third.keys()) | set(true_positives_fourth.keys()) | set(ground_truth_counts.keys()))
    plot_true_positives_and_ground_truths(true_positives_final, true_positives_additional, true_positives_third, true_positives_fourth, ground_truth_counts, features)
    
    # Calculate false positives and false negatives
    false_positives_final, false_negatives_final, incorrect_fp_final, incorrect_fn_final = count_false_positives_negatives(ground_truths['Features'], final_df['Classes'])
    false_positives_additional, false_negatives_additional, incorrect_fp_additional, incorrect_fn_additional = count_false_positives_negatives(ground_truths['Features'], additional_final_df['Classes'])
    false_positives_third, false_negatives_third, incorrect_fp_third, incorrect_fn_third = count_false_positives_negatives(ground_truths['Features'], third_final_df['Classes'])
    false_positives_fourth, false_negatives_fourth, incorrect_fp_fourth, incorrect_fn_fourth = count_false_positives_negatives(ground_truths['Features'], fourth_final_df['Classes'])
    
    features_fp = list(set(false_positives_final.keys()) | set(false_positives_additional.keys()) | set(false_positives_third.keys()) | set(false_positives_fourth.keys()))
    plot_false_positives(false_positives_final, false_positives_additional, false_positives_third, false_positives_fourth, features_fp)
    
    features_fn = list(set(false_negatives_final.keys()) | set(false_negatives_additional.keys()) | set(false_negatives_third.keys()) | set(false_negatives_fourth.keys()))
    plot_false_negatives(false_negatives_final, false_negatives_additional, false_negatives_third, false_negatives_fourth, features_fn)
    
    # Classify frames and calculate confidence levels and counts for each technique
   # final_confidences, final_counts = classify_frames_and_confidence_with_counts(
    #    ground_truths['Features'], final_df['Classes'], final_df['Confidence Level'])
    #additional_confidences, additional_counts = classify_frames_and_confidence_with_counts(
     #   ground_truths['Features'], additional_final_df['Classes'], additional_final_df['Confidence Level'])
    #third_confidences, third_counts = classify_frames_and_confidence_with_counts(
     #   ground_truths['Features'], third_final_df['Classes'], third_final_df['Confidence Level'])
    #fourth_confidences, fourth_counts = classify_frames_and_confidence_with_counts(
     #   ground_truths['Features'], fourth_final_df['Classes'], fourth_final_df['Confidence Level'])
    
    confidence_data = {
        dataset1: process_confidence_and_counts_data(final_df, ground_truths)[0],
        dataset2: process_confidence_and_counts_data(additional_final_df, ground_truths)[0],
        dataset3: process_confidence_and_counts_data(third_final_df, ground_truths)[0],
        dataset4: process_confidence_and_counts_data(fourth_final_df, ground_truths)[0],
    }
    
    counts_data = {
        dataset1: process_confidence_and_counts_data(final_df, ground_truths)[1],
        dataset2: process_confidence_and_counts_data(additional_final_df, ground_truths)[1],
        dataset3: process_confidence_and_counts_data(third_final_df, ground_truths)[1],
        dataset4: process_confidence_and_counts_data(fourth_final_df, ground_truths)[1],
    }
    
    # Plot confidence comparison with counts
    techniques = [dataset1, dataset2, dataset3, dataset4]
    plot_confidence_comparison_mean(confidence_data, techniques)
    plot_confidence_and_counts_comparison(confidence_data, counts_data, techniques)

# Example usage
if __name__ == "__main__":
    # Specify paths to your CSV files
    ground_truths_path = ground_truths_dataset
    final_df_path = dataset1_path
    additional_final_df_path = dataset2_path
    third_final_df_path = dataset3_path
    fourth_final_df_path = dataset4_path

    run_analysis(ground_truths_path, final_df_path, additional_final_df_path, third_final_df_path, fourth_final_df_path)

