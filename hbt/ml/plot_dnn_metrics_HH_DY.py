import os
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import  roc_auc_score
from training_pi_3cls import (
    prepare_input_data, data_path, particle_names, change_to_tf_datasets, split_dataset, open_parquet_files,
)
import tensorflow as tf
from sklearn.metrics import confusion_matrix


thisdir = os.path.realpath(os.path.dirname(__file__))

def main():
    input = os.path.join(thisdir, "dnn_models", "kaputt_HH_DY_actf_tanh_100_epochs_7_layers_128_nodes")

    model_history = os.path.join(input, "history.parquet")

    history = ak.from_parquet(model_history)

    epochs = np.array(range(len(history.loss)))

    plt.plot(epochs, history.val_binary_accuracy.to_numpy(), label="val_binary_accuracy")
    input_array_combined = ak.Array([])
    target_array_combined = ak.Array([])
    sample_weights = ak.Array([])
    for dataset in data_path.keys():
        input_array, target_array, weights_array= prepare_input_data(dataset)
        input_array_combined = ak.concatenate([input_array_combined, input_array], axis=0)
        target_array_combined = ak.concatenate([target_array_combined, target_array], axis=0)

        weight = (weights_array/ak.sum(weights_array))*1E5
        sample_weights = ak.concatenate([sample_weights, weight], axis=0)
    
    tf_dataset_combined = change_to_tf_datasets(input_array_combined, target_array_combined, sample_weights)
    train, test = split_dataset(tf_dataset_combined)
    model = tf.keras.models.load_model(input)

    #nn_score = model(input_array)
    from IPython import embed; embed()
    # true & predicted 
    y_true = [y for x, y, w in test.unbatch().as_numpy_iterator()]
    weights = [w for x, y, w in test.unbatch().as_numpy_iterator()]
    true_values = np.concatenate(y_true, axis=0).reshape(len(y_true), 3)
    predicted = model.predict(test)

    #from IPython import embed; embed()
    # auc value
    auc = roc_auc_score(y_true=y_true, 
        y_score=predicted, sample_weight=weights, multi_class='ovr')
    # Roc curve
    fpr, tpr, _ = roc_curve(true_values[:, 0], y_score=predicted[:, 0], sample_weight=weights)
    
    plt.clf()
    plt.plot(fpr, tpr, label="Class 0 vs. rest")
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend()
    plt.savefig("roc_HH.png")


    # def plot_multi_class_roc():
    #     for cls in [0, 1, 2]:
    #         fpr, tpr, _ = roc_curve(true_values[:, cls], y_score=predicted[:, cls])
    #         auc = roc_auc_score(true_values[:, cls], predicted[:,cls])
    #         plt.plot(fpr, tpr, label=f"Class {cls} vs. rest (AUC: {auc:0.3f}")
    #     plt.legend()
    #     plt.title("ROC Curve for Multi-Class Classification")
    #     plt.xlabel("False Positive Rate", fontsize=14)
    #     plt.ylabel("True Positive Rate", fontsize=14)
    #     plt.savefig("ROC Curve for Multi-Class Classification.png")


    #     final_predicted_cls = np.argmax(predicted, axis=-1)
    #     final_truth_cls = np.argmax(true_values, axis=-1)
    #     plt.clf()
    fig, ax0 = plt.subplots()
    for cls in [0, 1]:
        fpr, tpr, _ = roc_curve(true_values[:, cls], y_score=predicted[:, cls], sample_weight=weights)
        auc = roc_auc_score(true_values[:, cls], predicted[:,cls], sample_weight=weights)
        ax0.plot(fpr, tpr, label=f"Class {cls} vs. rest (AUC: {auc:0.3f}")
    ax0.legend(loc="best")
    fig.suptitle("ROC Curve for Multi-Class Classification", size=14)
    ax0.set_xlabel("False Positive Rate", fontsize=14)
    ax0.set_ylabel("True Positive Rate", fontsize=14)
    fig.savefig("roc_Curve_for_Multi-Class_Classification.png")


    final_predicted_cls = np.argmax(predicted, axis=-1)
    final_truth_cls = np.argmax(true_values, axis=-1)
    
    from IPython import embed; embed()
    # confusin matrix
    matrix = confusion_matrix(y_true=final_truth_cls, y_pred=final_predicted_cls,
     labels=[0, 1, 2], normalize="true")
    labels = ["HH", "DY"]
    fig, ax = plt.subplots()

    im = ax.imshow(matrix, interpolation="none")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, round(matrix[i, j], 2),
                           ha="center", va="center", color="w")
    #ax.legend()
    ax.set_title("confusin matrix")
    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)        
    fig.savefig("confusion.png")


    # distribution
    fi, axs = plt.subplots(1, 2, sharey=True, figsize=(9,5))
    label_list = [r"HH", r"DY"] 
    for cls in [0, 1]:
        for j in [0, 1]:
            axs[cls].hist(predicted[final_truth_cls==j][:, cls], bins=50, label=label_list[j], density=True, histtype="step")
    
        axs[cls].legend()
        axs[cls].set_title(f"{label_list[cls]} prediction")
    fi.savefig("distributions_normalized.png")

 


#sklearn.metrics.roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True)

if __name__ == '__main__':
    main()