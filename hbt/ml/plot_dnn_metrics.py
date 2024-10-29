import os
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import  roc_auc_score
from training_pi_3cls import (
    remove_empty_floats, features, data_path, particle_names, change_to_tf_datasets, split_dataset, open_parquet_files,
)
import tensorflow as tf
from sklearn.metrics import confusion_matrix


thisdir = os.path.realpath(os.path.dirname(__file__))

def combine_particle_columns(ak_array, input_names,features):
    #from IPython import embed; embed()
    # mask_1 = ak.flatten(ak.fill_none(ak_array[input_names[1]][features].zfrac > 0.5, [False], axis=0)) & ak.flatten(ak.fill_none(ak_array[input_names[0]][features].zfrac > 0.5, [False], axis=0))
    # return ak.concatenate([ak_array[name][mask_1][features] for name in input_names],axis=0)
    return ak.concatenate([ak_array[name][features] for name in input_names],axis=0)
#from IPython import embed; embed()
def combine_weights_columns(ak_array, weight, input_names = particle_names, features=features):
    # mask_1 = ak.flatten(ak.fill_none(ak_array[input_names[1]][features].zfrac > 0.5, [False], axis=0)) & ak.flatten(ak.fill_none(ak_array[input_names[0]][features].zfrac > 0.5, [False], axis=0))
    # return ak.concatenate([ak_array[mask_1][weight],ak_array[mask_1][weight]],axis=0)
    return ak.concatenate([ak_array[weight],ak_array[weight]],axis=0)

def prepare_input_data(
    dataset_name,
    data_path=data_path,
    particle_names=particle_names,
    features=features,
):

    data = open_parquet_files(data_path[dataset_name])
    input_array = combine_particle_columns(data, particle_names,features)  
    # input_array = remove_empty_floats(input_array)
    
    
    # weight='normalization_weight'
    weight="mc_weight"
    weight_array = combine_weights_columns(data,weight)
    input_array, weight_array = remove_empty_floats(input_array, weight_array)


    target_array = np.zeros(shape=(len(input_array),len(data_path.keys())))
    target_index = -9999
    for ikey,key in enumerate(data_path.keys()):

        if key == dataset_name:
            target_index = ikey
    target_array[:,target_index] = np.ones(len(input_array))
    
    return input_array, target_array, weight_array


def main():
    input = os.path.join(thisdir, "dnn_models", "actf_elu_500_epochs_4_layers_128_nodes_lr_0.001_nW")

    model_history = os.path.join(input, "history.parquet")

    history = ak.from_parquet(model_history)

    epochs = np.array(range(len(history.loss)))

    plt.plot(epochs, history.val_categorical_accuracy.to_numpy(), label="val_categorical_accuracy")


    fig, axx= plt.subplots()
    axx.plot(epochs, history.val_loss.to_numpy(), label="validation loss")
    axx.set_title('validation loss')
    axx.set_xlabel('epochs')
    axx.set_ylabel('validaton loss')
    fig.savefig('validaton_loss.png')

    fig, axs= plt.subplots()
    axs.plot(epochs, history.loss.to_numpy(), label="loss")
    axs.set_title('loss')
    axs.set_xlabel('epochs')
    axs.set_ylabel('loss')
    fig.savefig('loss.png')

    #from IPython import embed; embed()
    input_array_combined = ak.Array([])
    target_array_combined = ak.Array([])
    sample_weights = ak.Array([])
    for dataset in data_path.keys():
        input_array, target_array, weights_array= prepare_input_data(dataset)
        input_array_combined = ak.concatenate([input_array_combined, input_array], axis=0)
        target_array_combined = ak.concatenate([target_array_combined, target_array], axis=0)
        #TODO mask for zfrac
        
        # pion_neg_A12= (input_array.zfrac > 0.5) & (input_array.pdgId == -211)
        # pion_neg_A34= (input_array.zfrac < 0.5) & (input_array.pdgId == -211)
        # pion_pos_A13= (input_array.zfrac < 0.5) & (input_array.pdgId == 211)
        # pion_pos_A24= (input_array.zfrac > 0.5) & (input_array.pdgId == 211)
        # from IPython import embed; embed()
        # Area1= pion_neg_A12 | pion_pos_A13
        # Area2= pion_neg_A12 | pion_pos_A24
        # Area3= pion_neg_A34 | pion_pos_A13
        # Area4= pion_neg_A34 | pion_pos_A24
        weight = (weights_array/ak.sum(weights_array))*1E5
        sample_weights = ak.concatenate([sample_weights, weight], axis=0)



    labels = ak.argmax(target_array_combined, axis=1)
    unique_labels = set(labels)
    new_weights = np.zeros(len(labels), dtype=np.float32)
    for label in unique_labels:
        mask = labels==label
        new_weights[mask] = len(labels) / (len(unique_labels) * ak.sum(mask))     
    
    tf_dataset_combined = change_to_tf_datasets(input_array_combined, target_array_combined, new_weights)
    train, test = split_dataset(tf_dataset_combined)
    #from IPython import embed; embed()
    model = tf.keras.models.load_model(input)

    #nn_score = model(input_array)

    from IPython import embed; embed()
    # true & predicted 
    y_true = [y for x, y, w in test.unbatch().as_numpy_iterator()]
    weights = [w for x, y, w in test.unbatch().as_numpy_iterator()]
    true_values = np.concatenate(y_true, axis=0).reshape(len(y_true), 3)
    predicted = model.predict(test)

    


    def plot_multi_class(true_values, predicted, weights=None, postfix=""):
        fig, ax0 = plt.subplots()
        for cls in [0, 1, 2]:
            fpr, tpr, _ = roc_curve(true_values[:, cls], y_score=predicted[:, cls])
            auc = roc_auc_score(true_values[:, cls], predicted[:,cls])
            ax0.plot(fpr, tpr, label=f"Class {cls} vs. rest (AUC: {auc:0.3f}")
        ax0.legend(loc="best")
        fig.suptitle("ROC Curve for Multi-Class Classification", size=14)
        ax0.set_xlabel("False Positive Rate", fontsize=14)
        ax0.set_ylabel("True Positive Rate", fontsize=14)
        fig.savefig(f"roc_Curve_for_Multi-Class_Classification_{postfix}.png")


    plot_multi_class(true_values, predicted)

    final_predicted_cls = np.argmax(predicted, axis=-1)
    final_truth_cls = np.argmax(true_values, axis=-1)
    
    from IPython import embed; embed()
    # confusin matrix
    def confusion(y_true, y_pred, labels=[0, 1, 2], normalize="true"):
        fig, ax = plt.subplots()
        matrix = confusion_matrix(y_true=final_truth_cls, y_pred=final_predicted_cls, labels=[0, 1, 2], normalize="true")
        labels = ["HH", r"t$\bar{t}$", "DY"]
        im = ax.imshow(matrix, cmap="rainbow", interpolation="none")

        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
    
        for i in range(len(labels)):
            for j in range(len(labels)):
                # if 
                text = ax.text(j, i, round(matrix[i, j], 2),
                               ha="center", va="center", color="w")
        #ax.legend()
        ax.set_title("confusion matrix")
        ax.set_xlabel("Predicted label", fontsize=14)
        ax.set_ylabel("True label", fontsize=14)        
        fig.savefig("confusion.png")


    matrix = confusion_matrix(y_true=final_truth_cls, y_pred=final_predicted_cls,
     #sample_weight=weights,
     labels=[0, 1, 2], normalize="true")
    labels = ["HH", r"t$\bar{t}$", "DY"]
    fig, ax = plt.subplots()

    im = ax.imshow(matrix,  cmap="cool", interpolation="none")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, round(matrix[i, j], 2),
                           ha="center", va="center", color="w")
    #ax.legend()
    ax.set_title("confusion matrix")
    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)        
    fig.savefig("confusion.png")


    # distribution
    def distribution(predicted, final_truth_cls):
        fi, axs = plt.subplots(1, 3, sharey=True, figsize=(9,5))
        label_list = [r"HH", r"t$\bar{t}$", r"DY"] 
        for cls in [0, 1, 2]:
            for j in [0, 1, 2]:
                axs[cls].hist(predicted[final_truth_cls==j][:, cls], bins=50, label=label_list[j], density=True, histtype="step")
    
            axs[cls].legend()
            axs[cls].set_title(f"{label_list[cls]} prediction")
            axs[cls].set_xlim(0, 1)
        fi.savefig("distributions_normalized.png")
    


 


#sklearn.metrics.roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True)

if __name__ == '__main__':
    main()