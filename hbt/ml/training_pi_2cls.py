import tensorflow as tf
import keras
from keras import layers
import awkward as ak
import numpy as np
import os
import matplotlib.pyplot as plt

# import create_dnn_plots as dnnplots

thisdir = os.path.realpath(os.path.dirname(__file__))

EMPTY_FLOAT = np.array(-99999.0)


data_path = {
"hh_ggf_bbtautau":["/nfs/dust/cms/user/yamralim/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11_limited/hh_ggf_bbtautau_madgraph/nominal/calib__default/sel__gen_pt_eta_selection/prod__empty/v2/data_0.parquet"],
#"tt_dl_powheg":["/nfs/dust/cms/user/yamralim/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11_limited/tt_dl_powheg/nominal/calib__default/sel__gen_pt_eta_selection/prod__empty/v2/data_0.parquet"],
"dy_lep_pt50To100_amcatnlo":["/nfs/dust/cms/user/yamralim/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11_limited/dy_lep_pt50To100_amcatnlo/nominal/calib__default/sel__gen_pt_eta_selection/prod__empty/v2/data_0.parquet",]
}





particle_names = ["pion_neg",
                "pion_pos"]

input_names = [
                'eta', 
                # 'mass', 
                'phi', 
                'pt', 
                # 'pdgId', 
                # 'status', 
                # 'statusFlags',
                'zfrac'
                
            ]





def open_parquet_files(files):
    return ak.concatenate([ak.from_parquet(file) for file in files],axis=0)



 #    # scale the weights
 # if "weights" in events_dict.keys():
 #    N_events_procs = np.sum(target_array, axis=0)
 #    weights_scaler = np.mean(N_events_procs)
 #    for i in range(n_op_nodes):
 #        proc_mask = np.where(target_array[:, i] == 1, True, False)

 #        proc_weights_sum = np.sum(events_dict["weights"][proc_mask])
 #        events_dict["weights"] = np.where(proc_mask, events_dict["weights"] / proc_weights_sum, events_dict["weights"])
 #    events_dict["weights"] *= weights_scaler




def combine_particle_columns(ak_array, input_names,feature):
    return ak.concatenate([ak_array[name][feature] for name in input_names],axis=0)

def combine_weights_columns(ak_array, weight):
    return ak.concatenate([ak_array[weight],ak_array[weight]],axis=0)
#from IPython import embed; embed()


# def set_indices(input_array):
#     indices_none_filled_input_array = ak.local_index(
#         input_array,
#         [
#         'eta'==None,
#         'phi'==None,
#         'pt'==None,
#         'zfrac'==None
#         ],
#         axis=0
#     )
    
#     numpyed_indices_array= indices_none_filled_input_array.to_numpy()[ndices_none_filled_input_array.eta.to_numpy() !=None]

#     return ak.Array(numpyed_indices_array)  

def remove_empty_floats(input_array, weights_array):
    none_filled_input_array = ak.fill_none(
        input_array, 
        [{
        'eta':EMPTY_FLOAT,
        'phi':EMPTY_FLOAT,
        'pt':EMPTY_FLOAT,
        'zfrac':EMPTY_FLOAT
        }], 
        axis=0
    )
    
    


    numpyed_input_array = none_filled_input_array.to_numpy()[none_filled_input_array.pt.to_numpy()!=EMPTY_FLOAT]
    weights_array = weights_array.to_numpy()[ak.flatten(none_filled_input_array.pt.to_numpy())!=EMPTY_FLOAT]

    return ak.Array(numpyed_input_array), ak.Array(weights_array)
  

def standardize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean)/(std)

def prepare_input_data(
    dataset_name,
    data_path=data_path,
    particle_names=particle_names,
    input_names=input_names,
):

    data = open_parquet_files(data_path[dataset_name])
    input_array = combine_particle_columns(data, particle_names,input_names)  
    
    weight='normalization_weight'
    weight_array = combine_weights_columns(data,weight)
    input_array, weight_array = remove_empty_floats(input_array, weight_array)

    target_array = np.zeros(shape=(len(input_array),len(data_path.keys())))
    target_index = -9999
    for ikey,key in enumerate(data_path.keys()):

        if key == dataset_name:
            target_index = ikey
    target_array[:,target_index] = np.ones(len(input_array))
    
    return input_array, target_array, weight_array



def change_to_tf_datasets(input_array_combined, target_array_combined, sample_weights_combined):
    # from IPython import embed; embed()
    all_pions_flattened_array = ak.fill_none(ak.flatten(input_array_combined, axis=0), EMPTY_FLOAT)
    dtype = all_pions_flattened_array["pt"].type.content
    x = ak_2_regular_array(input_array_combined, input_names, dtype=dtype)
    y = target_array_to_output_tensor(target_array_combined, dtype=dtype)


    # shuffle ?

    dataset_x = tf.data.Dataset.from_tensor_slices((x))
    dataset_y = tf.data.Dataset.from_tensor_slices((y))
    dataset_weights = tf.data.Dataset.from_tensor_slices((sample_weights_combined))

    return tf.data.Dataset.zip((dataset_x, dataset_y, dataset_weights))
    


def ak_2_regular_array(input_array, input_names, dtype):
    
    all_pions_flattened_array = ak.fill_none(ak.flatten(input_array, axis=0), EMPTY_FLOAT)
    
    # change type of decayMode column to float -> BEWARE : this changes the ordering of the fields, now decayMode is the last one
    #all_pions_flattened_array["decayMode"] = ak.enforce_type(all_pions_flattened_array["decayMode"], all_pions_flattened_array["dxy"].type.content)
    #all_pions_flattened_array["decayMode"] = ak.enforce_type(all_pions_flattened_array["decayMode"], dtype)
    # from IPython import embed; embed()
    # change ak array to structured numpy array
    structured_numpy_array = all_pions_flattened_array.to_numpy()
    
    # do preprocessing
    for feature in input_names:
        standardized = standardize(structured_numpy_array[feature])
        structured_numpy_array[feature] = standardized
    
    # change structured numpy array to numpy array and then to a tensor
    numpy_array = structured_numpy_array.view(np.float64).reshape(structured_numpy_array.shape + (-1,))
    input_tensor = tf.constant(numpy_array)
    
    # max_shape = ak.max(ak.count(input_array[input_name],axis=1))
    # padded_array = ak.fill_none(ak.to_regular(ak.pad_none(ak_array,max_shape)),     EMPTY_FLOAT)
    #from IPython import embed;embed()
  
    # tf.Tensor(ak.to_numpy(padded_array))
    
    return input_tensor


# def target_array_to_output_tensor(target_array, dtype): 
#     all_pions_flattened_array_out = ak.fill_none(ak.flatten(target_array, axis=1), EMPTY_FLOAT)
#     # all_pions_flattened_array_out["charge"] = ak.enforce_type(all_pions_flattened_array_out["charge"], all_pions_flattened_array["dxy"].type.content)
#     all_pions_flattened_array_out["charge"] = ak.enforce_type(all_pions_flattened_array_out["charge"], dtype)

#     structured_numpy_array_out = all_pions_flattened_array_out.to_numpy()
#     numpy_array_out = structured_numpy_array_out.view(np.float32).reshape(structured_numpy_array_out.shape + (-1,))
#     numpy_array_out[numpy_array_out<0] = 0
#     target_tensor = tf.constant(numpy_array_out)
#     return target_tensor
def target_array_to_output_tensor(target_array, dtype): 
    
    
    target_tensor = tf.constant(target_array, dtype=tf.float64)
    return target_tensor



def split_dataset(dataset, split_ratio=0.2, batch_size=256):
    num_samples = dataset.cardinality().numpy()
    dataset = dataset.shuffle(buffer_size=num_samples, seed=123456, reshuffle_each_iteration=False)

    num_test_samples = int((1-split_ratio) * num_samples)
    train= dataset.take(num_test_samples)
    test = dataset.skip(num_test_samples).batch(batch_size)
    train = train.shuffle(buffer_size=num_test_samples,reshuffle_each_iteration=True).batch(batch_size)

    
    return train, test
    
if __name__ == '__main__':
    input_array_combined = ak.Array([])
    target_array_combined = ak.Array([])
    sample_weights = ak.Array([])
    #from IPython import embed; embed()
    for dataset in data_path.keys():
        input_array, target_array, weights_array = prepare_input_data(dataset)
        input_array_combined = ak.concatenate([input_array_combined, input_array], axis=0)
        target_array_combined = ak.concatenate([target_array_combined, target_array], axis=0)
        weight = (weights_array/ak.sum(weights_array))#*1E5
        sample_weights = ak.concatenate([sample_weights, weight], axis=0)
    
    #from IPython import embed; embed()

    
    tf_dataset_combined = change_to_tf_datasets(input_array_combined, target_array_combined, sample_weights)
    train, test = split_dataset(tf_dataset_combined)
 
    
    kernel_regularizer=tf.keras.regularizers.L2(0.01)

    

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.001, patience=20, verbose=0,
        mode='min', baseline=None, restore_best_weights=True)
        
    model_name = "kaputt_HH_DY_actf_tanh_100_epochs_7_layers_128_nodes"
    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(train.element_spec[0].shape[1],)),
            layers.Dense(128, name="layer1", kernel_regularizer=kernel_regularizer),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.tanh),
            layers.Dense(128, name="layer2", kernel_regularizer=kernel_regularizer),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.tanh),
            layers.Dense(128, name="layer3", kernel_regularizer=kernel_regularizer),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.tanh),
            layers.Dense(128, name="layer4", kernel_regularizer=kernel_regularizer),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.tanh),
            layers.Dense(128, name="layer5", kernel_regularizer=kernel_regularizer),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.tanh),
            layers.Dense(128, name="layer6", kernel_regularizer=kernel_regularizer),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.tanh),
            layers.Dense(128, name="layer7", kernel_regularizer=kernel_regularizer),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.tanh),
            layers.Dense(len(data_path.keys()), activation="softmax", name="output"),

        ]
    )

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(optimizer='sgd',
                loss=loss_fn,
                
                metrics=[
                    'accuracy',
                    'BinaryCrossentropy',
                    'BinaryAccuracy',
                ],
                weighted_metrics=[
                    'accuracy',
                    'BinaryCrossentropy',
                    'BinaryAccuracy',
                ])
    
    history = model.fit(train, validation_data=test, epochs=100, callbacks=[early_stopping])

    
    dnn_output_path = os.path.join(thisdir, "dnn_models")
    if not os.path.exists(dnn_output_path):
        os.makedirs(dnn_output_path)
    final_path = os.path.join(dnn_output_path, f"{model_name}")
    model.save(final_path)
    
    # save training history
    hist_array = ak.Array(history.history)
    ak.to_parquet(hist_array, os.path.join(final_path, "history.parquet"))
