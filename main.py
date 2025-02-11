import libemg
from libemg.utils import make_regex
import numpy as np
import matplotlib.pyplot as plt
import os

# some parameters/ dataset details:
window_size = 200
window_increment = 100
num_subjects = 22

if __name__ == "__main__":
    dataset = libemg.datasets._3DCDataset()
    
    fe = libemg.feature_extractor.FeatureExtractor()
    feature_list = fe.get_feature_list()
    feature_list.remove("FUZZYEN")
    feature_list.remove("SAMPEN")
    feature_group_list = fe.get_feature_groups()
    feature_group_list.pop("TSTD")
    om = libemg.offline_metrics.OfflineMetrics()
    metrics = ["CA"]

    results = np.zeros((num_subjects, len(feature_list)+len(feature_group_list)))
        
    subject_list = list(range(num_subjects))
    if not os.path.exists("results.npy"):
        for s in subject_list:
            data = dataset.prepare_data(subjects=[s])
            train_odh = data['Train']
            test_odh = data['Test']
            train_windows, train_metadata = train_odh.parse_windows(window_size, window_increment)
            test_windows,  test_metadata  = test_odh.parse_windows(window_size, window_increment)

            # get all features
            for f in range(len(feature_list)+len(feature_group_list)):
                if f < len(feature_list):
                    feature = feature_list[f]
                    train_features = fe.extract_features([feature], train_windows)
                    test_features = fe.extract_features([feature], test_windows)
                else:
                    feature = list(feature_group_list.keys())[f-len(feature_list)]
                    train_features = fe.extract_feature_group(feature, train_windows)
                    test_features = fe.extract_feature_group(feature, test_windows)
                feature_dictionary = {
                    "training_features": train_features,
                    "training_labels": train_metadata["classes"]
                }
            
                # train classifier
                clf = libemg.emg_predictor.EMGClassifier("LDA")
                clf.fit(feature_dictionary.copy())

                preds = clf.run(test_features)
                
                # test classifier
                results[s,f] = om.extract_offline_metrics(metrics, test_metadata["classes"], preds[0])[metrics[0]] * 100
                print("Subject: {}, Feature: {}, Accuracy: {}".format(s+1, feature, results[s,f]))
        
        np.save("results.npy", results)
    else:
        results = np.load("results.npy")
    mean_feature_accuracy = results.mean(axis=0)
    std_feature_accuracy  = results.std(axis=0)

    plt.figure(figsize=(10,5))
    plt.bar(feature_list+list(feature_group_list.keys()), mean_feature_accuracy, yerr=std_feature_accuracy)
    plt.grid()
    plt.xlabel("Features")
    plt.ylabel("Accuracy")
    plt.xticks(feature_list+list(feature_group_list.keys()), rotation=90)

    plt.tight_layout()
    plt.savefig("results.png")
