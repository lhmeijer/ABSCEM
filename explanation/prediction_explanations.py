import numpy as np
import json


def get_indices_of_correct_wrong_predictions(neural_language_model):

    x_training = np.array(neural_language_model.internal_data_loader.word_embeddings_training_all)
    train_aspects = np.array(neural_language_model.internal_data_loader.aspect_indices_training)
    y_training = np.array(neural_language_model.internal_data_loader.polarity_matrix_training)
    tr_sentence_id = np.array(neural_language_model.internal_data_loader.sentence_id_in_training)

    x_test = np.array(neural_language_model.internal_data_loader.word_embeddings_test_all)
    test_aspects = np.array(neural_language_model.internal_data_loader.aspect_indices_test)
    y_test = np.array(neural_language_model.internal_data_loader.polarity_matrix_test)
    te_sentence_id = np.array(neural_language_model.internal_data_loader.sentence_id_in_test)

    tr_x_left_part, tr_x_target_part, tr_x_right_part, tr_x_left_sen_len, tr_x_tar_len, tr_x_right_sen_len = \
        neural_language_model.internal_data_loader.split_embeddings(x_training, train_aspects,
                                                                    neural_language_model.config.max_sentence_length,
                                                                    neural_language_model.config.max_target_length)

    tr_y_pred, _ = neural_language_model.predict(np.array(tr_x_left_part), np.array(tr_x_target_part),
                                                 np.array(tr_x_right_part), np.array(tr_x_left_sen_len),
                                                 np.array(tr_x_tar_len), np.array(tr_x_right_sen_len))

    te_x_left_part, te_x_target_part, te_x_right_part, te_x_left_sen_len, te_x_tar_len, te_x_right_sen_len = \
        neural_language_model.internal_data_loader.split_embeddings(x_test, test_aspects,
                                                                    neural_language_model.config.max_sentence_length,
                                                                    neural_language_model.config.max_target_length)

    te_y_pred, _ = neural_language_model.predict(np.array(te_x_left_part), np.array(te_x_target_part),
                                                 np.array(te_x_right_part), np.array(te_x_left_sen_len),
                                                 np.array(te_x_tar_len), np.array(te_x_right_sen_len))

    with open(neural_language_model.config.remaining_data, 'r') as file:
        for line in file:
            indices_ontology = json.loads(line)

    tr_able_to_pred_indices = indices_ontology['tr_able_to_pred']
    te_able_to_pred_indices = indices_ontology['te_able_to_pred']
    tr_not_able_to_pred_indices = indices_ontology['tr_not_able_to_pred']
    te_not_able_to_pred_indices = indices_ontology['te_not_able_to_pred']

    tr_counter_true_predicted = np.zeros(y_training.shape[1], dtype=int)
    tr_counter_atp_true_predicted = np.zeros(y_training.shape[1], dtype=int)
    tr_counter_natp_true_predicted = np.zeros(y_training.shape[1], dtype=int)

    tr_correct_predicted = []
    tr_wrong_predicted = []
    tr_counter_correct_predicted = np.zeros(y_training.shape[1], dtype=int)
    tr_counter_wrong_predicted = np.zeros(y_training.shape[1], dtype=int)

    tr_atp_correct_predicted = []
    tr_atp_wrong_predicted = []
    tr_counter_atp_correct_predicted = np.zeros(y_training.shape[1], dtype=int)
    tr_counter_atp_wrong_predicted = np.zeros(y_training.shape[1], dtype=int)

    tr_natp_correct_predicted = []
    tr_natp_wrong_predicted = []
    tr_counter_natp_correct_predicted = np.zeros(y_training.shape[1], dtype=int)
    tr_counter_natp_wrong_predicted = np.zeros(y_training.shape[1], dtype=int)

    for i in range(x_training.shape[0]):

        pred_y = np.argmax(tr_y_pred[i])
        true_y = np.argmax(y_training[i])
        tr_counter_true_predicted[true_y] += 1

        if pred_y == true_y:
            tr_correct_predicted.append(i)
            tr_counter_correct_predicted[pred_y] += 1

            if i in tr_able_to_pred_indices:
                tr_atp_correct_predicted.append(i)
                tr_counter_atp_correct_predicted[pred_y] += 1
                tr_counter_atp_true_predicted[true_y] += 1

            elif i in tr_not_able_to_pred_indices:
                tr_natp_correct_predicted.append(i)
                tr_counter_natp_correct_predicted[pred_y] += 1
                tr_counter_natp_true_predicted[true_y] += 1
        else:
            tr_wrong_predicted.append(i)
            tr_counter_wrong_predicted[pred_y] += 1

            if i in tr_able_to_pred_indices:
                tr_atp_wrong_predicted.append(i)
                tr_counter_atp_wrong_predicted[pred_y] += 1
                tr_counter_atp_true_predicted[true_y] += 1

            elif i in tr_not_able_to_pred_indices:
                tr_natp_wrong_predicted.append(i)
                tr_counter_natp_wrong_predicted[pred_y] += 1
                tr_counter_natp_true_predicted[true_y] += 1

    te_counter_true_predicted = np.zeros(y_test.shape[1], dtype=int)
    te_counter_atp_true_predicted = np.zeros(y_test.shape[1], dtype=int)
    te_counter_natp_true_predicted = np.zeros(y_test.shape[1], dtype=int)

    te_correct_predicted = []
    te_wrong_predicted = []
    te_counter_correct_predicted = np.zeros(y_test.shape[1], dtype=int)
    te_counter_wrong_predicted = np.zeros(y_test.shape[1], dtype=int)

    te_atp_correct_predicted = []
    te_atp_wrong_predicted = []
    te_counter_atp_correct_predicted = np.zeros(y_test.shape[1], dtype=int)
    te_counter_atp_wrong_predicted = np.zeros(y_test.shape[1], dtype=int)

    te_natp_correct_predicted = []
    te_natp_wrong_predicted = []
    te_counter_natp_correct_predicted = np.zeros(y_test.shape[1], dtype=int)
    te_counter_natp_wrong_predicted = np.zeros(y_test.shape[1], dtype=int)

    for i in range(x_test.shape[0]):

        pred_y = np.argmax(te_y_pred[i])
        true_y = np.argmax(y_test[i])
        te_counter_true_predicted[true_y] += 1

        if pred_y == true_y:
            te_correct_predicted.append(i)
            te_counter_correct_predicted[pred_y] += 1

            if i in te_able_to_pred_indices:
                te_atp_correct_predicted.append(i)
                te_counter_atp_correct_predicted[pred_y] += 1
                te_counter_atp_true_predicted[true_y] += 1

            elif i in te_not_able_to_pred_indices:
                te_natp_correct_predicted.append(i)
                te_counter_natp_correct_predicted[pred_y] += 1
                te_counter_natp_true_predicted[true_y] += 1
        else:
            te_wrong_predicted.append(i)
            te_counter_wrong_predicted[pred_y] += 1

            if i in te_able_to_pred_indices:
                te_atp_wrong_predicted.append(i)
                te_counter_atp_wrong_predicted[pred_y] += 1
                te_counter_atp_true_predicted[true_y] += 1

            elif i in te_not_able_to_pred_indices:
                te_natp_wrong_predicted.append(i)
                te_counter_natp_wrong_predicted[pred_y] += 1
                te_counter_natp_true_predicted[true_y] += 1

    indices = {
        'tr_correct_predicted': tr_correct_predicted,
        'tr_wrong_predicted': tr_wrong_predicted,
        'tr_atp_correct_predicted': tr_atp_correct_predicted,
        'tr_atp_wrong_predicted': tr_atp_wrong_predicted,
        'tr_natp_correct_predicted': tr_natp_correct_predicted,
        'tr_natp_wrong_predicted': tr_natp_wrong_predicted,
        'te_correct_predicted': te_correct_predicted,
        'te_wrong_predicted': te_wrong_predicted,
        'te_atp_correct_predicted': te_atp_correct_predicted,
        'te_atp_wrong_predicted': te_atp_wrong_predicted,
        'te_natp_correct_predicted': te_natp_correct_predicted,
        'te_natp_wrong_predicted': te_natp_wrong_predicted
    }

    with open(neural_language_model.config.file_of_indices, 'w') as outfile:
        json.dump(indices, outfile, ensure_ascii=False)

    results = {
        'tr_counter_true_predicted': tr_counter_true_predicted.tolist(),
        'tr_counter_atp_true_predicted': tr_counter_atp_true_predicted.tolist(),
        'tr_counter_natp_true_predicted': tr_counter_natp_true_predicted.tolist(),
        'tr_counter_correct_predicted': tr_counter_correct_predicted.tolist(),
        'tr_counter_wrong_predicted': tr_counter_wrong_predicted.tolist(),
        'tr_counter_atp_correct_predicted': tr_counter_atp_correct_predicted.tolist(),
        'tr_counter_atp_wrong_predicted': tr_counter_atp_wrong_predicted.tolist(),
        'tr_counter_natp_correct_predicted': tr_counter_natp_correct_predicted.tolist(),
        'tr_counter_natp_wrong_predicted': tr_counter_natp_wrong_predicted.tolist(),
        'te_counter_true_predicted': te_counter_true_predicted.tolist(),
        'te_counter_atp_true_predicted': te_counter_atp_true_predicted.tolist(),
        'te_counter_natp_true_predicted': te_counter_natp_true_predicted.tolist(),
        'te_counter_correct_predicted': te_counter_correct_predicted.tolist(),
        'te_counter_wrong_predicted': te_counter_wrong_predicted.tolist(),
        'te_counter_atp_correct_predicted': te_counter_atp_correct_predicted.tolist(),
        'te_counter_atp_wrong_predicted': te_counter_atp_wrong_predicted.tolist(),
        'te_counter_natp_correct_predicted': te_counter_natp_correct_predicted.tolist(),
        'te_counter_natp_wrong_predicted': te_counter_natp_wrong_predicted.tolist()
    }
    print("results ", results)
    with open(neural_language_model.config.file_hybrid_results, 'w') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=0)

    results_lengths = {
        'tr_mean_left_length': np.mean(tr_x_left_sen_len),
        'tr_mean_right_length': np.mean(tr_x_right_sen_len),
        'tr_standard_deviation_left_length': np.std(tr_x_left_sen_len),
        'tr_standard_deviation_right_length': np.std(tr_x_right_sen_len),
        'tr_correct_predicted_mean_left_length': np.mean(tr_x_left_sen_len[tr_correct_predicted]),
        'tr_correct_predicted_mean_right_length': np.mean(tr_x_right_sen_len[tr_correct_predicted]),
        'tr_correct_predicted_standard_deviation_left_length': np.std(tr_x_left_sen_len[tr_correct_predicted]),
        'tr_correct_predicted_standard_deviation_right_length': np.std(tr_x_right_sen_len[tr_correct_predicted]),
        'tr_wrong_predicted_mean_left_length': np.mean(tr_x_left_sen_len[tr_wrong_predicted]),
        'tr_wrong_predicted_mean_right_length': np.mean(tr_x_right_sen_len[tr_wrong_predicted]),
        'tr_wrong_predicted_standard_deviation_left_length': np.std(tr_x_left_sen_len[tr_wrong_predicted]),
        'tr_wrong_predicted_standard_deviation_right_length': np.std(tr_x_right_sen_len[tr_wrong_predicted]),
        'tr_atp_correct_predicted_mean_left_length': np.mean(tr_x_left_sen_len[tr_atp_correct_predicted]),
        'tr_atp_correct_predicted_mean_right_length': np.mean(tr_x_right_sen_len[tr_atp_correct_predicted]),
        'tr_atp_correct_predicted_standard_deviation_left_length': np.std(tr_x_left_sen_len[tr_atp_correct_predicted]),
        'tr_atp_correct_predicted_standard_deviation_right_length': np.std(
            tr_x_right_sen_len[tr_atp_correct_predicted]),
        'tr_atp_wrong_predicted_mean_left_length': np.mean(tr_x_left_sen_len[tr_atp_wrong_predicted]),
        'tr_atp_wrong_predicted_mean_right_length': np.mean(tr_x_right_sen_len[tr_atp_wrong_predicted]),
        'tr_atp_wrong_predicted_standard_deviation_left_length': np.std(tr_x_left_sen_len[tr_atp_wrong_predicted]),
        'tr_atp_wrong_predicted_standard_deviation_right_length': np.std(tr_x_right_sen_len[tr_atp_wrong_predicted]),
        'tr_natp_correct_predicted_mean_left_length': np.mean(tr_x_left_sen_len[tr_natp_correct_predicted]),
        'tr_natp_correct_predicted_mean_right_length': np.mean(tr_x_right_sen_len[tr_natp_correct_predicted]),
        'tr_natp_correct_predicted_standard_deviation_left_length': np.std(
            tr_x_left_sen_len[tr_natp_correct_predicted]),
        'tr_natp_correct_predicted_standard_deviation_right_length': np.std(
            tr_x_right_sen_len[tr_natp_correct_predicted]),
        'tr_natp_wrong_predicted_mean_left_length': np.mean(tr_x_left_sen_len[tr_natp_wrong_predicted]),
        'tr_natp_wrong_predicted_mean_right_length': np.mean(tr_x_right_sen_len[tr_natp_wrong_predicted]),
        'tr_natp_wrong_predicted_standard_deviation_left_length': np.std(tr_x_left_sen_len[tr_natp_wrong_predicted]),
        'tr_natp_wrong_predicted_standard_deviation_right_length': np.std(tr_x_right_sen_len[tr_natp_wrong_predicted]),
        'te_mean_left_length': np.mean(te_x_left_sen_len),
        'te_mean_right_length': np.mean(te_x_right_sen_len),
        'te_standard_deviation_left_length': np.std(te_x_left_sen_len),
        'te_standard_deviation_right_length': np.std(te_x_right_sen_len),
        'te_correct_predicted_mean_left_length': np.mean(te_x_left_sen_len[te_correct_predicted]),
        'te_correct_predicted_mean_right_length': np.mean(te_x_right_sen_len[te_correct_predicted]),
        'te_correct_predicted_standard_deviation_left_length': np.std(te_x_left_sen_len[te_correct_predicted]),
        'te_correct_predicted_standard_deviation_right_length': np.std(te_x_right_sen_len[te_correct_predicted]),
        'te_wrong_predicted_mean_left_length': np.mean(te_x_left_sen_len[te_wrong_predicted]),
        'te_wrong_predicted_mean_right_length': np.mean(te_x_right_sen_len[te_wrong_predicted]),
        'te_wrong_predicted_standard_deviation_left_length': np.std(te_x_left_sen_len[te_wrong_predicted]),
        'te_wrong_predicted_standard_deviation_right_length': np.std(te_x_right_sen_len[te_wrong_predicted]),
        'te_atp_correct_predicted_mean_left_length': np.mean(te_x_left_sen_len[te_atp_correct_predicted]),
        'te_atp_correct_predicted_mean_right_length': np.mean(te_x_right_sen_len[te_atp_correct_predicted]),
        'te_atp_correct_predicted_standard_deviation_left_length': np.std(te_x_left_sen_len[te_atp_correct_predicted]),
        'te_atp_correct_predicted_standard_deviation_right_length': np.std(
            te_x_right_sen_len[te_atp_correct_predicted]),
        'te_atp_wrong_predicted_mean_left_length': np.mean(te_x_left_sen_len[te_atp_wrong_predicted]),
        'te_atp_wrong_predicted_mean_right_length': np.mean(te_x_right_sen_len[te_atp_wrong_predicted]),
        'te_atp_wrong_predicted_standard_deviation_left_length': np.std(te_x_left_sen_len[te_atp_wrong_predicted]),
        'te_atp_wrong_predicted_standard_deviation_right_length': np.std(te_x_right_sen_len[te_atp_wrong_predicted]),
        'te_natp_correct_predicted_mean_left_length': np.mean(te_x_left_sen_len[te_natp_correct_predicted]),
        'te_natp_correct_predicted_mean_right_length': np.mean(te_x_right_sen_len[te_natp_correct_predicted]),
        'te_natp_correct_predicted_standard_deviation_left_length': np.std(
            te_x_left_sen_len[te_natp_correct_predicted]),
        'te_natp_correct_predicted_standard_deviation_right_length': np.std(
            te_x_right_sen_len[te_natp_correct_predicted]),
        'te_natp_wrong_predicted_mean_left_length': np.mean(te_x_left_sen_len[te_natp_wrong_predicted]),
        'te_natp_wrong_predicted_mean_right_length': np.mean(te_x_right_sen_len[te_natp_wrong_predicted]),
        'te_natp_wrong_predicted_standard_deviation_left_length': np.std(te_x_left_sen_len[te_natp_wrong_predicted]),
        'te_natp_wrong_predicted_standard_deviation_right_length': np.std(te_x_right_sen_len[te_natp_wrong_predicted])
    }

    with open(neural_language_model.config.file_hybrid_lengths, 'w') as outfile:
        json.dump(results_lengths, outfile, ensure_ascii=False, indent=0)

    sentence_ids = {
        'tr_correct_predicted': tr_sentence_id[tr_correct_predicted].tolist(),
        'tr_wrong_predicted': tr_sentence_id[tr_wrong_predicted].tolist(),
        'tr_atp_correct_predicted': tr_sentence_id[tr_atp_correct_predicted].tolist(),
        'tr_atp_wrong_predicted': tr_sentence_id[tr_atp_wrong_predicted].tolist(),
        'tr_natp_correct_predicted': tr_sentence_id[tr_natp_correct_predicted].tolist(),
        'tr_natp_wrong_predicted': tr_sentence_id[tr_natp_wrong_predicted].tolist(),
        'te_correct_predicted': te_sentence_id[te_correct_predicted].tolist(),
        'te_wrong_predicted':  te_sentence_id[te_wrong_predicted].tolist(),
        'te_atp_correct_predicted':  te_sentence_id[te_atp_correct_predicted].tolist(),
        'te_atp_wrong_predicted':  te_sentence_id[te_atp_wrong_predicted].tolist(),
        'te_natp_correct_predicted':  te_sentence_id[te_natp_correct_predicted].tolist(),
        'te_natp_wrong_predicted': te_sentence_id[te_natp_wrong_predicted].tolist()
    }

    with open(neural_language_model.config.file_hybrid_ids, 'w') as outfile:
        json.dump(sentence_ids, outfile, ensure_ascii=False, indent=0)
