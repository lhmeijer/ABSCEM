from config import DiagnosticClassifierPOSConfig, DiagnosticClassifierAspectSentimentConfig, \
    DiagnosticClassifierRelationConfig, DiagnosticClassifierWordSentimentConfig, DiagnosticClassifierMentionConfig, \
    DiagnosticClassifierFullAspectSentimentConfig
import numpy as np
import matplotlib.pyplot as plt
import json


class DiagnosticClassificationPlot:

    def __init__(self, neural_language_model, diagnostic_classifiers):
        self.neural_language_model = neural_language_model
        self.diagnostic_classifiers = diagnostic_classifiers

    def plot(self):

        print("I am plotting the graphs of the diagnostic classifiers accuracies")

        configurations = {
            'word_sentiments': DiagnosticClassifierWordSentimentConfig,
            'aspect_sentiments': DiagnosticClassifierAspectSentimentConfig,
            'relations': DiagnosticClassifierRelationConfig,
            'pos_tags': DiagnosticClassifierPOSConfig,
            'mentions': DiagnosticClassifierMentionConfig,
            'full_aspect_sentiment': DiagnosticClassifierFullAspectSentimentConfig
        }

        for interest, value in self.diagnostic_classifiers.items():

            if value:

                name_of_file_training = configurations[interest].get_file_of_results(
                    self.neural_language_model.config.name_of_model, 'tr_correct_pred')
                with open(name_of_file_training, 'r') as file:
                    training_results = json.load(file)

                name_of_file_correct = configurations[interest].get_file_of_results(
                    self.neural_language_model.config.name_of_model, 'te_correct_pred')
                with open(name_of_file_correct, 'r') as file:
                    correct_results = json.load(file)
                self.plot_diagnostic_classifier(configurations[interest], correct_results, 'te_correct_pred',
                                                training_results)

                name_of_file_incorrect = configurations[interest].get_file_of_results(
                    self.neural_language_model.config.name_of_model, 'te_incorrect_pred')
                with open(name_of_file_incorrect, 'r') as file:
                    incorrect_results = json.load(file)
                self.plot_diagnostic_classifier(configurations[interest], incorrect_results, 'te_incorrect_pred',
                                                training_results)

    def plot_diagnostic_classifier(self, diagnostic_config, results, name, training):

        if "LCR_Rot_hop_model" in self.neural_language_model.config.name_of_model:
            n_points = 2 + self.neural_language_model.config.n_iterations_hop
            labels = ['$e$', '$h$']
            x = [1, 2]

            for i in range(self.neural_language_model.config.n_iterations_hop):
                labels.append('$r_{' + str(i + 1) + "}$")
                x.append((i + 3))
        else:
            n_points = 3
            labels = ['$e$', '$h$', '$r$']
            x = [1, 2, 3]

        n_lines = len(diagnostic_config.features_names) + 1
        accuracies = np.zeros((n_lines, n_points))
        std_deviations = np.zeros((n_lines, n_points))

        for i in range(n_lines):

            if i != n_lines - 1:
                accuracies[i][0] = results['acc_mean_correct_embeddings'][i]
                accuracies[i][1] = results['acc_mean_correct_hidden_states'][i]
                std_deviations[i][0] = results['acc_std_correct_embeddings'][i]
                std_deviations[i][1] = results['acc_std_correct_hidden_states'][i]

            else:
                accuracies[i][0] = training['acc_mean_correct_embeddings'][-1]
                accuracies[i][1] = training['acc_mean_correct_hidden_states'][-1]
                std_deviations[i][0] = training['acc_std_correct_embeddings'][-1]
                std_deviations[i][1] = training['acc_std_correct_hidden_states'][-1]

            if "LCR_Rot_hop_model" in self.neural_language_model.config.name_of_model:

                for j in range(self.neural_language_model.config.n_iterations_hop):
                    if i != n_lines - 1:
                        accuracies[i][j + 2] = results['acc_mean_correct_weighted_hidden_states_' + str(j)][i]
                        std_deviations[i][j + 2] = results['acc_std_correct_weighted_hidden_states_' + str(j)][i]
                    else:
                        accuracies[i][j + 2] = training['acc_mean_correct_weighted_hidden_states_' + str(j)][-1]
                        std_deviations[i][j + 2] = training['acc_std_correct_weighted_hidden_states_' + str(j)][-1]
            else:
                if i != n_lines - 1:
                    accuracies[i][2] = results['acc_mean_correct_weighted_hidden_states'][i]
                    std_deviations[i][2] = results['acc_std_correct_weighted_hidden_states'][i]
                else:
                    accuracies[i][2] = training['acc_mean_correct_weighted_hidden_states'][-1]
                    std_deviations[i][2] = training['acc_std_correct_weighted_hidden_states'][-1]

        minimum_value = np.amin(accuracies)

        for i in range(n_lines):
            if i == n_lines - 1:
                plt.errorbar(x, accuracies[i], yerr=std_deviations[i], marker='o',  markersize=12, linewidth=3,
                             label="Overall training set", color='black')
            elif i == n_lines - 2:
                plt.errorbar(x, accuracies[i], yerr=std_deviations[i], marker='o', markersize=12, linewidth=3,
                             label=diagnostic_config.features_names[i], color='grey')
            else:
                plt.errorbar(x, accuracies[i], yerr=std_deviations[i], marker='o', markersize=12, linewidth=3,
                             label=diagnostic_config.features_names[i])

        plt.legend(loc='lower left', fontsize='large', ncol=2)
        plt.xticks(x, labels, fontsize=14)
        plt.ylabel('prediction accuracy', fontsize=14)
        plt.xlabel('hidden layers', fontsize=14)
        plt.ylim([0.28, 1.02])
        plt.yticks(fontsize=14)
        file = diagnostic_config.get_file_of_plots(self.neural_language_model.config.name_of_model, name)
        plt.savefig(file)
        plt.close()
