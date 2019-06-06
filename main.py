from config import Config, OntologyConfig, CabascConfig, LCR_RotConfig, LCR_RotInverseConfig, LCR_RotHopConfig, \
    DiagnosticClassifierConfig, LocalInterpretableConfig, SVMConfig
from data_setup.external_data_loader import ExternalDataLoader
from data_setup.internal_data_loader import InternalDataLoader
from abs_classifiers.ontology_reasoner import OntologyReasoner
from abs_classifiers.cabasc import CABASCModel
from abs_classifiers.lcr_rot import LCRRot
from abs_classifiers.lcr_rot_inverse import LCRRotInverse
from abs_classifiers.lcr_rot_hop import LCRRotHopModel
from abs_classifiers.svm import SVM
from diagnostic_classifier.diagnostic_classifier import DiagnosticClassifier
from local_interpretable_model.local_interpretable_model import LocalInterpretableModel
import os


def main():

    # Load external data if the internal files are not available
    if not os.path.isfile(Config.internal_train_data) or not os.path.isfile(Config.internal_test_data):
        external_data_loader = ExternalDataLoader(Config)
        # external_data_loader.load_external_data(load_external_file_name=Config.external_train_data,
        #                                         write_internal_file_name=Config.internal_train_data)
        external_data_loader.load_external_data(load_external_file_name=Config.external_test_data,
                                                write_internal_file_name=Config.internal_test_data)

    # Run internal data loader to set up all different data sets, you always have to load these files
    # internal_data_loader = InternalDataLoader(Config)
    # internal_data_loader.load_internal_training_data(load_internal_file_name=Config.internal_train_data)
    # internal_data_loader.load_internal_test_data(load_internal_file_name=Config.internal_test_data)
    #
    # # Aspect-Based Sentiment Classifiers, which do you want on or off
    # ontology = True
    # svm = True
    # cabasc = True
    # lcr_rot = True
    # lcr_rot_inverse = True
    # lcr_rot_hop = True
    #
    # # Do you want to run a hybrid model, the ontology reasoner always runs the hybrid form
    # Config.hybrid = True
    #
    # # Do you want to run cross-validation rounds, please specify the cross_validation_rounds in config.py
    # Config.cross_validation = True
    #
    # # Diagnostic Classifier, do you want it on or off, please specify in config.py the tasks you want to predict
    # diagnostic_classifier = True
    #
    # # Local Interpretable model, do you want it on or off, please specify in config.py the classifier for word relevance
    # local_interpretable_model = True
    #
    # if ontology:
    #
    #     ontology_reasoner = OntologyReasoner(OntologyConfig, internal_data_loader)
    #     ontology_reasoner.run()
    #
    # if svm:
    #
    #     svm_model = SVM(SVMConfig, internal_data_loader)
    #     svm_model.run()
    #
    # if cabasc:
    #     cabasc_model = CABASCModel(CabascConfig, internal_data_loader)
    #     cabasc_model.run()
    #
    #     if diagnostic_classifier:
    #         diagnostic_classifier = DiagnosticClassifier(DiagnosticClassifierConfig, cabasc_model)
    #
    #     if local_interpretable_model:
    #         local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, cabasc_model)
    #
    # if lcr_rot:
    #     lcr_rot_model = LCRRot(LCR_RotConfig, internal_data_loader)
    #     lcr_rot_model.run()
    #
    #     if diagnostic_classifier:
    #         diagnostic_classifier = DiagnosticClassifier(DiagnosticClassifierConfig, lcr_rot_model)
    #
    #     if local_interpretable_model:
    #         local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, lcr_rot_model)
    #
    # if lcr_rot_inverse:
    #
    #     lcr_rot_inverse_model = LCRRotInverse(LCR_RotInverseConfig, internal_data_loader)
    #     lcr_rot_inverse_model.run()
    #
    #     if diagnostic_classifier:
    #         diagnostic_classifier = DiagnosticClassifier(DiagnosticClassifierConfig, lcr_rot_inverse_model)
    #
    #     if local_interpretable_model:
    #         local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, lcr_rot_inverse_model)
    #
    # if lcr_rot_hop:
    #     lcr_rot_hop_model = LCRRotHopModel(LCR_RotHopConfig, internal_data_loader)
    #     lcr_rot_hop_model.run()
    #
    #     if diagnostic_classifier:
    #         diagnostic_classifier = DiagnosticClassifier(DiagnosticClassifierConfig, lcr_rot_hop_model)
    #
    #     if local_interpretable_model:
    #         local_interpretable_model = LocalInterpretableModel(LocalInterpretableConfig, lcr_rot_hop_model)


if __name__ == '__main__':
    main()