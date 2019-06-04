import owlready2 as owl
from nltk.parse.stanford import StanfordDependencyParser
import nltk
import os

# path to the java runtime environment
nltk.internals.config_java('/usr/bin/java')
java_path = '/usr/bin/java'
os.environ['JAVAHOME'] = java_path


class OntologyTagging:

    def __init__(self):

        owl.onto_path.append("data/external_data")
        self.ontology = owl.get_ontology("ontology.owl").load()
        self.ontology_classes = set(self.ontology.classes())
        self.lexicon = {}

        for ontology_class in self.ontology_classes:
            self.lexicon[ontology_class] = ontology_class.lex

    def ontology_classes_tagging(self, lemmatized_sentence):

        number_words_in_sentence = len(lemmatized_sentence)
        word_classes = []

        for word_index in range(number_words_in_sentence):

            word_class = None
            for values in list(self.lexicon.values()):
                if lemmatized_sentence[word_index] in values:
                    word_class = list(self.lexicon.keys())[list(self.lexicon.values()).index(values)]
            word_classes.append(word_class)

        return word_classes

    def polarity_and_aspect_relation_tagging(self, ontology_classes_of_sentence, aspect_indices, aspect_categories):

        number_words_in_sentence = len(ontology_classes_of_sentence)
        number_of_aspects = len(aspect_indices)

        polarities_sentence = []
        relations_sentence = []

        for aspect_index in range(number_of_aspects):

            aspect_length = len(aspect_indices[aspect_index])
            aspect_class = ontology_classes_of_sentence[aspect_indices[aspect_index][0]]

            polarities = []
            relations = []

            if aspect_length > 1:

                for index in range(aspect_indices[aspect_index][1], aspect_indices[aspect_index][aspect_length - 2]):

                    if aspect_class is not None and ontology_classes_of_sentence[index] is not None:

                        aspect_class = owl.types.new_class(aspect_class.__name__ +
                                                           ontology_classes_of_sentence[index][0].__name__,
                                                           (aspect_class, ontology_classes_of_sentence[index][0]))

                    elif aspect_class is None and ontology_classes_of_sentence[index] is not None:
                        aspect_class = ontology_classes_of_sentence[index]

            for word_index in range(number_words_in_sentence):

                if word_index not in aspect_indices[aspect_index]:
                    word_class = ontology_classes_of_sentence[word_index]
                    category = aspect_categories[aspect_index]
                    relation, polarity = self.relation_between_word_aspect(word_class, category, aspect_class)
                else:
                    polarity = "AspectPolarity"
                    relation = False
                polarities.append(polarity)
                relations.append(relation)

            polarities_sentence.append(polarities)
            relations_sentence.append(relations)

        return polarities_sentence, relations_sentence

    def relation_between_word_aspect(self, word_class, category, aspect_class):

        if word_class is None:
            return False, "NoPolarity"

        positive_class = self.ontology.search(iri='*Positive')[0]
        negative_class = self.ontology.search(iri='*Negative')[0]

        sub_classes = word_class.ancestors()

        if self.ontology.PropertyMention in sub_classes:

            if self.ontology.GenericNegativePropertyMention in sub_classes:
                return True, "Negative"
            elif self.ontology.GenericPositivePropertyMention in sub_classes:
                return True, "Positive"
            elif self.ontology.Positive in sub_classes:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        return True, "Positive"
                return False, "NoPropertyPolarity"
            elif self.ontology.Negative in sub_classes:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        return True, "Negative"
                return False, "NoPropertyPolarity"
            else:
                if aspect_class is not None:
                    word_class = owl.types.new_class(word_class.__name__ + aspect_class.__name__,
                                                     (word_class, aspect_class))
                owl.sync_reasoner()
                positive = positive_class.__subclasscheck__(word_class)
                negative = negative_class.__subclasscheck__(word_class)

                if positive and not negative:
                    return True, "Positive"
                elif not positive and negative:
                    return True, "Negative"
                elif positive and negative:
                    return True, "UnknownPropertyPolarity"
                else:
                    return False, "NoPropertyPolarity"

        elif self.ontology.ActionMention in sub_classes:

            if self.ontology.GenericNegativeAction in sub_classes:
                return True, "Negative"
            elif self.ontology.GenericPositiveAction in sub_classes:
                return True, "Positive"
            elif self.ontology.Positive in sub_classes:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        return True, "Positive"
                return False, "NoActionPolarity"
            elif self.ontology.Negative in sub_classes:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        return True, "Negative"
                return False, "NoActionPolarity"
            else:
                raise Exception("Error in ActionMention ", sub_classes)

        elif self.ontology.EntityMention in sub_classes:

            for c in sub_classes:
                if c == owl.Thing:
                    continue
                elif category in c.aspect:
                    return True, "EntityPolarity"
            return False, "NoEntityPolarity"
        else:
            return False, "UnknownPolarity"

    def mention_tagging(self, ontology_classes_sentence):

        mentions = []
        words_of_sentence = len(ontology_classes_sentence)

        for word_index in range(words_of_sentence):

            if ontology_classes_sentence[word_index] is None:
                mentions.append("NoMention")
                continue

            sub_classes = ontology_classes_sentence[word_index].ancestors()

            if self.ontology.PropertyMention in sub_classes:
                mentions.append("PropertyMention")
            elif self.ontology.PersonMention in sub_classes:
                mentions.append("PersonMention")
            elif self.ontology.ServiceMention in sub_classes:
                mentions.append("ServiceMention")
            elif self.ontology.ActionMention in sub_classes:
                mentions.append("ActionMention")
            elif self.ontology.AmbienceMention in sub_classes:
                mentions.append("AmbienceMention")
            elif self.ontology.LocationMention in sub_classes:
                mentions.append("LocationMention")
            elif self.ontology.RestaurantMention in sub_classes:
                mentions.append("RestaurantMention")
            elif self.ontology.PriceMention in sub_classes:
                mentions.append("PriceMention")
            elif self.ontology.StyleOptionsMention in sub_classes:
                mentions.append("StyleOptionsMention")
            elif self.ontology.ExperienceMention in sub_classes:
                mentions.append("ExperienceMention")
            elif self.ontology.SustenanceMention in sub_classes:
                mentions.append("SustenanceMention")
            elif self.ontology.TimeMention in sub_classes:
                mentions.append("TimeMention")
            else:
                mentions.append("UnknownMention")
        return mentions

    # def negation_tagging(self, lemmatized_sentence):
    #
    #     # negation check with window and dependency graph
    #     path_to_jar = 'data/external_data/stanford-parser-full-2018-02-27/stanford-parser.jar'
    #     path_to_models_jar = 'data/external_data/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'
    #
    #     dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    #
    #     negations = ['not', 'n\'t', 'never']
    #     results = dependency_parser.raw_parse(' '.join(lemmatized_sentence))
    #     print("results ", results)
    #     dep = results.__next__()
    #     print("dep ", dep)
    #     result = list(dep.triples())
    #     print("result ", result)
    #     print("triple[0][0] ", result[0][0][0])
    #     print("triple[0][1] ", result[0][1])
