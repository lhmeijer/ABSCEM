import owlready2 as owl


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

    def polarity_and_aspect_relation_tagging(self, ontology_classes_of_sentence, aspect_indices, aspect_categories,
                                             aspect_dependencies):

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

                for index in range(1, aspect_length):

                    next_indices = aspect_indices[aspect_index][index]

                    if aspect_class is not None and ontology_classes_of_sentence[next_indices] is not None:

                        aspect_class = owl.types.new_class(aspect_class.__name__ +
                                                           ontology_classes_of_sentence[next_indices].__name__,
                                                           (aspect_class, ontology_classes_of_sentence[next_indices]))

                    elif aspect_class is None and ontology_classes_of_sentence[next_indices] is not None:
                        aspect_class = ontology_classes_of_sentence[next_indices]

            for word_index in range(number_words_in_sentence):

                if word_index not in aspect_indices[aspect_index]:
                    word_class = ontology_classes_of_sentence[word_index]
                    category = aspect_categories[aspect_index]
                    aspect_dependency = aspect_dependencies[aspect_index][word_index]
                    relation, polarity = self.relation_between_word_aspect(word_class, category, aspect_class,
                                                                           aspect_dependency)
                else:
                    polarity = [1, 0, 0]

                    # an aspect can have a relation with another aspect
                    if aspect_dependencies[aspect_index][word_index] != 'no':
                        relation = [1, 0]
                    else:
                        relation = [0, 1]

                polarities.append(polarity)
                relations.append(relation)

            polarities_sentence.append(polarities)
            relations_sentence.append(relations)

        return polarities_sentence, relations_sentence

    def relation_between_word_aspect(self, word_class, category, aspect_class, aspect_dependency):

        if aspect_dependency != 'no':
            aspect_relation = [1, 0]
        else:
            aspect_relation = [0, 1]

        if word_class is None:
            # aspect_relation, UnknownPolarity
            return aspect_relation, [1, 0, 0]

        positive_class = self.ontology.search(iri='*Positive')[0]
        negative_class = self.ontology.search(iri='*Negative')[0]

        sub_classes = word_class.ancestors()

        if self.ontology.PropertyMention in sub_classes:

            if self.ontology.GenericNegativePropertyMention in sub_classes:
                # True, Negative
                return [1, 0], [0, 1, 0]
            elif self.ontology.GenericPositivePropertyMention in sub_classes:
                # True, Positive
                return [1, 0], [0, 0, 1]
            elif self.ontology.Positive in sub_classes:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        # True, Positive
                        return [1, 0], [0, 0, 1]
                # aspect_relation, UnknownPolarity
                return aspect_relation, [1, 0, 0]
            elif self.ontology.Negative in sub_classes:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        # True, Negative
                        return [1, 0], [0, 1, 0]
                # aspect_relation, UnknownPolarity
                return aspect_relation, [1, 0, 0]
            else:
                if aspect_class is not None:
                    word_class = owl.types.new_class(word_class.__name__ + aspect_class.__name__,
                                                     (word_class, aspect_class))
                owl.sync_reasoner()
                positive = positive_class.__subclasscheck__(word_class)
                negative = negative_class.__subclasscheck__(word_class)

                if positive and not negative:
                    # True Positive
                    return [1, 0], [0, 0, 1]
                elif not positive and negative:
                    # True Negative
                    return [1, 0], [0, 1, 0]
                elif positive and negative:
                    # True UnknownPolarity
                    return [1, 0], [1, 0, 0]
                else:
                    # aspect_relation, UnknownPolarity
                    return aspect_relation, [1, 0, 0]

        elif self.ontology.ActionMention in sub_classes:

            if self.ontology.GenericNegativeAction in sub_classes:
                # True, Negative
                return [1, 0], [0, 1, 0]
            elif self.ontology.GenericPositiveAction in sub_classes:
                # True, Positive
                return [1, 0], [0, 0, 1]
            elif self.ontology.Positive in sub_classes:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        # True, Positive
                        return [1, 0], [0, 0, 1]
                # aspect_relation, UnknownPolarity
                return aspect_relation, [1, 0, 0]
            elif self.ontology.Negative in sub_classes:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        # True Negative
                        return [1, 0], [0, 1, 0]
                # aspect_relation, UnknownPolarity
                return aspect_relation, [1, 0, 0]
            else:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        # True UnknownPolarity
                        return [1, 0], [1, 0, 0]
                # aspect_relation, UnknownPolarity
                return aspect_relation, [1, 0, 0]

        elif self.ontology.EntityMention in sub_classes:

            if self.ontology.Positive in sub_classes:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        # True Positive
                        return [1, 0], [0, 0, 1]
                # aspect_relation, UnknownPolarity
                return aspect_relation, [1, 0, 0]
            elif self.ontology.Negative in sub_classes:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        # True Negative
                        return [1, 0], [0, 1, 0]
                # aspect_relation, UnknownPolarity
                return aspect_relation, [1, 0, 0]
            else:
                for c in sub_classes:
                    if c == owl.Thing:
                        continue
                    elif category in c.aspect:
                        # True UnknownPolarity
                        return [1, 0], [1, 0, 0]
                # aspect_relation, UnknownPolarity
                return aspect_relation, [1, 0, 0]
        else:
            # aspect_relation, UnknownPolarity
            return aspect_relation, [1, 0, 0]

    def mention_tagging(self, ontology_classes_sentence):

        mentions = []
        words_of_sentence = len(ontology_classes_sentence)

        for word_index in range(words_of_sentence):

            if ontology_classes_sentence[word_index] is None:
                # UnknownMention
                mentions.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                continue

            sub_classes = ontology_classes_sentence[word_index].ancestors()

            if self.ontology.PropertyMention in sub_classes:
                # PropertyMention
                mentions.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.ontology.PersonMention in sub_classes:
                # PersonMention
                mentions.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.ontology.ActionMention in sub_classes:
                # ActionMention
                mentions.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.ontology.ServiceMention in sub_classes:
                # ServiceMention
                mentions.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.ontology.AmbienceMention in sub_classes:
                # AmbienceMention
                mentions.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            elif self.ontology.LocationMention in sub_classes:
                # LocationMention
                mentions.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            elif self.ontology.RestaurantMention in sub_classes:
                # RestaurantMention
                mentions.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            elif self.ontology.PriceMention in sub_classes:
                # PriceMention
                mentions.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif self.ontology.StyleOptionsMention in sub_classes:
                # StyleOptionsMention
                mentions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif self.ontology.ExperienceMention in sub_classes:
                # ExperienceMention
                mentions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif self.ontology.SustenanceMention in sub_classes:
                # SustenanceMention
                mentions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            elif self.ontology.TimeMention in sub_classes:
                # TimeMention
                mentions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
            else:
                # UnknownMention
                mentions.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return mentions

