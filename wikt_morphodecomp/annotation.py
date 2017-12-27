__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import saf.constants as annot_const
from saf import Document, Sentence, Token
from saf.annotators import Annotator
from morphodecomp import decompose, decompose_ensemble, load_models, EnsembleMode
from config import DEFAULT_CONFIG_PATH

cache_config_paths = None
cache_models = None


def get_morpho_dict(word_list, config_paths, ensemble):
    global cache_config_paths
    global cache_models

    if (ensemble):
        if (config_paths == cache_config_paths):
            models = cache_models
        else:
            models = load_models(config_paths)
            cache_config_paths = config_paths
            cache_models = models

        morpho_analyses = decompose_ensemble(word_list, config_paths, models=models, mode=EnsembleMode.MAJORITY_OVERALL)
    else:
        morpho_analyses = decompose(word_list, config_paths[0])

    dict_analyses = dict([(anlz["word"], anlz) for anlz in morpho_analyses])

    return dict_analyses


class MorphoAnalysisAnnotator(Annotator):
    def annotate(self, annotable, ensemble=False, config_paths=(DEFAULT_CONFIG_PATH,)):
        if (annotable.__class__.__name__ == "Document"):
            return MorphoAnalysisAnnotator.annotate_document(annotable, ensemble, config_paths)
        elif (annotable.__class__.__name__ == "Sentence"):
            return MorphoAnalysisAnnotator.annotate_sentence(annotable, ensemble, config_paths)
        elif (annotable.__class__.__name__ == "Token"):
            return MorphoAnalysisAnnotator.annotate_token(annotable, ensemble, config_paths)

    @staticmethod
    def annotate_document(document, ensemble, config_paths):
        words = set()
        for sentence in document.sentences:
            for token in sentence.tokens:
                if (len(token.surface) >= 3):
                    words.add(token.surface.lower())

        morpho_dict = get_morpho_dict(list(words), config_paths, ensemble)

        for sentence in document.sentences:
            MorphoAnalysisAnnotator.annotate_sentence(sentence, ensemble, config_paths, morpho_dict)

        return document

    @staticmethod
    def annotate_sentence(sentence, ensemble, config_paths, morpho_dict=None):
        if (morpho_dict is None):
            morpho_dict = get_morpho_dict([token.surface.lower() for token in sentence.tokens if (len(token) >= 3)],
                                          config_paths, ensemble)

        for token in sentence.tokens:
            MorphoAnalysisAnnotator.annotate_token(token, ensemble, config_paths, morpho_dict)

        return sentence

    @staticmethod
    def annotate_token(token, ensemble, config_paths, morpho_dict=None):
        nondecomposable_annot = {"decomp": [token.surface], "confidence": 0.}

        if (len(token.surface) < 3):
            token.annotations[annot_const.MORPHO] = nondecomposable_annot
        else:
            if (morpho_dict is None):
                morpho_dict = get_morpho_dict([token.surface.lower()], config_paths, ensemble)

            if (token.surface in morpho_dict):
                token.annotations[annot_const.MORPHO] = morpho_dict[token.surface]
                if ("word" in token.annotations[annot_const.MORPHO]):
                    del token.annotations[annot_const.MORPHO]["word"]
            else:
                token.annotations[annot_const.MORPHO] = nondecomposable_annot

        return token
