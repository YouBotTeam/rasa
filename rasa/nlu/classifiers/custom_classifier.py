from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.model import Metadata
from typing import List, Dict, Text, Any, Optional
from rasa.shared.nlu.training_data.message import Message

MY_DIET_CLASSIFIER = 'my_diet_classifier'


class Classifier(IntentClassifier):

    def __init__(self, component_config, **kwargs):
        super().__init__(component_config)
        self.classifier_name = self.__class__.name
        self.classifier = None
        # self.train_set = None

    def preprocessing(self, tokens):
        pass

    def train(self, training_data, **kwargs):
        pass

    def process(self, message: Message, **kwargs) -> Dict[Text, Dict]:
        """

        :param **kwargs:
        :param message:
        :return:
            dict{
                    'intent':   intent_dict,
                    'entities': entities_dict
                }
        """
        pass

    def persist(self, file_name, model_dir):
        pass

    def load(self,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component = None,
        **kwargs: Any):
        pass

    def pred_to_rasa_output(self, value, confidence):
        pass
