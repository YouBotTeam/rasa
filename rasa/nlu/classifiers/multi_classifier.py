import logging
from copy import deepcopy
from typing import Any, Optional, Text, Dict, List, Type

from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.model import Metadata
from rasa.shared.nlu.constants import INTENT, ENTITIES
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

from rasa.utils.tensorflow.constants import EPOCHS, RANDOM_SEED

from rasa.nlu.classifiers.my_diet_classifier import PARAMS as DIET_PARAMS
from tensorflow.python.keras.callbacks import EarlyStopping

from rasa.nlu.classifiers.custom_classifier import MY_DIET_CLASSIFIER, Classifier
from rasa.nlu.classifiers.my_diet_classifier import My_Diet_Classifier
from rasa.nlu.utils.majority_voting import AVG, intent_maj_voting

logger = logging.getLogger(__name__)

MAJ_VOTING = 'maj_voting'
MODEL_NAME = 'multi_classifier'
INTENT_RANKING = 'intent_ranking'


class Multi_Classifier(IntentClassifier, EntityExtractor):
    """A new component"""

    name = "Multi_Classifier"

    defaults = {EPOCHS: 1,
                "classifiers": [MY_DIET_CLASSIFIER, MY_DIET_CLASSIFIER, MY_DIET_CLASSIFIER],
                "n_classifiers": 3,
                "strategy": MAJ_VOTING,
                "confidence": AVG,
                "random_seed": 2021,
                "tensorboard_log_level": 'epoch',
                "tensorboard_log_directory": './tensorboard/multi_classifier'
                }
    language_list = ['it']

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""
        return [Featurizer]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super(Multi_Classifier, self).__init__(component_config)
        self.component_config['model_name'] = 'multi-classifier'
        self.classifiers = self._init_classifiers()

    def _init_classifiers(self) -> Dict[Text, Classifier]:
        classifiers = {}

        for i, classifier_name in enumerate(self.component_config['classifiers']):

            # DIET CLASSIFIER
            if(classifier_name == MY_DIET_CLASSIFIER):
                name = f'My_DIET_Classifier_{i}'
                diet_params = deepcopy(DIET_PARAMS)
                diet_params.update(self.component_config)
                diet_params[RANDOM_SEED] = self.component_config[RANDOM_SEED] + i

                classifier = My_Diet_Classifier(diet_params)
                classifiers[name] = classifier

            # OTHER CLASSIFIER
            #
            #

        return classifiers

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        callbacks = [EarlyStopping(monitor="val_i_loss",
                                   mode="auto",
                                   min_delta=0.005,
                                   patience=3)]

        for classifier in self.classifiers.values():
            classifier.train(training_data, callbacks=callbacks)

    def process(self, message: Message, **kwargs: Any) -> None:
        intent_result, entities_result = {}, {}

        for i, key in enumerate(self.classifiers):
            classifier = self.classifiers[key]
            result_dict = classifier.process(message, )

            intent_result[key] = result_dict.get(INTENT)
            intent_result[key]['model_n'] = i
            entities_result[key] = result_dict.get(ENTITIES)
            # entities_result[key]['model_n'] = i

        final_intent, intent_models, model_idx = self.intent_voting(intent_result)
        final_entity = self.entity_voting(entities_result, model_idx)

        if final_intent is not None:
            logger.info(f'Message set - INTENT: {final_intent}')
            message.set(INTENT, final_intent, add_to_output=True)
            message.set(INTENT_RANKING, intent_models, add_to_output=True)
        else:
            logger.warning(f'Intent voting return None starting from {intent_result}')

        if final_entity is not None:
            logger.info(f'Message set - ENTITY: {final_entity}')
            message.set(ENTITIES, final_entity, add_to_output=True)
        else:
            logger.warning(f'Entity voting return None starting from {entities_result}')

        ## TEST
        #
        # label = {"id": hash(intent_name),
        #          "name": intent_name,
        #          "confidence": 0.95}
        # label_ranking = [
        #     {
        #         "id": hash(intent_name),
        #         "name": intent_name,
        #         "confidence": 0.94,
        #
        #         "extractor": extractor_name,
        #         "model_n": model_n
        #     }]
        # entities = []
        #
        # message.set(INTENT, label, add_to_output=True)
        # message.set(INTENT_RANKING, label_ranking, add_to_output=True)
        # message.set(ENTITIES, entities, add_to_output=True)
        ###

    def intent_voting(self, intent_result):
        for intent in intent_result:
            if intent is None:
                return None

        strategy = self.component_config['strategy']

        if strategy == MAJ_VOTING:
            maj_voting, models, model_idx = intent_maj_voting(intent_result)

        else:
            raise ValueError('Provided Strategy not found')

        return maj_voting, models, model_idx

    def entity_voting(self, entities_result, model_idx):
        for entity in entities_result.values():
            if (entity is None):
                return None

        return list(entities_result.values())[model_idx]

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:

        persist_dict = {}

        # JUST IN CASE THE CLASSIFIER IS DIET_CLASSIFIER
        for key in self.classifiers:
            classifier = self.classifiers[key]
            filename = key
            save_dict = classifier.persist(filename, model_dir)
            persist_dict[key] = save_dict

        # model_file_name = os.path.join(model_dir, MODEL_NAME)
        # pickle_dump(model_file_name, self)
        return {"files": persist_dict}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["Multi_Classifier"] = None,
        should_finetune: bool = False,
        **kwargs: Any,
    ) -> "Multi_Classifier":
        """Load this component from file."""

        # JUST IN CASE THE CLASSIFIER IS DIET_CLASSIFIER
        multi_classifier = Multi_Classifier(component_config=meta)

        for key in multi_classifier.classifiers:

            meta['file'] = meta['files'][key]['file']
            meta['params'] = meta['files'][key]['params']

            logger.debug(f'Loading model: {key}')
            multi_classifier.classifiers[key] = multi_classifier.classifiers[key].load(meta, model_dir)

            logger.debug(f'Model {key} loaded')

        return multi_classifier


