import json
import logging
import math
import os
import queue
import re
import time
from typing import List, Union, Dict, FrozenSet, Iterable

try:
    import nltk
    import numpy as np
    import torch
    import torch.multiprocessing as mp
    import tqdm
    has_easynmt = True
except ImportError:
    has_easynmt = False

from bpm_ai_core.translation.easy_nmt.opus_mt import OpusMT
from bpm_ai_core.translation.easy_nmt.util import http_get, fullname
from bpm_ai_core.translation.nmt import NMTModel
from bpm_ai_core.util.language import indentify_language

logger = logging.getLogger(__name__)

DOWNLOAD_URL = 'http://easynmt.net/models/v2'


class EasyNMT(NMTModel):
    """
    See https://github.com/UKPLab/EasyNMT.
    Copied and modified to remove dependency to fasttext.
    """

    def __init__(
            self,
            model_name: str = "opus-mt",
            cache_folder: str = None,
            translator=None,
            load_translator: bool = True,
            device=None,
            max_length: int = None,
            **kwargs
    ):
        """
        Easy-to-use, state-of-the-art machine translation
        :param model_name:  Model name (see Readme for available models)
        :param cache_folder: Which folder should be used for caching models. Can also be set via the EASYNMT_CACHE env. variable
        :param translator: Translator object. Set to None, to automatically load the model via the model name.
        :param load_translator: If set to false, it will only load the config but not the translation engine
        :param device: CPU / GPU device for PyTorch
        :param max_length: Max number of token per sentence for translation. Longer text will be truncated
        :param kwargs: Further optional parameters for the different models
        """
        if not has_easynmt:
            raise ImportError('easynmt dependencies are not installed')

        self._model_name = model_name
        self._fasttext_lang_id = None
        self._lang_pairs = frozenset()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.config = None

        if cache_folder is None:
            if 'EASYNMT_CACHE' in os.environ:
                cache_folder = os.environ['EASYNMT_CACHE']
            else:
                cache_folder = os.path.join(torch.hub._get_torch_home(), 'easynmt_v2')
        self._cache_folder = cache_folder

        if translator is not None:
            self.translator = translator
        else:
            if os.path.exists(model_name) and os.path.isdir(model_name):
                model_path = model_name
            else:
                model_name = model_name.lower()
                model_path = os.path.join(cache_folder, model_name)

                if not os.path.exists(model_path) or not os.listdir(model_path):
                    logger.info("Downloading EasyNMT model {} and saving it at {}".format(model_name, model_path))

                    model_path_tmp = model_path.rstrip("/").rstrip("\\") + "_part"
                    os.makedirs(model_path_tmp, exist_ok=True)

                    # Download easynmt.json
                    config_url = DOWNLOAD_URL + "/{}/easynmt.json".format(model_name)
                    config_path = os.path.join(model_path_tmp, 'easynmt.json')
                    http_get(config_url, config_path)

                    with open(config_path) as fIn:
                        downloaded_config = json.load(fIn)

                    if 'files' in downloaded_config:
                        for filename, url in downloaded_config['files'].items():
                            logger.info("Download {} from {}".format(filename, url))
                            http_get(url, os.path.join(model_path_tmp, filename))

                    ##Rename tmp path
                    try:
                        os.rename(model_path_tmp, model_path)
                    except Exception:
                        pass

            with open(os.path.join(model_path, 'easynmt.json')) as fIn:
                self.config = json.load(fIn)

            if 'lang_pairs' in self.config:
                self._lang_pairs = frozenset(self.config['lang_pairs'])

            if load_translator:
                self.translator = OpusMT(easynmt_path=model_path, **self.config['model_args'])
                self.translator.max_length = max_length

    async def _do_translate(self, text: str | list[str], target_language: str) -> str | list[str]:
        if isinstance(text, str):
            return self.do_translate(text, target_language, indentify_language(text))
        else:
            return [self.do_translate(t, target_language, indentify_language(t)) for t in text]

    def do_translate(
            self,
            documents: Union[str, List[str]],
            target_lang: str,
            source_lang: str,
            show_progress_bar: bool = False,
            beam_size: int = 5,
            batch_size: int = 16,
            perform_sentence_splitting: bool = True,
            paragraph_split: str = "\n",
            sentence_splitter=None,
            **kwargs
    ):
        """
        This method translates the given set of documents
        :param documents: If documents is a string, returns the translated document as string. If documents is a list of strings, translates all documents and returns a list.
        :param target_lang: Target language for the translation
        :param source_lang: Source language for all documents.
        :param show_progress_bar: If true, plot a progress bar on the progress for the translation
        :param beam_size: Size for beam search
        :param batch_size: Number of sentences to translate at the same time
        :param perform_sentence_splitting: Longer documents are broken down sentences, which are translated individually
        :param paragraph_split: Split symbol for paragraphs. No sentences can go across the paragraph_split symbol.
        :param sentence_splitter: Method used to split sentences. If None, uses the default self.sentence_splitting method
        :param kwargs: Optional arguments for the translator model
        :return: Returns a string or a list of string with the translated documents
        """

        # Method_args will store all passed arguments to method
        method_args = locals()
        del method_args['self']
        del method_args['kwargs']
        method_args.update(kwargs)

        if source_lang == target_lang:
            return documents

        is_single_doc = False
        if isinstance(documents, str):
            documents = [documents]
            is_single_doc = True

        if perform_sentence_splitting:
            if sentence_splitter is None:
                sentence_splitter = self.sentence_splitting

            # Split document into sentences
            start_time = time.time()
            splitted_sentences = []
            sent2doc = []
            for doc in documents:
                paragraphs = doc.split(paragraph_split) if paragraph_split is not None else [doc]
                for para in paragraphs:
                    for sent in sentence_splitter(para.strip(), source_lang):
                        sent = sent.strip()
                        if len(sent) > 0:
                            splitted_sentences.append(sent)
                sent2doc.append(len(splitted_sentences))
            # logger.info("Sentence splitting done after: {:.2f} sec".format(time.time() - start_time))
            # logger.info("Translate {} sentences".format(len(splitted_sentences)))

            translated_sentences = self.translate_sentences(splitted_sentences, target_lang=target_lang,
                                                            source_lang=source_lang,
                                                            show_progress_bar=show_progress_bar, beam_size=beam_size,
                                                            batch_size=batch_size, **kwargs)

            # Merge sentences back to documents
            start_time = time.time()
            translated_docs = []
            for doc_idx in range(len(documents)):
                start_idx = sent2doc[doc_idx - 1] if doc_idx > 0 else 0
                end_idx = sent2doc[doc_idx]
                translated_docs.append(
                    self._reconstruct_document(documents[doc_idx], splitted_sentences[start_idx:end_idx],
                                               translated_sentences[start_idx:end_idx]))

            # logger.info("Document reconstruction done after: {:.2f} sec".format(time.time() - start_time))
        else:
            translated_docs = self.translate_sentences(documents, target_lang=target_lang, source_lang=source_lang,
                                                       show_progress_bar=show_progress_bar, beam_size=beam_size,
                                                       batch_size=batch_size, **kwargs)

        if is_single_doc:
            translated_docs = translated_docs[0]

        return translated_docs

    @staticmethod
    def _reconstruct_document(doc, org_sent, translated_sent):
        """
        This method reconstructs the translated document and
        keeps white space in the beginning / at the end of sentences.
        """
        sent_idx = 0
        char_idx = 0
        translated_doc = ""
        while char_idx < len(doc):
            if sent_idx < len(org_sent) and doc[char_idx] == org_sent[sent_idx][0]:
                translated_doc += translated_sent[sent_idx]
                char_idx += len(org_sent[sent_idx])
                sent_idx += 1
            else:
                translated_doc += doc[char_idx]
                char_idx += 1
        return translated_doc

    def translate_sentences(
            self,
            sentences: Union[str, List[str]],
            target_lang: str,
            source_lang: str,
            show_progress_bar: bool = False,
            beam_size: int = 5,
            batch_size: int = 32,
            **kwargs
    ):
        """
        This method translates individual sentences.

        :param sentences: A single sentence or a list of sentences to be translated
        :param source_lang: Source language for all sentences.
        :param target_lang: Target language for the translation
        :param show_progress_bar: Show a progress bar
        :param beam_size: Size for beam search
        :param batch_size: Mini batch size
        :return: List of translated sentences
        """

        if source_lang == target_lang:
            return sentences

        is_single_sentence = False
        if isinstance(sentences, str):
            sentences = [sentences]
            is_single_sentence = True

        output = []

        # Sort by length to speed up processing
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        iterator = range(0, len(sentences_sorted), batch_size)
        if show_progress_bar:
            scale = min(batch_size, len(sentences))
            iterator = tqdm.tqdm(iterator, total=len(sentences) / scale, unit_scale=scale, smoothing=0)

        for start_idx in iterator:
            output.extend(self.translator.translate_sentences(sentences_sorted[start_idx:start_idx + batch_size],
                                                              source_lang=source_lang, target_lang=target_lang,
                                                              beam_size=beam_size, device=self.device, **kwargs))

        # Restore original sorting of sentences
        output = [output[idx] for idx in np.argsort(length_sorted_idx)]

        if is_single_sentence:
            output = output[0]

        return output

    def start_multi_process_pool(self, target_devices: List[str] = None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu'] * 4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(target=EasyNMT._encode_multi_process_worker,
                            args=(cuda_id, self, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    def translate_multi_process(
            self,
            pool: Dict[str, object],
            documents: List[str],
            show_progress_bar: bool = True,
            chunk_size: int = None, **kwargs
    ) -> List[str]:
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        """

        if chunk_size is None:
            chunk_size = min(math.ceil(len(documents) / len(pool["processes"]) / 10), 1000)

        logger.info("Chunk data into packages of size {}".format(chunk_size))

        input_queue = pool['input']
        last_chunk_id = 0

        for start_idx in range(0, len(documents), chunk_size):
            input_queue.put([last_chunk_id, documents[start_idx:start_idx + chunk_size], kwargs])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in
                               tqdm.tqdm(range(last_chunk_id), total=last_chunk_id, unit_scale=chunk_size, smoothing=0,
                                         disable=not show_progress_bar)], key=lambda chunk: chunk[0])
        translated = []
        for chunk in results_list:
            translated.extend(chunk[1])
        return translated

    def translate_stream(self, stream: Iterable[str], show_progress_bar: bool = True, chunk_size: int = 128,
                         **kwargs) -> List[str]:
        batch = []
        for doc in tqdm.tqdm(stream, smoothing=0.0, disable=not show_progress_bar):
            batch.append(doc)

            if len(batch) >= chunk_size:
                translated = self._do_translate(batch, show_progress_bar=False, **kwargs)
                for trans_doc in translated:
                    yield trans_doc
                batch = []

        if len(batch) > 0:
            translated = self._do_translate(batch, show_progress_bar=False, **kwargs)
            for trans_doc in translated:
                yield trans_doc

    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        """
        Internal working process to encode sentences in multi-process setup
        """
        model.device = target_device
        while True:
            try:
                id, documents, kwargs = input_queue.get()
                translated = model.do_translate(documents, **kwargs)
                results_queue.put([id, translated])
            except queue.Empty:
                break

    def sentence_splitting(self, text: str, lang: str = None):
        if lang in ['ar', 'jp', 'ko', 'zh']:
            sentences = list(re.findall(u'[^!?。\.]+[!?。\.]*', text, flags=re.U))
        else:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')

            sentences = nltk.sent_tokenize(text)

        return sentences

    @property
    def lang_pairs(self) -> FrozenSet[str]:
        """
        Returns all allowed languages directions for the loaded model
        """
        return self._lang_pairs

    def get_languages(self, source_lang: str = None, target_lang: str = None) -> List[str]:
        """
        Returns all available languages supported by the model
        :param source_lang:  If not None, then returns all languages to which we can translate for the given source_lang
        :param target_lang:  If not None, then returns all languages from which we can translate for the given target_lang
        :return: Sorted list with the determined languages
        """

        langs = set()
        for lang_pair in self.lang_pairs:
            source, target = lang_pair.split("-")

            if source_lang is None and target_lang is None:
                langs.add(source)
                langs.add(target)
            elif target_lang is not None and target == target_lang:
                langs.add(source)
            elif source_lang is not None and source == source_lang:
                langs.add(target)

        return sorted(list(langs))

    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        filepath = os.path.join(output_path, 'easynmt.json')

        config = {
            'model_class': fullname(self.translator),
            'lang_pairs': list(self.lang_pairs),
            'model_args': self.translator.save(output_path)
        }

        with open(filepath, 'w') as fOut:
            json.dump(config, fOut)
