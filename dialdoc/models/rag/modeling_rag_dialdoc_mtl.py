from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import OrderedDict
import warnings
import torch
import copy
from torch import nn
from transformers.models.rag.modeling_rag import (
    RagModel,
    RagTokenForGeneration,
    RetrievAugLMOutput,
    RetrievAugLMMarginOutput,
)
from transformers.models.rag.retrieval_rag import RagRetriever
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GreedySearchOutput, BeamSearchOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList, validate_stopping_criteria

from dialdoc.models.rag.configuration_rag_dialdoc import DialDocRagConfig
from dialdoc.models.rag.modeling_bart_mtl import BartForConditionalGenerationMTL

from transformers.utils import logging

logger = logging.get_logger(__name__)
TASKS = ['grounding', 'generation']

class DialDocRagModelMTL(RagModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional = None,
        **kwargs,
    ):
        self.config_class = DialDocRagConfig
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an question_encoder and a generator has to be provided."
        if config is None:
            config = DialDocRagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        super(RagModel, self).__init__(config)
        if question_encoder is None:
            from transformers.models.auto.modeling_auto import AutoModel

            question_encoder = AutoModel.from_config(config.question_encoder)

        if generator is None:
            from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM

            generator = BartForConditionalGenerationMTL(config.generator)

        self.retriever = retriever
        if self.retriever is not None:
            assert isinstance(
                retriever, RagRetriever
            ), f"`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`"
            self.retriever = retriever

        self.question_encoder = question_encoder
        self.generator = generator

        self.bm25 = kwargs.pop("bm25", None)
        if self.bm25:
            logger.info("Using BM25 inside RAG Model")

    @staticmethod
    def mean_pool(vector: torch.LongTensor):
        return vector.sum(axis=0) / vector.shape[0]

    @staticmethod
    def get_attn_mask(tokens_tensor: torch.LongTensor) -> torch.tensor:
        return tokens_tensor != 0

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        doc_scores=None,
        context_input_ids=None,
        context_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_retrieved=None,
        n_docs=None,
        domain=None,
    ):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_retrieved = output_retrieved if output_retrieved is not None else self.config.output_retrieved

        # whether retriever has to be used
        has_to_retrieve = (
            self.retriever is not None
            and (context_input_ids is None or context_attention_mask is None or doc_scores is None)
            and encoder_outputs is None
        )
        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is None:
            dialog_lengths = None
            if has_to_retrieve:
                question_enc_outputs = self.question_encoder(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
                )
                # if self.config.scoring_func in ['linear', 'linear2', 'linear3', 'nonlinear', 'reranking', 'reranking2']:
                if self.config.scoring_func != "original":
                    combined_out = question_enc_outputs.pooler_output

                    ## Get mask for current turn input ids
                    curr_turn_mask = torch.logical_xor(attention_mask, token_type_ids)
                    current_turn_input_ids = input_ids * curr_turn_mask
                    current_turn_only_out = self.question_encoder(
                        current_turn_input_ids, attention_mask=curr_turn_mask.long(), return_dict=True
                    )
                    current_turn_output = current_turn_only_out.pooler_output

                    ## Split the dpr sequence output
                    sequence_output = question_enc_outputs.hidden_states[-1]
                    attn_mask = self.get_attn_mask(input_ids)
                    ## Split sequence output, and pool each sequence
                    seq_out_0 = []  # last turn, if query; doc structure if passage
                    seq_out_1 = []  # dial history, if query; passage text if passage
                    dialog_lengths = []
                    for i in range(sequence_output.shape[0]):
                        seq_out_masked = sequence_output[i, attn_mask[i], :]
                        segment_masked = token_type_ids[i, attn_mask[i]]
                        seq_out_masked_0 = seq_out_masked[segment_masked == 0, :]
                        seq_out_masked_1 = seq_out_masked[segment_masked == 1, :]
                        dialog_lengths.append((len(seq_out_masked_0), len(seq_out_masked_1)))
                        ### perform pooling
                        seq_out_0.append(self.mean_pool(seq_out_masked_0))
                        seq_out_1.append(self.mean_pool(seq_out_masked_1))

                    pooled_output_q = torch.cat([seq.view(1, -1) for seq in seq_out_0], dim=0)
                    pooled_output_h = torch.cat([seq.view(1, -1) for seq in seq_out_1], dim=0)

                    if self.config.scoring_func in ["reranking_original", "current_original"]:
                        current_out = current_turn_output
                    else:
                        current_out = pooled_output_q

                    retriever_outputs = self.retriever(
                        input_ids,
                        combined_out.cpu().detach().to(torch.float32).numpy(),
                        current_out.cpu().detach().to(torch.float32).numpy(),
                        pooled_output_h.cpu().detach().to(torch.float32).numpy(),
                        prefix=self.generator.config.prefix,
                        n_docs=n_docs,
                        dialog_lengths=dialog_lengths,
                        domain=domain,
                        return_tensors="pt",
                    )
                else:
                    combined_out = question_enc_outputs[0]  # hidden states of question encoder

                    retriever_outputs = self.retriever(
                        input_ids,
                        combined_out.cpu().detach().to(torch.float32).numpy(),
                        combined_out.cpu().detach().to(torch.float32).numpy(),  ## sending dummy
                        combined_out.cpu().detach().to(torch.float32).numpy(),  ## sending dummy
                        prefix=self.generator.config.prefix,
                        n_docs=n_docs,
                        dialog_lengths=dialog_lengths,
                        domain=domain,
                        return_tensors="pt",
                        bm25=self.bm25,
                    )

                (
                    context_input_ids,
                    context_attention_mask,
                    retrieved_doc_embeds,
                    retrieved_doc_ids,
                    retrieved_doc_scores,
                ) = (
                    retriever_outputs["context_input_ids"],
                    retriever_outputs["context_attention_mask"],
                    retriever_outputs["retrieved_doc_embeds"],
                    retriever_outputs["doc_ids"],
                    retriever_outputs["doc_scores"],
                )

                # set to correct device
                retrieved_doc_embeds = retrieved_doc_embeds.to(combined_out)
                context_input_ids = context_input_ids.to(input_ids)
                context_attention_mask = context_attention_mask.to(input_ids)
                doc_scores = retrieved_doc_scores.to(combined_out)

                # compute doc_scores
                if self.config.scoring_func in [
                    "reranking",
                    "reranking2",
                    "original",
                    "reranking_original",
                    "current_original",
                    "current_pooled",
                ]:
                    doc_scores = torch.bmm(combined_out.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
                elif self.config.scoring_func in ["linear", "linear2", "linear3", "nonlinear"]:
                    doc_scores_curr = torch.bmm(
                        pooled_output_q.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)

                    doc_scores_hist = torch.bmm(
                        pooled_output_h.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)

                    if self.config.scoring_func == "linear":
                        doc_scores = doc_scores_curr + doc_scores_hist
                    elif self.config.scoring_func == "linear2":
                        doc_scores = doc_scores_curr + 0.5 * doc_scores_hist
                    elif self.config.scoring_func == "linear3":
                        # TODO: linear 3 scoring
                        doc_scores = doc_scores_curr + 0.5 * doc_scores_hist
                    else:  # nonlinear
                        bsz = doc_scores_curr.shape[0]
                        doc_scores_curr_flattened = doc_scores_curr.flatten().unsqueeze(
                            1
                        )  # from (B, n_docs) to (Bxn_docs, 1)
                        doc_scores_hist_flattened = doc_scores_hist.flatten().unsqueeze(
                            1
                        )  # from (B, n_docs) to (Bxn_docs, 1)
                        scorer_inp = torch.cat(
                            [doc_scores_curr_flattened, doc_scores_hist_flattened], dim=1
                        )  # (Bxn_docs, 2)
                        scores = self.retriever.nn_scorer(scorer_inp)
                        doc_scores = scores.reshape((bsz, -1))

            else:
                assert (
                    context_input_ids is not None
                ), "Make sure that `context_input_ids` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."
                assert (
                    context_attention_mask is not None
                ), "Make sure that `context_attention_mask` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."
                assert (
                    doc_scores is not None
                ), "Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."

        assert (
            doc_scores is not None
        ), "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."

        assert (
            doc_scores.shape[1] % n_docs
        ) == 0, f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}."

        # Decoder input without context documents
        for task in TASKS:
            if decoder_input_ids[task] is not None:
                decoder_input_ids[task] = decoder_input_ids[task].repeat_interleave(n_docs, dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)

        gen_outputs = self.generator(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=True,
        )

        if not has_to_retrieve:
            combined_out = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions

        if not has_to_retrieve or not output_retrieved:
            # don't output retrieved docs
            context_input_ids = (None,)
            context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None

        return RetrievAugLMOutput(
            logits=gen_outputs['logits'],
            doc_scores=doc_scores,
            past_key_values=gen_outputs['past_key_values'],
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=combined_out,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_enc_last_hidden_state=gen_outputs['encoder_last_hidden_state'],
            generator_enc_hidden_states=gen_outputs['encoder_hidden_states'],
            generator_enc_attentions=gen_outputs['encoder_attentions'],
            generator_dec_hidden_states=gen_outputs['decoder_hidden_states'],
            generator_dec_attentions=gen_outputs['decoder_attentions'],
            generator_cross_attentions=gen_outputs['cross_attentions'],
        )


class DialDocRagTokenForGenerationMTL(RagTokenForGeneration):
    config_class = DialDocRagConfig

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional = None,
        bm25: Optional = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = DialDocRagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        super(RagTokenForGeneration, self).__init__(config)
        # instantiate model
        if bm25:
            logger.info("Using bm25")
            self.rag = DialDocRagModelMTL(
                config=config, question_encoder=question_encoder, generator=generator, retriever=retriever, bm25=bm25
            )
            self.bm25 = bm25
        else:
            self.rag = DialDocRagModelMTL(
                config=config, question_encoder=question_encoder, generator=generator, retriever=retriever
            )
            self.bm25 = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_retrieved=None,
        do_marginalize=None,
        reduce_loss=None,
        labels=None,
        n_docs=None,
        domain=None,
        **kwargs,  # needs kwargs for generation
    ):
        r"""
        do_marginalize (:obj:`bool`, `optional`):
            If :obj:`True`, the logits are marginalized over all documents by making use of
            ``torch.nn.functional.log_softmax``.
        reduce_loss (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the NLL loss is reduced using the
            ``torch.Tensor.sum`` operation.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Legacy dictionary, which is required so that model can use `generate()` function.

        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
            >>> import torch

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

            >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
            >>> with tokenizer.as_target_tokenizer():
            ...    targets = tokenizer("In Paris, there are 10 million people.", return_tensors="pt")
            >>> input_ids = inputs["input_ids"]
            >>> labels = targets["input_ids"]
            >>> outputs = model(input_ids=input_ids, labels=labels)

            >>> # or use retriever separately
            >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
            >>> # 1. Encode
            >>> question_hidden_states = model.question_encoder(input_ids)[0]
            >>> # 2. Retrieve
            >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
            >>> doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)
            >>> # 3. Forward to generator
            >>> outputs = model(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, decoder_input_ids=labels)

            >>> # or directly generate
            >>> generated = model.generate(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores)
            >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        """
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_marginalize = do_marginalize if do_marginalize is not None else self.config.do_marginalize
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
            domain=domain,
        )

        loss = {}
        logits = outputs.logits
        for task in TASKS:
            if labels is not None:
                assert decoder_input_ids is not None
                loss[task] = self.get_nll(
                    outputs.logits[task],
                    outputs.doc_scores,
                    labels[task],
                    reduce_loss=reduce_loss,
                    epsilon=self.config.label_smoothing,
                    n_docs=n_docs,
                )

            if do_marginalize:
                logits[task] = self.marginalize(logits[task], outputs.doc_scores, n_docs)

        return RetrievAugLMMarginOutput(
            loss=loss, # Dict
            logits=logits, # Dict
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values, # Dict
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )

    @staticmethod
    def mean_pool(vector: torch.LongTensor):
        return vector.sum(axis=0) / vector.shape[0]

    @staticmethod
    def get_attn_mask(tokens_tensor: torch.LongTensor) -> torch.tensor:
        return tokens_tensor != 0

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        output_embeddings = self.get_output_embeddings()
        for i in range(len(TASKS)):
            if output_embeddings is not None and self.config.tie_word_embeddings:
                self._tie_or_clone_weights(output_embeddings[i], self.get_input_embeddings())

        if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        domain=None,
        max_length=None,
        min_length=None,
        early_stopping=None,
        use_cache=None,
        num_beams=None,
        num_beam_groups=None,
        diversity_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        repetition_penalty=None,
        bad_words_ids=None,
        num_return_sequences=None,
        decoder_start_token_id=None,
        n_docs=None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        **model_kwargs,
    ):
        # set default parameters
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        max_length = max_length if max_length is not None else self.config.max_length
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.generator.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.generator.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.generator.pad_token_id
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.generator.decoder_start_token_id
        )
        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.config.remove_invalid_values
        )

        # retrieve docs
        dialog_lengths = None
        if self.retriever is not None and context_input_ids is None:
            if self.config.scoring_func != "original":
                dpr_out = self.question_encoder(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
                )
                combined_out = dpr_out.pooler_output

                ## Get mask for current turn input ids
                curr_turn_mask = torch.logical_xor(attention_mask, token_type_ids)
                current_turn_input_ids = input_ids * curr_turn_mask
                current_turn_only_out = self.question_encoder(
                    current_turn_input_ids, attention_mask=curr_turn_mask.long(), return_dict=True
                )
                current_turn_output = current_turn_only_out.pooler_output

                ## Split the dpr sequence output
                sequence_output = dpr_out.hidden_states[-1]
                attn_mask = self.get_attn_mask(input_ids)
                ## Split sequence output, and pool each sequence
                seq_out_0 = []  # last turn, if query; doc structure if passage
                seq_out_1 = []  # dial history, if query; passage text if passage
                dialog_lengths = []
                for i in range(sequence_output.shape[0]):
                    seq_out_masked = sequence_output[i, attn_mask[i], :]
                    segment_masked = token_type_ids[i, attn_mask[i]]
                    seq_out_masked_0 = seq_out_masked[segment_masked == 0, :]
                    seq_out_masked_1 = seq_out_masked[segment_masked == 1, :]
                    dialog_lengths.append((len(seq_out_masked_0), len(seq_out_masked_1)))
                    ### perform pooling
                    seq_out_0.append(self.mean_pool(seq_out_masked_0))
                    seq_out_1.append(self.mean_pool(seq_out_masked_1))

                pooled_output_0 = torch.cat([seq.view(1, -1) for seq in seq_out_0], dim=0)
                pooled_output_1 = torch.cat([seq.view(1, -1) for seq in seq_out_1], dim=0)

                if self.config.scoring_func in ["reranking_original", "current_original"]:
                    current_out = current_turn_output
                else:
                    current_out = pooled_output_0

                out = self.retriever(
                    input_ids,
                    combined_out.cpu().detach().to(torch.float32).numpy(),
                    current_out.cpu().detach().to(torch.float32).numpy(),
                    pooled_output_1.cpu().detach().to(torch.float32).numpy(),
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    dialog_lengths=dialog_lengths,
                    domain=domain,
                    return_tensors="pt",
                )
            else:
                combined_out = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
                out = self.retriever(
                    input_ids,
                    combined_out.cpu().detach().to(torch.float32).numpy(),
                    combined_out.cpu().detach().to(torch.float32).numpy(),  ## sending dummy
                    combined_out.cpu().detach().to(torch.float32).numpy(),  ## sending dummy
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    dialog_lengths=dialog_lengths,
                    domain=domain,
                    return_tensors="pt",
                    bm25=self.bm25,
                )

            context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_scores = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
                out["doc_scores"],
            )

            # set to correct device
            retrieved_doc_embeds = retrieved_doc_embeds.to(combined_out)
            context_input_ids = context_input_ids.to(input_ids)
            context_attention_mask = context_attention_mask.to(input_ids)
            doc_scores = retrieved_doc_scores.to(combined_out)

            # compute doc_scores
            if self.config.scoring_func in [
                "reranking",
                "reranking2",
                "original",
                "reranking_original",
                "current_original",
                "current_pooled",
            ]:
                doc_scores = torch.bmm(combined_out.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
            elif self.config.scoring_func in ["linear", "linear2", "linear3", "nonlinear"]:
                doc_scores_curr = torch.bmm(
                    pooled_output_0.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                ).squeeze(1)

                doc_scores_hist = torch.bmm(
                    pooled_output_1.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                ).squeeze(1)

                if self.config.scoring_func == "linear":
                    doc_scores = doc_scores_curr + doc_scores_hist
                elif self.config.scoring_func == "linear2":
                    doc_scores = doc_scores_curr + 0.5 * doc_scores_hist
                elif self.config.scoring_func == "linear3":
                    # TODO
                    doc_scores = doc_scores_curr + 0.5 * doc_scores_hist
                else:  # nonlinear
                    bsz = doc_scores_curr.shape[0]
                    doc_scores_curr_flattened = doc_scores_curr.flatten().unsqueeze(
                        1
                    )  # from (B, n_docs) to (Bxn_docs, 1)
                    doc_scores_hist_flattened = doc_scores_hist.flatten().unsqueeze(
                        1
                    )  # from (B, n_docs) to (Bxn_docs, 1)
                    scorer_inp = torch.cat(
                        [doc_scores_curr_flattened, doc_scores_hist_flattened], dim=1
                    )  # (Bxn_docs, 2)
                    scores = self.retriever.nn_scorer(scorer_inp)
                    doc_scores = scores.reshape((bsz, -1))

        assert (
            context_input_ids.shape[0] % n_docs
        ) == 0, f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}."

        # batch_size
        batch_size = context_input_ids.shape[0] // n_docs

        encoder = self.rag.generator.get_encoder()
        encoder_outputs = encoder(input_ids=context_input_ids, attention_mask=context_attention_mask, return_dict=True)
 
        input_ids = torch.full(
            (batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        last_hidden_state = encoder_outputs["last_hidden_state"]

        def extend_enc_output(tensor, num_beams=None):
            # split into `batch_size`, `num_beams`, `num_docs`
            tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
            # repeat same last hidden states over `num_beams` dimension
            tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
            # merge `batch_size`, `num_beams`, `num_docs` dims again
            return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])

        # correctly extend last_hidden_state and attention mask
        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=num_beams)
        encoder_outputs["last_hidden_state"] = extend_enc_output(last_hidden_state, num_beams=num_beams)

        doc_scores = doc_scores.repeat_interleave(num_beams, dim=0)

        # define start_len & additional parameters
        model_kwargs["doc_scores"] = doc_scores
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = context_attention_mask
        model_kwargs["n_docs"] = n_docs

        pre_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=context_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
        )

        if num_beams == 1:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )
            return self.greedy_search(
                input_ids,
                logits_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        elif num_beams > 1:
            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
            seq_ids = {}
            for task in TASKS:
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    num_beams=num_beams,
                    device=self.device,
                    length_penalty=length_penalty,
                    do_early_stopping=early_stopping,
                    num_beam_hyps_to_keep=num_return_sequences,
                ) 
                seq_ids[task] = self.beam_search(
                    task,
                    input_ids,
                    beam_scorer,
                    logits_processor=pre_processor,
                    max_length=max_length,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    **model_kwargs,
                )
            return seq_ids
        else:
            raise ValueError(f"`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {num_beams}")

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        doc_scores=None,
        n_docs=None,
        **kwargs
    ):
        if past is not None:
            # if past is defined use only last decoder_input_ids
            decoder_input_ids = {task: decoder_input_ids[task][:, -1:] for task in TASKS}  

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "doc_scores": doc_scores,
            "context_attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past,
            "use_cache": use_cache,
            "do_marginalize": True,
            "n_docs": n_docs,
        }
    
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = {task: input_ids.new(input_ids.shape[0]).fill_(1) for task in TASKS}
        cur_len = input_ids.shape[-1]
        seq_ids = {task: input_ids for task in TASKS}
        stop = [0] * len(TASKS)

        # this_peer_finished = False  # used by synced_gpus only
        while True:

            # if synced_gpus:
            #     # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            #     # The following logic allows an early break if all peers finished generating their sequence
            #     this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            #     # send 0.0 if we finished, 1.0 otherwise
            #     dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            #     # did all peers finish? the reduced sum will be 0.0 then
            #     if this_peer_finished_flag.item() == 0.0:
            #         break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(copy.deepcopy(seq_ids), **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            for i, task in enumerate(TASKS):
                next_token_logits = outputs.logits[task][:, -1, :]

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate: # Should be false
                    if output_scores:
                        scores += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # pre-process distribution
                next_tokens_scores = logits_processor(seq_ids[task], next_token_logits)

                # argmax
                next_tokens = torch.argmax(next_tokens_scores, dim=-1)

                # finished sentences should have their next token be a padding token
                if eos_token_id is not None:
                    if pad_token_id is None:
                        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                    next_tokens = next_tokens * unfinished_sequences[task] + pad_token_id * (1 - unfinished_sequences[task])

                if not stop[i]:
                    # update generated ids, model inputs, and length for next step 
                    seq_ids[task] = torch.cat([seq_ids[task], next_tokens[:, None]], dim=-1)

                    # if eos_token was found in one sentence, set sentence to finished
                    if eos_token_id is not None:
                        unfinished_sequences[task] = unfinished_sequences[task].mul((next_tokens != eos_token_id).long())

                    # stop when each sentence is finished, or if we exceed the maximum length
                    if unfinished_sequences[task].max() == 0 or stopping_criteria(seq_ids[task], scores):
                        stop[i] = 1

            if sum(stop) == len(TASKS):
                break
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=seq_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=seq_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return seq_ids

    def beam_search(
        self,
        task: str,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            seq_ids = {t: input_ids for t in TASKS}
            model_inputs = self.prepare_inputs_for_generation(seq_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[task][:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"][task] = self._reorder_cache(model_kwargs["past"][task], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]
