"""Finetuning script for RAG models. Adapted from examples.seq2seq.finetune.py"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.accelerators.ddp_accelerator import DDPAccelerator
from pytorch_lightning.cluster_environments import TorchElasticEnvironment
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
    BatchEncoding,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    RagTokenizer,
    T5ForConditionalGeneration,
)
from transformers import logging as transformers_logging
from transformers.integrations import is_ray_available


if is_ray_available():
    import ray
    from distributed_ray_retriever import RagRayDistributedRetriever, RayRetriever


from callbacks_rag import (  # noqa: E402 # isort:skipq
    get_checkpoint_callback,
    get_early_stopping_callback,
    Seq2SeqLoggingCallback,
)

from dialdoc.models.rag.distributed_pytorch_retriever import RagPyTorchDistributedRetriever  # noqa: E402 # isort:skip
from dialdoc.models.rag.modeling_rag_dialdoc import RagTokenForGeneration, DialDocRagTokenForGeneration
from dialdoc.models.rag.modeling_rag_dialdoc_mtl import DialDocRagTokenForGenerationMTL
from dialdoc.models.rag.configuration_rag_dialdoc import DialDocRagConfig


from utils_rag import (
    calculate_exact_match,
    calculate_bleu,
    flatten_list,
    is_rag_model,
    lmap,
    pickle_save,
    save_json,
    set_extra_model_params,
    load_bm25,
)

from utils_rag_mtl import Seq2SeqDataset

# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import BaseTransformer, add_generic_args, generic_train  # noqa


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_info()

TASKS = ['grounding', 'generation']

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# In PTL >v1.0, `init_ddp_connection` method in the `LightningModule`
# is no longer used, and is moved into DDPAccelerator instead.
# We override DDPAccelerator to add our custom logic for initializing the
# retriever.
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/backends/test_accelerator_connector.py


class CustomAccel(DDPAccelerator):
    def __init__(self, trainer=None, **kwargs):
        # Trainer is set later.
        super().__init__(trainer, **kwargs)

    def init_ddp_connection(self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True):
        logger.info("Custom init_ddp_connection.")
        module = self.trainer.model
        if self.cluster_environment is None:
            self.cluster_environment = TorchElasticEnvironment()
        self.distributed_port = module.hparams.distributed_port
        os.environ["MASTER_PORT"] = str(self.distributed_port)
        super().init_ddp_connection(global_rank, world_size, is_slurm_managing_tasks)
        if module.is_rag_model:
            if module.distributed_retriever == "pytorch":
                module.model.rag.retriever.init_retrieval(self.distributed_port)
            elif module.distributed_retriever == "ray" and global_rank == 0:
                # For the Ray retriever, only initialize it once when global
                # rank is 0.
                module.model.rag.retriever.init_retrieval()


class GenerativeQAModule(BaseTransformer):
    mode = "generative_qa"
    loss_names = ["grounding_loss", "generation_loss"]
    metric_names = ["em", "bleu"]
    val_metric = "em"

    def __init__(self, hparams, **kwargs):
        # when loading from a pytorch lightning checkpoint, hparams are passed as dict
        if isinstance(hparams, dict):
            hparams = AttrDict(hparams)
        if hparams.model_type == "rag_sequence":
            self.model_class = RagSequenceForGeneration
        elif hparams.model_type == "rag_token":
            self.model_class = RagTokenForGeneration
        elif hparams.model_type == "rag_token_dialdoc":
            self.model_class = DialDocRagTokenForGeneration
        elif hparams.model_type == "rag_token_dialdoc_mtl":
            self.model_class = DialDocRagTokenForGenerationMTL 
        elif hparams.model_type == "bart":
            self.model_class = BartForConditionalGeneration
        else:
            self.model_class = T5ForConditionalGeneration
        self.is_rag_model = is_rag_model(hparams.model_type)

        config_class = DialDocRagConfig if self.is_rag_model else AutoConfig
        config = config_class.from_pretrained(hparams.model_name_or_path)

        # set retriever parameters
        # logger.info("Dataset name - {}".format(config.dataset))
        config.n_docs = hparams.n_docs
        config.do_marginalize = hparams.do_marginalize or config.do_marginalize
        config.scoring_func = hparams.scoring_func or config.scoring_func
        logger.info("Using scoring function - {}".format(config.scoring_func))
        config.segmentation = hparams.segmentation or config.segmentation
        config.max_combined_length = hparams.max_combined_length or config.max_combined_length
        config.max_source_length = hparams.max_source_length or config.max_source_length
        config.index_name = hparams.index_name or config.index_name
        config.passages_path = hparams.passages_path or config.passages_path
        config.index_path = hparams.index_path or config.index_path
        config.mapping_file = hparams.mapping_file or config.mapping_file
        config.use_dummy_dataset = hparams.use_dummy_dataset

        if hparams.bm25:
            bm25 = load_bm25(hparams.bm25)
            config.bm25 = hparams.bm25
        else:
            bm25 = None
            config.bm25 = None

        # set extra_model_params for generator configs and load_model
        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "attention_dropout", "dropout")
        if self.is_rag_model:
            if hparams.prefix is not None:
                config.generator.prefix = hparams.prefix
            config.label_smoothing = hparams.label_smoothing
            hparams, config.generator = set_extra_model_params(extra_model_params, hparams, config.generator)
            if hparams.distributed_retriever == "pytorch":
                retriever = RagPyTorchDistributedRetriever.from_pretrained(hparams.model_name_or_path, config=config)
            elif hparams.distributed_retriever == "ray":
                # The Ray retriever needs the handles to the retriever actors.
                retriever = RagRayDistributedRetriever.from_pretrained(
                    hparams.model_name_or_path, hparams.actor_handles, config=config
                )
            # import pdb; pdb.set_trace()
            model = self.model_class.from_pretrained(
                hparams.model_name_or_path, config=config, retriever=retriever, bm25=bm25
            )
            prefix = config.question_encoder.prefix
            model.bm25 = bm25
        else:
            if hparams.prefix is not None:
                config.prefix = hparams.prefix
            hparams, config = set_extra_model_params(extra_model_params, hparams, config)
            model = self.model_class.from_pretrained(hparams.model_name_or_path, config=config)
            prefix = config.prefix

        tokenizer = (
            RagTokenizer.from_pretrained(hparams.model_name_or_path)
            if self.is_rag_model
            else AutoTokenizer.from_pretrained(hparams.model_name_or_path)
        )

        super().__init__(hparams, config=config, tokenizer=tokenizer, model=model)

        # logger.info("Printing config")
        # logger.info(config)

        # save_git_info(self.hparams.output_dir)
        self.output_dir = Path(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        data_dir = self.hparams.data_dir
        segmentation = self.hparams.segmentation
        data_dict = {task: f'{data_dir}/dd-{task}-{segmentation}' for task in TASKS}
        self.dataset_kwargs: dict = dict(
            data_dir=data_dict,
            max_source_length=self.hparams.max_source_length,
            prefix=prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        # self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.distributed_port = self.hparams.distributed_port

        # For single GPU training, init_ddp_connection is not called.
        # So we need to initialize the retrievers here.
        if hparams.gpus <= 1:
            if hparams.distributed_retriever == "ray":
                self.model.retriever.init_retrieval()
            elif hparams.distributed_retriever == "pytorch":
                self.model.retriever.init_retrieval(self.distributed_port)

        self.distributed_retriever = hparams.distributed_retriever

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        source_ids, source_mask, token_type_ids, domain = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            batch["domain"]
        )

        rag_kwargs = {}
        decoder_input_ids, lm_labels = {}, {}
        for task in TASKS:
            target_ids = batch[f'{task}_decoder_input_ids']
            if isinstance(self.model, T5ForConditionalGeneration):
                decoder_input_ids = self.model._shift_right(target_ids)
                lm_labels = target_ids
            elif isinstance(self.model, BartForConditionalGeneration):
                decoder_input_ids = target_ids[:, :-1].contiguous()
                lm_labels = target_ids[:, 1:].clone()
            else:
                assert self.is_rag_model
                # generator = self.model.rag.generator
                # if isinstance(generator, T5ForConditionalGeneration):
                #     decoder_start_token_id = generator.config.decoder_start_token_id
                #     decoder_input_ids = (
                #         torch.cat(
                #             [torch.Tensor([[decoder_start_token_id]] * target_ids.shape[0]).to(target_ids), target_ids],
                #             dim=1,
                #         )
                #         if target_ids.shape[0] < self.target_lens["train"]
                #         else generator._shift_right(target_ids)
                #     )
                # elif isinstance(generator, BartForConditionalGeneration):
                decoder_input_ids[task] = target_ids
                lm_labels[task] = decoder_input_ids[task]
                rag_kwargs["reduce_loss"] = True

        assert decoder_input_ids is not None

        outputs = self(
            source_ids,
            attention_mask=source_mask,
            token_type_ids=token_type_ids,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            labels=lm_labels,
            domain=domain,
            **rag_kwargs,
        )
        
        grounding_loss = outputs['loss']['grounding']
        generation_loss = outputs['loss']['generation']
        return (grounding_loss, generation_loss)

    @property
    def pad(self) -> int:
        raise NotImplementedError("pad not implemented")

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        self.log_dict(logs, prog_bar=True)
        return loss_tensors[0] + loss_tensors[1]

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        for task in TASKS:
            file_path = self.output_dir / f'{task}_step_{self.global_step}_preds.txt'
            with file_path.open(mode='w') as f:
                for x in outputs:
                    for line in x[task]:
                        f.write(f'{line}\n')

    def calc_generative_metrics(self, preds, target) -> Dict:
        d_metrics = calculate_exact_match(preds, target)
        d_metrics.update(calculate_bleu(preds, target))
        return d_metrics

    def _generative_step(self, batch: dict) -> dict:
        # start_time = time.time()
        domain = None
        if "domain" in batch:
            domain = batch["domain"]
            del batch["domain"]
        batch = BatchEncoding(batch).to(device=self.model.device)
        batch["domain"] = domain
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            domain=domain,
            do_deduplication=False,  # rag specific parameter
            use_cache=True,
            min_length=1,
            max_length=self.target_lens["val"],
        )

        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        self.log_dict(base_metrics)

        pred_dict = {}
        for task in TASKS:
            # gen_time = (time.time() - start_time) / batch["input_ids"].shape[0]
            preds: List[str] = self.ids_to_clean_text(generated_ids[task])
            target: List[str] = self.ids_to_clean_text(batch[f"{task}_decoder_input_ids"])
            pred_dict[task] = preds

            gen_metrics: Dict = self.calc_generative_metrics(preds, target)
            summ_len = np.mean(lmap(len, generated_ids))
            gen_metrics['summ_len'] = summ_len
            self.log_dict({f'{task}_{key}': value for key, value in gen_metrics.items()})

        return pred_dict

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    # def test_epoch_end(self, outputs):
    #     return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = Seq2SeqDataset(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("checkpoint{}".format(self.global_step))
        self.model.config.save_step = self.global_step
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--scoring_func",
            default="original",
            type=str,
            help="different scoring function, `original`, `linear`, `nonlinear`, `reranking`",
        )
        parser.add_argument(
            "--segmentation",
            default="token",
            type=str,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--bm25",
            default=None,
            type=str,
            help="BM25 result folder",
        )
        parser.add_argument(
            "--max_combined_length",
            default=512,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_source_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=25,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument(
            "--prefix",
            type=str,
            default=None,
            help="Prefix added at the beginning of each text, typically used with T5-based models.",
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument(
            "--distributed-port", type=int, default=-1, required=False, help="Port number for distributed training."
        )
        parser.add_argument(
            "--model_type",
            choices=["rag_sequence", "rag_token", "rag_token_dialdoc", "rag_token_dialdoc_mtl", "bart", "t5"],
            type=str,
            help="RAG model type: sequence or token, if none specified, the type is inferred from the model_name_or_path",
        )
        return parser

    @staticmethod
    def add_retriever_specific_args(parser):
        parser.add_argument(
            "--n_docs",
            type=int,
            default=5,
            help="Number of documents to retrieve.",
        )
        parser.add_argument(
            "--index_name",
            type=str,
            default=None,
            help="Name of the index to use: 'hf' for a canonical dataset from the datasets library (default), 'custom' for a local index, or 'legacy' for the orignal one)",
        )
        parser.add_argument(
            "--passages_path",
            type=str,
            default=None,
            help="Path to the dataset of passages for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )
        parser.add_argument(
            "--index_path",
            type=str,
            default=None,
            help="Path to the faiss index for custom index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )
        parser.add_argument(
            "--mapping_file", type=str, default=None, help="Path to domain information for each sample"
        )
        parser.add_argument(
            "--distributed_retriever",
            choices=["ray", "pytorch"],
            type=str,
            default="pytorch",
            help="What implementation to use for distributed retriever? If "
            "pytorch is selected, the index is loaded on training "
            "worker 0, and torch.distributed is used to handle "
            "communication between training worker 0, and the other "
            "training workers. If ray is selected, the Ray library is "
            "used to create load the index on separate processes, "
            "and Ray handles the communication between the training "
            "workers and the retrieval actors.",
        )
        parser.add_argument(
            "--use_dummy_dataset",
            type=bool,
            default=False,
            help="Whether to use the dummy version of the dataset index. More info about custom indexes in the RagRetriever documentation as well as in `examples/rag/use_own_knowledge_dataset.py`",
        )
        parser.add_argument(
            "--do_marginalize",
            type=bool,
            default=False,
            help="",
        )
        return parser

    @staticmethod
    def add_ray_specific_args(parser):
        # Ray cluster address.
        parser.add_argument(
            "--ray-address",
            default="auto",
            type=str,
            help="The address of the Ray cluster to connect to. If not "
            "specified, Ray will attempt to automatically detect the "
            "cluster. Has no effect if pytorch is used as the distributed "
            "retriever.",
        )
        parser.add_argument(
            "--num_retrieval_workers",
            type=int,
            default=1,
            help="The number of retrieval actors to use when Ray is selected"
            "for the distributed retriever. Has no effect when "
            "distributed_retriever is set to pytorch.",
        )
        return parser


def main(args=None, model=None) -> GenerativeQAModule:
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GenerativeQAModule.add_model_specific_args(parser, os.getcwd())
    parser = GenerativeQAModule.add_retriever_specific_args(parser)

    args = args or parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    named_actors = []
    if args.distributed_retriever == "ray" and args.gpus > 1:
        if not is_ray_available():
            raise RuntimeError("Please install Ray to use the Ray " "distributed retriever.")
        # Connect to an existing Ray cluster.
        try:
            ray.init(address=args.ray_address)
        except (ConnectionError, ValueError):
            logger.warning(
                "Connection to Ray cluster failed. Make sure a Ray"
                "cluster is running by either using Ray's cluster "
                "launcher (`ray up`) or by manually starting Ray on "
                "each node via `ray start --head` for the head node "
                "and `ray start --address='<ip address>:6379'` for "
                "additional nodes. See "
                "https://docs.ray.io/en/master/cluster/index.html "
                "for more info."
            )
            raise

        # Create Ray actors only for rank 0.
        if ("LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == 0) and (
            "NODE_RANK" not in os.environ or os.environ["NODE_RANK"] == 0
        ):
            remote_cls = ray.remote(RayRetriever)
            named_actors = [
                remote_cls.options(name="retrieval_worker_{}".format(i)).remote()
                for i in range(args.num_retrieval_workers)
            ]
        else:
            logger.info(
                "Getting named actors for NODE_RANK {}, LOCAL_RANK {}".format(
                    os.environ["NODE_RANK"], os.environ["LOCAL_RANK"]
                )
            )
            named_actors = [ray.get_actor("retrieval_worker_{}".format(i)) for i in range(args.num_retrieval_workers)]
    args.actor_handles = named_actors
    assert args.actor_handles == named_actors

    if model is None:
        model: GenerativeQAModule = GenerativeQAModule(args)

    # dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        training_logger = True  # don't pollute wandb logs unnecessarily
    # elif args.logger_name == "wandb":
    #     from pytorch_lightning.loggers import WandbLogger

    #     project = os.environ.get("WANDB_PROJECT", dataset)
    #     training_logger = WandbLogger(name=model.output_dir.name, project=project)

    # elif args.logger_name == "wandb_shared":
    #     from pytorch_lightning.loggers import WandbLogger

    #     training_logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    es_callback = (
        get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
        if args.early_stopping_patience >= 0
        else False
    )

    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=model.output_dir)
    trainer.fit(model)

    # trainer: pl.Trainer = generic_train(
    #     model,
    #     args,
    #     logging_callback=Seq2SeqLoggingCallback(),
    #     checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
    #     early_stopping_callback=es_callback,
    #     logger=training_logger,
    #     accelerator=CustomAccel() if args.gpus > 1 else None,
    #     profiler=pl.profiler.AdvancedProfiler() if args.profile else None,
    # )

    if model.hparams.bm25:
        model.hparams.bm25 = None
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    if not args.do_predict:
        return model

    # test() without a model tests using the best checkpoint automatically
    trainer.test()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GenerativeQAModule.add_model_specific_args(parser, os.getcwd())
    parser = GenerativeQAModule.add_retriever_specific_args(parser)
    parser = GenerativeQAModule.add_ray_specific_args(parser)

    # Pytorch Lightning Profiler
    parser.add_argument(
        "--profile",
        action="store_true",
        help="If True, use pytorch_lightning.profiler.AdvancedProfiler to profile the Trainer.",
    )

    args = parser.parse_args()

    main(args)