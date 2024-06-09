"""
    Speechbrain training script for fine-tuning SSL models on EmoBench task
    Based on https://github.com/speechbrain/speechbrain/blob/develop/recipes/IEMOCAP/emotion_recognition/train_with_wav2vec2.py
    Author:
        Mingjie Chen, University of Sheffield, 2024        

"""

import os
import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from scoring_metric import scoring_all, scoring_ua_wa, output_score

import torch
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore")

class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier.
        """
        batch = batch.to(self.device)
        feats, _  = batch.feat
        mask = (feats.sum(dim=2) != 0.0).float()
        outputs = self.modules.output_mlp(feats, mask)
        outputs = self.hparams.log_softmax(outputs)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        emoid, _ = batch.emo_encoded

        """to meet the input form of nll loss"""
        emoid = emoid.squeeze(1)
        loss = self.hparams.compute_cost(predictions, emoid)
        if stage != sb.Stage.TRAIN:
            #self.error_metrics.append(batch.id, predictions, emoid)
            self.error_metrics.append((batch.id, predictions.data.cpu().numpy(), emoid.data.cpu().numpy()))

        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()

        self.optimizer.zero_grad()

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            #self.error_metrics = self.hparams.error_stats()
            self.error_metrics = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            ua, wa, micro_f1, macro_f1 = scoring_ua_wa(self.error_metrics)
            stats = {
                "loss": stage_loss,
                "wa": wa,
                'ua': ua,
                'macro_f1': macro_f1
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            new_lr = self.hparams.lr_annealing.get_next_value()
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)


            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": new_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_checkpoint(
                meta=stats
            )
            
            scores, num_samples = scoring_all(self.error_metrics)
            os.makedirs(self.hparams.valid_scores_dir, exist_ok = True)
            scores_path = os.path.join(self.hparams.valid_scores_dir, f'epoch_{epoch}.txt')
            output_score(scores, scores_path, epoch, num_samples, 'IEMOCAP')

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            epoch = self.hparams.epoch_counter.current
            scores, num_samples = scoring_all(self.error_metrics)
            os.makedirs(self.hparams.test_scores_dir, exist_ok = True)
            scores_path = os.path.join(self.hparams.test_scores_dir, f'epoch_{epoch}.txt')
            output_score(scores, scores_path, epoch, num_samples, 'IEMOCAP')

    def on_evaluate_start(self, max_key=None, min_key=None):
        pass

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        #self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        avg_test_loss_list = []
        checkpoints = self.checkpointer.list_checkpoints()
        checkpoints.sort()
        
        tested_epoch = set()
        with open(self.hparams.train_log, 'r') as f:
            for line in f:
                if 'Epoch loaded' in line:
                    tested_epoch.add(int(line.split("-")[0].split()[-1]))
        for checkpoint in checkpoints:
            self.checkpointer.load_checkpoint(checkpoint)
            # skip tested epoch
            epoch = self.hparams.epoch_counter.current
            if epoch in tested_epoch:
                continue
            self.modules.eval()
            avg_test_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    test_set,
                    dynamic_ncols=True,
                    disable=not progressbar,
                    colour=self.tqdm_barcolor["test"],
                ):
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                    avg_test_loss = self.update_average(loss, avg_test_loss)

                    # Profile only if desired (steps allow the profiler to know when all is warmed up)
                    if self.profiler is not None:
                        if self.profiler.record_steps:
                            self.profiler.step()

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                self.on_stage_end(sb.Stage.TEST, avg_test_loss, None)
                avg_test_loss_list.append(avg_test_loss)
            self.step = 0
        return avg_test_loss_list
    
    def init_optimizers(self):
        "Initializes the model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none)


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """
    
    
    # Define audio pipeline
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("feat")
    def audio_pipeline(id):
        """Load the audio features, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        feat_path = os.path.join(hparams['feat_dir'], id + '.npy')
        feat = np.load(feat_path)
        feat = torch.FloatTensor(feat)
        yield feat

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(hparams['out_n_neurons'])
    # Define label pipeline:
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        emo_encoded = label_encoder.encode_label_torch(emo)
        yield emo_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["save_json_train"],
        "valid": hparams["save_json_valid"],
        "test": hparams["save_json_test"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "feat", "emo_encoded"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
    
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="emo",
    )

    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from dataset_prepare import prepare_data

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_data,
            kwargs={
                "train_annotation": hparams["train_annotation"],
                "valid_annotation": hparams["valid_annotation"],
                "test_annotation": hparams["test_annotation"],
                "save_json_train": hparams["save_json_train"],
                "save_json_valid": hparams["save_json_valid"],
                "save_json_test": hparams["save_json_test"],
                "split_ratio": hparams["split_ratio"],
                "seed": hparams['seed'],
                "label_map": hparams['label_map'],
                "feat_dir": hparams['feat_dir']
            },
        )

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        max_key="wa",
        test_loader_kwargs=hparams["dataloader_options"],
    )
