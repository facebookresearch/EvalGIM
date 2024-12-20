![alt text](https://github.com/facebookresearch/EvalGIM/blob/main/visual.png?raw=true)

# 🦾 EvalGIM: A Library for Evaluating Generative Image Models

EvalGIM (pronounced "EvalGym") makes it easier to evaluate text-to-image generative models. In addition, EvalGIM is customizable, allowing the user to define their own metrics, datasets, and visualizations for evaluation. Below, we walk through the Quick Start for the repo, how to replicate Evaluation Exercises presented in our accompanying [paper](https://arxiv.org/abs/2412.10604), perform more advanced evaluations, and customize the library for your own use cases.

Note: This code is made available under a CC-by-NC license, however you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models or licenses governing third-party datasets

## Quick start
To get started with EvalGIM, we include a simple tutorial for evaluating trade-offs in model performance across classifier guidance scales.
First install the library then proceed through the three steps of generating images, running evaluations, and visualizing results. 
All commands can be run from the root directory.
As an even easier onboarding experience, you can skip to third step and generate plots using pre-existing CSVs of metrics data. 

To replicate results in the Evaluation Exercises [paper](https://arxiv.org/abs/2412.10604), skip ahead to the next Section.
To explore additional settings for generation, evaluation, and visualization, proceed to Section after.

### 0. Install library and related packages

Run the following from the root directory: 
```
conda create -n evaluation_library python=3.10 -y
conda activate evaluation_library
pip install -e .
```

In order to evaluate models with existing, public datasets, users can download the datasets containing image generation prompts. In addition, the user can (optionally) download real images for calculating marginal metrics. 
These datasets paths can be added in `evaluation_library/data/paths.yaml`, pointing to local versions of the datasets that you wish to evaluate. 
For the rest of the Quick Start, you will only need the COCO 2014 Validation Dataset. 

Note: Quick Start only sets out an example of how to use this library, the Quick Start guide and use of the COCO 2014 Validation Dataset is not a pre-requisite for use. See information on customization below. 

### 1. Generate images
Use the following command to generate images from a HuggingFace model with a sample of 5000 prompts from the COCO dataset (the default `--dataset_name` parameter).
This command will run a sweep across classifier guidance scales 2.0, 5.0, and 7.5. 
Pass your desired model as `model_id`.

For each guidance scale, the command will launch 128 jobs running on Volta32 GPUs by default. Depending on the generative model you use, it can take ~15min-1hr (assuming resources are allocated in a timely manner) to generate all 5000 images. 
You can reduce the number of jobs by setting the `--num_jobs` parameter or reduce the number of generated images using the `--num_samples` parameter.
```
python -m evaluation_library.generate --model_id {model_id} --num_samples 5000 --batch_size 4 --sweep
```
Generated images and logs will be saved in `./projects/generated`, with a separate folder for each guidance scale with the name `{model_id}__coco_txt_dataset__cfg{guidance_scale}`.

For generating images on a local GPU, use the `--local` flag:
```
python -m evaluation_library.generate --model_id {model_id} --num_samples 50 --batch_size 4 --local
```
> Alternative: You can skip to Step 3 if you wish to avoid generation and evaluation and simply reproduce plots. 

### 2. Run evaluation

To perform evaluations with the generated images, run the following command.
Note that this will run evaluations with the *full* COCO validation dataset when computing marginal metrics like FID and precision, recall, coverage, and diversity. See the following Section for guidance on how to customize EvalGIM for your own use cases.
```
python -m evaluation_library.evaluate --model_id {model_id} --sweep
```

You can use this to  this will perform evaluations with FID, precision/recall/density/coverage, and CLIPScore or modify to just the evaluation metrics you need.
Results are saved in `./projects/generated/evals/results.csv`.
To 
run evaluation on a local GPU, add the `--local` flag and specify the `--generated_images_path`:
```bash
python -m evaluation_library.evaluate --model_id {model_id} --local --generated_images_path [add path]
```

> Alternative: You can skip to Step 3 if you wish to avoid evaluation and simply reproduce plots.

### 3. Perform visualization

To perform visualizations with the evaluations, run the following command.
```
python -m evaluation_library.visualizations.pareto_fronts --csv_path ./projects/generated/evals/results.csv --metrics_for_three_axis precision coverage clipscore --sweep_over cfg_scale
```
(If you skipped the previous evaluation step, you can use the following command to visualize Pareto Fronts of a model throughout training time: `python -m evaluation_library.visualizations.pareto_fronts --csv_path ./projects/onboarding/results.csv --metrics_for_three_axis precision coverage vqascore`)

This will create a visualization PNG showing the trade-offs in quality, diversity, and consistency across guidance scales that is saved in the same folder. 

## Running Evaluation Exercises

To utilize Evaluation Exercises presented in the [EvalGIM paper](https://arxiv.org/abs/2412.10604), see the `projects/evaluation_exercises/` folder where there are notebooks with three-step executions for generating images, evaluating outputs, and visualizing results for each Exercise. 
For these Exercises, you can leverage the HuggingFace `diffusers` library to more easily select models for analysis.
* Tradeoffs Evaluation Exercise: `./projects/evaluation_exercises/tradeoffs/evaluation_exercise.ipynb`
* Groups Evaluation Exercise: `./projects/evaluation_exercises/groups/evaluation_exercise.ipynb`
* Ranking Robustness Evaluation Exercise: `./projects/evaluation_exercises/rankingrobustness/evaluation_exercise.ipynb`
* Prompt Types Evaluation Exercise: `./projects/evaluation_exercises/prompttypes/evaluation_exercise.ipynb`
  
Alternatively, if you wish to replicate plots in the paper with the exact results, you can perform visualizations with `./projects/evaluation_exercise/visualize.ipynb`. 
This notebook leverages evaluation results saved in `./projects/evaluation_exercise/data/`.

## Additional use-cases
The library supports additional evaluation functionality beyond those listed in Quick Start. 
We recommend that the user starts with introductory commands included in the Quick Start and in the Evaluation Exercises.
Then, the user can read through this Section for examples of more complex or customized evaluations that you can run using the library.

### 1. Generate images

The default image generation command (`python -m evaluation_library.generate --model_id {model_id}`) can be customized in several ways. 
To control the guidance scale when generating images, use the `--cfg_scale` flag. 
Alternatively, sweeps over guidance scales [2.0, 5.0, 7.5] can be initiated with the `--sweep` flag. 
You can customize the number of images by using the `--num_samples` flag for up to 50,000 images. . In order to generate fewer images for any dataset you use, use the `--num_samples` flag. 
Customize the save path for the images with the `--output_path` parameter. 
Determine the number of jobs with the `--num_jobs` and batch size with the `--batch_size` arguments. 

### 2. Run evaluation

Evaluations can be run with publicly available datasets that are supported by EvalGIM by using the `--dataset_name` parameter, should the user wish to use them. Alternatively, users can add their own datasets, as discussed in the next Section. 
The following is the current set of supported datasets:
* COCO: `coco_txt_dataset`
* CC12M: `cc12m_validation_dataset`
* GeoDE: `geode_dataset`
* ImageNet: `imagenet_validation_dataset`
* TIFA-160 (prompt-only): `tifa160_dataset`

In addition, EvalGIM includes support for uniformly sub-sampled versions of any dataset you select to enable ``apples-to-apples`` comparisons. In order to sub-sample, we select only a subset of images from the user-downloaded data source rather than selecting all of the data. No new dataset is created.
For more guidance on how to use these datasets, you can reference the Prompt Types Evaluation Exercise at `./projects/evaluation_exercises/prompttypes/evaluation_exercise.ipynb`. 
* COCO: `evaluation_library.data.real_datasets_balanced.COCO15K`
* CC12M: `evaluation_library.data.real_datasets_balanced.CC12MValidation15K`
* GeoDE: `evaluation_library.data.real_datasets_balanced.GeoDE15K`
* ImageNet: `'evaluation_library.data.real_datasets_balanced.ImageNetValidation15K`


Evaluations can be customized to include alternative metrics, such as VQAScore, or disaggregated group measurements, such as by geographic region. 

To customize which metrics are run, use the `--marginal_metrics` and `--conditional_metrics` arguments. 
Metrics should be passed in as a comma-separated string. 
For example, the following can be used to evaluate with only FID for marginal metrics:
```
python evaluate.py --model_id model_a --dataset_name geode_dataset --cfg_scale 7.5 --output_path ./projects/testing --marginal_metrics fid_torchmetrics
```

To be able to evaluate images generated from prompts without an accompanying dataset of real images, use `none` in the `--marginal_metrics` field, as in the following example: 
```
python -m evaluation_library.evaluate --model_id {model_name} --dataset tifa160_dataset --marginal_metrics none --conditional_metrics dsg
```

In addition, users can also run disaggregated evaluations across subgroups.
This requires the use of an evaluation dataset that contains group information, such as GeoDE. 
To initiate group measurements, select and download an evaluation dataset that contains group information, then use the `--groups` flag, with group names passed in as a comma-separated string.
These groups should correspond to the `group` attribute in the `RealImageDataset` and `RealAttributeDataset`.
For example, the following can be used for grouped evaluations using the GeoDE dataset.
```
python evaluate.py --model_id model_a --dataset_name geode_dataset --cfg_scale 7.5 --output_path ./projects/testing --groups Africa,Americas,EastAsia,Europe,SouthEastAsia,WestAsia
```
> Note: Currently VQAScore and DSG metrics do not support metrics disaggregated by groups.

The following is the current set of supported metrics:

**Marginal Metrics**:
* FID (torchmetrics implementation; [source](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/fid.py#L182-L465), [license](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/fid.py#L1-L13))  : `fid_torchmetrics`
* FID (torcheval implementation; [source](https://pytorch.org/torcheval/main/_modules/torcheval/metrics/image/fid.html#FrechetInceptionDistance.__init__), [license](https://github.com/pytorch/torcheval/blob/main/LICENSE)): `fid_torcheval`
* Precision/recall/diversity/coverage ([source](https://github.com/clovaai/generative-evaluation-prdc), [license](https://github.com/clovaai/generative-evaluation-prdc/blob/master/LICENSE.md)): `prdc`

**Conditional Metics**:
* CLIPScore (torchmetrics implementation, [source](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/multimodal/clip_score.py#L43), [license](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/multimodal/clip_score.py#L1-L13)): `clipscore`
* Davidsonian Scene Graph ([source](https://github.com/j-min/DSG), [license](https://github.com/j-min/DSG/blob/main/LICENSE)): `dsg`
* VQAScore ([source](https://github.com/linzhiqiu/t2v_metrics), [license](https://github.com/linzhiqiu/t2v_metrics/blob/main/LICENSE)): `vqascore`
> Note: DSG requires a set of pre-computed question-answer graphs. The library currently supports the TIFA 160 dataset for DSG measurements.

#### Performing evaluations with VQAScore
Because VQAScore requires specific resources, an additional flag is required to run it.
In addition, VQAScore does not currently support grouped measurements. 

### 3. Perform visualization

Visualizations can be run with scripts contained in `evaluation_library/visualizations/`. 
For an example of how to use run each visualization, the related Evaluation Exercise can be referenced:
* Pareto Fronts: Example in the Tradeoffs Evaluation Exercise (`./projects/evaluation_exercises/tradeoffs/evaluation_exercise.ipynb`)
* Radar plots: Example in the Groups Evaluation Exercise (`./projects/evaluation_exercises/groups/evaluation_exercise.ipynb`)
* Ranking table: Example in the Ranking Robustness Evaluation Exercise (`./projects/evaluation_exercises/rankingrobustness/evaluation_exercise.ipynb`)
* Scatter plots across datasets: Example in the Prompt Types Evaluation Exercise (`./projects/evaluation_exercises/prompttypes/evaluation_exercise.ipynb`)

## Making additions to EvalGIM

We welcome contributions and customization of EvalGIM.

### Add your own models
While any model in the `diffusers` library can be run with EvalGIM out-of-the-box, the library can also be integrated directly into existing model training pipelines, allowing for more thorough monitoring of model performance over training time. 
This can be done by creating random latents to sample a text-to-image pipeline by adapting the `evaluation_library/generate.sample()` function. 
In addition, the user can implement custom seeding methods, for example, if they wish to define a given seed for each unique prompt.

### Add your own datasets
EvalGIM also supports the addition of custom datasets. 
For additional real image datasets used in the computation of marginal metrics, developers can leverage the `RealImageDataset()` class, which requires a set of real images and optionally supports class- or group-level metadata. 

To incorporate new image prompts, developers may use the `RealAttributeDataset()` class, which requires prompt strings and optionally supports generation conditions, class- or group-level labels, and metadata corresponding to metric calculation (such as question-answer graphs for DSG calculations).

```
    
class RealImageDataset(ABC):
    """Dataset of real images, used for computing marginal metrics.
    """

    @abstractmethod
    def __getitem__(self, idx) -> RealImageDatapoint:
        """Returns RealImageDatapoint containing
            image: Tensor
            class_label: Optional[str]
            group: Optional[List[str]]
        """

class RealAttributeDataset(ABC):
    """Dataset of prompts and metadata, used for generating images and computing metrics.
    """

    @abstractmethod
    def __getitem__(self, idx) -> RealAttributeDatapoint:
        """Returns RealAttributeDatapoint containing
            prompt: str
            condition: Condition
            class_label: Optional[str]
            group: Optional[List[str]]
            questions: Optional[List[str]]
            answers: Optional[List[str]]
            dependencies: Optional[Condition]
        """
\end{lstlisting}
```

To call these custom datasets, provide the full path, e.g. `data.datasets.[dataset file name].[dataset name]` and name your datasets with `[dataset name]RealImageDataset` and `[dataset name]RealAttributeDataset`.
For an example, see the implementation of the balanced datasets in `evaluation_library/data/real_datasets_balanced.py` and their use in the Prompt Types Evaluation Exercise (`./projects/evaluation_exercises/prompttypes/evaluation_exercise.ipynb`).

### Add your own metrics

EvaLGIM builds on `torchmetrics` as the framework for metrics. 
Following this implementation, intermediate metric values are updated with each batch of real or generated images then computed holistically with the `texttt{compute()` function once all batches have been added. 
`update_real_images()` and `update_generated_images()`  which take as inputs batches of real and generated images, respectively, and their associated metadata. 
Reference-free metrics use only an `update()` function, which is equivalent to `update_generated_images()`. 

```   
class MarginalMetric(Metric):
    def update_real_images(
        self,
        reference_images: torch.Tensor,
        real_image_datapoint_batch: dict,
    ) -> None:
    
    def update_generated_images(
        self,
        generated_images: torch.Tensor, 
        real_attribute_datapoint_batch: dict
    ) -> None:

    def compute(self) -> dict:
        return {"metric_name": super().compute()}

class ReferenceFreeMetric(Metric):
    def update(
    		self, 
        generated_images_batch: torch.Tensor, 
        real_attribute_datapoint_batch: dict
    ) -> None:

    def compute(self) -> dict:
        return {"metric_name": super().compute()}
```
> Note that `torchmetrics` have special data `states` for supporting distributed evaluations. It is recommended to review the `torchmetrics` documentation for implementing a metric [here](https://lightning.ai/docs/torchmetrics/stable/pages/implement.html) for more guidance. 

Additionally, EvalGIM supports the addition of metrics that can be disaggregated by subgroups.
For guidance on how to adapt your new metric to support subgroup measurements, the PRDC implementation can be viewed as an example for marginal metrics (`evaluation_library/metrics/PRDC.py`) and the CLIPScore implemetation can be used as an example for conditional metrics (`evaluation_library/metrics/customCLIPScore.py`).

### Add your own visualizations

Visualization scripts are stored in the `evaluation_library/visualizations/` directory and take as inputs a CSV with evaluation results for each model-dataset-hyperparameter combination and possible visualization parameters, such as metrics to display. 
New visualizations can be added following this scheme.

### Other

Instead of directly pushing changes to the `main` branch, please:
- Create a new branch from `origin/main` and make the necessary code changes.
- Submit a PR from your branch to `main`, including a concise description and a test plan for your changes.
- We encourage you to add tests for your changes. Units tests can be included in the `tests/` folder that will be run for every PR. This makes sure that your changes don't break any existing code and that future changes will not break your code.

**Code Formatting**
At the root of the repo, run:
```bash
# linting
ruff check .
# formatting
ruff format .
```

**Support**
If your usecase is not included here, please feel free to open an issue or reach out to Melissa Hall (melissahall@meta.com) for additional support. 

# License Info
The majority of EvalGIM is licensed under CC-BY-NC 4.0, however portions of the project are available under separate license terms: [FID](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/fid.py#L1-L13), [CLIPScore](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/multimodal/clip_score.py#L1-L13), [DSG](https://github.com/j-min/DSG/blob/main/LICENSE), and [VQAScore](https://github.com/linzhiqiu/t2v_metrics/blob/main/LICENSE) are licensed Apache 2.0; [Precision/Recall/Density/Coverage](https://github.com/clovaai/generative-evaluation-prdc/blob/master/LICENSE.md) is licensed MIT.

# Citation
**EvalGIM Library**

```
@misc{hall2024evalgimlibraryevaluatinggenerative,
      title={EvalGIM: A Library for Evaluating Generative Image Models}, 
      author={Melissa Hall and Oscar Mañas and Reyhane Askari and Mark Ibrahim and Candace Ross and Pietro Astolfi and Tariq Berrada Ifriqi and Marton Havasi and Yohann Benchetrit and Karen Ullrich and Carolina Braga and Abhishek Charnalia and Maeve Ryan and Mike Rabbat and Michal Drozdzal and Jakob Verbeek and Adriana Romero Soriano},
      year={2024},
      eprint={2412.10604},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.10604}, 
}
```

**Consistency-diversity-realism Pareto fronts of conditional image generative models**
```
@misc{astolfi2024consistencydiversityrealismparetofrontsconditional,
      title={Consistency-diversity-realism Pareto fronts of conditional image generative models}, 
      author={Pietro Astolfi and Marlene Careil and Melissa Hall and Oscar Mañas and Matthew Muckley and Jakob Verbeek and Adriana Romero Soriano and Michal Drozdzal},
      year={2024},
      eprint={2406.10429},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.10429}, 
}
```

**DIG In: Evaluating disparities in image generations with indicators for geographic diversity**
```
@misc{hall2024diginevaluatingdisparities,
      title={DIG In: Evaluating Disparities in Image Generations with Indicators for Geographic Diversity}, 
      author={Melissa Hall and Candace Ross and Adina Williams and Nicolas Carion and Michal Drozdzal and Adriana Romero Soriano},
      year={2024},
      eprint={2308.06198},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2308.06198}, 
}
```
