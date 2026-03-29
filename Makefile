SHELL := /bin/bash
VENV_NAME := tsr
# CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_NAME)
CONDA_ACTIVATE := true
PYTHON := $(CONDA_ACTIVATE) && python
PIP := $(CONDA_ACTIVATE) && pip3
# Stacked single-node multi-worker: https://pytorch.org/docs/stable/elastic/run.html#stacked-single-node-multi-worker 
TORCHRUN = $(CONDA_ACTIVATE) && torchrun --rdzv_backend=c10d --rdzv_endpoint localhost:0 --nnodes=1 --nproc_per_node=$(NGPU)

# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

#
# Virtual Environment Targets
#
clean:
> rm -f .venv_done

.done_venv: clean
> conda create -n $(VENV_NAME) python=3.9 -y
> $(PIP) install -r requirements.txt
> $(PIP) install -e .
> touch $@

#
# Download pretrained and UniTable model weights
#
WEIGHTS_PATH = experiments/unitable_weights
M_VQVAE_1M = $(WEIGHTS_PATH)/vqvae_1m.pt
M_VQVAE_2M = $(WEIGHTS_PATH)/vqvae_2m.pt
M_SSP_1M_BASE = $(WEIGHTS_PATH)/ssp_1m_base.pt
M_SSP_1M_LARGE = $(WEIGHTS_PATH)/ssp_1m_large.pt
M_SSP_2M_BASE = $(WEIGHTS_PATH)/ssp_2m_base.pt
M_SSP_2M_LARGE = $(WEIGHTS_PATH)/ssp_2m_large.pt
UNITABLE_HTML = $(WEIGHTS_PATH)/unitable_large_structure.pt
UNITABLE_BBOX = $(WEIGHTS_PATH)/unitable_large_bbox.pt
UNITABLE_CELL = $(WEIGHTS_PATH)/unitable_large_content.pt

.done_download_weights:
ifeq ("$(words $(wildcard $(WEIGHTS_PATH)/*.pt))", "9")
> $(info All 9 model weights have already been downloaded to $(WEIGHTS_PATH).)
else
> $(info There should be 9 weights file under $(WEIGHTS_PATH), but only $(words $(wildcard $(WEIGHTS_PATH)/*.pt)) are found.)
> $(info Begin downloading weights from HuggingFace ...)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/vqvae_1m.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/vqvae_2m.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/ssp_1m_base.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/ssp_1m_large.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/ssp_2m_base.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/ssp_2m_large.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/unitable_large_structure.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/unitable_large_bbox.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/unitable_large_content.pt -P $(WEIGHTS_PATH)
> $(info Completed!)
endif 

#
# Python Targets
#
include CONFIG.mk
SRC := src
# BEST_MODEL = "../$(word 1,$(subst -, ,$*))/model/best.pt"
# /root/autodl-tmp/TSR/experiments/ssp_2m_mini_html_base/model/best.pt
BEST_MODEL = /root/autodl-tmp/TSR/experiments/ssp_2m_mini_html_base/model/best.pt
RESULT_JSON := html.json
TEDS_STRUCTURE = -f "../experiments/$*/$(RESULT_JSON)" -s

######################
NGPU := 1  # number of gpus used in the experiments

.SECONDARY:

# vq-vae and self-supervised pretraining
experiments/%/.done_pretrain:
> @echo "Using experiment configurations from variable EXP_$*"
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train"
> touch $@

# #################Train########################
# # finetuning from SSP weights for table structure, cell bbox and cell content
# experiments/%/.done_finetune:
# > @echo "Finetuning phase 1 - using experiment configurations from variable EXP_$*"
# # > cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train" ++trainer.trainer.beit_pretrained_weights="../$(M_SSP_2M_BASE)"
# > cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train" ++trainer.trainer.beit_pretrained_weights="../$(M_SSP_2M_BASE)" ++trainer.train.dataloader.batch_size=24 ++trainer.valid.dataloader.batch_size=24
# > @echo "Finetuning phase 2 - starting from epoch 4"
# > cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train" ++trainer.trainer.snapshot="epoch3_snapshot.pt" ++trainer.trainer.beit_pretrained_weights=null
# > touch $@

experiments/%/.done_finetune:
> @echo "Finetuning phase 1 (SKIPPED) - Resuming directly from Epoch 3 snapshot..."
# 这里的 Phase 1 被注释掉了，不会执行重头训练
# > cd $(SRC) && export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train" ++trainer.trainer.beit_pretrained_weights="../$(M_SSP_2M_BASE)"
> @echo "Finetuning phase 2 - starting from epoch 4"
> cd $(SRC) && export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train" ++trainer.trainer.snapshot="epoch3_snapshot.pt" ++trainer.trainer.beit_pretrained_weights=null
> touch $@

#################Eval########################
# First:
experiments/pubtabnet/.done_%:
> @echo "Testing model $* on pubtabnet"
> @rm -f $(@D)/$*/*
> @mkdir -p $(@D)
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="test" \
  ++hydra.run.dir="../$(@D)" $(PUBTABNET) ++trainer.trainer.model_weights=$(BEST_MODEL)
# > cd $(SRC) && $(PYTHON) -m utils.engine -f ../$(@D)/$* -t bbox
> cd $(SRC) && PYTHONPATH=.. $(PYTHON) -m utils.engine -f ../$(@D)/$* -t bbox
> touch $@

# Scend:
# cd ./src/utils/
# python coco_map.py -f /you_path/final.json


# ========================================================
# 专为你的 PubTabNet 数据定制的 Large 训练任务
# ========================================================
# 1. 显存保护：设为 24 防止显存溢出
BATCH_LARGE_CUSTOM = ++trainer.train.dataloader.batch_size=12 ++trainer.valid.dataloader.batch_size=12

# 2. 核心配置：只读 PubTabNet，但用 Large 架构和权重
EXP_pub_bbox_large_final := $(TRAIN_pub_bbox) $(ARCH_LARGE) \
    $(WEIGHTS_mtim_2m_large) $(LOCK_MTIM_4) $(BATCH_LARGE_CUSTOM) $(LR_cosine216k_warm27k)
    