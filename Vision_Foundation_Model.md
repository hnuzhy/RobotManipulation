# ⭐Vision Foundation Model
*currently, we treat `SAM` as the only `Vision Foundation Model` and collect many other related works*

## Materials

* [**(github)** Awesome-Segment-Anything: the first comprehensive survey on Meta AI's Segment Anything Model (SAM).](https://github.com/liliu-avril/Awesome-Segment-Anything)
* [**(github)** Segment-anything related awesome extensions/projects/repos](https://github.com/JerryX1110/awesome-segment-anything-extensions)
* [**(github)** Tracking and collecting papers/projects/others related to Segment Anything](https://github.com/Hedlen/awesome-segment-anything)
* [**(github)** (segment-anything-2 real-time) Run Segment Anything Model 2 on a live video stream](https://github.com/Gy920/segment-anything-2-real-time)
* [**(github)** VILA: Optimized Vision Language Models, VILA is a family of SOTA VLMs for diverse multimodal AI tasks across the edge, data center, and cloud.](https://github.com/NVlabs/VILA)


## Papers

***

### ⭐1) Segment Anything Series
*for the `instance segmentation` task*

* 👍**SAM(arxiv2023.04)(ICCV2023 Best Paper)** Segment Anything [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2304.02643)][[project homepage](https://segment-anything.com/)][[publication link](https://ai.facebook.com/research/publications/segment-anything/)][[blogs](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)][[code|official](https://github.com/facebookresearch/segment-anything)][`Meta AI`, `Facebook`]

* **SSA(2023.04)** Semantic Segment Anything [[demo link](https://replicate.com/cjwbw/semantic-segment-anything)][[code|official](https://github.com/fudan-zvg/Semantic-Segment-Anything)][`Fudan`]

* **Grounded-SAM(2023.04)** Grounded Segment Anything [[code|official](https://github.com/IDEA-Research/Grounded-Segment-Anything)][`IDEA-Research`]

* 👍**Anything-3D(2023.04)** Segment-Anything + 3D. Let's lift anything to 3D [[code|official](https://github.com/Anything-of-anything/Anything-3D)][`Anything-3D-Objects`, `Anything-3DNovel-View`, `Anything-NeRF`, `Any-3DFace`][`Any-3DFace` is based on SAM and [HRN (CVPR2023)](https://younglbw.github.io/HRN-homepage/)]

* 👍**3D-Box-Segment-Anything(2023.04)** 3D-Box via Segment Anything [[code|official](https://github.com/dvlab-research/3D-Box-Segment-Anything)][`It extends Segment Anything to 3D perception by combining it with [VoxelNeXt (CVPR2023)](https://github.com/dvlab-research/VoxelNeXt)`]

* **SALT(2023.04)** Segment Anything Labelling Tool (SALT) [[code|official](https://github.com/anuragxel/salt)][`Uses the Segment-Anything Model By Meta AI and adds a barebones interface to label images and saves the masks in the COCO format`]

* **SA3D(2023.04)** Segment Anything in 3D with NeRFs [[arxiv link](https://arxiv.org/abs/2304.12308)][[project link](https://jumpat.github.io/SA3D/)][[code|official](https://github.com/Jumpat/SegmentAnythingin3D)]

* 👍**Inpaint-Anything(arxiv2023.04)** Inpaint Anything: Segment Anything Meets Image Inpainting [[arxiv link](https://arxiv.org/abs/2304.06790)][[code|official](https://github.com/geekyutao/Inpaint-Anything)][[HuggingFace link](https://huggingface.co/spaces/InpaintAI/Inpaint-Anything)]

* **SAM-Survey(arxiv2023.05)** A Comprehensive Survey on Segment Anything Model for Vision and Beyond [[arxiv link](https://arxiv.org/abs/2305.08196)]

* **PerSAM(ICLR2024)(arxiv2023.05)** Personalize Segment Anything Model with One Shot [[openreview link](https://openreview.net/forum?id=6Gzkhoc6YS)][[arxiv link](https://arxiv.org/abs/2305.03048)][[code|official](https://github.com/ZrrSkywalker/Personalize-SAM)][`CUHK MMLab + Shanghai Artificial Intelligence Laboratory + Institute of Automation, Chinese Academy of Sciences + CFCS, School of CS, Peking University + CPII of InnoHK`][`Reference Detection and Segmentation`]

* **SAM3D(ICCVW2023)(arxiv2023.06)** SAM3D: Segment Anything in 3D Scenes [[arxiv link](https://arxiv.org/abs/2306.03908)][[code|official](https://github.com/Pointcept/SegmentAnything3D)][`University of Hong Kong`]

* **HQ-SAM(NIPS2023)(arxiv2023.06)** Segment Anything in High Quality [[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f828e38160f31935cfe9f67503ad17c-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2306.01567)][[code|official](https://github.com/SysCV/SAM-HQ)][`ETH Zurich & HKUST`, it proposes HQ-SAM to upgrade SAM for high-quality zero-shot segmentation.]

* **Semantic-SAM(arxiv2023.07)** Semantic-SAM: Segment and Recognize Anything at Any Granularity [[arxiv link](https://arxiv.org/abs/2307.04767)][[code|official](https://github.com/UX-Decoder/Semantic-SAM)][`HKUST`,  It introduces a universal image segmentation model to enable segment and recognize anything at any desired granularity. The authors have `trained on the whole SA-1B dataset` and the model can reproduce SAM and beyond it.]

* **PseCo(CVPR2024)(arxiv2023.11)** Point, Segment and Count: A Generalized Framework for Object Counting [[arxiv link](https://arxiv.org/abs/2311.12386)][[code|official](https://github.com/Hzzone/PseCo)][`Fudan University`, `few-shot/zero-shot object counting/detection`]

* **SAM-6D(CVPR2024)(arxiv2023.11)** SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation [[arxiv link](https://arxiv.org/abs/2311.15707)][[code|official](https://github.com/JiehongLin/SAM-6D)][`South China University of Technology`, the first author [`Jiehong Lin (林杰鸿)`](https://jiehonglin.github.io/)]

* **SAI3D(CVPR2024)(arxiv2023.12)** SAI3D: Segment Any Instance in 3D Scenes [[arxiv link](https://arxiv.org/abs/2312.11557)][[project link](https://yd-yin.github.io/SAI3D/)][[code|official](https://github.com/yd-yin/SAI3D)][`Peking University`, the first author [`Yingda Yin 尹英达`](https://yd-yin.github.io/)]

* **GroundedSAM(arxiv2024.01)** Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks [[arxiv link](https://arxiv.org/abs/2401.14159)][[code|official 1](https://github.com/IDEA-Research/Grounded-SAM-2)][[code|official 2](https://github.com/IDEA-Research/Grounded-Segment-Anything)][`International Digital Economy Academy (IDEA)`]

* **VRP-SAM(CVPR2024)(arxiv2024.02)** VRP-SAM: SAM with Visual Reference Prompt [[paper link](http://openaccess.thecvf.com/content/CVPR2024/html/Sun_VRP-SAM_SAM_with_Visual_Reference_Prompt_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2402.17726)][[code|official](https://github.com/syp2ysy/VRP-SAM)][`Nanjing University of Science and Technology + Baidu VIS + Beihang University + Australian National University`][`Reference Detection and Segmentation`]

* **UOIS-SAM(arxiv2024.09)** Adapting Segment Anything Model for Unseen Object Instance Segmentation [[arxiv link](https://arxiv.org/abs/2409.15481)][`CUHK`][`Unseen Object Instance Segmentation (UOIS) SAM`]

* **SAMPart3D(arxiv2024.11)** SAMPart3D: Segment Any Part in 3D Objects [[arxiv link](https://arxiv.org/abs/2411.07184)][[project link](https://yhyang-myron.github.io/SAMPart3D-website/)][[code|official](https://github.com/Pointcept/SAMPart3D)][`The University of Hong Kong + VAST`]

* **VLP-SAM(arxiv2025.02)** VLP-SAM: Vision and Language reference Prompt into SAM [[arxiv link](https://arxiv.org/abs/2502.00719)][[code|official](https://github.com/kosukesakurai1/VLP-SAM)][`Waseda University`][It is still based on the large `sam_vit_h_4b8939.pth` weight, and not user-friendly for real-time inference.][`Reference Detection and Segmentation`]



***

### ⭐2) Vision Transformer Series
*for the `Theories & Backbones`, `Self-Supervsied Learning` and `Joint, Lightweight & Efficient Training ` researches*

#### 2.1) Theories and Backbones

* **(ICCV2019)** Local Relation Networks for Image Recognition [[paper link](http://openaccess.thecvf.com/content_ICCV_2019/html/Hu_Local_Relation_Networks_for_Image_Recognition_ICCV_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1904.11491)][[code|official](https://github.com/microsoft/Swin-Transformer/tree/LR-Net)][`Microsoft + THU`, `the first full-attention visual backbone`]

* 👍**ViT(ICLR2021 Oral)(arxiv2020.10)** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [[paper link](https://openreview.net/forum?id=YicbFdNTTy)][[arxiv link](https://arxiv.org/abs/2010.11929)][[code|official](https://github.com/google-research/vision_transformer )]

* **LeViT (ICCV2021)** LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.html)][[code|official](https://github.com/facebookresearch/LeViT)][`facebookresearch`]

* 👍**Swin-Transformer (Shifted Window)(ICCV2021)** Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper)][[arxiv link](https://arxiv.org/abs/2103.14030)][[code|official](https://github.com/microsoft/swin-transformer)][[Swin Transformers inference implemented in FasterTransformer by Nvidia](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/swin_guide.md)][`Microsoft`]

* **Swin-Transformer-V2 (CVPR2022)** Swin Transformer V2: Scaling Up Capacity and Resolution [[paper link](http://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2111.09883)][[code|official](https://github.com/microsoft/swin-transformer)][`Microsoft`]

* **TinyViT(ECCV2022)(arxiv2022.07)** TinyViT: Fast Pretraining Distillation for Small Vision Transformers [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19803-8_5)][[arxiv link](https://arxiv.org/abs/2207.10666)][[code|official](https://github.com/microsoft/Cream/tree/main/TinyViT)][`Microsoft Research + Microsoft Cloud+AI`]

* **EfficientViT (CVPR2023)(arxiv2023.05)** EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2305.07027)][[code|official](https://github.com/microsoft/Cream/tree/main/EfficientViT)][`CUHK + Microsoft`]

* **RoFormer / RoPE (Neurocomputing2024)(arxiv2021.04)(with very high influence)** RoFormer: Enhanced transformer with Rotary Position Embedding [[paper link](https://www.sciencedirect.com/science/article/abs/pii/S0925231223011864)][[arxiv link](https://arxiv.org/abs/2104.09864)][[code|official](https://huggingface.co/docs/transformers/model_doc/roformer)][`Zhuiyi Technology`; It is widely used in modern `transformer` designs, for example using `RoPE` to get the reformed [`RoPE-ViT`](https://github.com/naver-ai/rope-vit) by the work [`(arxiv2024.03) Rotary Position Embedding for Vision Transformer`](https://arxiv.org/abs/2403.13298)]

* 👍**StarNet(CVPR2024)(arxiv2024.03)** Rewrite the Stars [[arxiv link](https://arxiv.org/abs/2403.19967)][[weixin blog](https://mp.weixin.qq.com/s/SemsRFsrGQ0WJf_yQN6p4A)][[code|official](https://github.com/ma-xu/Rewrite-the-Stars)][`microsoft`; superior than transformer-based conunterparts `FasterViT`, `EdgeViT`, and `Mobile-Former`]

* **CoPE(arxiv2024.05)** Contextual Position Encoding: Learning to Count What's Important [[arxiv link](https://arxiv.org/abs/2405.18719)][`FAIR at Meta`; It aims to enhance the performance of `RoPE`]

* 👍**YOCO(arxiv2024.05)** You Only Cache Once: Decoder-Decoder Architectures for Language Models [[arxiv link](https://arxiv.org/abs/2405.05254)][[weixin blog](https://mp.weixin.qq.com/s/X4HSyEreN4L4xTizC-_mow)][[code|official](https://github.com/microsoft/unilm/tree/master/YOCO)][`microsoft`; partially based on [`Flash-Attention`](https://github.com/Dao-AILab/flash-attention)]


#### 2.2) Self-Supervsied Learning

* 👍**SimCLR (ICML2020)** A Simple Framework for Contrastive Learning of Visual Representations [[paper link](http://proceedings.mlr.press/v119/chen20j.html)][[paperswithcode link](https://paperswithcode.com/paper/a-simple-framework-for-contrastive-learning)][[code|official](https://github.com/google-research/simclr)][[official blog](https://blog.research.google/2020/04/advancing-self-supervised-and-semi.html)][`Geoffrey Hinton`, `Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* 👍**MoCo (CVPR2020)** Momentum Contrast for Unsupervised Visual Representation Learning [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html)][[arxiv link](http://arxiv.org/abs/1911.05722)][[code|official](https://github.com/facebookresearch/moco)][`Kaiming He + Ross Girshick`, `Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* 👍**BYOL (NIPS2020)** Bootstrap your own latent: A new approach to self-supervised Learning [[paper link](https://papers.nips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)][[arxiv link](https://arxiv.org/abs/2006.07733)][[code|official](https://github.com/deepmind/deepmind-research/tree/master/byol)][`Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* 👍**SwAV (NIPS2020)** Unsupervised Learning of Visual Features by Contrasting Cluster Assignments [[paper link](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)][[arxiv link](https://arxiv.org/abs/2006.09882)]
[[code|official](https://github.com/facebookresearch/swav)][including `contrastive learning`]

* **CARE(NIPS2021)** Revitalizing CNN Attention via Transformers in Self-Supervised Visual Representation Learning [[paper link](https://proceedings.neurips.cc/paper_files/paper/2021/hash/21be992eb8016e541a15953eee90760e-Abstract.html)][[openreview link](https://openreview.net/forum?id=sRojdWhXJx)][[arxiv link](https://arxiv.org/abs/2110.05340)][[`Chongjian GE 葛崇剑`](https://chongjiange.github.io/)][In order to make the training process of `Mean-Teacher` more stable, it slowly increases α from 0.999 to 1 through `cosine` design.]

* 👍👍**DINO (ICCV2021)** Emerging Properties in Self-Supervised Vision Transformers [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)][`ViT-based`, `a form of self-distillation with no labels`, `self-supervised pre-training`]

* **MoCo-v3(ICCV2021)** An Empirical Study of Training Self-Supervised Vision Transformers [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_An_Empirical_Study_of_Training_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)][`ViT-based`, `self-supervised pre-training`]

* 👍**SimSiam (CVPR2021)** Exploring Simple Siamese Representation Learning [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2011.10566)][[code|official](https://github.com/facebookresearch/simsiam)][`Kaiming He`, `Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* **SimMIM (CVPR2022)(arxiv2021.11)** SimMIM: A Simple Framework for Masked Image Modeling [[arxiv link](https://arxiv.org/abs/2111.09886)][[code|official](https://github.com/microsoft/SimMIM)][`Microsoft`, `a self-supervised approach that enables SwinV2-G`]

* 👍**MAE (CVPR2022)** Masked Autoencoders Are Scalable Vision Learners [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)][`ViT-based`, `FAIR`, `He Kaiming`， `It reconstructs the original signal given its partial observation`, `self-supervised pre-training`]

* **RSP(ICML2024)(arxiv2024.06)** Visual Representation Learning with Stochastic Frame Prediction [[arxiv link](https://arxiv.org/abs/2406.07398)][[project link](https://sites.google.com/view/2024rsp)][[code|official](https://github.com/huiwon-jang/RSP)][`KAIST + UC Berkeley + Dyson Robot Learning Lab`][It can be used for `Vision-based Robot Learning` inlcuding the `RLBench`]

#### 2.3) Joint, Lightweight and Efficient Training 

* **EfficientTrain++(TPAMI2024)(arxiv2024.05)** EfficientTrain++: Generalized Curriculum Learning for Efficient Visual Backbone Training [[paper link](https://ieeexplore.ieee.org/abstract/document/10530470/)][[arxiv link](https://arxiv.org/pdf/2405.08768)][[weixin blog](https://mp.weixin.qq.com/s/FJj0F2NcW9ftmT_lbO1R3w)][[code|official](https://github.com/LeapLabTHU/EfficientTrain)][`THU + BAAI`, used the `generalized curriculum learning`][The conference (EfficientTrain, ICCV2023) version [EfficientTrain: Exploring Generalized Curriculum Learning for Training Visual Backbones](https://arxiv.org/abs/2211.09703)]

* **DI-MaskDINO(NIPS2024)(arxiv2024.10)*** DI-MaskDINO: A Joint Object Detection and Instance Segmentation Model [[openreview link](https://openreview.net/forum?id=srQxkSPJLW)][[arxiv link](https://arxiv.org/abs/2410.16707)][`Chongqing University + Tsinghua University`; `Jifeng Dai`]


***

### ⭐3) Vision-Language Series
*also known as `Vision Language Models`, `Vision Language Representation Learning` and `Large Multimodal Models`*

#### 3.1) VLMs Distillation Series
*for the `Vision Foundation Models Distillation` task*

* 👍**AM-RADIO(CVPR2024)(arxiv2023.12)** AM-RADIO: Agglomerative Vision Foundation Model -- Reduce All Domains Into One [[paper link](http://openaccess.thecvf.com/content/CVPR2024/html/Ranzinger_AM-RADIO_Agglomerative_Vision_Foundation_Model_Reduce_All_Domains_Into_One_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2312.06709)][[code|official](https://github.com/NVlabs/RADIO)][`NVIDIA`]

* **TroL(arxiv2024.06)** TroL: Traversal of Layers for Large Language and Vision Models [[arxiv link](https://arxiv.org/abs/2406.12246)][[code|official](https://github.com/ByungKwanLee/TroL)][`KAIST`][It is one of the `Numerous Efficient Approaches`.]

* **UNIC(ECCV2024)(arxiv2024.08)*** UNIC: Universal Classification Models via Multi-teacher Distillation [[arxiv link](https://arxiv.org/abs/2408.05088)][[project link](https://europe.naverlabs.com/research/publications-enhanced/unic-universal-classification-models-via-multi-teacher-distillation/)][[code|official](https://github.com/naver/unic)][`NAVER LABS Europe`]

* 👍**JAFAR(arxiv2025.06)** JAFAR: Jack up Any Feature at Any Resolution [[arxiv link](https://arxiv.org/abs/2506.11136)][[project link](https://jafar-upsampler.github.io/)][[code|official](https://github.com/PaulCouairon/JAFAR)][`Sorbonne Université + Thales cortAIx Labs + Valeo.ai`][`Foundation Vision Encoders`]

* 👍**Grafting(arxiv2025.06)** Exploring Diffusion Transformer Designs via Grafting [[arxiv link](https://arxiv.org/abs/2506.05340)][[project link](https://grafting.stanford.edu/)][[code|official](https://github.com/keshik6/grafting)][`Stanford University + Liquid AI + Together AI + UC San Diego + Northwestern University + Google DeepMind + Salesforce Research`; `Fei-Fei Li`][`Diffusion Transformer`, `Network Architecture Optimization`]

#### 3.2) VLMs Pretraining/Application Series
 
* **Flamingo(NIPS2022)(arxiv2022.04)** Flamingo: a Visual Language Model for Few-Shot Learning [[paper link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2204.14198)][[blog|official](https://deepmind.google/discover/blog/tackling-multiple-tasks-with-a-single-visual-language-model/)][`DeepMind`, Flamingo is a family of `Visual Language Models (VLM)`]

* **OpenFlamingo(arxiv2023.08)** OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models [[arxiv link](https://arxiv.org/abs/2308.01390)][[code|official](https://github.com/mlfoundations/open_flamingo)][[blog 1 | official](https://laion.ai/blog/open-flamingo/)][[blog 2 | official](https://laion.ai/blog/open-flamingo-v2/)][`University of Washington` and `Stanford University`]

* **TiC-CLIP(ICLR2024)(arxiv2023.10)** TiC-CLIP: Continual Training of CLIP Models [[openreview link](https://openreview.net/forum?id=TLADT8Wrhn)][[arxiv link](https://arxiv.org/abs/2310.16226)][[code|official](https://github.com/apple/ml-tic-clip)][`Apple`, based on the code of [`OpenCLIP`](https://github.com/mlfoundations/open_clip)]

* **Cluster Masking(CVPR2024)(arxiv2024.05)** Efficient Vision-Language Pre-training by Cluster Masking [[paper link]()][[arxiv link](https://arxiv.org/abs/2405.08815)][[project link](https://zxp46.github.io/cluster-masking/)][[code|official](https://github.com/Zi-hao-Wei/Efficient-Vision-Language-Pre-training-by-Cluster-Masking)][`University of Michigan`]

* 👍**Florence-2(CVPR2024)(arxiv2023.11)** Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Florence-2_Advancing_a_Unified_Representation_for_a_Variety_of_Vision_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2311.06242)][[project link](https://blog.roboflow.com/florence-2/)][[code|official](https://github.com/kijai/ComfyUI-Florence2)][[code|huggingface](https://huggingface.co/microsoft/Florence-2-large)][`Azure AI, Microsoft`]

* **LQMFormer(CVPR2024)** LQMFormer: Language-aware Query Mask Transformer for Referring Image Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Shah_LQMFormer_Language-aware_Query_Mask_Transformer_for_Referring_Image_Segmentation_CVPR_2024_paper.html)][`Johns Hopkins University`; `Referring Image Segmentation (RIS)` aims to segment objects from an image based on a language description.][`Vision Language Application`]

* **ImOV3D(NIPS2024)(arxiv2024.10)** ImOV3D: Learning Open-Vocabulary Point Clouds 3D Object Detection from Only 2D Images [[openreview link](https://openreview.net/forum?id=RCO9fRP8AJ)][[arxiv link](https://arxiv.org/abs/2410.24001)][[code|official](https://github.com/yangtiming/ImOV3D)][`Shanghai Qi Zhi Institute + IIIS, Tsinghua University + Shanghai AI Lab`]

* **RefHuman(NIPS2024)(arxiv2024.10)** Referring Human Pose and Mask Estimation in the Wild [[openreview link](https://openreview.net/forum?id=fXEi3LVflp)][[arxiv link](https://arxiv.org/abs/2410.20508)][[code|official](https://github.com/bo-miao/RefHuman)][`University of Western Australia + Xidian University + Hunan University + Griffith University`][`Vision Language Application`]

* 👍**Florence-VL(CVPR2025)(arxiv2024.12)** Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion [[paper link](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Florence-VL_Enhancing_Vision-Language_Models_with_Generative_Vision_Encoder_and_Depth-Breadth_CVPR_2025_paper.html)][[arxiv link](https://arxiv.org/abs/2412.04424)][[project link](https://jiuhaichen.github.io/florence-vl.github.io/)][[code|official](https://github.com/JiuhaiChen/Florence-VL)][`University of Maryland + Microsoft Research`]

* 👍**GroundingSuite(ICCV2025)(arxiv2025.03)** GroundingSuite: Measuring Complex Multi-Granular Pixel Grounding [[arxiv link](https://arxiv.org/abs/2503.10596)][[code|official](https://github.com/hustvl/GroundingSuite)][`Huazhong University of Science and Technology + vivo AI Lab`; `Xinggang Wang`]



***

### ⭐4) Tracking Anything Series
*for the `2d/3d visual tracking` task*

* 👍**SMITE(ICLR2025)(arxiv2024.10)** SMITE: Segment Me In TimE [[arxiv link](https://arxiv.org/abs/2410.18538)][[project link](https://segment-me-in-time.github.io/)][[code|official](https://github.com/alimohammadiamirhossein/smite/)][[weixin blog](https://mp.weixin.qq.com/s/b2b6NxyaVpjGO8_pgL7KFA)][`Simon Fraser University + Autodesk Research + University of Toronto + Google DeepMind`]

* **EfficientTAM(arxiv2024.11)** Efficient Track Anything [[arxiv link](https://arxiv.org/abs/2411.18933)][[project link](https://yformer.github.io/efficient-track-anything/)][[code|official](https://github.com/yformer/EfficientTAM)][`Meta AI + Nanyang Technological University`]


***

### ⭐5) Depth Anything Series
*for the `monocular depth estimation` and `stereo matching` task*

* **DepthAnything(CVPR2024)(arxiv2024.01)** Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Depth_Anything_Unleashing_the_Power_of_Large-Scale_Unlabeled_Data_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2401.10891)][[project link](https://depth-anything.github.io/)][[code|official](https://github.com/LiheYoung/Depth-Anything)][`HKU + TikTok + CUHK + ZJU`][the first author is [`Lihe Yang`](https://liheyoung.github.io/)][two keys factors: following the `scaling law of large dataset` and the `semi-supervised learning` technique][It harnesses large-scale unlabeled data to speed up data scaling-up and increase the data coverage]
  
* **DepthAnythingV2(NIPS2025)(arxiv2024.06)** Depth Anything V2 [[arxiv link](https://arxiv.org/abs/2406.09414)][[project link](https://depth-anything-v2.github.io/)][[code|official](https://github.com/DepthAnything/Depth-Anything-V2)][`HKU + TikTok`][the first author is [`Lihe Yang`](https://liheyoung.github.io/)][two keys factors: following the `scaling law of large dataset` and the `semi-supervised learning` technique][It demonstrates “precise synthetic data + pseudo-labeled real data” is a more promising roadmap than labeled real data]

* **FoundationStereo(CVPR2025)(arxiv2025.01)** FoundationStereo: Zero-Shot Stereo Matching [[arxiv link](https://arxiv.org/abs/2501.09898)][[project link](https://nvlabs.github.io/FoundationStereo/)][[code|official](https://github.com/NVlabs/FoundationStereo)][`NVIDIA`]


***

### ⭐6) Mesh Anything Series
*for the `3D mesh generation` task*

* **MeshAnything(arxiv2024.06)** MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers [[arxiv link](https://arxiv.org/abs/2406.10163)][[project link](https://buaacyw.github.io/mesh-anything/)][[blog link](https://zhuanlan.zhihu.com/p/706166825)][[code|official](https://github.com/buaacyw/MeshAnything)][`S-Lab, Nanyang Technological University, + others`]

* **MeshAnythingV2(arxiv2024.08)** MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization [[arxiv link](https://arxiv.org/abs/2408.02555)][[project link](https://buaacyw.github.io/meshanything-v2/)][[blog link](https://baijiahao.baidu.com/s?id=1807065134602050319)][[code|official](https://github.com/buaacyw/MeshAnythingV2)][`S-Lab, Nanyang Technological University, + others`]

***

### ⭐7) Foundation Pose Series
*for the popular `6D Object Pose Estimation` task*

* **FoundationPose(CVPR2024 Highlight)(arxiv2023.12)** FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2312.08344)][[project link](https://nvlabs.github.io/FoundationPose/)][[code|official](https://github.com/NVlabs/FoundationPose)][`NVIDIA`]

* **OrientAnything(arxiv2024.12)** Orient Anything: Learning Robust Object Orientation Estimation from Rendering 3D Models [[arxiv link](https://arxiv.org/abs/2412.18605)][[project link](https://orient-anything.github.io/)][[code|official](https://github.com/SpatialVision/Orient-Anything)][`Zhejiang University + Sea AI Lab + The University of Hong Kong`]

* **FoundationPose++(year2025.03)** FoundationPose++: Simple Tricks Boost FoundationPose Performance in High-Dynamic Scenes [[code|official](https://github.com/teal024/FoundationPose-plus-plus)][`Real-Time 6D Pose Tracker in High-Dynamic Scenes`]

* **DynOPETs(arxiv2025.03)** DynOPETs: A Versatile Benchmark for Dynamic Object Pose Estimation and Tracking in Moving Camera Scenarios [[arxiv link](https://arxiv.org/abs/2503.19625)][[project link](https://stay332.github.io/DynOPETs/)][[code|official](https://github.com/Launch-on-Titania/DynOPETs)][`ShanghaiTech University, Mobile Perception Lab + Fudan University, Multi-Agent Robotic Systems Lab`]

* **CAP-Net(CVPR2025 Highlight)(arxiv2025.04)** CAP-Net: A Unified Network for 6D Pose and Size Estimation of Categorical Articulated Parts from a Single RGB-D Image [[paper link](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_CAP-Net_A_Unified_Network_for_6D_Pose_and_Size_Estimation_CVPR_2025_paper.html)][[arxiv link](https://arxiv.org/pdf/2504.11230)][[project link](https://shanehuanghz.github.io/CAPNet/)][[code|official](https://github.com/ShaneHuangHZ/CAPNet)][`Fudan University + Huawei, Noah’s Ark Lab`]

* **UA-Pose(CVPR2025)(arxiv2025.06)** UA-Pose: Uncertainty-Aware 6D Object Pose Estimation and Online Object Completion with Partial References [[paper link](https://openaccess.thecvf.com/content/CVPR2025/html/Li_UA-Pose_Uncertainty-Aware_6D_Object_Pose_Estimation_and_Online_Object_Completion_CVPR_2025_paper.html)][[arxiv link](https://arxiv.org/abs/2506.07996)][[project link](https://minfenli.github.io/UA-Pose/)][[code|official](https://github.com/minfenli/UA-Pose)][`Carnegie Mellon University + Stony Brook University + National Tsing Hua University + Amazon`]


* **** [[paper link]()][[arxiv link]()][[project link]()][[code|official]()]

