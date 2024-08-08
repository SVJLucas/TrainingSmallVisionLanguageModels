# Training Small Vision Language Models: A Comprehensive Guide

Welcome to the **Training Small Vision Language Models** repository. This repository offers a guide and example for training small-scale vision-language models, with a focus on implementing and training MODA (Multimodal Object Description Assistant). MODA is an advanced AI model designed to describe fashion items by effectively combining visual and textual data.

## Vision-Language Models (VLMs)

<p align="center">
  <img width="700" alt="VLMs" src="https://github.com/user-attachments/assets/07788713-90e6-4d7b-b083-415502745688">
</p> 

Vision-Language Models (VLMs) have seen significant advancements by integrating visual and textual data. They leverage pretrained backbones to reduce training costs while maintaining high performance. Examples of such models include LLaVA[4] and PaLI[5]. 

### Key VLMs from Pretrained Backbones. 

- **LLaVA** utilizes CLIP and Vicuna to handle diverse tasks such as visual question answering and image captioning. It achieves efficient resource utilization and incorporates reinforcement learning from human feedback .
- **PaLI** supports over 100 languages by combining large Vision Transformers (ViT) with mT5 text models, trained on the extensive WebLI dataset, demonstrating robust multilingual capabilities .
- **PaliGemma**, inspired by PaLI-3, integrates the SigLIP vision model and the Gemma language model, focusing on multimodal pretraining and high-resolution tasks to balance performance and efficiency .
- **Frozen** and **MiniGPT-4** effectively align visual features with text token embeddings, reducing computational requirements and enhancing versatility  .



While these models are highly efficient, their large number of parameters often prevents their use in low-compute environments or for tasks that can be achieved with fewer parameters. This highlights the need for smaller models tailored to specific tasks, which can still perform well without requiring extensive computational resources.

## MODA - Multimodal Object Description Assistant

MODA (Multimodal Object Description Assistant) addresses this need by being a specialized, task-specific VLM designed for fashion item descriptions. Combining FashionCLIP[1] for image encoding and OPT-125M[3] for text generation, MODA is tailored for accuracy in its domain while maintaining a lightweight architecture with only 280 million parameters. This small size allows MODA to run efficiently, even without a GPU, making it highly accessible for specific applications where resource constraints are a consideration. This specialization underscores MODA's advantage in delivering detailed and accurate fashion descriptions with minimal computational overhead.



<p align="center">
  <img width="1000" alt="MODA" src="https://github.com/user-attachments/assets/c5fff5bf-13a8-4999-8c11-27736503581e">
</p> 

MODA is built using FashionCLIP[1], a model that combines the capabilities of CLIP (Contrastive Language-Image Pre-training)[2] with fashion-specific datasets, and OPT (Open Pre-trained Transformers)[3] from Meta, which is a large language model. By leveraging these technologies, MODA provides detailed and accurate descriptions of various fashion objects.

### MODA Architecture


<p align="center">
  <img width="700" alt="Architecture" src="https://github.com/user-attachments/assets/4010d574-1a82-4594-82b9-5071e0459daa">
</p> 

### Training

The MODA model is designed to generate detailed descriptions for fashion items by integrating visual and textual data. This hybrid approach leverages the capabilities of both the FashionCLIP[1] model and the OPT-125M[3] language model. The FashionCLIP[1] model processes the input image to generate an image embedding, which is then projected to a suitable size through a non-linear projection layer. Simultaneously, the OPT[3] tokenizer converts the text description into token embeddings. These image and text embeddings are combined into a sequence of embeddings, which are then fed into the OPT-125M[3] model. This pre-trained transformer model processes the combined embeddings to generate or refine the descriptive text, enhancing the detail and accuracy of fashion item descriptions.


<p align="center">
  <img width="1000" alt="Training Scheme" src="https://github.com/user-attachments/assets/9f846bb8-13c5-413b-8de6-7e05677879ae">
</p> 

The training process employs the AdamW optimizer with a learning rate of 1e-3, and the learning rate is adjusted using a StepLR scheduler. During training, the FashionCLIP[1] had its parameters frozen, and only the non-linear projection and the language model were trained. The CrossEntropyLoss function is used to compute the loss, and training is conducted on a CUDA-enabled GPU when available. The model is trained for 20 epochs, taking 1 hour and 44 minutes to train on a Google Colab A100, with gradient accumulation and periodic model saving to ensure stability and performance.

<p align="center">
  <img width="700" alt="Training Chart" src="https://github.com/user-attachments/assets/85813aac-d8a0-4c7e-9ef9-fb81f5f62b3d">
</p> 

The training chart shows a sharp decline in both training and validation losses in the initial steps, indicating rapid learning and convergence. As training progresses, losses continue to decrease and stabilize at lower values, demonstrating effective learning and good generalization to validation data. The close alignment between the training (blue line) and validation (orange line) loss curves suggests that the model is not overfitting and maintains good performance on the validation set.

### References

1. Chia, P.J., Attanasio, G., Bianchi, F. et al. Contrastive language and vision learning of general fashion concepts. Sci Rep 12, 18958 (2022). [https://doi.org/10.1038/s41598-022-23052-9](https://doi.org/10.1038/s41598-022-23052-9)
2. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv preprint arXiv:2103.00020. [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
3. Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X.V., Mihaylov, T., Ott, M., Shleifer, S., Shuster, K., Simig, D., Koura, P.S., Sridhar, A., Wang, T., Zettlemoyer, L. (2022). OPT: Open Pre-trained Transformer Language Models. arXiv preprint arXiv:2205.01068. [https://arxiv.org/abs/2205.01068](https://arxiv.org/abs/2205.01068)
4. Liu, H., Li, C., Wu, Q., Lee, Y.J. Visual instruction tuning. Advances in Neural Information Processing Systems, 36, 34892–34916 (2023). [https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf)
5. Chen, J., Zhu, D., Shen, X., Li, X., Liu, Z., Zhang, P., Krishnamoorthi, R., Chandra, V., Xiong, Y., Elhoseiny, M. MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478 (2023). [https://arxiv.org/abs/2310.09478](https://arxiv.org/abs/2310.09478)
6. Beyer, L., Steiner, A., Pinto, A.S., Kolesnikov, A., Wang, X., Salz, D., Neumann, M., Alabdulmohsin, I., Tschannen, M., Bugliarello, E., Unterthiner, T., Keysers, D., Koppula, S., Liu, F., Grycner, A., Gritsenko, A., Houlsby, N., Kumar, M., Rong, K., Eisenschlos, J., Kabra, R., Bauer, M., Bošnjak, M., Chen, X., Minderer, M., Voigtlaender, P., Bica, I., Balazevic, I., Puigcerver, J., Papalampidi, P., Henaff, O., Xiong, X., Soricut, R., Harmsen, J., Zhai, X. PaliGemma: A versatile 3B VLM for transfer. arXiv preprint arXiv:2407.07726 (2024). [https://arxiv.org/abs/2407.07726](https://arxiv.org/abs/2407.07726)
7. Tsimpoukelli, M., Menick, J.L., Cabi, S., Eslami, S.M., Vinyals, O., Hill, F. Multimodal few-shot learning with frozen language models. Advances in Neural Information Processing Systems, 34, 200–212 (2021). [https://proceedings.neurips.cc/paper_files/paper/2021/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf)
8. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023). [https://arxiv.org/abs/2304.10592](https://arxiv.org/abs/2304.10592)



# Training Small Vision Language Models

Welcome to the TrainingSmallVisionLanguageModels repository. This repository serves as a comprehensive example and guide for training small-scale vision-language models, focusing on the implementation and training of MODA (Multimodal Object Description Assistant). MODA is an advanced AI model designed to describe fashion items by effectively combining visual and textual data.

## Vision-Language Models (VLMs)

Vision-Language Models (VLMs) have significantly advanced by integrating visual and textual data, leveraging pretrained backbones to reduce training costs while maintaining high performance. Models like LLaVA[4] and PaLI[5] exemplify this trend. LLaVA uses CLIP and Vicuna to handle diverse tasks such as visual question answering and image captioning with efficient resource utilization and reinforcement learning from human feedback [4]. PaLI supports over 100 languages by combining large ViT models and mT5 text models, trained on the extensive WebLI dataset, demonstrating robust multilingual capabilities [5]. PaliGemma, inspired by PaLI-3, integrates the SigLIP vision model and the Gemma language model, focusing on multimodal pretraining and high-resolution tasks to balance performance and efficiency [6]. Leveraging pretrained backbones, models like Frozen and MiniGPT-4 efficiently align visual features with text token embeddings, reducing computational requirements and enhancing versatility [7][8].


<p align="center">
  <img width="700" alt="VLMs" src="https://github.com/user-attachments/assets/07788713-90e6-4d7b-b083-415502745688">
</p> 

**Although these models are highly efficient, their large number of parameters often prevents their use in low-compute environments or for tasks that can be achieved with fewer parameters. This highlights the need for smaller models tailored to specific tasks, which can still perform well without requiring extensive computational resources.**


## MODA - Multimodal Object Description Assistant

MODA (Multimodal Object Description Assistant) addresses this need by being a specialized, task-specific VLM designed for fashion item descriptions. Combining FashionCLIP [1] for image encoding and OPT-125M [3] for text generation, MODA is tailored for accuracy in its domain while maintaining a lightweight architecture with only 280 million parameters. This small size allows MODA to run efficiently, even without a GPU, making it highly accessible for specific applications where resource constraints are a consideration. This specialization underscores MODA's advantage in delivering detailed and accurate fashion descriptions with minimal computational overhead.

<p align="center">
  <img width="700" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/c5fff5bf-13a8-4999-8c11-27736503581e">
</p> 

MODA (Multimodal Object Description Assistant) is an AI model that describes fashion items by combining visual and textual data. Vision Language Models (VLMs) like MODA integrate visual and linguistic information to enhance understanding and generation of descriptive text for images. MODA is built using FashionCLIP [1], a model that combines the capabilities of CLIP (Contrastive Language-Image Pre-training) [2] with fashion-specific datasets, and OPT (Open Pre-trained Transformers) [3] from Meta, which is a large language model. By leveraging these technologies, MODA provides detailed and accurate descriptions of various fashion objects.

### MODA Architecture
<p align="center">
  <img width="700" alt="Architecture" src="https://github.com/user-attachments/assets/4010d574-1a82-4594-82b9-5071e0459daa">
</p> 

### Training

The MODA model is designed to generate detailed descriptions for fashion items by integrating visual and textual data. This hybrid approach leverages the capabilities of both the FashionCLIP [1] model and the OPT-125M [3] language model. The FashionCLIP [1] model processes the input image to generate an image embedding, which is then projected to a suitable size through a non-linear projection layer. Simultaneously, the OPT [3] tokenizer converts the text description into token embeddings. These image and text embeddings are combined into a sequence of embeddings, which are then fed into the OPT-125M [3] model. This pre-trained transformer model processes the combined embeddings to generate or refine the descriptive text, enhancing the detail and accuracy of fashion item descriptions.

<p align="center">
  <img width="1200" alt="logo_dark_stacked_KzUurne (1)" src="https://github.com/user-attachments/assets/9f846bb8-13c5-413b-8de6-7e05677879ae">
</p> 

The training process employs the AdamW optimizer with a learning rate of 1e-3, and the learning rate is adjusted using a StepLR scheduler. During training, the FashionCLIP [1] had its parameters frozen, and we trained only the non-linear projection and the language model. The CrossEntropyLoss function is used to compute the loss, and training is conducted on a CUDA-enabled GPU when available. The model is trained for 20 epochs, taking 1 hour and 44 minutes to train on a Google Colab A100, with gradient accumulation and periodic model saving to ensure stability and performance.

<p align="center">
  <img width="600" alt="training_chart" src="https://github.com/user-attachments/assets/85813aac-d8a0-4c7e-9ef9-fb81f5f62b3d">
</p> 

The training chart shows a sharp decline in both training and validation losses in the initial steps, indicating rapid learning and convergence. As training progresses, losses continue to decrease and stabilize at lower values, demonstrating effective learning and good generalization to validation data. The close alignment between the training (blue line) and validation (orange line) loss curves suggests that the model is not overfitting and maintains good performance on the validation set.

### References

1. Chia, P.J., Attanasio, G., Bianchi, F. et al. Contrastive language and vision learning of general fashion concepts. Sci Rep 12, 18958 (2022). https://doi.org/10.1038/s41598-022-23052-9
2. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv preprint arXiv:2103.00020. https://arxiv.org/abs/2103.00020
3. Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X.V., Mihaylov, T., Ott, M., Shleifer, S., Shuster, K., Simig, D., Koura, P.S., Sridhar, A., Wang, T., Zettlemoyer, L. (2022). OPT: Open Pre-trained Transformer Language Models. arXiv preprint arXiv:2205.01068. https://arxiv.org/abs/2205.01068
4. Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf.
5. Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023. URL https://arxiv.org/abs/2310.09478.
6. Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, Xiaohua Zhai. PaliGemma: A versatile 3B VLM for transfer. arXiv preprint arXiv:2407.07726, 2024. URL https://arxiv.org/abs/2407.07726.
7. Maria Tsimpoukelli, Jacob L. Menick, Serkan Cabi, S.M. Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. Advances in Neural Information Processing Systems, 34:200–212, 2021. URL https://proceedings.neurips.cc/paper/2021/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf.
8. Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. URL https://arxiv.org/abs/2304.10592.
