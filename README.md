# Training Small Vision Language Models: A Comprehensive Guide

Welcome to the **Training Small Vision Language Models** repository. This repository offers a guide and example for training small-scale vision-language models, with a focus on implementing and training MODA (Multimodal Object Description Assistant). MODA is an advanced AI model designed to describe fashion items by effectively combining visual and textual data.

## Vision-Language Models (VLMs)

<p align="center">
  <img width="700" alt="VLMs" src="https://github.com/user-attachments/assets/07788713-90e6-4d7b-b083-415502745688">
</p> 

In recent years, advancements in language modeling have led to the development of powerful Large Language Models (LLMs) such as Llama, Mistral, ChatGPT, and Gemini, which can handle a wide variety of tasks. These models, originally limited to text inputs, are now being extended to visual inputs, creating new possibilities for AI applications. Integrating vision and language efficiently is crucial to making this technology more accessible and cost-effective. In this context, the training of Vision-Language Models (VLMs) using pretrained backbones emerges as a powerful solution to simplify training and reduce costs while maintaining high performance. These pretrained models can efficiently process and align visual and textual information, enabling more effective development of VLMs. Models like LLaVA and PaLI demonstrate the effectiveness of this approach, highlighting the potential for optimized training and improved functionality in VLMs.

### Key VLMs from Pretrained Backbones. 

- **LLaVA** utilizes CLIP and Vicuna to handle diverse tasks such as visual question answering and image captioning. It achieves efficient resource utilization and incorporates reinforcement learning from human feedback .
- **PaLI** supports over 100 languages by combining large Vision Transformers (ViT) with mT5 text models, trained on the extensive WebLI dataset, demonstrating robust multilingual capabilities .
- **PaliGemma**, inspired by PaLI-3, integrates the SigLIP vision model and the Gemma language model, focusing on multimodal pretraining and high-resolution tasks to balance performance and efficiency .
- **Frozen** and **MiniGPT-4** effectively align visual features with text token embeddings, reducing computational requirements and enhancing versatility  .

While these models are highly efficient, their large number of parameters often prevents their use in low-compute environments or for tasks that can be achieved with fewer parameters. This highlights the need for smaller models tailored to specific tasks, which can still perform well without requiring extensive computational resources.

## MODA - Multimodal Object Description Assistant


MODA (Multimodal Object Description Assistant) addresses the need for specialized, task-specific Vision-Language Models (VLMs) designed for fashion item descriptions. With only 280 million parameters, MODA maintains a lightweight architecture that allows it to run efficiently, even without a GPU, making it highly accessible for applications with resource constraints. This specialization highlights MODA's advantage in delivering detailed and accurate fashion descriptions with minimal computational overhead.

<p align="center">
  <img width="1000" alt="MODA" src="https://github.com/user-attachments/assets/c5fff5bf-13a8-4999-8c11-27736503581e">
</p> 


MODA is built using FashionCLIP[1], a model that integrates the capabilities of CLIP (Contrastive Language-Image Pre-training)[2] with fashion-specific datasets, and OPT (Open Pre-trained Transformers)[3] from Meta, a large language model. By leveraging these advanced technologies, MODA provides precise and detailed descriptions of various fashion objects, demonstrating the effectiveness of combining state-of-the-art image and text models for specialized applications.

### Product Image Dataset for E-commerce Applications

This dataset contains over 2900 product images, categorized under Apparel and Footwear, and includes items for Boys, Girls, Men, and Women. The dataset includes a `fashion.csv` file with metadata such as title, description, category, and gender. It is suitable for various applications like category classification, visual similarity-based recommendation systems, custom named entity recognition for attributes like color and gender.

<p align="center">
  <img width="700" alt="Dataset" src="https://github.com/user-attachments/assets/ad1b3888-8c72-4362-babf-ac79382bb0f7">
</p> 

The dataset's high-resolution images and metadata support better product organization and customer recommendations on e-commerce platforms. This dataset is a useful for developing machine learning models and algorithms focused on improving product image recognition and recommendation systems. For more information and access, visit the [dataset page](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images/data).


### MODA Architecture

MODA (Multimodal Object Description Assistant) uses a specialized architecture to generate detailed descriptions of fashion items by combining FashionCLIP for image encoding and OPT-125M as decoder for text generation. Its non-linear projection increases and then reduces the dimensionality of image embeddings, capturing complex patterns and enhancing representation quality. This approach allows MODA to deliver high accuracy in fashion-specific tasks with only 280 million parameters, making it highly efficient and capable of running without a GPU.

<p align="center">
  <img width="700" alt="Architecture" src="https://github.com/user-attachments/assets/43b201d3-b1ee-402d-9a4a-2a6f66c2476d">
</p> 

In comparison, models like LLaVA and PaLI adopt a more versatile approach, leveraging pretrained backbones to align visual and textual modalities for a wide range of tasks. LLaVA integrates CLIP and Vicuna with a simple linear projection for efficient resource utilization, while PaLI combines ViT and mT5 models to support over 100 languages. PaliGemma follows a similar strategy with multimodal pretraining and high-resolution tasks, balancing performance and computational efficiency (Google AI, 2023). MODA’s non-linear projection method provides a distinct advantage in capturing intricate visual details, enhancing its performance in generating accurate and detailed descriptions compared to the linear projections used by other models.


### Training

The training of the MODA model aims to optimize its ability to generate detailed descriptions of fashion items by effectively integrating visual and textual data. The process starts with the FashionCLIP[1] model, which processes the input image to generate an embedding. This embedding is then projected to a suitable size through a non-linear projection layer, enhancing its representation quality. Simultaneously, the OPT[3] tokenizer converts the text description (image label) into token embeddings. These image and text embeddings are combined into a sequence and fed into the OPT-125M[3] model, a pre-trained transformer, to predict the next token in the sequence. Then, the model's prediction performance is evaluated using the CrossEntropy loss function.

<p align="center">
  <img width="1000" alt="Training Scheme" src="https://github.com/user-attachments/assets/9f846bb8-13c5-413b-8de6-7e05677879ae">
</p> 

The training process employs the AdamW optimizer with a learning rate of 1e-3, and the learning rate is adjusted using a StepLR scheduler. During training, similar to LLaVA, the FashionCLIP[1] encoder had its parameters frozen, and only the non-linear projection and the language model were trained. The model is trained for 20 epochs, taking 1 hour and 44 minutes to train on a Google Colab A100, with gradient accumulation and periodic model saving to ensure stability and performance.

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

