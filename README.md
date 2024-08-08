# Training Small Vision Language Models: A Comprehensive Guide

Welcome to the **Training Small Vision Language Models** repository. This repository offers a guide and example for training small-scale vision-language models, with a focus on implementing and training MODA (Multimodal Object Description Assistant). MODA is an advanced AI model designed to describe fashion items by effectively combining visual and textual data.

## Vision-Language Models (VLMs)

<p align="center">
  <img width="700" alt="VLMs" src="https://github.com/user-attachments/assets/07788713-90e6-4d7b-b083-415502745688">
</p> 

In recent years, advancements in language modeling have led to the development of powerful Large Language Models (LLMs) such as Llama[1], Mistral[2], ChatGPT[3], and Gemini[4], which can handle a wide variety of tasks. These models, originally limited to text inputs, are now being extended to visual inputs, creating new possibilities for AI applications. Integrating vision and language efficiently is crucial to making this technology more accessible and cost-effective. In this context, the training of Vision-Language Models (VLMs) using pretrained backbones emerges as a powerful solution to simplify training and reduce costs while maintaining high performance[5]. These pretrained models can efficiently process and align visual and textual information, enabling more effective development of VLMs. Models like LLaVA[6] and PaLI[7] demonstrate the effectiveness of this approach, highlighting the potential for optimized training and improved functionality in VLMs.

In the context of VLMs from Pretrained Backbones, the most known ones are: 

- **LLaVA**[6] (7B and 13B parameters) utilizes CLIP (Contrastive Language-Image Pre-training)[8] and Vicuna[9] to handle diverse tasks such as visual question answering and image captioning. It achieves efficient resource utilization and incorporates reinforcement learning from human feedback .
- **PaLI**[7] (3B, 15B and 17B parameters) supports over 100 languages by combining large Vision Transformers (ViT)[10] with mT5[11] text models, trained on the extensive WebLI dataset, demonstrating robust multilingual capabilities .
- **PaliGemma**[12] (3B parameters) , inspired by PaLI-3[13], integrates the SigLIP[14] vision model and the Gemma[15] language model, focusing on multimodal pretraining and high-resolution tasks to balance performance and efficiency .
- **MiniGPT-4**[16] (7B and 13B parameters) effectively align visual features with text token embeddings, reducing computational requirements and enhancing versatility  .

While these models are highly efficient, their large number of parameters often prevents their use in low-compute environments or for tasks that can be achieved with fewer parameters. This highlights the need for smaller models tailored to specific tasks, which can still perform well without requiring extensive computational resources.

## MODA - Multimodal Object Description Assistant


MODA (Multimodal Object Description Assistant) addresses the need for specialized, task-specific Vision-Language Models (VLMs) designed for fashion item descriptions. With only 280 million parameters, MODA maintains a lightweight architecture that allows it to run efficiently, even without a GPU, making it highly accessible for applications with resource constraints. This specialization highlights MODA's advantage in delivering detailed and accurate fashion descriptions with minimal computational overhead.

<p align="center">
  <img width="1000" alt="MODA" src="https://github.com/user-attachments/assets/c5fff5bf-13a8-4999-8c11-27736503581e">
</p> 


MODA is built using FashionCLIP[17], a model that integrates the capabilities of CLIP (Contrastive Language-Image Pre-training)[8] with fashion-specific datasets, and OPT (Open Pre-trained Transformers)[18] from Meta, a language model. By leveraging these advanced technologies, MODA provides precise and detailed descriptions of various fashion objects, demonstrating the effectiveness of combining state-of-the-art image and text models for specialized applications.

### Product Image Dataset for E-commerce Applications

This dataset contains over 2900 product images, categorized under Apparel and Footwear, and includes items for Boys, Girls, Men, and Women. The dataset includes a `fashion.csv` file with metadata such as title, description, category, and gender. It is suitable for various applications like category classification, visual similarity-based recommendation systems, custom named entity recognition for attributes like color and gender. Bellow, there are some samples from the [Image Dataset for E-commerce Applications](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images/data):

<p align="center">
  <img width="700" alt="Dataset" src="https://github.com/user-attachments/assets/ad1b3888-8c72-4362-babf-ac79382bb0f7">
</p> 

The dataset's high-resolution images and metadata support better product organization and customer recommendations on e-commerce platforms. This dataset is a useful for developing machine learning models and algorithms focused on improving product image recognition and recommendation systems.


### MODA Architecture

MODA (Multimodal Object Description Assistant) uses a specialized architecture to generate detailed descriptions of fashion items by combining FashionCLIP[17] for image encoding and OPT-125M[18] as decoder for text generation. Its non-linear projection increases and then reduces the dimensionality of image embeddings, capturing complex patterns and enhancing representation quality. This approach allows MODA to deliver accuracy in fashion-specific tasks with only 280 million parameters, making it efficient and capable of running without a GPU.

<p align="center">
  <img width="700" alt="Architecture" src="https://github.com/user-attachments/assets/43b201d3-b1ee-402d-9a4a-2a6f66c2476d">
</p> 

In comparison, models like LLaVA[6] and PaLI[7] adopt a more versatile approach, leveraging pretrained backbones to align visual and textual modalities for a wide range of tasks. LLaVA[6] integrates CLIP[8] and Vicuna[9] with a simple linear projection for efficient resource utilization, while PaLI combines Vision Transformers (ViT)[10] and mT5[11] models also via linear transformation for the same reason. Since MODA is a small model, the use of non-linearity does not impact resource utilization significantly. MODA’s non-linear projection method provides a distinct advantage in capturing intricate visual details, enhancing its performance in generating accurate and detailed descriptions compared to the linear projections used by other models.


### Training

The training of the MODA model aims to optimize its ability to generate detailed descriptions of fashion items by effectively integrating visual and textual data. The process starts with the FashionCLIP[17] model, which processes the input image to generate an embedding. This embedding is then projected to a suitable size through a non-linear projection layer, enhancing its representation quality. Simultaneously, the OPT[18] tokenizer converts the text description (image label) into token embeddings. These image and text embeddings are combined into a sequence and fed into the OPT-125M[18] model, a pre-trained transformer, to predict the next token in the sequence. Then, the model's prediction performance is evaluated using the CrossEntropy loss function.

<p align="center">
  <img width="1000" alt="Training Scheme" src="https://github.com/user-attachments/assets/9f846bb8-13c5-413b-8de6-7e05677879ae">
</p> 

The training process uses the AdamW[19] optimizer, which starts with a learning rate of 1e-3. To ensure the training is efficient, the learning rate is periodically reduced, dropping by a fixed percentage at regular intervals. During training, similar to LLaVA[6], the FashionCLIP[17] encoder had its parameters frozen, and only the non-linear projection and the language model were trained. The model is trained for 20 epochs, taking 1 hour and 44 minutes to train on a Google Colab A100, with gradient accumulation and periodic model saving to ensure stability and performance.

<p align="center">
  <img width="700" alt="Training Chart" src="https://github.com/user-attachments/assets/85813aac-d8a0-4c7e-9ef9-fb81f5f62b3d">
</p> 

The training chart shows a sharp decline in both training and validation losses in the initial steps, indicating rapid learning and convergence. As training progresses, losses continue to decrease and stabilize at lower values, demonstrating effective learning and good generalization to validation data. The close alignment between the training (blue line) and validation (orange line) loss curves suggests that the model is not overfitting and maintains good performance on the validation set.

## References

1. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., Lample, G. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971. [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)

2. Jiang, A.Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D.S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Renard Lavaud, L., Lachaux, M.-A., Stock, P., Le Scao, T., Lavril, T., Wang, T., Lacroix, T., El Sayed, W. (2023). Mistral 7B. arXiv preprint arXiv:2310.06825. [https://arxiv.org/abs/2310.06825](https://arxiv.org/abs/2310.06825)

3. OpenAI. ChatGPT: An AI Language Model for Conversational Applications. ChatGPT is an advanced AI language model developed by OpenAI, designed to understand and generate human-like text based on the input it receives. It has applications in various fields such as customer service, education, content creation, and more. [https://openai.com/index/chatgpt/](https://openai.com/index/chatgpt/)

4. DeepMind. Gemini: Advanced AI for Research and Applications. Gemini, developed by DeepMind, is a cutting-edge AI system designed to tackle complex problems across various domains. It leverages state-of-the-art machine learning techniques to enhance capabilities in research, healthcare, and technology. [https://deepmind.google/technologies/gemini/](https://deepmind.google/technologies/gemini/)

5. Jayakumar, S., Guo, C., Bouchacourt, D., Al-Tahan, H., Padthe, K., Sharma, V., Xu, H., Tan, X.E., Richards, M., Lavoie, S., Astolfi, P., Hemmat, R.A., Chen, J., Tirumala, K., Assouel, R., Moayeri, M., Talattof, A., Chaudhuri, K., Liu, Z., Chen, X., Garrido, Q., Ullrich, K., Agrawal, A., Saenko, K., Celikyilmaz, A., Chandra, V. (2024). An Introduction to Vision-Language Modeling. arXiv preprint arXiv:2405.17247. [https://arxiv.org/abs/2405.17247](https://arxiv.org/abs/2405.17247)

6. Liu, H., Li, C., Wu, Q., Lee, Y.J. Visual instruction tuning. Advances in Neural Information Processing Systems, 36, 34892–34916 (2023). [https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf)

7. Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A.J., Padlewski, P., Salz, D., Goodman, S., Grycner, A., Mustafa, B., Beyer, L., Kolesnikov, A., Puigcerver, J., Ding, N., Rong, K., Akbari, H., Mishra, G., Xue, L., Thapliyal, A., Bradbury, J., Kuo, W., Seyedhosseini, M., Jia, C., Karagol Ayan, B., Riquelme, C., Steiner, A., Angelova, A., Zhai, X., Houlsby, N., Soricut, R. (2023). PaLI: A Jointly-Scaled Multilingual Language-Image Model. arXiv preprint arXiv:2209.06794. [https://arxiv.org/abs/2209.06794](https://arxiv.org/abs/2209.06794)

8. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv preprint arXiv:2103.00020. [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

9. Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J.E., Stoica, I., Xing, E.P. (2023). Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality. [https://lmsys.org/blog/2023-03-30-vicuna/](https://lmsys.org/blog/2023-03-30-vicuna/)

10. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929. [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

11. Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., Barua, A., Raffel, C. (2021). mT5: A massively multilingual pre-trained text-to-text transformer. arXiv preprint arXiv:2010.11934. [https://arxiv.org/abs/2010.11934](https://arxiv.org/abs/2010.11934)

12. Beyer, L., Steiner, A., Pinto, A.S., Kolesnikov, A., Wang, X., Salz, D., Neumann, M., Alabdulmohsin, I., Tschannen, M., Bugliarello, E., Unterthiner, T., Keysers, D., Koppula, S., Liu, F., Grycner, A., Gritsenko, A., Houlsby, N., Kumar, M., Rong, K., Eisenschlos, J., Kabra, R., Bauer, M., Bošnjak, M., Chen, X., Minderer, M., Voigtlaender, P., Bica, I., Balazevic, I., Puigcerver, J., Papalampidi, P., Henaff, O., Xiong, X., Soricut, R., Harmsen, J., Zhai, X. PaliGemma: A versatile 3B VLM for transfer. arXiv preprint arXiv:2407.07726 (2024). [https://arxiv.org/abs/2407.07726](https://arxiv.org/abs/2407.07726)

13. Chen, X., Wang, X., Beyer, L., Kolesnikov, A., Wu, J., Voigtlaender, P., Mustafa, B., Goodman, S., Alabdulmohsin, I., Padlewski, P., Salz, D., Xiong, X., Vlasic, D., Pavetic, F., Rong, K., Yu, T., Keysers, D., Zhai, X., Soricut, R. (2023). PaLI-3 Vision Language Models: Smaller, Faster, Stronger. arXiv preprint arXiv:2310.09199. [https://arxiv.org/abs/2310.09199](https://arxiv.org/abs/2310.09199)

14. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L. (2023). Sigmoid Loss for Language Image Pre-Training. arXiv preprint arXiv:2303.15343. [https://arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343)

15. Gemma Team, Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., Sifre, L., Rivière, M., Kale, M.S., Love, J., Tafti, P., Hussenot, L., Sessa, P.G., Chowdhery, A., Roberts, A., Barua, A., Botev, A., Castro-Ros, A., Slone, A., Héliou, A., Tacchetti, A., Bulanova, A., Paterson, A., Tsai, B., Shahriari, B., Le Lan, C., Choquette-Choo, C.A., Crepy, C., Cer, D., Ippolito, D., Reid, D., Buchatskaya, E., Ni, E., Noland, E., Yan, G., Tucker, G., Muraru, G.-C., Rozhdestvenskiy, G., Michalewski, H., Tenney, I., Grishchenko, I., Austin, J., Keeling, J., Labanowski, J., Lespiau, J.-B., Stanway, J., Brennan, J., Chen, J., Ferret, J., Chiu, J., Mao-Jones, J., Lee, K., Yu, K., Millican, K., Sjoesund, L.L., Lee, L., Dixon, L., Reid, M., Mikuła, M., Wirth, M., Sharman, M., Chinaev, N., Thain, N., Bachem, O., Chang, O., Wahltinez, O., Bailey, P., Michel, P., Yotov, P., Chaabouni, R., Comanescu, R., Jana, R., Anil, R., McIlroy, R., Liu, R., Mullins, R., Smith, S.L., Borgeaud, S., Girgin, S., Douglas, S., Pandya, S., Shakeri, S., De, S., Klimenko, T., Hennigan, T., Feinberg, V., Stokowiec, W., Chen, Y., Ahmed, Z., Gong, Z., Warkentin, T., Peran, L., Giang, M., Farabet, C., Vinyals, O., Dean, J., Kavukcuoglu, K., Hassabis, D., Ghahramani, Z., Eck, D., Barral, J., Pereira, F., Collins, E., Joulin, A., Fiedel, N., Senter, E., Andreev, A., Kenealy, K. (2024). Gemma: Open Models Based on Gemini Research and Technology. arXiv preprint arXiv:2403.08295. [https://arxiv.org/abs/2403.08295](https://arxiv.org/abs/2403.08295)

16. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023). [https://arxiv.org/abs/2304.10592](https://arxiv.org/abs/2304.10592)

17. Chia, P.J., Attanasio, G., Bianchi, F. et al. Contrastive language and vision learning of general fashion concepts. Sci Rep 12, 18958 (2022). [https://doi.org/10.1038/s41598-022-23052-9](https://doi.org/10.1038/s41598-022-23052-9)
    
18. Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X.V., Mihaylov, T., Ott, M., Shleifer, S., Shuster, K., Simig, D., Koura, P.S., Sridhar, A., Wang, T., Zettlemoyer, L. (2022). OPT: Open Pre-trained Transformer Language Models. arXiv preprint arXiv:2205.01068. [https://arxiv.org/abs/2205.01068](https://arxiv.org/abs/2205.01068)

19. Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization. arXiv preprint arXiv:1711.05101. [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)


