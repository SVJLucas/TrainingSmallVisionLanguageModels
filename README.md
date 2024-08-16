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


## Understanding the Dataset Creation Pipeline

Creating high-quality datasets is crucial for training machine learning models, especially in specialized domains like fashion. The dataset creation pipeline illustrated below demonstrates a multi-step process that leverages advanced models like Phi-3 Vision and Phi-3 Instruct to produce refined and structured datasets.

<p align="center">
  <img width="700" alt="Dataset Pipeline" src="https://github.com/user-attachments/assets/ff2f586d-50ac-4c7c-8cd6-327a043d9174">
</p> 

### Step 1: **Generating an Initial Image Description with Phi-3 Vision**

<p align="center">
  <img width="700" alt="Dataset" src="https://github.com/user-attachments/assets/0ca50ae0-5b0c-4129-bcf2-3481b4307234">
</p> 


The first step in the dataset creation pipeline leverages the power of Phi-3 Vision to generate concise and structured descriptions of fashion items directly from images. This process is based on the principle of Datafree distillation, which aims to create synthetic datasets by extracting the most relevant attributes of a fashion item, such as its gender association, category, color, and intended usage. The approach focuses solely on the item itself, deliberately ignoring unnecessary details like the person wearing the item or any visible text within the image. This ensures that the resulting descriptions are both clear and focused, capturing only the essential characteristics of the item. To guide Phi-3 Vision in this task, the item image is combined with a carefully crafted template, shown bellow:

<p align="center">
  <img width="700" alt="Prompt I" src="https://github.com/user-attachments/assets/b1934cc2-9031-4f21-95e6-10778af499db">
</p> 

This prompt instructs the model to analyze the image and produce a brief yet comprehensive description that encapsulates the item's key features. The output is formatted in a simple JSON structure, making it easy to integrate into various data processing or machine learning workflows. For example, Phi-3 Vision might generate a description like "A pink women's blouse with lace trim, perfect for formal occasions," capturing the item's color, gender association, and appropriate usage.

- **Why it Matters:** By generating these initial descriptions using a model licensed under MIT, such as Phi-3 Vision, we can use the outputs as training data for other Vision-Language Models (VLMs). Phi-3 Vision works by extracting visual features from the image and converting them into concise textual descriptions. This process not only ensures that the dataset is highly relevant and accurate, but it also provides a robust foundation for further refinement and enhancement. The transformation of raw image data into structured, machine-readable descriptions is essential for training AI models.

However, **it’s important to note that Phi-3 Vision was initially trained with a strong focus on Optical Character Recognition (OCR)**. For exemple, if we took the image from the training sampled data, as bellow:

<p align="center">
  <img width="400" alt="Sample Item" src="https://github.com/user-attachments/assets/5b05da3c-d9f3-458a-ae67-5fa378836eea">
</p>


Phi-3 Vision will output the following result:


<p align="center">
  <img width="700" alt="Sample Item Description by Phi-3 Vision" src="https://github.com/user-attachments/assets/7c1375e3-d27c-48e6-af6f-76b513ac8c7c">
</p> 



This presents a challenge because the inclusion of text from the image is not desired in our model's outputs. The presence of OCR-derived text could complicate the training process and lead to less accurate descriptions, , especially for a smaller model. To address this, we incorporate an additional step where Phi-3 Instruct is used. This model utilizes metadata to refine the initial descriptions, ensuring that any OCR text inadvertently captured by Phi-3 Vision is removed, thereby aligning the final dataset with our specific goals.


### Step 2: **Refining the Description with Metadata Using Phi-3 Instruct**

<p align="center">
  <img width="1000" alt="Pipeline's Second Part" src="https://github.com/user-attachments/assets/a148bcb2-5ab2-4c76-a8d3-67aee142c919">
</p> 



After generating the initial description with Phi-3 Vision, the process moves into the refinement stage, where the description is combined with additional metadata to create a more precise and comprehensive final output. This is achieved by feeding the initial description and metadata into a second template, "Prompt II," designed to guide Phi-3 Instruct in refining the description, as shown bellow. The metadata includes key details about the item, such as gender, master category, subcategory, article type, and usage, which are crucial for ensuring that the final description is both accurate and contextually relevant, avoiding possible hallucinations from Phi-3 Vision.

<p align="center">
  <img width="700" alt="Prompt II" src="https://github.com/user-attachments/assets/e6db2654-3519-4604-a57d-c7c4cdf8487e">
</p> 

The refinement process carried out by Phi-3 Instruct is essential because it not only enhances the description by integrating these additional data points but also addresses potential issues that may arise from the initial analysis. As mentioned before, Phi-3 Vision was trained with a strong focus on Optical Character Recognition (OCR), which might cause it to include unwanted text from the image in the description. Since this text is not relevant to our dataset and could complicate the training of other Vision-Language Models (VLMs), Phi-3 Instruct plays a vital role in filtering out any OCR-derived content. By doing so, it ensures that the final dataset is free from unnecessary elements and fully aligned with the intended use case.

- **Why it Matters:** This refinement step is crucial because it ensures that the final dataset is both detailed and aligned with the necessary categorizations. By removing any irrelevant OCR text and integrating metadata through sophisticated refinement techniques, the resulting dataset becomes a powerful tool for advanced machine learning applications. This meticulous approach guarantees that the dataset is clean, focused, and ready for use in training AI models, thereby enhancing their accuracy and effectiveness.

### Step 3: **Creating the Final Dataset**

The culmination of this meticulous process is the creation of a final dataset that pairs each refined item description with its corresponding image. By integrating the outputs from Phi-3 Vision and Phi-3 Instruct, we ensure that each fashion item is represented with both visual and textual precision. The images provide the necessary visual context, while the descriptions, refined through the use of metadata and advanced processing techniques, encapsulate the most relevant features of each item in a concise and structured format.

- **Why it Matters:** The significance of this final dataset cannot be overstated. It serves as a robust and versatile resource that can be leveraged across various data-driven applications in the fashion industry, from training AI models to enhancing product recommendation systems and streamlining catalog management. By ensuring that the dataset is both accurate and comprehensive, this step guarantees that the insights and efficiencies gained from the earlier stages of the process are fully realized in practical, real-world scenarios. Ultimately, this high-quality dataset is a cornerstone for driving innovation and improving performance in a competitive and data-centric market.




## MODA Architecture

MODA (Multimodal Object Description Assistant) uses a specialized architecture to generate detailed descriptions of fashion items by combining FashionCLIP[17] for image encoding and OPT-125M[18] as decoder for text generation. Its non-linear projection increases and then reduces the dimensionality of image embeddings, capturing complex patterns and enhancing representation quality. This approach allows MODA to deliver accuracy in fashion-specific tasks with only 280 million parameters, making it efficient and capable of running without a GPU.

<p align="center">
  <img width="900" alt="Architecture" src="https://github.com/user-attachments/assets/ab91f5f9-60e3-462e-b759-8dce3f4fa53c">
</p> 


In comparison, models like LLaVA[6] and PaLI[7] adopt a more versatile approach, leveraging pretrained backbones to align visual and textual modalities for a wide range of tasks. LLaVA[6] integrates CLIP[8] and Vicuna[9] with a simple linear projection for efficient resource utilization, while PaLI combines Vision Transformers (ViT)[10] and mT5[11] models also via linear transformation for the same reason. Since MODA is a small model, the use of non-linearity does not impact resource utilization significantly. MODA’s non-linear projection method provides a distinct advantage in capturing intricate visual details, enhancing its performance in generating accurate and detailed descriptions compared to the linear projections used by other models.


## Training

The training of the MODA model aims to optimize its ability to generate detailed descriptions of fashion items by effectively integrating visual and textual data. The process starts with the FashionCLIP[17] model, which processes the input image to generate an embedding. This embedding is then projected to a suitable size through a non-linear projection layer, enhancing its representation quality. Simultaneously, the OPT[18] tokenizer converts the text description (image label) into token embeddings. These image and text embeddings are combined into a sequence and fed into the OPT-125M[18] model, a pre-trained transformer, to predict the next token in the sequence. Then, the model's prediction performance is evaluated using the CrossEntropy loss function.

<p align="center">
  <img width="1000" alt="Training Scheme" src="https://github.com/user-attachments/assets/55543651-b762-45a7-8750-ec4469bf77d5">
</p> 


The training process uses the AdamW[19] optimizer, which starts with a learning rate of 1e-3. To ensure the training is efficient, the learning rate is periodically reduced, dropping by a fixed percentage at regular intervals. During training, similar to LLaVA[6], the FashionCLIP[17] encoder had its parameters frozen, and only the non-linear projection and the language model were trained. The model is trained for 40 epochs, taking 5 hours and 54 minutes to train on a Google Colab A100, with gradient accumulation and periodic model saving to ensure stability and performance.

<p align="center">
  <img width="700" alt="Training Chart" src="https://github.com/user-attachments/assets/6f25ee6b-5a49-4f24-abde-d1df29247f76">
</p> 

The training chart shows a sharp decline in both training and validation losses in the initial steps, indicating rapid learning and convergence. As training progresses, losses continue to decrease and stabilize at lower values, demonstrating effective learning and good generalization to validation data. The close alignment between the training (blue line) and validation (orange line) loss curves suggests that the model is not overfitting and maintains good performance on the validation set.

## Evaluation

Below, we present some sample descriptions generated by MODA on the test set. These examples showcase the model's ability to interpret and describe fashion items, highlighting its strengths in identifying various attributes such as color, style, category, and gender. Each generated description is a reflection of the model’s understanding of the visual features and its ability to translate them into coherent and contextually relevant text. The subsequent analysis will delve into how effectively MODA performs across a range of metrics designed to measure its descriptive capabilities.

<p align="center">
  <img width="1200" alt="Test Samples" src="https://github.com/user-attachments/assets/a66a7e7f-7a6d-4050-bd76-3e6b7e49794c">
</p> 

Following the generation of these descriptions, a comprehensive evaluation framework was applied to assess MODA’s performance. The model was evaluated across five critical metrics: **Color Accuracy**, **Style Recognition**, **Category Accuracy**, **Gender Accuracy**, and **Factual Consistency**. This framework ensures that the model produces not only accurate but also contextually relevant descriptions, which are essential for a wide range of fashion-related applications. The evaluation was conducted by **GPT-4o mini**, serving as an impartial assessor, and focused on analyzing MODA's performance on the test set images. The detailed methodology and scoring system used in this evaluation are outlined below:



### 1. **Color Accuracy**

**Definition**: The accuracy with which the model identifies the colors present in a fashion item. This metric is vital for the correct representation of the item's visual characteristics.

**Scoring Methodology**: 
- The model is scored on a scale from 0 to 10 based on the ratio of correctly identified colors to the total number of colors in the item.

  <p align="center">
    <img width="300" alt="Training Chart" src="https://github.com/user-attachments/assets/b1a113d9-b2eb-444c-ba02-38f6d3fb9763">
  </p> 

- For instance, if the item contains three colors and the model accurately identifies two, the score would be calculated as follows:
  <p align="center">
    <img width="180" alt="Training Chart" src="https://github.com/user-attachments/assets/bbf505a3-5d4e-4576-9c08-4739ea62a0d4">
  </p> 


### 2. **Style Recognition**

**Definition**: This metric evaluates the model's ability to correctly identify the fashion style of an item, such as whether it is casual, formal, or sporty. Accurate style recognition is crucial for categorizing fashion items within their intended use and audience.

**Scoring Methodology**: 
- A binary scoring system is employed:
  - **10** if the style is correctly identified.
  - **0** if the style is incorrectly identified.

### 3. **Category Accuracy**

**Definition**: Category Accuracy measures the model’s performance in identifying the correct category of the fashion item, such as whether it is classified as a dress, shirt, or pants. This metric is essential for accurate item categorization within a fashion dataset.

**Scoring Methodology**: 
- Similar to Style Recognition, this metric uses binary scoring:
  - **10** if the category is correctly identified.
  - **0** if the category is incorrectly identified.

### 4. **Gender Accuracy**

**Definition**: This metric assesses the model's ability to accurately determine the intended gender for the fashion item, such as whether it is designed for men, women, or is unisex. Accurate gender identification is vital for appropriate product placement and targeting in the fashion industry.

**Scoring Methodology**: 
- Binary scoring is also applied here:
  - **10** if the gender is correctly identified.
  - **0** if the gender is incorrectly identified.

### 5. **Factual Consistency**

**Definition**: Factual Consistency evaluates the model’s accuracy in mentioning relevant details while avoiding hallucinations—incorrect or irrelevant details not present in the image. This metric is critical for ensuring the trustworthiness of the model's outputs.

**Scoring Methodology**: 
- The score is determined by the proportion of relevant details to the total details mentioned by the model.

  <p align="center">
    <img width="300" alt="Training Chart" src="https://github.com/user-attachments/assets/1de4bb10-a998-49cc-b7d7-7ee76a824f4f">
  </p> 

- A score of **10** indicates that no irrelevant details were mentioned, ensuring the output's factual integrity, while a score of **0** indicates that all mentioned details were irrelevant.

The overall grade represents a cumulative assessment of the model's performance across all five metrics. This aggregate score provides a holistic evaluation of the model's capability to accurately and effectively describe fashion items, ensuring both precision and relevance in its outputs. The evaluation across all metrics can be seen in the chart below:

<p align="center">
  <img width="800" alt="Training Chart" src="https://github.com/user-attachments/assets/99588c47-053e-4c02-9dce-431914674f5d">
</p> 

This chart illustrates the evolution of the model's performance across several key evaluation metrics as it processes increasing amounts of data, measured in millions of tokens. Metrics such as **Color Accuracy**, **Style Recognition**, **Category Accuracy**, and **Gender Accuracy** exhibit rapid improvement within the first few million tokens, stabilizing at high levels of accuracy early in the training process. These results suggest that the model is particularly adept at learning to identify colors, styles, categories, and gender-related attributes quickly and consistently, with Gender Accuracy reaching near-perfect performance almost immediately.

However, **Factual Consistency** shows a more gradual and variable improvement, indicating that the model had more difficulty maintaining accuracy in detail relevance, with fluctuations suggesting ongoing challenges in reducing hallucinations. The Overall Grade follows the general trend of individual metrics, rising rapidly before stabilizing, reflecting solid overall performance. To address the variability in Factual Consistency, further refinement could involve training the model with the entire dataset or increasing the number of tokens per image, similar to approaches used in models like InternLM-XComposer2-4KHD. This could enhance the model's ability to maintain consistent accuracy in detailed descriptions.

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


