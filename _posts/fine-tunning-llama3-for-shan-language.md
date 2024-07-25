---
tags: ["llama" ,"shan" ,"LLM" ,"generative-ai", "finetune"]
date: July 25, 2024
title: Fine-Tuning Llama3 Large Language Model for Shan language - NLLB + Quantized + Ollama
subtitle: Instruct note to fine-tuning Llama3 for Shan language with NLLB translation datasets, Quantization and Ollama
image: https://i.pinimg.com/564x/9e/49/01/9e49015a36c3effbf97b3707ad944d9b.jpg
link: blog/fine-tuning-llama-for-shan-language
description: ၸၢမ်းႁဵတ်းႁႂ်ႈ llama3 ပွင်ႇၸႂ်လိၵ်ႈတႆး။
---

Thank to AI-Commandos [LLaMa2lang convenience scripts](https://github.com/AI-Commandos/LLaMa2lang) to finetune (chat-)LLaMa3

ပွင်ႈၵႂၢမ်းႁူဝ်ၼႆႉ ပဵၼ်မၢႆတႃႇတွင်း လွင်ႈလဵပ်ႈႁဵၼ်းတူၺ်း တႃႇတေႁဵတ်းႁႂ်ႈ LLM (Large Language Model) မိူၼ်ၼင်ႇ Llama3 ႁူႉပွင်ႇၸႂ် လႄႈတွပ်ႇၶေႃႈထၢမ်လိၵ်ႈတႆး ၵႂၢမ်းတႆးလႆႈ။

တွၼ်ႈတႃႇလိၵ်ႈတႆး ၵႂၢမ်းတႆး ယင်းတိုၵ်ႉမီးလွင်ႈၶဵင်ႇတႃႉ လႄႈလွင်ႈလူဝ်ႇလဵပ်ႈႁဵၼ်းထႅင်ႈတင်းၼမ် တႃႇတေၶဵၼ်ႇၽိုတ်ႉဢီး ႁဵတ်းႁႂ်ႈလိၵ်ႈတႆး ၵႂၢမ်းတႆးႁဝ်းၸႂ်ႉလႆႈလီၼႂ်းၵၢပ်ႈပၢၼ် AI ၼႆလႄႈ ၼႆႉပဵၼ်ပွင်ႈၵႂၢမ်းမၢႆတွင်း (Study Note) တႃႇလဵပ်ႈႁဵၼ်းတူၺ်းပွတ်းဢွၼ်ႇတွၼ်ႈၼိုင်ႈၵူၺ်းၶႃႈ။

## Why Llama

Llama (Large Language Model Meta AI) ပဵၼ် generative ai မေႃႇတႄႇလ် ဢၼ်ၼိုင်ႈ ဢၼ် Meta (Facebook) ၶဝ်ၶူင်သၢင်ႈဢွၵ်ႇမႃးဝႆႉ လႄႈပဵၼ် မေႃႇတႄႇလ်ဢၼ်ပိုတ်ႇငဝ်ႈတိုၼ်း (Open-Source) ႁႂ်ႈဢဝ်ၸႂ်ႉတိုဝ်းလႆႈၵႂၢင်ႈၵႂၢင်ႈၶႂၢင်ၶႂၢင် ႁူမ်ႈတင်းႁဵတ်း Fine-tunnine လႆႈ။

Llama ပဵၼ်မေႃႇတႄႇလ်ဢၼ်ၼိုင်ႈဢၼ်လႆႈဝႃႈၵတ်ႉၶႅၼ်ႇ ၼႂ်းၵႄႈၵၢင်မေႃႇတႄႇလ်ဢၼ်ဢမ်ႇပိုတ်ႇငဝ်ႈတိုၼ်းတၢင်ႇဢၼ် မိူၼ်ၼင်ႇ GPT3, GPT4, Gemini, Claud.ai ၸိူဝ်းၼႆႉ။

Llama ပဵၼ် multilingual pre-trained model ဢၼ်ၸႂ်ႉၶေႃႈမုၼ်းယႂ်ႇလူင် လႄႈႁႅင်းတိုၼ်းငိုၼ်းၼမ် တႃႇတေႁဵတ်းဢွၵ်ႇပၼ်မႃး checkpoint ဢၼ်ဢဝ်ၵႂႃႇၸႂ်ႉလႆႈလၢႆလၢႆတီႈ မိူၼ်ၼင်ႇ LangChain, Ollama။

ၵူၺ်းၵႃႈဝႃႈ တွၼ်ႈတႃႇလိၵ်ႈတႆးတႄႉ လႆႈဝႃႈၸဵမ်ႁႅင်းၶေႃႈမုၼ်း လႄႈလွင်ႈၵိုင်ႇတၢၼ်ႇတႃႇတေႁဵတ်းဢွၵ်ႇပႆႇတဵမ်ထူၼ် လႄႈ မေႃႇတႄႇလ်တင်းၼမ်ဢၼ်ဢွၵ်ႇမႃးၼၼ်ႉၵေႃႈ ပႆႇႁူႉပွင်ႇၸႂ်လႆႈလိၵ်ႈတႆးလီလီၼၼ်ႉယဝ်ႉ။

## Low-resources Language

ၽႃႇသႃႇလိၵ်ႈလၢႆးၼႂ်းၵမ်ႇၽႃႇ ဢမ်ႇၵွမ်ႉၵႃႈလိၵ်ႈတႆးၵူၺ်း လၢႆလၢႆၽႃႇသႃႇ လႆႈထုၵ်ႇၼပ်ႉဝႃႈပဵၼ် ၽႃႇသႃႇဢၼ်မီးၶေႃႈမုၼ်းၵမ်ႉဢေႇ (Low-Resources Language) ၼၼ်ႉပွင်ႇဝႃႈ ၶေႃႈမုၼ်းလိၵ်ႈလၢႆ ၼမ်ႉၵႂၢမ်း သဵင်လၢတ်ႈ ဢၼ်ပဵၼ် digital-format လႄႈၽွမ်ႉၸႂ်ႉၼႂ်းၶၵ်ႉၵၢၼ် NLP လႄႈ Machine-Learning, Deep-Learning training ၼၼ်ႉဢမ်ႇပႆႇမီးၼမ်ပဵင်းပေႃး။

ၶၵ်ႉၵၢၼ် NLP လႄႈ Machine-Learning, Deep-Learning training ၸိူဝ်းၼၼ်ႉပဵၼ်ၶၵ်ႉၵၢၼ်ဢၼ် တွင်ႉမႆႈၶေႃႈမုၼ်း (Data hungry) တႃႇပွၼ်ႈသွၼ်ပၼ်မၼ်းတင်းၼမ် ဢၼ်ႉၵႆႉႁွင်ႉဝႃႈ large-scale datasets ၶေႃႈၼႆႉတေပိူင်ႈၵၼ်တင်းၶေႃႈၵႂၢမ်းဢၼ်ၵႆႉၺိၼ်းမိူဝ်ႈပူၼ်ႉမႃးဝႃႈ Big-Data ("3 Vs": Volume, Variety, and Velocity.) ၵူၺ်းၶေႃႈမုၼ်းတွၼ်ႈတႃႇ training datasets ၼႆႉလိူဝ်သေလူဝ်ႇမီးတၢင်းယႂ်ႇၼမ်ယဝ်ႉယင်းလူဝ်ႇမီးထႅင်ႈ quality, diversity, and relevance။

## Why and What is Fine-tuning

ၼင်ႇဝႃႈမႃးၼႂ်းတွၼ်ႈ Low-resources language ၼၼ်ႉယဝ်ႉ ပေႃးဝႃႈဢမ်ႇမီးၶေႃႈမုၼ်းတီႈလီ လႄႈၶိုၵ်ႉယႂ်ႇၼၼ်ႉ မေႃႇတႄႇလ်ဢၼ်ဢွၵ်ႇမႃးၼၼ်ႉၵေႃႈ တိုၼ်းဝႃႈတေဢမ်ႇၶိုၵ်ႉၶႅမ်ႈၸႂ်ႉလႆႈ။

လိူဝ်သေလူဝ်ႇၶေႃႈမုၼ်းၶိုၵ်ႉယႂ်ႇယဝ်ႉ တႃႇတေ train deep-learning model သေဢၼ်ဢၼ်တႄႇတီႈငဝ်ႈမၼ်းၼၼ်ႉ လႆႈၸႂ်ႉၶၢဝ်းယၢမ်း လႄႈငိုၼ်းလူင်းတိုၼ်းၼမ် ၵႃႈၶၼ်ယႂ်ႇ၊ ယွၼ်ႉၼၼ်လႄႈ ၸင်ႇမီးလွၵ်းလၢႆးဢၼ်ႁွင်ႉဝႃႈ Fine-Tune ၼၼ်ႉမႃး။

Fine-Tune ၼၼ်ႉပဵၼ်လွၵ်းလၢႆးၼိုင်ႈ ဢၼ်ႁွင်ႉဝႃႈ transfer-learning ၵၢၼ်သိုပ်ႇသူင်ႇတၢင်းႁူႉ ဢၼ် pre-trained မေႃႇတႄႇလ်ၼၼ်ႉလႆႉထုၵ်ႇၾိုၵ်းသွၼ် ႁဵၼ်းႁူႉမႃး၊ မိူၼ်ၼင်ႇ Llama, GPT3, GPT4 ၸိူဝ်းၼၼ်ႉ လုၵ်ႉတီႈၽူႈၶူင်သၢင်ႈၶဝ်သေ ၸႂ်ႉတင်းတိုၼ်းလၢင်း ၶၢဝ်းယၢမ်း လႄႈၶေႃႈမုၼ်းၶိုၵ်ႉယႂ်ႇ တူဝ်ႈဢိၼ်ႇတႃႇၼႅတ်ႉတင်းလုမ်ႈၾႃႉ ၾိုၵ်းသွၼ်မႃးဝႆႉယဝ်ႉ၊ ၵၢၼ် fine-tune မေႃႇတႄႇလ်ၵေႃႈမိူၼ်ၼင်ႇၵၢၼ် ၶိုၼ်းလုပ်ႈၶျေႃး ပွတ်ႈလပ်ႉထႅင်ႈႁႂ်ႈမၼ်းႁဵၼ်းႁူႉ ၸွမ်းၼင်ႇၶေႃႈမုၼ်းႁဝ်းမီးထႅင်ႈၼၼ်ႉယဝ်ႉ။

ယွၼ်ႉၼၼ်လႄႈ တွၼ်ႈတႃႇၽႃႇသႃႇလိၵ်ႈလၢႆးတႆးႁဝ်း မိူဝ်ႈဢၼ်ပႆႇမီးတိုၼ်းလၢင်း လႄႈၶေႃႈမုၼ်းၶိုၵ်ႉယႂ်ႇ တႃႇပွၼ်ႈသွၼ်ပၼ်မၼ်းၼၼ်ႉ လွၵ်းလၢႆး Fine-tune တေပဵၼ်လၢႆးဝႆး လႄႈၵိုင်ႇငၢမ်ႇၸွမ်းငဝ်းလၢႆးယဝ်ႉ။

## Datasets and Fine-tune processes

> Code လႄႈလွၵ်းလၢႆး fine-tune တွၼ်ႈၼႆႉၸႂ်ႉတိုဝ်း [AI-Commandos/LLaMa2lang Convenience scripts](https://github.com/AI-Commandos/LLaMa2lang) ဢၼ်တူင်ႇဝူင်းၸွႆႈၵၼ်ပိုၼ်ၽႄဝႆႉ ပိူဝ်ႈတႃႇ Optimize လႄႈ လႆႈၼမ်ႉတွၼ်းၼႂ်းမေႃႇတႄႇလ်သုင်သုတ်း။
>
> Code ဢၼ်ႁၢင်ႈႁႅၼ်းဝႆႉတွၼ်ႈတႃႇၽႃႇသႃႇတႆး [https://github.com/NoerNova/LLaMa2lang.git](https://github.com/NoerNova/LLaMa2lang.git)

ၶေႃႈမုၼ်းဢၼ်တႃႇတေၸႂ်ႉၼႂ်းၶၵ်ႉတွၼ်ႈၵၢၼ် fine-tune Llama ဢမ်ႇၼၼ်တီႈၼႆႈပဵၼ် Chat-Llama ၼၼ်ႉ ႁဝ်းလူဝ်ႇၶေႃႈမုၼ်း ထၢမ်-တွပ်ႇ ပိူဝ်ႈတႃႇႁႂ်ႈမၼ်းႁဵၼ်းႁူႉလႆႈဝႃႈ ပေႃးမီးၶေႃႈထၢမ် မၼ်းတေလႆႈတွပ်ႇၸိူင်ႉႁိုဝ်။

(သူၼ်ၸႂ်လဵပ်ႈႁဵၼ်း - [Fine-tune GPT2 for Shan text-generator](blog/fine-tuning-gpt2-for-shan-language))

![Sample datasets {caption: prompter-assistant dataset}](/assets/fine-tuning-llama-for-shan-language/Screenshot-2567-07-25-at-02.19.07.png)

![Sample datasets {caption: prompter-assistant dataset}](/assets/fine-tuning-llama-for-shan-language/Screenshot-2567-07-25-at-02.19.42.png)

ၼင်ႇၶေႃႈမုၼ်းၽၢႆႇၼိူဝ်ၼၼ်ႉ ပေႃးၽူႈထၢမ် (prompter) လၢတ်ႈဝႃႈ "မႂ်ႇသုင်ၶႃႈ" ၼႆ assistant တေလႆႈတွပ်ႇဝႃႈၸိူင်ႉႁိုဝ် ၼႆၼၼ်ႉယဝ်ႉ။

ၶေႃႈမုၼ်းၸိူင်ႉၼႆၼႆႉ လႆႈၸႂ်ႉၶၢဝ်းယၢမ်းလႄႈလွင်ႈသိုပ်ႇႁႃၶေႃႈမုၼ်းတင်းၼမ် တႃႇတေလႆႈၶေႃႈ ထၢမ်-တွပ်ႇ ဢၼ်ပဵၼ်ၶေႃႈမုၼ်းမၢၼ်ႇမႅၼ်ႈ လႄႈလွင်ႈထတ်းတူဝ်ၽိတ်းထုၵ်ႇမၼ်း၊ ၶေႃႈမုၼ်းဢၼ်ပဵၼ်ၽႃႇသႃႇတႆးၵေႃႈ တိုၵ်ႉယူႇၼႂ်းၶၵ်ႉတွၼ်ႈၵၢၼ်ၵဵပ်းႁွမ်း လႄႈဢမ်ႇပႆႇမီးလွင်ႈၶိုၵ်ႉၼမ်။

ၵွပ်ႈၼၼ်ၼႂ်းတွၼ်ႈၼႆႉ ႁဝ်းတေၸႂ်ႉလွၵ်ႉလၢႆးၵၢၼ်ပိၼ်ႇၽႃႇသႃႇၸုမ်ႇၶေႃႈမုၼ်း ဢၼ်ၸိုဝ်ႈဝႃႈ [OASST1 (OpenAssistant)](https://huggingface.co/datasets/OpenAssistant/oasst1) ဢၼ်ဢိင်ၼိူဝ် AI model [NLLB - သိုပ်ႇလူ](blog/meta-NLLB-shan-machine-translations) ၵၢၼ်ပိၼ်ႇၽႃႇသႃႇဢၼ်ဢွၵ်ႇမႃးဢွၼ်တၢင်းၼၼ်ႉ။

### 0. Pre-requirements

1. ၵၢၼ် fine-tune LLM model ၼႆႉၸႂ်ႉတိုဝ်းႁႅင်း computational တင်းၼမ်လႄႈ သင်ဝႃႈၸႂ်ႉတိုဝ်း Graphic card မိူၼ်ၼင်ႇ Nvidia GPU ဢၼ်မီး VRAM လႄႈ CUDA တေတိူဝ်းဝႆး လႄႈလီလိူဝ်။
2. ၸႂ်ႉတိုဝ်း python venv ဢမ်ႇၼၼ် anaconda, miniconda တေႁဵတ်းႁႂ်ႈ setup fine-tune environment လႆႈငၢႆႈ။
3. install [pytorch - https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) use of CUDA preferable.

### 1. Clone project and install requirements

```bash
git clone https://github.com/NoerNova/LLaMa2lang.git
```

```bash
pip install -r requirements.txt
```

### 2. Translate base dataset

တီႈၼႆႈႁဝ်းတေၸႂ်ႉတိုဝ်းမေႃႇတႄႇလ် NLLB တႃႇပိၼ်ႇၽႃႇသႃႇၸုမ်ႇၶေႃႈမုၼ်း OASST1 လႄႈ ၼႂ်း translators/nllb.py ၼၼ်ႉထႅမ်သႂ်ႇပၼ်ပႃး