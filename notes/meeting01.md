# Meeting 25.01.2024

present Ilpo, Valtteri and Timo

## Project ideas

1. Deep fake detection from video sequence
    - Possibly also audio?
    - Deepfake detection where most of the frames are real but some are fake!
    - Articles:
        - Dataset: <https://github.com/ControlNet/AV-Deepfake1M>
        - AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset (<https://arxiv.org/abs/2311.15308>)
        - DF-TransFusion: Multimodal Deepfake Detection via Lip-Audio Cross-Attention and Facial Self-Attention (<https://arxiv.org/abs/2309.06511>)
        - Joint Audio-Visual Attention with Contrastive Learning for More General Deepfake Detection (<https://andor.tuni.fi/permalink/358FIN_TAMPO/176jdvt/cdi_crossref_primary_10_1145_3625100>)

2. Operational Transformer
    - Switch normal neurons to operational neurons in a transformer. Some, all, whatever?
    - NLP task

3. Predicting the electricity price
   - Use ONNs for this
   - Try to find a mix of features to use to predict the data
   - Combining multiple datasets (on time axis)
   - Data: https://www.fingrid.fi/

   - Articles:
     - Spot price prediction with feature extraction, and hybrid models https://www.sciencedirect.com/science/article/pii/S037877962100434X
     - Transfer Learning vois olla hyvä lähestymistapa alkuun jos halutaan viljellä Op.Perceptroneita https://www.sciencedirect.com/science/article/pii/S2352467723000048
     - A two-stage supervised learning approach for electricity price forecasting by leveraging different data sources https://www.sciencedirect.com/science/article/pii/S0306261919305380?ref=pdf_download&fr=RR-2&rr=84d3db3f5a6c70fc
     - Yleisesti mietin: Miks ML ennustamista ei rakennella usean aiheeseen liittyvän datasetin varaan, vaan usein 1 in 1 out tyyppinen. Oisko tää just Hybrid modelin heiniä, et katotaan yksittäisten datasettien ennusteiden mukaan joku haluttu erillinen ennuste?
     - Sähkössä just: suhdeluvut sisäänoston, eri tuotantomuotojen, ja kulutuksen mukaan, sääennuste  (vesi tuuli ja kulutus), ydinenergian huoltosammutukset

## Transformer resources

1. "Original" transformer paper: <https://arxiv.org/abs/1706.03762>

2. Andrew Karpahty minGPT
    - <https://github.com/karpathy/minGPT>
    - <https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy>

3. Nice summarisation: <https://arxiv.org/pdf/2401.02038.pdf>

4. Nice illustrations: <https://jalammar.github.io/illustrated-transformer/>

5. Nice blog post by Lilian Weng: <https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html>

6. The transformers family v2: <https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/>

## Next steps

Find a few papers related to each topic.

Topics: 1. Ilpo 2. Timo 3. Valtteri
